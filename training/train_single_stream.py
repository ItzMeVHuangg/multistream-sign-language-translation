import sys
import os
import argparse
import yaml
import time
from pathlib import Path
import multiprocessing
import math
import tempfile

# Fix CUDA OOM fragmentation: must be set before any CUDA allocation
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
# Avoid /dev/shm issues on shared nodes by redirecting torch shared memory
os.environ.setdefault('TORCH_SHM_DIR', str(Path(tempfile.gettempdir()) / 'torch_shm'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import PhoenixDataset, collate_fn, build_vocabularies, build_transforms
from models.stream1_rgb_ustm import Stream1_RGB_USTM
from models.stream2_hypergraph import Stream2_Hypergraph
from models.stream3_crossattn import Stream3_CrossAttn
from models.bilstm_ctc import CTCCriterion
from utils.metrics import compute_wer
from training.train_cslr import set_seed, _worker_init_fn, get_cosine_schedule_with_warmup, PhoenixWithKeypoints, collate_fn_kpts

# Safer multiprocessing settings for HPC nodes with limited /dev/shm
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except (RuntimeError, ValueError):
    pass
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

def build_single_optimizer(model: nn.Module, cfg: dict, stream_id: int) -> AdamW:
    tc       = cfg['three_stream']
    base_lr  = tc['learning_rate']
    wd       = tc['weight_decay']

    if stream_id == 1:
        s1_cfg = cfg.get('stream1', {})
        base_lr = s1_cfg.get('learning_rate', base_lr)
        wd = s1_cfg.get('weight_decay', wd)
        swin_sc = s1_cfg.get('swin_lr_scale', tc.get('swin_lr_scale', 0.05))
        optim_name = str(s1_cfg.get('optimizer', 'AdamW')).lower()
        if s1_cfg.get('single_lr', False):
            if optim_name == 'sgd':
                return SGD(
                    model.parameters(),
                    lr=base_lr,
                    momentum=s1_cfg.get('momentum', 0.1),
                    nesterov=s1_cfg.get('nesterov', False),
                    weight_decay=wd,
                )
            if optim_name == 'adam':
                return Adam(model.parameters(), lr=base_lr, weight_decay=wd)
            return AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
    else:
        swin_sc = tc.get('swin_lr_scale', 0.05)

    if stream_id == 2:
        s2_cfg = cfg.get('stream2', {})
        s2_lr = s2_cfg.get('learning_rate', cfg.get('cslr', {}).get('learning_rate', base_lr))
        s2_wd = s2_cfg.get('weight_decay', wd)
        return AdamW(model.parameters(), lr=s2_lr, weight_decay=s2_wd)

    swin_ids = set()
    
    if hasattr(model, 'swin_patch_embed'): # Stream 1
        swin_modules = [model.swin_patch_embed, model.swin_pos_drop,
                        model.swin_stage1, model.swin_merge1,
                        model.swin_stage2, model.swin_merge2,
                        model.swin_stage3, model.swin_merge3,
                        model.swin_stage4, model.swin_norm]
    elif hasattr(model, 'swin'): # Stream 3
        swin_modules = [model.swin]
    else:
        swin_modules = []

    for m in swin_modules:
        swin_ids.update(id(p) for p in m.parameters())

    swin_params, rest_params = [], []
    seen_ids = set()
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        pid = id(p)
        if pid in seen_ids: continue
        seen_ids.add(pid)
        
        if pid in swin_ids:
            swin_params.append(p)
        else:
            rest_params.append(p)

    param_groups = [
        {'params': swin_params, 'lr': base_lr * swin_sc, 'name': 'swin'},
        {'params': rest_params, 'lr': base_lr,            'name': 'rest'},
    ]
    if stream_id == 1 and str(cfg.get('stream1', {}).get('optimizer', 'AdamW')).lower() == 'adam':
        return Adam(param_groups, weight_decay=wd)
    return AdamW(param_groups, weight_decay=wd)


def _stream2_lr(cfg: dict) -> float:
    tc = cfg['three_stream']
    return cfg.get('stream2', {}).get(
        'learning_rate',
        cfg.get('cslr', {}).get('learning_rate', tc['learning_rate']),
    )


def _stream1_lrs(cfg: dict) -> tuple[float, float]:
    tc = cfg['three_stream']
    s1_cfg = cfg.get('stream1', {})
    rest_lr = s1_cfg.get('learning_rate', tc['learning_rate'])
    swin_scale = s1_cfg.get('swin_lr_scale', tc.get('swin_lr_scale', 0.05))
    return rest_lr * swin_scale, rest_lr


def _reset_stream1_resume_lrs(optimizer, scheduler, cfg: dict):
    swin_lr, rest_lr = _stream1_lrs(cfg)
    for group in optimizer.param_groups:
        name = group.get('name', '')
        lr = swin_lr if name == 'swin' and len(optimizer.param_groups) > 1 else rest_lr
        group['lr'] = lr
        group['initial_lr'] = lr
    if hasattr(scheduler, 'base_lrs'):
        scheduler.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
    if hasattr(scheduler, '_last_lr'):
        scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
    print(f"  [resume] Stream 1 optimizer LR reset to swin={swin_lr:.3e}, rest={rest_lr:.3e}")


def _stream1_protocol_signature(cfg: dict) -> dict:
    return {
        'frames_root': cfg.get('paths', {}).get('frames_root'),
        'optimizer': cfg.get('stream1', {}).get('optimizer'),
        'single_lr': cfg.get('stream1', {}).get('single_lr'),
        'learning_rate': cfg.get('stream1', {}).get('learning_rate'),
        'conv_ctc_weight': cfg.get('stream1', {}).get('conv_ctc_weight'),
        'seq_ctc_weight': cfg.get('stream1', {}).get('seq_ctc_weight'),
        'decode_mode': cfg.get('stream1', {}).get('decode_mode'),
    }


def _reset_stream2_resume_lrs(optimizer, scheduler, cfg: dict):
    lr = _stream2_lr(cfg)
    for group in optimizer.param_groups:
        group['lr'] = lr
        group['initial_lr'] = lr
    if hasattr(scheduler, 'base_lrs'):
        scheduler.base_lrs = [lr for _ in scheduler.base_lrs]
    if hasattr(scheduler, '_last_lr'):
        scheduler._last_lr = [lr for _ in scheduler._last_lr]
    print(f"  [resume] Stream 2 optimizer LR reset to {lr:.3e}")

def train_single_stream(cfg: dict, stream_id: int, resume_path: str = None):
    seed = cfg.get('seed', 42)
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Stream {stream_id}] Device: {device}')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    paths = cfg.get('paths', {})
    frames_root = Path(paths.get('frames_root')) if paths.get('frames_root') else None
    annot_root = Path(paths.get('annot_root')) if paths.get('annot_root') else None

    gloss_vocab, text_vocab = build_vocabularies(annot_root=annot_root)
    num_classes = len(gloss_vocab)

    dc = cfg['data']
    tc = cfg['three_stream']
    kpts_root = Path(cfg['paths']['mediapipe_kpts_root'])
    max_fr = dc['max_frames']
    kp_dim = cfg.get('stream2', {}).get('keypoint_dim', 225)

    cache_root = cfg.get('paths', {}).get('frame_cache_root') or os.environ.get('PHOENIX_FRAME_CACHE')
    cache_root = Path(cache_root) if cache_root else None
    if cache_root:
        print(f"[Cache] Frame cache root: {cache_root}")

    def make_ds(split):
        aug = dc.get('augmentation', {})
        temporal_scale = aug.get('temporal_scale', {})
        temporal_scale_range = None
        clip_aug_crop = None
        if split == 'train':
            if temporal_scale:
                temporal_scale_range = (
                    temporal_scale.get('min_scale', 0.8),
                    temporal_scale.get('max_scale', 1.2),
                )
            if aug.get('random_crop', False):
                clip_aug_crop = (dc['img_height'], dc['img_width'])

        base = PhoenixDataset(
            split=split,
            gloss_vocab=gloss_vocab,
            text_vocab=text_vocab,
            max_frames=max_fr,
            temporal_stride=dc['temporal_stride'],
            transform=build_transforms(split, dc['img_height'], dc['img_width']),
            return_translation=False,
            temporal_scale_range=temporal_scale_range,
            clip_aug_crop=clip_aug_crop,
            img_h=dc['img_height'],
            img_w=dc['img_width'],
            cache_root=cache_root,
            cache_read=True,
            cache_write=True,
            frames_root=frames_root,
            annot_root=annot_root,
        )
        # Load frames for streams 1 & 3, keypoints for 2 & 3
        load_frames = (stream_id in [1, 3])
        load_kpts   = (stream_id in [2, 3])
        return PhoenixWithKeypoints(base, kpts_root, max_fr, kp_dim, load_frames=load_frames, load_kpts=load_kpts)

    train_ds = make_ds('train')
    dev_ds   = make_ds('dev')

    g = torch.Generator()
    g.manual_seed(seed)
    
    if stream_id == 1:
        batch_size = cfg.get('stream1', {}).get('batch_size', tc['batch_size'])
    else:
        batch_size = tc['batch_size']

    pin_memory = False

    def build_loaders(_num_workers: int):
        _prefetch_factor = dc.get('prefetch_factor', 1)
        if _num_workers > 0:
            print(f"[DataLoader] Using {_num_workers} workers (spawn + file_system, prefetch={_prefetch_factor})")
        else:
            print("[DataLoader] Using single-process data loading (num_workers=0)")

        _worker_kwargs = dict(
            worker_init_fn=_worker_init_fn,
            generator=g,
            persistent_workers=False,
        ) if _num_workers > 0 else {}

        _train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=_num_workers, pin_memory=pin_memory,
            prefetch_factor=_prefetch_factor if _num_workers > 0 else None,
            collate_fn=collate_fn_kpts, drop_last=True,
            **_worker_kwargs,
        )
        _dev_loader = DataLoader(
            dev_ds, batch_size=batch_size, shuffle=False,
            num_workers=_num_workers, pin_memory=pin_memory,
            prefetch_factor=_prefetch_factor if _num_workers > 0 else None,
            collate_fn=collate_fn_kpts,
            **_worker_kwargs,
        )
        return _train_loader, _dev_loader

    num_workers = dc.get('num_workers', 4)
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        num_workers = min(num_workers, max(0, int(slurm_cpus)))
    train_loader, dev_loader = build_loaders(num_workers)

    if stream_id == 1:
        s_cfg = cfg.get('stream1', {})
        model = Stream1_RGB_USTM(
            pretrained=s_cfg.get('pretrained', True),
            num_classes=num_classes,
            d_model=s_cfg.get('d_model', 1024),
            tape_kernel=s_cfg.get('tape_kernel', 3),
            tape_dilation_short=s_cfg.get('tape_dilation_short', 1),
            tape_dilation_long=s_cfg.get('tape_dilation_long', 3),
            tape_max_len=s_cfg.get('tape_max_len', 512),
            tape_dropout=s_cfg.get('tape_dropout', 0.1),
            temporal_dropout=s_cfg.get('temporal_dropout', 0.1),
            temporal_lstm_layers=s_cfg.get('temporal_lstm_layers', 1),
            proj_dropout=s_cfg.get('proj_dropout', 0.0),
            ctc_dropout=s_cfg.get('ctc_dropout', 0.1),
            use_checkpoint=s_cfg.get('use_checkpoint', True),
            backbone=s_cfg.get('backbone', 'swint'),
            temporal_head=s_cfg.get('temporal_head', 'mstcn'),
            conv_type=s_cfg.get('conv_type', 3),
            adapter=s_cfg.get('adapter', 3),
            hidden_size=s_cfg.get('hidden_size', s_cfg.get('d_model', 1024)),
            weight_norm=s_cfg.get('weight_norm', True),
            share_classifier=s_cfg.get('share_classifier', True),
            output_norm=s_cfg.get('output_norm', False),
        )
    elif stream_id == 2:
        s_cfg = cfg.get('stream2', {})
        model = Stream2_Hypergraph(
            keypoint_dim=s_cfg.get('keypoint_dim', 225),
            num_joints=s_cfg.get('num_joints', 75),
            keypoint_layout=s_cfg.get('keypoint_layout', 'pose_hands'),
            node_dim=s_cfg.get('node_dim', 128),
            d_model=s_cfg.get('d_model', 512),
            num_proto=s_cfg.get('num_proto', 8),
            knn_k=s_cfg.get('knn_k', 8),
            ms_kernels=s_cfg.get('ms_kernels', [1, 3, 5, 7]),
            dropout=s_cfg.get('dropout', 0.1),
            ctc_dropout=s_cfg.get('ctc_dropout', 0.1),
            normalize_keypoints=s_cfg.get('normalize_keypoints', True),
            temporal_lstm_layers=s_cfg.get('temporal_lstm_layers', 1),
            cgp_channels=s_cfg.get('cgp_channels', [64, 128, 128, 256]),
            num_classes=num_classes,
        )
    elif stream_id == 3:
        s_cfg = cfg.get('stream3', {})
        model = Stream3_CrossAttn(
            keypoint_dim=s_cfg.get('keypoint_dim', 225),
            d_model=s_cfg.get('d_model', 512),
            nhead=s_cfg.get('nhead', 8),
            bilstm_hidden=s_cfg.get('bilstm_hidden', 512),
            bilstm_layers=s_cfg.get('bilstm_layers', 2),
            num_classes=num_classes,
            attn_dropout=s_cfg.get('attn_dropout', 0.1),
            bilstm_dropout=s_cfg.get('bilstm_dropout', 0.3),
            ctc_dropout=s_cfg.get('ctc_dropout', 0.1),
            pretrained=s_cfg.get('pretrained', True),
            use_checkpoint=s_cfg.get('use_checkpoint', True),
            swin_ref=None,
        )
    else:
        raise ValueError("Stream ID must be 1, 2 or 3")

    model = model.to(device)
    print(f'[Model] Total params: {sum(p.numel() for p in model.parameters()):,}')

    criterion = CTCCriterion(blank_idx=cfg['cslr']['ctc_blank_idx'])
    optimizer = build_single_optimizer(model, cfg, stream_id)
    print('[Optimizer] ' + ', '.join(
        f"group{i}:{group['lr']:.3e}" for i, group in enumerate(optimizer.param_groups)
    ))

    num_epochs = cfg.get('stream1', {}).get('num_epochs', tc['num_epochs']) if stream_id == 1 else tc['num_epochs']
    steps_per_ep = len(train_loader)
    if stream_id == 1:
        grad_acc = cfg.get('stream1', {}).get('grad_accumulation_steps', tc.get('grad_accumulation_steps', 4))
    else:
        grad_acc = tc.get('grad_accumulation_steps', 4)
    opt_steps_per_ep = math.ceil(steps_per_ep / max(1, grad_acc))
    total_steps = num_epochs * opt_steps_per_ep
    s1_cfg = cfg.get('stream1', {})
    stream1_epoch_scheduler = stream_id == 1 and s1_cfg.get('scheduler', '').lower() == 'multistep'
    if stream1_epoch_scheduler:
        scheduler = MultiStepLR(
            optimizer,
            milestones=s1_cfg.get('step', [20, 30, 40, 50]),
            gamma=s1_cfg.get('gamma', 0.2),
        )
    else:
        warmup_epochs = s1_cfg.get('warmup_epochs', tc.get('warmup_epochs', 5)) if stream_id == 1 else tc.get('warmup_epochs', 5)
        warmup_steps = warmup_epochs * opt_steps_per_ep
        max_group_lr = max(group['lr'] for group in optimizer.param_groups)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps, tc.get('eta_min', 1e-6) / max(max_group_lr, 1e-12)
        )
    scaler = torch.amp.GradScaler('cuda') if (device.type == 'cuda' and tc.get('use_amp', True)) else None

    ckpt_dir = Path(cfg['paths']['checkpoint_dir']) / f'stream{stream_id}_only'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt   = ckpt_dir / f'best_stream{stream_id}.pth'
    newest_ckpt = ckpt_dir / f'newest_stream{stream_id}.pth'
    best_wer  = float('inf')
    start_epoch = 0

    # Auto-resume: priority to --resume, then newest checkpoint
    _resume_target = resume_path or (str(newest_ckpt) if newest_ckpt.exists() else None)
    if _resume_target:
        print(f"  [resume] Loading checkpoint from {_resume_target}")
        checkpoint = torch.load(_resume_target, map_location=device)
        try:
            if stream_id == 1 and resume_path is None:
                old_sig = _stream1_protocol_signature(checkpoint.get('cfg', {}))
                new_sig = _stream1_protocol_signature(cfg)
                if old_sig != new_sig:
                    raise RuntimeError(
                        "stream1 protocol changed; refusing auto-resume. "
                        f"old={old_sig}, new={new_sig}"
                    )
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if stream_id == 1:
                _reset_stream1_resume_lrs(optimizer, scheduler, cfg)
            elif stream_id == 2:
                _reset_stream2_resume_lrs(optimizer, scheduler, cfg)
            start_epoch = checkpoint['epoch'] + 1
            saved_best = checkpoint.get('best_wer', float('inf'))
            saved_wer = checkpoint.get('wer', float('inf'))
            best_wer = min(saved_best, saved_wer)
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            print(f"  [resume] Resuming from epoch {start_epoch} (Best WER: {best_wer*100:.2f}%)")
        except RuntimeError as e:
            if resume_path is not None:
                raise
            print(f"  [resume] Auto-resume skipped: checkpoint is incompatible with current model ({e})")
            print("  [resume] Starting from scratch. Keep old checkpoints only for comparison/eval.")
            best_wer = float('inf')
            start_epoch = 0

    from utils.ctc_decoder import batch_ctc_decode
    blank = cfg['cslr']['ctc_blank_idx']

    def _progress_miniters(loader):
        try:
            return max(1, len(loader) // 20)
        except TypeError:
            return 1

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        while True:
            try:
                pbar = tqdm(
                    train_loader,
                    desc=f"Train E{epoch+1}",
                    leave=False,
                    miniters=_progress_miniters(train_loader),
                )
                for step, batch in enumerate(pbar):
                    frames = batch['frames'].to(device, non_blocking=pin_memory)
                    keypoints = batch['keypoints'].to(device, non_blocking=pin_memory)
                    frame_lens = batch['frame_lens'].to(device, non_blocking=pin_memory)
                    gloss = batch['gloss'].to(device, non_blocking=pin_memory)
                    gloss_lens = batch['gloss_lens'].to(device, non_blocking=pin_memory)

                    with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                        try:
                            if stream_id == 1:
                                log_probs, _, adj_lens = model(frames, frame_lens)
                            elif stream_id == 2:
                                log_probs, _, adj_lens = model(keypoints, frame_lens)
                            else: # 3
                                log_probs, _, adj_lens = model(frames, keypoints, frame_lens)

                            T_out = log_probs.shape[0]
                            loss = criterion(log_probs, gloss, adj_lens.clamp(max=T_out), gloss_lens)
                            if stream_id == 1:
                                seq_w = cfg.get('stream1', {}).get('seq_ctc_weight', 1.0)
                                conv_w = cfg.get('stream1', {}).get('conv_ctc_weight', 0.0)
                                loss = loss * seq_w
                                conv_log_probs = getattr(model, 'last_conv_log_probs', None)
                                conv_lens = getattr(model, 'last_conv_lens', None)
                                if conv_w > 0.0 and conv_log_probs is not None and conv_lens is not None:
                                    loss = loss + conv_w * criterion(
                                        conv_log_probs,
                                        gloss,
                                        conv_lens.clamp(max=conv_log_probs.shape[0]),
                                        gloss_lens,
                                    )
                            loss_scaled = loss / grad_acc

                        except torch.cuda.OutOfMemoryError:
                            print(f"\n  [OOM] CUDA out of memory at step {step}, skipping batch and clearing cache.")
                            torch.cuda.empty_cache()
                            optimizer.zero_grad()
                            continue

                    if scaler is not None:
                        scaler.scale(loss_scaled).backward()
                        if (step + 1) % grad_acc == 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get('gradient_clip', 1.0))
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            if not stream1_epoch_scheduler:
                                scheduler.step()
                    else:
                        loss_scaled.backward()
                        if (step + 1) % grad_acc == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get('gradient_clip', 1.0))
                            optimizer.step()
                            optimizer.zero_grad()
                            if not stream1_epoch_scheduler:
                                scheduler.step()
                    epoch_loss += loss.item()
                    num_batches += 1
                    pbar.set_postfix(loss=f'{loss.item():.4f}')

                break
            except (RuntimeError, OSError) as e:
                msg = str(e)
                is_resource_error = any(term in msg for term in [
                    "shared memory", 
                    "unable to open shared memory object", 
                    "Too many open files", 
                    "Errno 23",
                    "DataLoader worker",
                    "resource temporarily unavailable"
                ])
                if is_resource_error and num_workers > 0:
                    print(f"\n[DataLoader] Resource/shared memory error in train ({e}). Falling back to num_workers=0 and retrying epoch.")
                    num_workers = 0
                    train_loader, dev_loader = build_loaders(num_workers)
                    continue
                raise
        avg_train_loss = epoch_loss / max(num_batches, 1)
        if stream1_epoch_scheduler:
            scheduler.step()

        # Release reserved-but-unallocated GPU memory after each epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        while True:
            try:
                model.eval()
                hyp, ref = [], []
                hyp_video_ids = []
                pred_token_count = 0
                ref_token_count = 0
                blank_frames = 0
                total_frames = 0
                val_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                        dev_loader,
                        desc=f"Eval E{epoch+1}",
                        leave=False,
                        miniters=_progress_miniters(dev_loader),
                    ):
                        frames = batch['frames'].to(device, non_blocking=pin_memory)
                        keypoints = batch['keypoints'].to(device, non_blocking=pin_memory)
                        frame_lens = batch['frame_lens'].to(device, non_blocking=pin_memory)
                        gloss = batch['gloss'].to(device, non_blocking=pin_memory)
                        gloss_lens = batch['gloss_lens'].to(device, non_blocking=pin_memory)

                        if stream_id == 1:
                            log_probs, _, adj_lens = model(frames, frame_lens)
                        elif stream_id == 2:
                            log_probs, _, adj_lens = model(keypoints, frame_lens)
                        else:
                            log_probs, _, adj_lens = model(frames, keypoints, frame_lens)
                            
                        T_out = log_probs.shape[0]
                        cl = adj_lens.clamp(max=T_out)
                        
                        loss = criterion(log_probs, gloss, cl, gloss_lens)
                        val_loss += loss.item()

                        best_path = log_probs.argmax(dim=-1).transpose(0, 1)
                        for b in range(best_path.shape[0]):
                            actual_t = int(cl[b].item())
                            if actual_t > 0:
                                path = best_path[b, :actual_t]
                                blank_frames += int((path == blank).sum().item())
                                total_frames += actual_t
                        
                        decode_mode = cfg.get('stream1', {}).get('decode_mode', 'greedy') if stream_id == 1 else 'greedy'
                        beam_width = cfg.get('stream1', {}).get('beam_width', cfg.get('eval', {}).get('beam_width', 4))
                        preds = batch_ctc_decode(log_probs, cl, blank_idx=blank, mode=decode_mode, beam_width=beam_width)
                        B = frames.shape[0]
                        for b in range(B):
                            r = gloss_vocab.decode(gloss[b, :gloss_lens[b].item()].tolist(), skip_special=False)
                            ref.append(r)
                            decoded_pred = gloss_vocab.decode(preds[b])
                            hyp.append(decoded_pred)
                            hyp_video_ids.append(batch['video_ids'][b])
                            pred_token_count += len(decoded_pred)
                            ref_token_count += len(r)
                            
                wer = compute_wer(hyp, ref)
                break
            except (RuntimeError, OSError) as e:
                msg = str(e)
                is_resource_error = any(term in msg for term in [
                    "shared memory", 
                    "unable to open shared memory object", 
                    "Too many open files", 
                    "Errno 23",
                    "DataLoader worker",
                    "resource temporarily unavailable"
                ])
                if is_resource_error and num_workers > 0:
                    print(f"\n[DataLoader] Resource/shared memory error in eval ({e}). Falling back to num_workers=0 and retrying eval.")
                    num_workers = 0
                    train_loader, dev_loader = build_loaders(num_workers)
                    continue
                raise
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss/len(dev_loader):.4f} | WER: {wer*100:.2f}%")
        avg_pred_len = pred_token_count / max(len(hyp), 1)
        avg_ref_len = ref_token_count / max(len(ref), 1)
        blank_ratio = blank_frames / max(total_frames, 1)
        print(f"  [decode] avg_pred_len={avg_pred_len:.2f} avg_ref_len={avg_ref_len:.2f} blank_frames={blank_ratio*100:.1f}%")
        
        improved = wer < best_wer
        if improved:
            best_wer = wer

        # Save newest checkpoint every epoch. Store best_wer after the current
        # epoch update so auto-resume does not revive stale metadata.
        newest_state = {
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_wer':  best_wer,
            'wer':       wer,
            'cfg':       cfg,
        }
        if scaler is not None:
            newest_state['scaler'] = scaler.state_dict()
        torch.save(newest_state, newest_ckpt)

        # Save best checkpoint if WER improves
        if improved:
            torch.save(newest_state, best_ckpt)
            print(f"  [ckpt] Best saved (WER={best_wer*100:.2f}%)")

        # Cleanup old checkpoints (keep only best and newest)
        if stream_id == 1:
            _keep = {best_ckpt.name, newest_ckpt.name}
            for _old in ckpt_dir.glob('*.pth'):
                if _old.name not in _keep:
                    try:
                        _old.unlink()
                        print(f"  [ckpt] Removed old checkpoint: {_old.name}")
                    except OSError as e:
                        print(f"  [ckpt] Warning: could not remove {_old.name}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--stream', type=int, choices=[1, 2, 3], required=True, help="ID of the stream to train individually (1, 2, or 3)")
    parser.add_argument('--resume', type=str, default=None, help="Path to .pth file to resume training")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_single_stream(cfg, args.stream, args.resume)
