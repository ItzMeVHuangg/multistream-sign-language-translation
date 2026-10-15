
import os
import random
import sys
import argparse
import math
import time
import yaml
from pathlib import Path
from typing import List, Optional
import tempfile

os.environ.setdefault('TORCH_SHM_DIR', str(Path(tempfile.gettempdir()) / 'torch_shm'))

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            print("[TensorBoard] tensorboard is not installed; logging disabled.")

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except (RuntimeError, ValueError):
    pass

from data.dataset    import PhoenixDataset, collate_fn, build_vocabularies, build_transforms, Vocabulary
from models.rover_ensemble import ThreeStreamCSLR
from models.bilstm_ctc import CTCCriterion
from utils.metrics   import compute_wer


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def _worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def _progress_miniters(loader):
    try:
        return max(1, len(loader) // 20)
    except TypeError:
        return 1


# -----------------------------------------------------------------------------
# Dataset wrapper that also returns MediaPipe keypoints
# -----------------------------------------------------------------------------

class PhoenixWithKeypoints(Dataset):
    """
    Wraps PhoenixDataset and loads pre-extracted keypoints from .npy files.

    Keypoint files: <kpts_root>/<split>/<video_id>.npy  shape (T_raw, 225)
    Keypoints are temporally aligned and padded to match max_frames.
    """

    def __init__(
        self,
        base_dataset:  PhoenixDataset,
        kpts_root:     Path,
        max_frames:    int,
        keypoint_dim:  int = 225,
        load_frames:   bool = True,
        load_kpts:     bool = True,
    ):
        self.base         = base_dataset
        self.kpts_root    = kpts_root / base_dataset.split
        self.max_frames   = max_frames
        self.kp_dim       = keypoint_dim
        self.load_frames  = load_frames
        self.load_kpts    = load_kpts

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        if self.load_frames:
            item = self.base[idx]
        else:
            # Skip heavy frame loading if only keypoints are needed
            sample = self.base.samples[idx]
            gloss_ids = torch.tensor(self.base.gloss_vocab.encode(sample["gloss"]), dtype=torch.long)
            item = {
                "video_id":  sample["video_id"],
                "frames":    torch.zeros((self.max_frames, 3, 224, 224)), # dummy
                "frame_len": torch.tensor(0, dtype=torch.long), # will be updated by kpts
                "gloss":     gloss_ids,
                "gloss_len": torch.tensor(len(sample["gloss"]), dtype=torch.long),
            }

        if not self.load_kpts:
            item['keypoints'] = torch.zeros((self.max_frames, self.kp_dim))
            return item

        video_id   = item['video_id']
        kpt_path = self.kpts_root / f"{video_id}.npy"
        if kpt_path.exists():
            kpts_raw = np.load(kpt_path, allow_pickle=False)
            # If frames are not loaded, get frame_len from raw kpts or metadata
            f_len = item['frame_len'].item() if self.load_frames else kpts_raw.shape[0]
            kpts = self._align(kpts_raw, f_len, self.max_frames)
            if not self.load_frames:
                item['frame_len'] = torch.tensor(f_len, dtype=torch.long)
        else:
            kpts = np.zeros((self.max_frames, self.kp_dim), dtype=np.float32)

        item['keypoints'] = torch.from_numpy(kpts)
        return item

    def _align(
        self,
        kpts:       np.ndarray,   # (T_raw, D)
        frame_len:  int,          # target actual length
        max_frames: int,
    ) -> np.ndarray:
        T_raw, D = kpts.shape

        # Resample to frame_len using linear interpolation
        if T_raw != frame_len and T_raw > 0:
            idx_f = np.linspace(0, T_raw - 1, frame_len)
            idx_l = np.floor(idx_f).astype(int).clip(0, T_raw - 1)
            idx_r = np.ceil(idx_f).astype(int).clip(0, T_raw - 1)
            alpha = (idx_f - idx_l)[:, None]
            kpts  = kpts[idx_l] * (1 - alpha) + kpts[idx_r] * alpha

        # Truncate or pad to max_frames
        out = np.zeros((max_frames, D), dtype=np.float32)
        n   = min(len(kpts), max_frames)
        out[:n] = kpts[:n]
        return out


def collate_fn_kpts(batch: List[dict]) -> dict:
    """Extended collate that also stacks keypoints."""
    out = collate_fn(batch)
    out['keypoints'] = torch.stack([b['keypoints'] for b in batch])   # (B, T, 225)
    return out


# -----------------------------------------------------------------------------
# Differential LR optimizer
# -----------------------------------------------------------------------------

def build_optimizer(model: ThreeStreamCSLR, cfg: dict) -> AdamW:
    """
    Three param groups:
        swin_params   : lr = base_lr * swin_lr_scale     (very small, pretrained)
        stream2_params: lr = base_lr * stream2_lr_scale  (full lr, no pretrained)
        rest_params   : lr = base_lr                      (TAPE, proj, heads, BiLSTM)
    """
    tc       = cfg['three_stream']
    base_lr  = tc['learning_rate']
    swin_sc  = tc.get('swin_lr_scale', 0.05)
    s2_sc    = tc.get('stream2_lr_scale', 1.0)
    wd       = tc['weight_decay']

    # Collect Swin parameter ids (stream1 + stream3 swin, shared if share_swin=True)
    swin_ids = set()
    for m in [model.stream1.swin_patch_embed, model.stream1.swin_pos_drop,
              model.stream1.swin_stage1, model.stream1.swin_merge1,
              model.stream1.swin_stage2, model.stream1.swin_merge2,
              model.stream1.swin_stage3, model.stream1.swin_merge3,
              model.stream1.swin_stage4, model.stream1.swin_norm]:
        swin_ids.update(id(p) for p in m.parameters())

    # Stream3 swin (may share same params as stream1 if share_swin=True)
    for p in model.stream3.swin.parameters():
        swin_ids.add(id(p))

    # Stream2 ids
    s2_ids = set(id(p) for p in model.stream2.parameters())

    swin_params, s2_params, rest_params = [], [], []
    seen_ids = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)

        if pid in swin_ids:
            swin_params.append(p)
        elif pid in s2_ids:
            s2_params.append(p)
        else:
            rest_params.append(p)

    return AdamW([
        {'params': swin_params, 'lr': base_lr * swin_sc, 'name': 'swin'},
        {'params': s2_params,   'lr': base_lr * s2_sc,   'name': 'stream2'},
        {'params': rest_params, 'lr': base_lr,            'name': 'rest'},
    ], weight_decay=wd)


# -----------------------------------------------------------------------------
# LR scheduler: cosine with warmup
# -----------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int,
    eta_min_ratio: float = 0.0,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(eta_min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# -----------------------------------------------------------------------------
# Train one epoch
# -----------------------------------------------------------------------------

def train_one_epoch(
    model:       ThreeStreamCSLR,
    loader:      DataLoader,
    optimizer:   AdamW,
    scheduler:   LambdaLR,
    criterion:   CTCCriterion,
    scaler,
    device:      torch.device,
    cfg:         dict,
    epoch:       int,
    writer:      SummaryWriter,
    global_step: int,
) -> tuple:
    model.train()

    tc        = cfg['three_stream']
    if tc.get('freeze_streams_for_fusion', False):
        # Keep frozen pretrained streams deterministic while training only
        # fusion_logits. Otherwise Dropout/BatchNorm in the frozen streams can
        # change the inputs seen by the fusion head and update BN statistics.
        model.stream1.eval()
        model.stream2.eval()
        model.stream3.eval()
    elif tc.get('freeze_stream2', False):
        model.stream2.eval()

    grad_acc  = tc.get('grad_accumulation_steps', 4)
    clip_grad = tc.get('gradient_clip', 1.0)
    w         = tc.get('loss_weights', [1.0, 1.0, 1.0])
    fusion_w  = tc.get('fusion_ctc_weight', 0.0)

    epoch_loss  = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(
        loader,
        desc=f"[3Stream Train] Epoch {epoch+1}",
        leave=False,
        miniters=_progress_miniters(loader),
    )
    for step, batch in enumerate(pbar):
        frames     = batch['frames'].to(device)        # (B, T, C, H, W)
        keypoints  = batch['keypoints'].to(device)     # (B, T, 225)
        frame_lens = batch['frame_lens'].to(device)    # (B,)
        gloss      = batch['gloss'].to(device)         # (B, G)
        gloss_lens = batch['gloss_lens'].to(device)    # (B,)

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            out = model(frames, keypoints, frame_lens)

            # Per-stream CTC losses
            def ctc_loss_for(lp, lens):
                T_out = lp.shape[0]
                input_lens = lens.clamp(max=T_out)
                return criterion(lp, gloss, input_lens, gloss_lens)

            l1 = ctc_loss_for(out['s1_log_probs'], out['s1_lens'])
            l2 = ctc_loss_for(out['s2_log_probs'], out['s2_lens'])
            l3 = ctc_loss_for(out['s3_log_probs'], out['s3_lens'])
            lf = ctc_loss_for(out['fused_log_probs'], out['fused_lens'])

            loss = w[0] * l1 + w[1] * l2 + w[2] * l3 + fusion_w * lf
            loss_scaled = loss / grad_acc

        if scaler is not None:
            scaler.scale(loss_scaled).backward()
            if (step + 1) % grad_acc == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss_scaled.backward()
            if (step + 1) % grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad()

        if (step + 1) % grad_acc == 0:
            scheduler.step()
        epoch_loss  += loss.item()
        num_batches += 1
        global_step += 1

        if writer and global_step % 10 == 0:
            writer.add_scalar('3stream/loss_total',  loss.item(),   global_step)
            writer.add_scalar('3stream/loss_stream1', l1.item(),    global_step)
            writer.add_scalar('3stream/loss_stream2', l2.item(),    global_step)
            writer.add_scalar('3stream/loss_stream3', l3.item(),    global_step)
            writer.add_scalar('3stream/loss_fused',   lf.item(),    global_step)
            for idx, fw in enumerate(out['fusion_weights'].detach().cpu().tolist(), start=1):
                writer.add_scalar(f'3stream/fusion_weight_s{idx}', fw, global_step)
            if 'fusion_temperatures' in out:
                for idx, temp in enumerate(out['fusion_temperatures'].detach().cpu().tolist(), start=1):
                    writer.add_scalar(f'3stream/fusion_temperature_s{idx}', temp, global_step)
            if 'fusion_blank_biases' in out:
                for idx, bias in enumerate(out['fusion_blank_biases'].detach().cpu().tolist(), start=1):
                    writer.add_scalar(f'3stream/fusion_blank_bias_s{idx}', bias, global_step)
            writer.add_scalar('3stream/lr_swin',
                              optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('3stream/lr_rest',
                              optimizer.param_groups[2]['lr'], global_step)

        pbar.set_postfix(
            loss=f'{loss.item():.4f}',
            l1=f'{l1.item():.3f}',
            l2=f'{l2.item():.3f}',
            l3=f'{l3.item():.3f}',
            lf=f'{lf.item():.3f}',
        )

    return epoch_loss / max(num_batches, 1), global_step


# -----------------------------------------------------------------------------
# Evaluation with ROVER
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:      ThreeStreamCSLR,
    loader:     DataLoader,
    criterion:  CTCCriterion,
    device:     torch.device,
    gloss_vocab: Vocabulary,
    cfg:        dict,
) -> tuple:
    """Returns (avg_total_loss, wer_rover, wer_fused, wer_s1, wer_s2, wer_s3)."""
    model.eval()

    tc      = cfg['three_stream']
    w       = tc.get('loss_weights', [1.0, 1.0, 1.0])
    fusion_w = tc.get('fusion_ctc_weight', 0.0)

    total_loss = 0.0
    all_hyp_rover, all_hyp_fused = [], []
    all_hyp_s1, all_hyp_s2, all_hyp_s3, all_ref = [], [], [], []

    for batch in tqdm(
        loader,
        desc='[3Stream Eval]',
        leave=False,
        miniters=_progress_miniters(loader),
    ):
        frames     = batch['frames'].to(device)
        keypoints  = batch['keypoints'].to(device)
        frame_lens = batch['frame_lens'].to(device)
        gloss      = batch['gloss'].to(device)
        gloss_lens = batch['gloss_lens'].to(device)

        out = model(frames, keypoints, frame_lens)

        def ctc_loss_for(lp, lens):
            return criterion(lp, gloss, lens.clamp(max=lp.shape[0]), gloss_lens)

        l1 = ctc_loss_for(out['s1_log_probs'], out['s1_lens'])
        l2 = ctc_loss_for(out['s2_log_probs'], out['s2_lens'])
        l3 = ctc_loss_for(out['s3_log_probs'], out['s3_lens'])
        lf = ctc_loss_for(out['fused_log_probs'], out['fused_lens'])
        total_loss += (w[0]*l1 + w[1]*l2 + w[2]*l3 + fusion_w*lf).item()

        # ROVER ensemble decode
        rover_seqs = model.rover(
            [out['s1_log_probs'], out['s2_log_probs'], out['s3_log_probs']],
            [out['s1_lens'],      out['s2_lens'],      out['s3_lens']],
        )

        # Per-stream greedy decode for diagnostics
        from utils.ctc_decoder import batch_ctc_decode
        blank = cfg['cslr']['ctc_blank_idx']

        def greedy_decode_stream(lp, lens):
            return batch_ctc_decode(lp, lens.clamp(max=lp.shape[0]),
                                    blank_idx=blank, mode='greedy')

        preds_s1 = greedy_decode_stream(out['s1_log_probs'], out['s1_lens'])
        preds_s2 = greedy_decode_stream(out['s2_log_probs'], out['s2_lens'])
        preds_s3 = greedy_decode_stream(out['s3_log_probs'], out['s3_lens'])
        preds_fused = greedy_decode_stream(out['fused_log_probs'], out['fused_lens'])

        B = frames.shape[0]
        for b in range(B):
            ref = gloss_vocab.decode(
                gloss[b, :gloss_lens[b].item()].tolist(), skip_special=False
            )
            all_ref.append(ref)
            all_hyp_rover.append(gloss_vocab.decode(rover_seqs[b]))
            all_hyp_fused.append(gloss_vocab.decode(preds_fused[b]))
            all_hyp_s1.append(gloss_vocab.decode(preds_s1[b]))
            all_hyp_s2.append(gloss_vocab.decode(preds_s2[b]))
            all_hyp_s3.append(gloss_vocab.decode(preds_s3[b]))

    avg_loss  = total_loss / len(loader)
    wer_rover = compute_wer(all_hyp_rover, all_ref)
    wer_fused = compute_wer(all_hyp_fused, all_ref)
    wer_s1    = compute_wer(all_hyp_s1,    all_ref)
    wer_s2    = compute_wer(all_hyp_s2,    all_ref)
    wer_s3    = compute_wer(all_hyp_s3,    all_ref)

    return avg_loss, wer_rover, wer_fused, wer_s1, wer_s2, wer_s3


# -----------------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------------

def _resolve_stream_ckpts(cfg: dict, ckpt1: Optional[str], ckpt2: Optional[str], ckpt3: Optional[str]) -> tuple:
    ckpt_root = Path(cfg['paths']['checkpoint_dir'])
    return (
        ckpt1 or str(ckpt_root / 'stream1_only' / 'best_stream1.pth'),
        ckpt2 or str(ckpt_root / 'stream2_only' / 'best_stream2.pth'),
        ckpt3 or str(ckpt_root / 'stream3_only' / 'best_stream3.pth'),
    )


def _checkpoint_state(ckpt_path: str, device: torch.device) -> dict:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    return checkpoint.get('model', checkpoint)


def _filter_state_dict(state: dict, skip_prefixes: tuple = ()) -> dict:
    if not skip_prefixes:
        return state
    return {
        key: value
        for key, value in state.items()
        if not key.startswith(skip_prefixes)
    }


def _load_stream_weights(
    module: nn.Module,
    ckpt_path: str,
    device: torch.device,
    label: str,
    skip_prefixes: tuple = (),
    required: bool = True,
    min_load_ratio: float = 0.98,
):
    if not ckpt_path:
        return
    if not os.path.exists(ckpt_path):
        msg = f"{label} checkpoint not found: {ckpt_path}"
        if required:
            raise FileNotFoundError(msg)
        print(f"[3Stream] WARNING: {msg}")
        return

    print(f"[3Stream] Loading {label} weights from {ckpt_path}...")
    state = _filter_state_dict(_checkpoint_state(ckpt_path, device), skip_prefixes)
    if skip_prefixes:
        print(f"[3Stream]   skipped prefixes for {label}: {', '.join(skip_prefixes)}")
    target_state = module.state_dict()
    expected_keys = [
        key for key in target_state
        if not (skip_prefixes and key.startswith(skip_prefixes))
    ]
    mismatched = [
        key for key, value in state.items()
        if key in target_state and value.shape != target_state[key].shape
    ]
    if mismatched:
        state = {key: value for key, value in state.items() if key not in mismatched}
        print(f"[3Stream] WARNING: {label} shape-mismatched keys skipped: {len(mismatched)}")
    missing, unexpected = module.load_state_dict(state, strict=False)
    loaded_keys = [key for key in state if key in target_state and target_state[key].shape == state[key].shape]
    load_ratio = len(loaded_keys) / max(1, len(expected_keys))
    print(
        f"[3Stream]   loaded {len(loaded_keys)}/{len(expected_keys)} expected keys "
        f"({load_ratio*100:.1f}%)"
    )
    if load_ratio < min_load_ratio:
        msg = (
            f"{label} checkpoint looks incompatible with the current model "
            f"(loaded {load_ratio*100:.1f}%, require {min_load_ratio*100:.1f}%). "
            "Use the exact stream code/config that produced this checkpoint."
        )
        if required:
            raise RuntimeError(msg)
        print(f"[3Stream] WARNING: {msg}")
    if missing:
        print(f"[3Stream] WARNING: {label} missing keys: {len(missing)}")
    if unexpected:
        print(f"[3Stream] WARNING: {label} unexpected keys: {len(unexpected)}")


def _freeze_module(module: nn.Module, label: str):
    for p in module.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in module.parameters())
    print(f"[3Stream] Frozen {label}: {n_params:,} params")


def _freeze_streams_for_fusion(model: ThreeStreamCSLR):
    _freeze_module(model.stream1, 'Stream 1 (RGB-USTM)')
    _freeze_module(model.stream2, 'Stream 2 (Hypergraph)')
    _freeze_module(model.stream3, 'Stream 3 (CrossAttn)')
    model.fusion_logits.requires_grad = True
    if hasattr(model, 'fusion_log_temps'):
        model.fusion_log_temps.requires_grad = True
    if hasattr(model, 'fusion_blank_bias'):
        model.fusion_blank_bias.requires_grad = True
    print("[3Stream] Training only learnable calibrated logit-fusion parameters")


def _save_joint_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler,
    best_wer: float,
    current_wer: float,
    no_improve: int,
    global_step: int,
    gloss_vocab: Vocabulary,
    text_vocab: Vocabulary,
    cfg: dict,
    seed: int,
    metrics: Optional[dict] = None,
):
    state = {
        'epoch':        epoch,
        'model':        model.state_dict(),
        'optimizer':    optimizer.state_dict(),
        'scheduler':    scheduler.state_dict(),
        'best_wer':     best_wer,
        'wer':          current_wer,
        'no_improve':   no_improve,
        'global_step':  global_step,
        'gloss_vocab':  gloss_vocab,
        'text_vocab':   text_vocab,
        'cfg':          cfg,
        'seed':         seed,
    }
    if metrics is not None:
        state['metrics'] = metrics
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    torch.save(state, path)


def train_cslr(
    cfg: dict,
    ckpt1: Optional[str] = None,
    ckpt2: Optional[str] = None,
    ckpt3: Optional[str] = None,
    resume_path: Optional[str] = None,
) -> float:
    seed = cfg.get('seed', 42)
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[3Stream] Device: {device}')
    if device.type == 'cuda':
        print(f'[3Stream] GPU  : {torch.cuda.get_device_name(0)}')
        print(f'[3Stream] VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    # ------ Vocabulary ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    paths = cfg.get('paths', {})
    frames_root = Path(paths.get('frames_root')) if paths.get('frames_root') else None
    annot_root = Path(paths.get('annot_root')) if paths.get('annot_root') else None

    gloss_vocab, text_vocab = build_vocabularies(annot_root=annot_root)
    num_classes = len(gloss_vocab)
    print(f'[Vocab] Gloss: {num_classes}')

    # ------ Datasets ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    dc       = cfg['data']
    tc       = cfg['three_stream']
    kpts_root = Path(cfg['paths']['mediapipe_kpts_root'])
    max_fr   = dc['max_frames']
    kp_dim   = cfg['stream2']['keypoint_dim']

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
            split              = split,
            gloss_vocab        = gloss_vocab,
            text_vocab         = text_vocab,
            max_frames         = max_fr,
            temporal_stride    = dc['temporal_stride'],
            transform          = build_transforms(split, dc['img_height'], dc['img_width']),
            return_translation = False,
            temporal_scale_range = temporal_scale_range,
            clip_aug_crop      = clip_aug_crop,
            img_h              = dc['img_height'],
            img_w              = dc['img_width'],
            cache_root          = cache_root,
            cache_read          = True,
            cache_write         = True,
            frames_root         = frames_root,
            annot_root          = annot_root,
        )
        return PhoenixWithKeypoints(base, kpts_root, max_fr, kp_dim)

    train_ds = make_ds('train')
    dev_ds   = make_ds('dev')

    batch_size = tc['batch_size']
    g = torch.Generator()
    g.manual_seed(seed)

    def build_loaders(_num_workers: int):
        if _num_workers > 0:
            print(f"[DataLoader] Using {_num_workers} workers")
        else:
            print("[DataLoader] Using single-process data loading (num_workers=0)")
        _train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers     = _num_workers,
            pin_memory      = dc.get('pin_memory', True),
            prefetch_factor = dc.get('prefetch_factor', 2) if _num_workers > 0 else None,
            collate_fn      = collate_fn_kpts,
            drop_last       = True,
            generator       = g,
            worker_init_fn  = _worker_init_fn,
            persistent_workers = (_num_workers > 0),
        )
        _dev_loader = DataLoader(
            dev_ds, batch_size=batch_size, shuffle=False,
            num_workers     = _num_workers,
            pin_memory      = dc.get('pin_memory', True),
            prefetch_factor = dc.get('prefetch_factor', 2) if _num_workers > 0 else None,
            collate_fn      = collate_fn_kpts,
            worker_init_fn  = _worker_init_fn,
            persistent_workers = (_num_workers > 0),
        )
        return _train_loader, _dev_loader

    num_workers = dc['num_workers']
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        num_workers = min(num_workers, max(0, int(slurm_cpus)))
    train_loader, dev_loader = build_loaders(num_workers)

    # ------ Model ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    model = ThreeStreamCSLR(
        num_classes = num_classes,
        cfg         = cfg,
        share_swin  = tc.get('share_swin', True),
    ).to(device)

    ckpt1, ckpt2, ckpt3 = _resolve_stream_ckpts(cfg, ckpt1, ckpt2, ckpt3)
    _load_stream_weights(model.stream1, ckpt1, device, 'Stream 1 (RGB-USTM)')
    _load_stream_weights(model.stream2, ckpt2, device, 'Stream 2 (Hypergraph)')

    # Stream 3 can share Stream 1's Swin backbone in joint training. In that
    # case, do not let best_stream3.pth overwrite the Swin weights that came
    # from best_stream1.pth; load only Stream 3's fusion/temporal/CTC layers.
    s3_skip_prefixes = ('swin.',) if tc.get('share_swin', True) else ()
    _load_stream_weights(
        model.stream3,
        ckpt3,
        device,
        'Stream 3 (CrossAttn)',
        skip_prefixes=s3_skip_prefixes,
        min_load_ratio=0.70 if s3_skip_prefixes else 0.98,
    )

    if tc.get('freeze_stream2', False):
        _freeze_module(model.stream2, 'Stream 2 (Hypergraph)')
    if tc.get('freeze_streams_for_fusion', False):
        _freeze_streams_for_fusion(model)

    total_p = sum(p.numel() for p in model.parameters())
    # Count unique params (shared Swin counted once)
    unique_p = sum(p.numel() for p in set(model.parameters()))
    trainable_p = sum(p.numel() for p in set(model.parameters()) if p.requires_grad)
    print(f'[Model] Total params (with sharing): {unique_p:,}')
    print(f'[Model] Total params (raw sum)      : {total_p:,}')
    print(f'[Model] Trainable params           : {trainable_p:,}')

    # ------ Loss, optimizer, scheduler ------------------------------------------------------------------------------------------------------------------------------------
    criterion = CTCCriterion(blank_idx=cfg['cslr']['ctc_blank_idx'])
    optimizer = build_optimizer(model, cfg)
    print('[Optimizer] ' + ', '.join(
        f"{group.get('name', i)}:params={sum(p.numel() for p in group['params']):,},lr={group['lr']:.3e}"
        for i, group in enumerate(optimizer.param_groups)
    ))
    if hasattr(model, 'fusion_logits'):
        init_fw = torch.softmax(model.fusion_logits.detach().cpu(), dim=0).tolist()
        print('[Fusion] initial weights: ' + ', '.join(f's{i+1}={w:.3f}' for i, w in enumerate(init_fw)))
    if hasattr(model, 'fusion_log_temps'):
        init_temps = model.fusion_log_temps.detach().cpu().exp().clamp(0.5, 2.0).tolist()
        print('[Fusion] initial temperatures: ' + ', '.join(f's{i+1}={t:.3f}' for i, t in enumerate(init_temps)))
    if hasattr(model, 'fusion_blank_bias'):
        init_bias = model.fusion_blank_bias.detach().cpu().clamp(-5.0, 5.0).tolist()
        print('[Fusion] initial blank bias: ' + ', '.join(f's{i+1}={b:.3f}' for i, b in enumerate(init_bias)))

    num_epochs   = tc['num_epochs']
    steps_per_ep = len(train_loader)
    grad_acc     = tc.get('grad_accumulation_steps', 4)
    opt_steps_per_ep = math.ceil(steps_per_ep / max(1, grad_acc))
    warmup_steps = tc.get('warmup_epochs', 5) * opt_steps_per_ep
    total_steps  = num_epochs * opt_steps_per_ep
    eta_ratio    = tc.get('eta_min', 1e-6) / tc['learning_rate']

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, eta_ratio
    )
    scaler = torch.amp.GradScaler('cuda') if (device.type == 'cuda' and tc.get('use_amp', True)) else None

    # ------ Checkpoints & logging ---------------------------------------------------------------------------------------------------------------------------------------------------
    ckpt_dir = Path(cfg['paths']['checkpoint_dir']) / 'three_stream'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer   = SummaryWriter(log_dir=str(Path(cfg['paths']['log_dir']) / 'three_stream'))
    best_ckpt = ckpt_dir / 'best_three_stream.pth'
    best_fused_ckpt = ckpt_dir / 'best_three_stream_fused.pth'
    best_rover_ckpt = ckpt_dir / 'best_three_stream_rover.pth'
    newest_ckpt = ckpt_dir / 'newest_three_stream.pth'

    es_cfg    = tc.get('early_stopping', {})
    do_es     = es_cfg.get('enabled', True)
    patience  = es_cfg.get('patience', 15)
    no_improve = 0
    best_wer   = float('inf')
    best_fused_wer = float('inf')
    best_rover_wer = float('inf')
    monitor_metric = tc.get('monitor', 'rover').lower()
    if monitor_metric not in {'rover', 'fused'}:
        print(f"[3Stream] WARNING: unknown monitor={monitor_metric}; using rover")
        monitor_metric = 'rover'
    global_step = 0
    start_epoch = 0

    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Joint resume checkpoint not found: {resume_path}")
        print(f"[3Stream] Resuming joint training from {resume_path}")
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(resume['model'], strict=False)
        if missing:
            print(f"[3Stream] WARNING: resume missing keys: {len(missing)}")
        if unexpected:
            print(f"[3Stream] WARNING: resume unexpected keys: {len(unexpected)}")
        if 'optimizer' in resume:
            try:
                optimizer.load_state_dict(resume['optimizer'])
            except ValueError as e:
                print(f"[3Stream] WARNING: optimizer state not loaded ({e}); using fresh optimizer.")
        if 'scheduler' in resume:
            try:
                scheduler.load_state_dict(resume['scheduler'])
            except Exception as e:
                print(f"[3Stream] WARNING: scheduler state not loaded ({e}); using fresh scheduler.")
        if scaler is not None and 'scaler' in resume:
            scaler.load_state_dict(resume['scaler'])
        start_epoch = int(resume.get('epoch', -1)) + 1
        best_wer = float(resume.get('best_wer', resume.get('wer', best_wer)))
        resume_metrics = resume.get('metrics', {})
        best_fused_wer = float(resume_metrics.get('best_fused_wer', best_fused_wer))
        best_rover_wer = float(resume_metrics.get('best_rover_wer', best_rover_wer))
        no_improve = int(resume.get('no_improve', 0))
        global_step = int(resume.get('global_step', start_epoch * len(train_loader)))
        print(
            f"[3Stream] Resume state: next epoch={start_epoch + 1}, "
            f"best {monitor_metric.upper()} WER={best_wer * 100:.2f}%, global_step={global_step}"
        )

    print(f'\n[3Stream] Training {num_epochs} epochs | '
          f'eff. batch={batch_size * grad_acc} | '
          f'warmup={tc.get("warmup_epochs", 5)} epochs')

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()

        while True:
            try:
                train_loss, global_step = train_one_epoch(
                    model, train_loader, optimizer, scheduler,
                    criterion, scaler, device, cfg, epoch, writer, global_step,
                )
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

        while True:
            try:
                val_loss, wer_rv, wer_fused, wer_s1, wer_s2, wer_s3 = evaluate(
                    model, dev_loader, criterion, device, gloss_vocab, cfg
                )
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

        elapsed = time.time() - t0
        print(
            f'Epoch [{epoch+1:3d}/{num_epochs}] '
            f'loss={train_loss:.4f} | val={val_loss:.4f} | '
            f'WER rover={wer_rv*100:.2f}% '
            f'fused={wer_fused*100:.2f}% '
            f's1={wer_s1*100:.2f}% s2={wer_s2*100:.2f}% s3={wer_s3*100:.2f}% | '
            f'{elapsed:.0f}s'
        )

        writer.add_scalar('3stream/val_loss',   val_loss, epoch)
        writer.add_scalar('3stream/wer_rover',  wer_rv,   epoch)
        writer.add_scalar('3stream/wer_fused',  wer_fused, epoch)
        writer.add_scalar('3stream/wer_s1',     wer_s1,   epoch)
        writer.add_scalar('3stream/wer_s2',     wer_s2,   epoch)
        writer.add_scalar('3stream/wer_s3',     wer_s3,   epoch)

        monitor_wer = wer_fused if monitor_metric == 'fused' else wer_rv
        current_no_improve = no_improve + (0 if monitor_wer < best_wer else 1)
        metrics = {
            'wer_rover': wer_rv,
            'wer_fused': wer_fused,
            'wer_s1': wer_s1,
            'wer_s2': wer_s2,
            'wer_s3': wer_s3,
            'best_rover_wer': min(best_rover_wer, wer_rv),
            'best_fused_wer': min(best_fused_wer, wer_fused),
            'monitor': monitor_metric,
        }
        _save_joint_checkpoint(
            newest_ckpt, epoch, model, optimizer, scheduler, scaler,
            min(best_wer, monitor_wer), monitor_wer, current_no_improve, global_step,
            gloss_vocab, text_vocab, cfg, seed, metrics=metrics,
        )

        if wer_rv < best_rover_wer:
            best_rover_wer = wer_rv
            metrics['best_rover_wer'] = best_rover_wer
            _save_joint_checkpoint(
                best_rover_ckpt, epoch, model, optimizer, scheduler, scaler,
                best_rover_wer, wer_rv, no_improve, global_step,
                gloss_vocab, text_vocab, cfg, seed, metrics=metrics,
            )
            print(f'  [ckpt] Best ROVER saved (WER={best_rover_wer*100:.2f}%)')

        if wer_fused < best_fused_wer:
            best_fused_wer = wer_fused
            metrics['best_fused_wer'] = best_fused_wer
            _save_joint_checkpoint(
                best_fused_ckpt, epoch, model, optimizer, scheduler, scaler,
                best_fused_wer, wer_fused, no_improve, global_step,
                gloss_vocab, text_vocab, cfg, seed, metrics=metrics,
            )
            print(f'  [ckpt] Best FUSED saved (WER={best_fused_wer*100:.2f}%)')

        if monitor_wer < best_wer:
            best_wer   = monitor_wer
            no_improve = 0
            _save_joint_checkpoint(
                best_ckpt, epoch, model, optimizer, scheduler, scaler,
                best_wer, monitor_wer, no_improve, global_step,
                gloss_vocab, text_vocab, cfg, seed, metrics=metrics,
            )
            print(f'  [ckpt] Best monitor saved ({monitor_metric.upper()} WER={best_wer*100:.2f}%)')
        else:
            no_improve += 1
            print(f'  [no improve] {no_improve}/{patience}')

        if do_es and no_improve >= patience:
            print(f'\n[3Stream] Early stopping at epoch {epoch+1}.')
            break

    writer.close()
    if best_ckpt.exists():
        saved = torch.load(best_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])

    print(f'\n[3Stream] Done. Best {monitor_metric.upper()} WER = {best_wer*100:.2f}%')
    return best_wer


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--ckpt1', default=None, help='Path to pre-trained Stream 1 checkpoint')
    parser.add_argument('--ckpt2', default=None, help='Path to pre-trained Stream 2 checkpoint')
    parser.add_argument('--ckpt3', default=None, help='Path to pre-trained Stream 3 checkpoint')
    parser.add_argument('--resume', default=None, help='Path to joint ThreeStreamCSLR checkpoint to resume')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cslr(cfg, args.ckpt1, args.ckpt2, args.ckpt3, args.resume)
