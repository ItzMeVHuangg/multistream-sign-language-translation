"""
eval_three_stream.py
====================
Evaluate the Three-Stream CSLR pipeline using three independently-trained
stream checkpoints (best_stream1.pth, best_stream2.pth, best_stream3.pth).

Strategy
--------
Each stream was trained independently with train_single_stream.py, so the
saved state_dicts map to individual Stream1/2/3 modules — NOT ThreeStreamCSLR.
This script:
  1. Instantiates each stream independently.
  2. Loads its checkpoint (strict=True).
  3. Wraps them into an ad-hoc ensemble evaluator that mirrors ROVER logic
     from ThreeStreamCSLR without re-training.

Usage
-----
  python -m training.eval_three_stream \\
      --config   configs/config.yaml \\
      --ckpt1    checkpoints/stream1_only/best_stream1.pth \\
      --ckpt2    checkpoints/stream2_only/best_stream2.pth \\
      --ckpt3    checkpoints/stream3_only/best_stream3.pth \\
      --split    test \\
      --decode   beam \\
      --beam_width 10 \\
      --batch_size 2

Metrics reported
----------------
  - WER (%) per stream  (greedy decode for diagnostic)
  - WER (%) ROVER ensemble (beam decode + bigram LM)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from statistics import mean
from typing import List, Tuple

# ── env tweaks before any torch import ─────────────────────────────────────
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TORCH_SHM_DIR', str(Path(tempfile.gettempdir()) / 'torch_shm'))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset            import (PhoenixDataset, build_vocabularies,
                                     build_transforms)
from models.stream1_rgb_ustm import Stream1_RGB_USTM
from models.stream2_hypergraph import Stream2_Hypergraph
from models.stream3_crossattn import Stream3_CrossAttn
from models.rover_ensemble   import ROVEREnsemble
from utils.ctc_decoder        import batch_ctc_decode
from utils.metrics            import compute_wer
from training.train_cslr      import PhoenixWithKeypoints, collate_fn_kpts

try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except (RuntimeError, ValueError):
    pass


def _file_sha1(path: str, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            digest.update(block)
    return digest.hexdigest()[:12]


# ───────────────────────────────────────────────────────────────────────────
# Dataset / DataLoader helpers
# ───────────────────────────────────────────────────────────────────────────

def build_eval_loader(
    cfg: dict,
    split: str,
    gloss_vocab,
    text_vocab,
    batch_size: int,
) -> DataLoader:
    dc        = cfg['data']
    kpts_root = Path(cfg['paths']['mediapipe_kpts_root'])
    max_fr    = dc['max_frames']
    kp_dim    = cfg.get('stream2', {}).get('keypoint_dim', 225)

    cache_root = (cfg.get('paths', {}).get('frame_cache_root')
                  or os.environ.get('PHOENIX_FRAME_CACHE'))
    cache_root = Path(cache_root) if cache_root else None

    base_ds = PhoenixDataset(
        split              = split,
        gloss_vocab        = gloss_vocab,
        text_vocab         = text_vocab,
        max_frames         = max_fr,
        temporal_stride    = dc['temporal_stride'],
        transform          = build_transforms(split, dc['img_height'], dc['img_width']),
        return_translation = False,
        img_h              = dc['img_height'],
        img_w              = dc['img_width'],
        cache_root         = cache_root,
        cache_read         = True,
        cache_write        = False,
        frames_root        = cfg.get('paths', {}).get('frames_root'),
        annot_root         = cfg.get('paths', {}).get('annot_root'),
    )
    ds = PhoenixWithKeypoints(base_ds, kpts_root, max_fr, kp_dim,
                               load_frames=True, load_kpts=True)

    num_workers = dc.get('num_workers', 4)
    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = False,
        prefetch_factor = dc.get('prefetch_factor', 1) if num_workers > 0 else None,
        collate_fn  = collate_fn_kpts,
    )
    return loader


# ───────────────────────────────────────────────────────────────────────────
# Model builders — each stream loaded independently
# ───────────────────────────────────────────────────────────────────────────

def _load_stream1(cfg: dict, num_classes: int, ckpt_path: str,
                  device: torch.device) -> Stream1_RGB_USTM:
    print(f'  [Stream 1] Checkpoint path: {Path(ckpt_path).resolve()} | sha1={_file_sha1(ckpt_path)}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_cfg = ckpt.get('cfg', cfg)
    s_cfg = ckpt_cfg.get('stream1', cfg.get('stream1', {}))
    model = Stream1_RGB_USTM(
        pretrained          = False,          # weights come from checkpoint
        num_classes         = num_classes,
        d_model             = s_cfg.get('d_model', 512),
        tape_kernel         = s_cfg.get('tape_kernel', 3),
        tape_dilation_short = s_cfg.get('tape_dilation_short', 1),
        tape_dilation_long  = s_cfg.get('tape_dilation_long', 3),
        tape_max_len        = s_cfg.get('tape_max_len', 512),
        tape_dropout        = s_cfg.get('tape_dropout', 0.1),
        temporal_dropout    = s_cfg.get('temporal_dropout', 0.1),
        temporal_lstm_layers = s_cfg.get('temporal_lstm_layers', 1),
        proj_dropout        = s_cfg.get('proj_dropout', 0.1),
        ctc_dropout         = s_cfg.get('ctc_dropout', 0.1),
        use_checkpoint      = False,          # no grad-ckpt at inference
        backbone            = s_cfg.get('backbone', 'swint'),
        temporal_head       = s_cfg.get('temporal_head', 'mstcn'),
        conv_type           = s_cfg.get('conv_type', 3),
        adapter             = s_cfg.get('adapter', 3),
        hidden_size         = s_cfg.get('hidden_size', s_cfg.get('d_model', 512)),
        weight_norm         = s_cfg.get('weight_norm', True),
        share_classifier    = s_cfg.get('share_classifier', True),
        output_norm         = s_cfg.get('output_norm', False),
    )
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=True)
    print(f'  [Stream 1] Loaded from {Path(ckpt_path).name}'
          f'  (epoch={ckpt.get("epoch","?")}, WER={ckpt.get("wer", ckpt.get("best_wer","?"))})')
    return model.to(device).eval()


def _load_stream2(cfg: dict, num_classes: int, ckpt_path: str,
                  device: torch.device) -> Stream2_Hypergraph:
    print(f'  [Stream 2] Checkpoint path: {Path(ckpt_path).resolve()} | sha1={_file_sha1(ckpt_path)}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_cfg = ckpt.get('cfg', cfg)
    s_cfg = ckpt_cfg.get('stream2', cfg.get('stream2', {}))
    model = Stream2_Hypergraph(
        keypoint_dim = s_cfg.get('keypoint_dim', 225),
        num_joints   = s_cfg.get('num_joints', 75),
        keypoint_layout = s_cfg.get('keypoint_layout', 'pose_hands'),
        node_dim     = s_cfg.get('node_dim', 128),
        d_model      = s_cfg.get('d_model', 512),
        num_proto    = s_cfg.get('num_proto', 16),
        knn_k        = s_cfg.get('knn_k', 5),
        ms_kernels   = s_cfg.get('ms_kernels', [1, 3, 5, 7]),
        dropout      = s_cfg.get('dropout', 0.1),
        ctc_dropout  = s_cfg.get('ctc_dropout', 0.1),
        normalize_keypoints = s_cfg.get('normalize_keypoints', True),
        temporal_lstm_layers = s_cfg.get('temporal_lstm_layers', 1),
        num_classes  = num_classes,
    )
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=True)
    print(f'  [Stream 2] Loaded from {Path(ckpt_path).name}'
          f'  (epoch={ckpt.get("epoch","?")}, WER={ckpt.get("wer", ckpt.get("best_wer","?"))})')
    return model.to(device).eval()


def _load_stream3(cfg: dict, num_classes: int, ckpt_path: str,
                  device: torch.device) -> Stream3_CrossAttn:
    print(f'  [Stream 3] Checkpoint path: {Path(ckpt_path).resolve()} | sha1={_file_sha1(ckpt_path)}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_cfg = ckpt.get('cfg', cfg)
    s_cfg = ckpt_cfg.get('stream3', cfg.get('stream3', {}))
    model = Stream3_CrossAttn(
        keypoint_dim   = s_cfg.get('keypoint_dim', 225),
        d_model        = s_cfg.get('d_model', 512),
        nhead          = s_cfg.get('nhead', 8),
        bilstm_hidden  = s_cfg.get('bilstm_hidden', 512),
        bilstm_layers  = s_cfg.get('bilstm_layers', 2),
        num_classes    = num_classes,
        attn_dropout   = s_cfg.get('attn_dropout', 0.1),
        bilstm_dropout = s_cfg.get('bilstm_dropout', 0.3),
        ctc_dropout    = s_cfg.get('ctc_dropout', 0.1),
        pretrained     = False,               # weights come from checkpoint
        use_checkpoint = False,
        swin_ref       = None,                # independent Swin at eval time
    )
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=True)
    print(f'  [Stream 3] Loaded from {Path(ckpt_path).name}'
          f'  (epoch={ckpt.get("epoch","?")}, WER={ckpt.get("wer", ckpt.get("best_wer","?"))})')
    return model.to(device).eval()


# ───────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    stream1:     Stream1_RGB_USTM,
    stream2:     Stream2_Hypergraph,
    stream3:     Stream3_CrossAttn,
    rover:       ROVEREnsemble,
    loader:      DataLoader,
    gloss_vocab,
    cfg:         dict,
    decode_mode: str  = 'beam',
    beam_width:  int  = 10,
    device:      torch.device = torch.device('cpu'),
) -> dict:
    """
    Run all three streams + ROVER ensemble over *loader*.

    Returns a dict with keys:
        wer_s1, wer_s2, wer_s3, wer_rover   (float, 0–1)
        hyp_s1, hyp_s2, hyp_s3, hyp_rover   (List[List[str]])
        ref                                   (List[List[str]])
    """
    blank  = cfg['cslr']['ctc_blank_idx']
    eval_cfg = cfg.get('eval', {})
    s1_blank_bias = float(eval_cfg.get('s1_blank_bias', 0.0))

    all_ref      : List[List[str]] = []
    all_hyp_s1   : List[List[str]] = []
    all_hyp_s2   : List[List[str]] = []
    all_hyp_s3   : List[List[str]] = []
    all_hyp_rover: List[List[str]] = []
    blank_frames = {'s1': 0, 's2': 0, 's3': 0}
    total_frames = {'s1': 0, 's2': 0, 's3': 0}

    for batch in tqdm(loader, desc='[3Stream Eval]', leave=True):
        frames     = batch['frames'].to(device)
        keypoints  = batch['keypoints'].to(device)
        frame_lens = batch['frame_lens'].to(device)
        gloss      = batch['gloss'].to(device)
        gloss_lens = batch['gloss_lens'].to(device)

        # ── forward each stream ──────────────────────────────────────────
        lp1, _, l1 = stream1(frames, frame_lens)
        lp2, _, l2 = stream2(keypoints, frame_lens)
        lp3, _, l3 = stream3(frames, keypoints, frame_lens)

        if s1_blank_bias != 0.0:
            lp1 = lp1.clone()
            lp1[:, :, blank] += s1_blank_bias
            lp1 = torch.log_softmax(lp1, dim=-1)

        # Clamp lengths to actual output length
        l1 = l1.clamp(max=lp1.shape[0])
        l2 = l2.clamp(max=lp2.shape[0])
        l3 = l3.clamp(max=lp3.shape[0])

        for name, lp, lens in [('s1', lp1, l1), ('s2', lp2, l2), ('s3', lp3, l3)]:
            best_path = lp.argmax(dim=-1).transpose(0, 1)
            for b in range(best_path.shape[0]):
                actual_t = int(lens[b].item())
                if actual_t > 0:
                    path = best_path[b, :actual_t]
                    blank_frames[name] += int((path == blank).sum().item())
                    total_frames[name] += actual_t

        # ── ROVER ensemble ───────────────────────────────────────────────
        rover_seqs = rover([lp1, lp2, lp3], [l1, l2, l3])

        # ── Per-stream decode (diagnostic) ───────────────────────────────
        def _decode(lp: torch.Tensor, lens: torch.Tensor) -> List[List[int]]:
            return batch_ctc_decode(
                lp, lens,
                blank_idx  = blank,
                mode       = decode_mode,
                beam_width = beam_width,
            )

        preds_s1 = _decode(lp1, l1)
        preds_s2 = _decode(lp2, l2)
        preds_s3 = _decode(lp3, l3)

        # ── Collect results ───────────────────────────────────────────────
        B = frames.shape[0]
        for b in range(B):
            ref = gloss_vocab.decode(
                gloss[b, :gloss_lens[b].item()].tolist(), skip_special=False
            )
            all_ref.append(ref)
            all_hyp_s1.append(gloss_vocab.decode(preds_s1[b]))
            all_hyp_s2.append(gloss_vocab.decode(preds_s2[b]))
            all_hyp_s3.append(gloss_vocab.decode(preds_s3[b]))
            all_hyp_rover.append(gloss_vocab.decode(rover_seqs[b]))

    wer_s1 = compute_wer(all_hyp_s1, all_ref)
    wer_s2 = compute_wer(all_hyp_s2, all_ref)
    wer_s3 = compute_wer(all_hyp_s3, all_ref)
    wer_rover = compute_wer(all_hyp_rover, all_ref)

    def _avg_len(items):
        return mean(len(x) for x in items) if items else 0.0

    diagnostics = {
        'avg_ref_len': _avg_len(all_ref),
        'avg_s1_len': _avg_len(all_hyp_s1),
        'avg_s2_len': _avg_len(all_hyp_s2),
        'avg_s3_len': _avg_len(all_hyp_s3),
        'avg_rover_len': _avg_len(all_hyp_rover),
        'blank_s1': blank_frames['s1'] / max(total_frames['s1'], 1),
        'blank_s2': blank_frames['s2'] / max(total_frames['s2'], 1),
        'blank_s3': blank_frames['s3'] / max(total_frames['s3'], 1),
    }

    return {
        'wer_s1':    wer_s1,
        'wer_s2':    wer_s2,
        'wer_s3':    wer_s3,
        'wer_rover': wer_rover,
        'hyp_s1':    all_hyp_s1,
        'hyp_s2':    all_hyp_s2,
        'hyp_s3':    all_hyp_s3,
        'hyp_rover': all_hyp_rover,
        'ref':       all_ref,
        'n_samples': len(all_ref),
        'diagnostics': diagnostics,
    }


# ───────────────────────────────────────────────────────────────────────────
# Pretty print
# ───────────────────────────────────────────────────────────────────────────

def print_results(results: dict, split: str, decode_mode: str, beam_width: int):
    n = results['n_samples']
    sep = '=' * 68
    print(f'\n{sep}')
    print(f'  Three-Stream CSLR Evaluation  |  split={split}  |  n={n}')
    print(f'  CTC decode: {decode_mode} (beam_width={beam_width})')
    diag = results.get('diagnostics', {})
    if diag:
        print(
            f'  lens avg ref={diag["avg_ref_len"]:.2f} '
            f's1={diag["avg_s1_len"]:.2f} s2={diag["avg_s2_len"]:.2f} '
            f's3={diag["avg_s3_len"]:.2f} rover={diag["avg_rover_len"]:.2f}'
        )
        print(
            f'  blank frames s1={diag["blank_s1"]*100:.1f}% '
            f's2={diag["blank_s2"]*100:.1f}% s3={diag["blank_s3"]*100:.1f}%'
        )
    print(sep)
    print(f'  Stream 1 (RGB-USTM)        WER : {results["wer_s1"]*100:6.2f} %')
    print(f'  Stream 2 (Hypergraph)      WER : {results["wer_s2"]*100:6.2f} %')
    print(f'  Stream 3 (CrossAttn)       WER : {results["wer_s3"]*100:6.2f} %')
    print(f'  ─' * 34)
    print(f'  ROVER Ensemble             WER : {results["wer_rover"]*100:6.2f} %  ★')
    print(sep)


def save_hypotheses(results: dict, out_dir: Path, split: str):
    """Save per-stream and ensemble hypotheses to text files for later analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, hyps: List[List[str]]):
        fpath = out_dir / f'{split}_{name}.txt'
        with open(fpath, 'w', encoding='utf-8') as f:
            for tokens in hyps:
                f.write(' '.join(tokens) + '\n')
        print(f'  Saved: {fpath}')

    _write('hyp_stream1', results['hyp_s1'])
    _write('hyp_stream2', results['hyp_s2'])
    _write('hyp_stream3', results['hyp_s3'])
    _write('hyp_rover',   results['hyp_rover'])
    _write('ref',         results['ref'])


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Three-Stream CSLR with independent checkpoints + ROVER')
    parser.add_argument('--config',      default='configs/config.yaml',
                        help='Path to config YAML')
    parser.add_argument('--ckpt1',       default='checkpoints/stream1_only/best_stream1.pth',
                        help='Stream 1 checkpoint')
    parser.add_argument('--ckpt2',       default='checkpoints/stream2_only/best_stream2.pth',
                        help='Stream 2 checkpoint')
    parser.add_argument('--ckpt3',       default='checkpoints/stream3_only/best_stream3.pth',
                        help='Stream 3 checkpoint')
    parser.add_argument('--split',       default='test',
                        choices=['train', 'dev', 'test'])
    parser.add_argument('--decode',      default='beam',
                        choices=['greedy', 'beam'],
                        help='Per-stream CTC decode mode (ROVER always uses beam internally)')
    parser.add_argument('--beam_width',  type=int, default=10)
    parser.add_argument('--lm_weight',   type=float, default=None,
                        help='Override ensemble.lm_weight from config')
    parser.add_argument('--no_lm',       action='store_true',
                        help='Disable bigram LM in ROVER (ablation)')
    parser.add_argument('--batch_size',  type=int, default=None,
                        help='Override data batch_size (useful for low-VRAM GPUs)')
    parser.add_argument('--s1_blank_bias', type=float, default=None,
                        help='Add this bias to Stream 1 blank logit before decode/ROVER')
    parser.add_argument('--save_hyps',   action='store_true',
                        help='Save hypothesis files to output/eval_hyps/')
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.s1_blank_bias is not None:
        cfg.setdefault('eval', {})['s1_blank_bias'] = args.s1_blank_bias

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n[eval_three_stream] Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU : {torch.cuda.get_device_name(0)}')
        print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    # ── Vocabularies ─────────────────────────────────────────────────────
    gloss_vocab, text_vocab = build_vocabularies()
    num_classes = len(gloss_vocab)
    print(f'\n[Vocab] Gloss vocabulary: {num_classes} tokens')

    # ── DataLoader ───────────────────────────────────────────────────────
    batch_size = args.batch_size or cfg['three_stream'].get('batch_size', 2)
    print(f'\n[Data] Split={args.split}  batch_size={batch_size}')
    loader = build_eval_loader(cfg, args.split, gloss_vocab, text_vocab, batch_size)
    print(f'  {len(loader.dataset)} samples  /  {len(loader)} batches')

    # ── Load streams ─────────────────────────────────────────────────────
    print('\n[Models] Loading stream checkpoints …')
    stream1 = _load_stream1(cfg, num_classes, args.ckpt1, device)
    stream2 = _load_stream2(cfg, num_classes, args.ckpt2, device)
    stream3 = _load_stream3(cfg, num_classes, args.ckpt3, device)

    # ── ROVER ─────────────────────────────────────────────────────────────
    ens_cfg    = cfg.get('ensemble', {})
    lm_weight  = args.lm_weight if args.lm_weight is not None else ens_cfg.get('lm_weight', 0.3)
    use_lm     = not args.no_lm and ens_cfg.get('use_lm', True)
    rover_bw   = ens_cfg.get('beam_width', args.beam_width)

    rover = ROVEREnsemble(
        vocab_size  = num_classes,
        blank_idx   = cfg['cslr']['ctc_blank_idx'],
        beam_width  = rover_bw,
        lm_weight   = lm_weight,
        use_lm      = use_lm,
        stream_weights = ens_cfg.get('stream_weights', [1.0, 1.0, 1.0]),
    ).to(device)
    print(f'\n[ROVER] beam_width={rover_bw} | use_lm={use_lm} | lm_weight={lm_weight} | '
          f'stream_weights={ens_cfg.get("stream_weights", [1.0, 1.0, 1.0])}')

    # ── Evaluate ──────────────────────────────────────────────────────────
    print(f'\n[Eval] Running evaluation on split="{args.split}" …\n')
    if cfg.get('eval', {}).get('s1_blank_bias', 0.0):
        print(f'[Eval] Stream 1 blank bias: {cfg["eval"]["s1_blank_bias"]:+.2f}')
    results = evaluate(
        stream1, stream2, stream3, rover,
        loader, gloss_vocab, cfg,
        decode_mode = args.decode,
        beam_width  = args.beam_width,
        device      = device,
    )

    print_results(results, args.split, args.decode, args.beam_width)

    if args.save_hyps:
        out_dir = Path('output') / 'eval_hyps'
        print(f'\n[Save] Writing hypothesis files to {out_dir} …')
        save_hypotheses(results, out_dir, args.split)

    return results


if __name__ == '__main__':
    main()
