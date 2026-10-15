"""
Evaluate a joint ThreeStreamCSLR checkpoint on PHOENIX-2014T.

This script loads a checkpoint saved by training/train_cslr.py, i.e. a full
ThreeStreamCSLR state_dict containing stream1/2/3 plus calibrated logit-fusion
parameters. It reports WER for:
  - weighted ROVER output
  - calibrated fused-logit CTC output
  - each diagnostic stream output
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from statistics import mean
from typing import List

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_SHM_DIR", str(Path(tempfile.gettempdir()) / "torch_shm"))

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import PhoenixDataset, build_transforms, build_vocabularies
from models.rover_ensemble import ThreeStreamCSLR
from training.train_cslr import PhoenixWithKeypoints, collate_fn_kpts
from utils.ctc_decoder import batch_ctc_decode
from utils.metrics import compute_wer


def _merge_eval_cfg(file_cfg: dict, ckpt_cfg: dict) -> dict:
    """Use checkpoint architecture/training cfg, but keep current path settings."""
    cfg = dict(ckpt_cfg or file_cfg)
    cfg["paths"] = dict(file_cfg.get("paths", cfg.get("paths", {})))

    # Avoid loading external torchvision pretrained weights; checkpoint supplies them.
    cfg.setdefault("stream1", {})["pretrained"] = False
    cfg.setdefault("stream3", {})["pretrained"] = False
    return cfg


def build_loader(cfg: dict, split: str, gloss_vocab, text_vocab, batch_size: int) -> DataLoader:
    dc = cfg["data"]
    paths = cfg.get("paths", {})
    kpts_root = Path(paths["mediapipe_kpts_root"])
    max_frames = dc["max_frames"]
    kp_dim = cfg.get("stream2", {}).get("keypoint_dim", 225)

    cache_root = paths.get("frame_cache_root") or os.environ.get("PHOENIX_FRAME_CACHE")
    cache_root = Path(cache_root) if cache_root else None

    base_ds = PhoenixDataset(
        split=split,
        gloss_vocab=gloss_vocab,
        text_vocab=text_vocab,
        max_frames=max_frames,
        temporal_stride=dc["temporal_stride"],
        transform=build_transforms(split, dc["img_height"], dc["img_width"]),
        return_translation=False,
        img_h=dc["img_height"],
        img_w=dc["img_width"],
        cache_root=cache_root,
        cache_read=True,
        cache_write=False,
        frames_root=paths.get("frames_root"),
        annot_root=paths.get("annot_root"),
    )
    ds = PhoenixWithKeypoints(base_ds, kpts_root, max_frames, kp_dim, load_frames=True, load_kpts=True)

    num_workers = dc.get("num_workers", 4)
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        num_workers = min(num_workers, max(0, int(slurm_cpus)))

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=dc.get("prefetch_factor", 1) if num_workers > 0 else None,
        collate_fn=collate_fn_kpts,
    )


def _decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank: int, mode: str, beam_width: int):
    return batch_ctc_decode(
        log_probs,
        lengths.clamp(max=log_probs.shape[0]),
        blank_idx=blank,
        mode=mode,
        beam_width=beam_width,
    )


@torch.no_grad()
def evaluate(model: ThreeStreamCSLR, loader: DataLoader, gloss_vocab, cfg: dict, args, device) -> dict:
    model.eval()
    blank = cfg["cslr"]["ctc_blank_idx"]

    refs: List[List[str]] = []
    hyps = {"rover": [], "fused": [], "s1": [], "s2": [], "s3": []}
    blank_frames = {"s1": 0, "s2": 0, "s3": 0, "fused": 0}
    total_frames = {"s1": 0, "s2": 0, "s3": 0, "fused": 0}

    for batch in tqdm(loader, desc=f"[Joint Eval:{args.split}]", leave=True):
        frames = batch["frames"].to(device)
        keypoints = batch["keypoints"].to(device)
        frame_lens = batch["frame_lens"].to(device)
        gloss = batch["gloss"].to(device)
        gloss_lens = batch["gloss_lens"].to(device)

        out = model(frames, keypoints, frame_lens)
        lp_map = {
            "s1": (out["s1_log_probs"], out["s1_lens"]),
            "s2": (out["s2_log_probs"], out["s2_lens"]),
            "s3": (out["s3_log_probs"], out["s3_lens"]),
            "fused": (out["fused_log_probs"], out["fused_lens"]),
        }

        rover_ids = model.rover(
            [out["s1_log_probs"], out["s2_log_probs"], out["s3_log_probs"]],
            [out["s1_lens"], out["s2_lens"], out["s3_lens"]],
        )

        pred_ids = {
            name: _decode(lp, lens, blank, args.decode, args.beam_width)
            for name, (lp, lens) in lp_map.items()
        }

        for name, (lp, lens) in lp_map.items():
            lens = lens.clamp(max=lp.shape[0])
            best_path = lp.argmax(dim=-1).transpose(0, 1)
            for b in range(best_path.shape[0]):
                actual_t = int(lens[b].item())
                if actual_t > 0:
                    path = best_path[b, :actual_t]
                    blank_frames[name] += int((path == blank).sum().item())
                    total_frames[name] += actual_t

        for b in range(frames.shape[0]):
            ref = gloss_vocab.decode(gloss[b, : gloss_lens[b].item()].tolist(), skip_special=False)
            refs.append(ref)
            hyps["rover"].append(gloss_vocab.decode(rover_ids[b]))
            for name in ("fused", "s1", "s2", "s3"):
                hyps[name].append(gloss_vocab.decode(pred_ids[name][b]))

    wers = {name: compute_wer(seq, refs) for name, seq in hyps.items()}

    def avg_len(items):
        return mean(len(x) for x in items) if items else 0.0

    diagnostics = {
        "avg_ref_len": avg_len(refs),
        **{f"avg_{name}_len": avg_len(seq) for name, seq in hyps.items()},
        **{f"blank_{name}": blank_frames[name] / max(total_frames[name], 1) for name in blank_frames},
    }
    return {"wers": wers, "refs": refs, "hyps": hyps, "diagnostics": diagnostics}


def print_results(result: dict, split: str, ckpt_path: str):
    wers = result["wers"]
    diag = result["diagnostics"]
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"Joint ThreeStreamCSLR Evaluation | split={split}")
    print(f"Checkpoint: {ckpt_path}")
    print(
        f"lengths avg ref={diag['avg_ref_len']:.2f} "
        f"rover={diag['avg_rover_len']:.2f} fused={diag['avg_fused_len']:.2f} "
        f"s1={diag['avg_s1_len']:.2f} s2={diag['avg_s2_len']:.2f} s3={diag['avg_s3_len']:.2f}"
    )
    print(
        f"blank frames fused={diag['blank_fused']*100:.1f}% "
        f"s1={diag['blank_s1']*100:.1f}% s2={diag['blank_s2']*100:.1f}% "
        f"s3={diag['blank_s3']*100:.1f}%"
    )
    print(sep)
    print(f"ROVER output       WER: {wers['rover']*100:6.2f}%")
    print(f"Fused-logit output WER: {wers['fused']*100:6.2f}%")
    print(f"Stream 1           WER: {wers['s1']*100:6.2f}%")
    print(f"Stream 2           WER: {wers['s2']*100:6.2f}%")
    print(f"Stream 3           WER: {wers['s3']*100:6.2f}%")
    print(sep)


def save_hyps(result: dict, out_dir: Path, split: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, seqs in result["hyps"].items():
        path = out_dir / f"{split}_hyp_{name}.txt"
        path.write_text("\n".join(" ".join(tokens) for tokens in seqs) + "\n", encoding="utf-8")
        print(f"Saved: {path}")
    ref_path = out_dir / f"{split}_ref.txt"
    ref_path.write_text("\n".join(" ".join(tokens) for tokens in result["refs"]) + "\n", encoding="utf-8")
    print(f"Saved: {ref_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a joint ThreeStreamCSLR checkpoint")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--ckpt", default="checkpoints/three_stream/best_three_stream_rover.pth")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--decode", default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_hyps", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        file_cfg = yaml.safe_load(f)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _merge_eval_cfg(file_cfg, ckpt.get("cfg", file_cfg))

    paths = cfg.get("paths", {})
    annot_root = Path(paths["annot_root"]) if paths.get("annot_root") else None
    gloss_vocab, text_vocab = build_vocabularies(annot_root=annot_root)
    num_classes = len(gloss_vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Joint Eval] device={device}")
    if device.type == "cuda":
        print(f"[Joint Eval] gpu={torch.cuda.get_device_name(0)}")

    model = ThreeStreamCSLR(
        num_classes=num_classes,
        cfg=cfg,
        share_swin=cfg.get("three_stream", {}).get("share_swin", True),
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    if missing or unexpected:
        print(f"[Joint Eval] missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()

    batch_size = args.batch_size or cfg.get("three_stream", {}).get("batch_size", 2)
    loader = build_loader(cfg, args.split, gloss_vocab, text_vocab, batch_size)
    print(f"[Joint Eval] split={args.split} samples={len(loader.dataset)} batches={len(loader)} batch_size={batch_size}")
    print(f"[Joint Eval] checkpoint epoch={ckpt.get('epoch', '?')} wer={ckpt.get('wer', '?')}")
    if "metrics" in ckpt:
        print(f"[Joint Eval] checkpoint metrics={ckpt['metrics']}")

    result = evaluate(model, loader, gloss_vocab, cfg, args, device)
    print_results(result, args.split, args.ckpt)

    if args.save_hyps:
        save_hyps(result, Path("output") / "eval_joint_hyps", args.split)


if __name__ == "__main__":
    main()
