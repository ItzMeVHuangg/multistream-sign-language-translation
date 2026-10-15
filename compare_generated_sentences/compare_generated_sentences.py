import argparse
import copy
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("TORCH_SHM_DIR", str(Path(tempfile.gettempdir()) / "torch_shm"))

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except (RuntimeError, ValueError):
    pass

from ablations.main_pipeline.train_main_pipeline import (  # noqa: E402
    RoverMainMBartSLT,
    load_config as load_main_config,
)
from data.dataset import PhoenixDataset, build_transforms, build_vocabularies  # noqa: E402
from models.translator_llm import load_mbart50_tokenizer  # noqa: E402
from training.train_slt import PhoenixSLTWithKeypoints, build_mbart_collate  # noqa: E402


def progress_miniters(loader):
    try:
        return max(1, len(loader) // 20)
    except TypeError:
        return 1


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def disable_backbone_pretraining(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for key in ("stream1", "stream3"):
        if key in cfg:
            cfg[key]["pretrained"] = False
    return cfg


def build_loader(cfg: dict, split: str, tokenizer, gloss_vocab, text_vocab, eval_cfg: dict) -> DataLoader:
    paths = cfg.get("paths", {})
    dc = cfg["data"]
    annot_root = Path(paths.get("annot_root")) if paths.get("annot_root") else None
    frames_root = Path(paths.get("frames_root")) if paths.get("frames_root") else None
    kpts_root = Path(paths["mediapipe_kpts_root"])
    cache_root = paths.get("frame_cache_root") or os.environ.get("PHOENIX_FRAME_CACHE")

    base = PhoenixDataset(
        split=split,
        gloss_vocab=gloss_vocab,
        text_vocab=text_vocab,
        max_frames=dc["max_frames"],
        temporal_stride=dc["temporal_stride"],
        transform=build_transforms(split, dc["img_height"], dc["img_width"]),
        return_translation=True,
        img_h=dc["img_height"],
        img_w=dc["img_width"],
        cache_root=Path(cache_root) if cache_root else None,
        cache_read=True,
        cache_write=False,
        frames_root=frames_root,
        annot_root=annot_root,
    )
    ds = PhoenixSLTWithKeypoints(base, kpts_root, dc["max_frames"], cfg["stream2"]["keypoint_dim"])
    collate = build_mbart_collate(
        tokenizer,
        cfg["slt"].get("max_seq_len", cfg["llm"].get("max_gen_length", 128)),
    )
    num_workers = int(eval_cfg.get("num_workers", dc.get("num_workers", 0)))
    return DataLoader(
        ds,
        batch_size=int(eval_cfg.get("batch_size", cfg["slt"].get("batch_size", 2))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(eval_cfg.get("pin_memory", dc.get("pin_memory", True))),
        prefetch_factor=dc.get("prefetch_factor", 2) if num_workers > 0 else None,
        persistent_workers=(num_workers > 0 and dc.get("persistent_workers", True)),
        collate_fn=collate,
    )


def decode_gloss(gloss_vocab, ids: List[int]) -> str:
    return " ".join(gloss_vocab.decode([int(i) for i in ids], skip_special=True))


def decode_settings(cfg: dict) -> tuple[bool, int]:
    beam_width = int(cfg["main_pipeline"].get("slt_beam_width", cfg["llm"].get("num_beams", 4)))
    return beam_width > 1, beam_width


def load_model(eval_cfg: dict, device: torch.device):
    cfg = load_main_config(str(resolve_path(eval_cfg["model"]["config"])))
    model_cfg_for_init = disable_backbone_pretraining(cfg)

    paths = cfg.get("paths", {})
    annot_root = Path(paths.get("annot_root")) if paths.get("annot_root") else None
    gloss_vocab, text_vocab = build_vocabularies(annot_root=annot_root)

    tokenizer = load_mbart50_tokenizer(cfg["llm"].get("model_name", "facebook/mbart-large-50"))
    tokenizer.src_lang = cfg["llm"].get("src_lang", "de_DE")
    tokenizer.tgt_lang = cfg["llm"].get("tgt_lang", "de_DE")

    model = RoverMainMBartSLT(
        model_cfg_for_init,
        len(gloss_vocab),
        gloss_vocab=gloss_vocab,
        tokenizer=tokenizer,
    ).to(device)
    ckpt_path = resolve_path(eval_cfg["model"]["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Main pipeline checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    model.eval()

    return cfg, model, tokenizer, gloss_vocab, text_vocab, ckpt


@torch.no_grad()
def collect_samples(eval_cfg: dict) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CompareGenerated] Device: {device}")
    if device.type == "cuda":
        print(f"[CompareGenerated] GPU: {torch.cuda.get_device_name(0)}")

    cfg, model, tokenizer, gloss_vocab, text_vocab, ckpt = load_model(eval_cfg, device)
    loader = build_loader(
        cfg,
        eval_cfg["eval"].get("split", "test"),
        tokenizer,
        gloss_vocab,
        text_vocab,
        eval_cfg["eval"],
    )

    bos_idx = tokenizer.lang_code_to_id.get(tokenizer.tgt_lang, tokenizer.eos_token_id)
    eos_idx = tokenizer.eos_token_id
    use_beam, beam_width = decode_settings(cfg)
    sample_limit = int(eval_cfg["eval"].get("num_samples", 10))
    rows: list[dict] = []

    for batch in tqdm(loader, desc="[CompareGenerated]", leave=False, miniters=progress_miniters(loader)):
        frames = batch["frames"].to(device)
        keypoints = batch["keypoints"].to(device)
        frame_lens = batch["frame_lens"].to(device)
        gloss = batch["gloss"].to(device)
        gloss_lens = batch["gloss_lens"].to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and eval_cfg["eval"].get("use_amp", True))):
            cslr_out = model.cslr(frames, keypoints, frame_lens)
            pred_gloss_ids = model._rover_gloss(cslr_out, cfg)
            pred_ids = model.generate(
                frames,
                keypoints,
                frame_lens,
                gloss,
                bos_idx,
                eos_idx,
                use_beam=use_beam,
                cfg=cfg,
            )

        hyp_sents = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref_sents = batch["translation_texts"]
        video_ids = batch.get("video_ids", [""] * len(hyp_sents))

        for b, hyp in enumerate(hyp_sents):
            rows.append(
                {
                    "index": len(rows) + 1,
                    "video_id": video_ids[b],
                    "ref_gloss": decode_gloss(gloss_vocab, gloss[b, : gloss_lens[b].item()].tolist()),
                    "pred_gloss": decode_gloss(gloss_vocab, pred_gloss_ids[b]),
                    "ref_sentence": ref_sents[b],
                    "hyp_sentence": hyp,
                    "beam_width": beam_width,
                    "checkpoint_epoch": ckpt.get("epoch", ""),
                    "checkpoint_bleu4": ckpt.get("bleu", ckpt.get("best_bleu", "")),
                }
            )
            if len(rows) >= sample_limit:
                return rows

    return rows


def write_outputs(rows: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "comparison_samples.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = out_dir / "comparison_samples.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    md_path = out_dir / "comparison_samples.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Generated Sentence Comparison\n\n")
        for row in rows:
            f.write(f"## {row['index']}. {row['video_id']}\n\n")
            f.write(f"- Reference gloss: `{row['ref_gloss']}`\n")
            f.write(f"- Predicted gloss: `{row['pred_gloss']}`\n")
            f.write(f"- Reference sentence: {row['ref_sentence']}\n")
            f.write(f"- Generated sentence: {row['hyp_sentence']}\n\n")

    print(f"[CompareGenerated] Wrote {len(rows)} samples to {out_dir}")
    print(f"  - {md_path}")
    print(f"  - {csv_path}")
    print(f"  - {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description="Export main-pipeline gloss and sentence comparison samples.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_compare_generated_sentences.yaml")))
    args = parser.parse_args()

    eval_cfg = load_yaml(resolve_path(args.config))
    rows = collect_samples(eval_cfg)
    if not rows:
        raise RuntimeError("No samples were collected. Check split, dataset paths, and checkpoint.")
    write_outputs(rows, resolve_path(eval_cfg.get("output_dir", "output/compare_generated_sentences")))


if __name__ == "__main__":
    main()
