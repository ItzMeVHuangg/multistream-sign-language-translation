import argparse
import copy
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

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
from ablations.metrics_fallback import safe_translation_metrics  # noqa: E402
from data.dataset import PhoenixDataset, build_transforms, build_vocabularies  # noqa: E402
from models.translator_llm import load_mbart50_tokenizer  # noqa: E402
from training.train_slt import PhoenixSLTWithKeypoints, build_mbart_collate  # noqa: E402
from visual_feature_ablation.train_main_novf import RoverMainNoVisualMBartSLT  # noqa: E402


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


def load_model(variant_name: str, variant_cfg: dict, device: torch.device):
    cfg = load_main_config(str(resolve_path(variant_cfg["config"])))
    model_cfg_for_init = disable_backbone_pretraining(cfg)

    paths = cfg.get("paths", {})
    annot_root = Path(paths.get("annot_root")) if paths.get("annot_root") else None
    gloss_vocab, text_vocab = build_vocabularies(annot_root=annot_root)

    model_cls = (
        RoverMainNoVisualMBartSLT
        if cfg.get("main_pipeline", {}).get("disable_visual_features", False)
        else RoverMainMBartSLT
    )
    model = model_cls(model_cfg_for_init, len(gloss_vocab)).to(device)
    ckpt_path = resolve_path(variant_cfg["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{variant_name} checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    model.eval()

    tokenizer = load_mbart50_tokenizer(cfg["llm"].get("model_name", "facebook/mbart-large-50"))
    tokenizer.src_lang = cfg["llm"].get("src_lang", "de_DE")
    tokenizer.tgt_lang = cfg["llm"].get("tgt_lang", "de_DE")
    return cfg, model, tokenizer, gloss_vocab, text_vocab, ckpt


def decode_settings(cfg: dict) -> Tuple[bool, int]:
    beam_width = int(cfg["main_pipeline"].get("slt_beam_width", cfg["llm"].get("num_beams", 4)))
    return beam_width > 1, beam_width


@torch.no_grad()
def evaluate_split(
    variant_name: str,
    cfg: dict,
    model,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[dict, List[dict]]:
    bos_idx = tokenizer.lang_code_to_id.get(tokenizer.tgt_lang, tokenizer.eos_token_id)
    eos_idx = tokenizer.eos_token_id
    use_beam, beam_width = decode_settings(cfg)

    hypotheses: List[str] = []
    references: List[str] = []
    rows: List[dict] = []

    for batch in tqdm(loader, desc=f"[Eval {variant_name}]", leave=False, miniters=progress_miniters(loader)):
        frames = batch["frames"].to(device)
        keypoints = batch["keypoints"].to(device)
        frame_lens = batch["frame_lens"].to(device)
        gloss = batch["gloss"].to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and use_amp)):
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

        hyps = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = batch["translation_texts"]
        video_ids = batch.get("video_ids", [""] * len(hyps))
        hypotheses.extend(hyps)
        references.extend(refs)
        for video_id, hyp, ref in zip(video_ids, hyps, refs):
            rows.append({"video_id": video_id, "hyp": hyp, "ref": ref})

    bleu, rouge, meteor = safe_translation_metrics(hypotheses, references)
    metrics = {
        "num_samples": len(hypotheses),
        "beam_width": beam_width,
        "disable_visual_features": bool(cfg["main_pipeline"].get("disable_visual_features", False)),
        "bleu1": bleu["bleu1"],
        "bleu2": bleu["bleu2"],
        "bleu3": bleu["bleu3"],
        "bleu4": bleu["bleu"],
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "rouge1_pct": rouge["rouge1"] * 100.0,
        "rouge2_pct": rouge["rouge2"] * 100.0,
        "rougeL_pct": rouge["rougeL"] * 100.0,
        "meteor": meteor,
    }
    return metrics, rows


def write_predictions(path: Path, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            f.write(f"[{i}] video_id: {row['video_id']}\n")
            f.write(f"[{i}] hyp: {row['hyp']}\n")
            f.write(f"[{i}] ref: {row['ref']}\n\n")


def write_summary(out_dir: Path, results: Dict[str, Dict[str, dict]]):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    fields = [
        "variant",
        "split",
        "disable_visual_features",
        "num_samples",
        "beam_width",
        "bleu1",
        "bleu2",
        "bleu3",
        "bleu4",
        "rouge1_pct",
        "rouge2_pct",
        "rougeL_pct",
        "meteor",
    ]
    with open(out_dir / "metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for variant_name, by_split in results.items():
            for split, metrics in by_split.items():
                row = {"variant": variant_name, "split": split}
                row.update({k: metrics[k] for k in fields if k in metrics})
                writer.writerow(row)

    with open(out_dir / "metrics_delta.csv", "w", encoding="utf-8", newline="") as f:
        fields_delta = ["split", "metric", "with_visual", "no_visual", "delta_no_visual_minus_with_visual"]
        writer = csv.DictWriter(f, fieldnames=fields_delta)
        writer.writeheader()
        if "with_visual" in results and "no_visual" in results:
            for split in sorted(set(results["with_visual"]) & set(results["no_visual"])):
                for metric in ("bleu1", "bleu2", "bleu3", "bleu4", "rougeL_pct", "meteor"):
                    with_v = results["with_visual"][split][metric]
                    no_v = results["no_visual"][split][metric]
                    writer.writerow(
                        {
                            "split": split,
                            "metric": metric,
                            "with_visual": with_v,
                            "no_visual": no_v,
                            "delta_no_visual_minus_with_visual": no_v - with_v,
                        }
                    )

    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        for variant_name, by_split in results.items():
            f.write(f"\n{variant_name}\n")
            f.write("-" * len(variant_name) + "\n")
            for split, m in by_split.items():
                f.write(
                    f"{split}: BLEU-1={m['bleu1']:.2f} BLEU-2={m['bleu2']:.2f} "
                    f"BLEU-3={m['bleu3']:.2f} BLEU-4={m['bleu4']:.2f} | "
                    f"ROUGE-L={m['rougeL_pct']:.2f} | METEOR={m['meteor']:.4f} | "
                    f"disable_visual={m['disable_visual_features']}\n"
                )


def main():
    parser = argparse.ArgumentParser(description="Compare main pipeline with and without visual features.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_eval_visual_feature.yaml")))
    args = parser.parse_args()

    eval_cfg = load_yaml(resolve_path(args.config))
    out_dir = resolve_path(eval_cfg.get("output_dir", "output/visual_feature_ablation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VisualFeatureEval] Device: {device}")
    if device.type == "cuda":
        print(f"[VisualFeatureEval] GPU: {torch.cuda.get_device_name(0)}")

    results: Dict[str, Dict[str, dict]] = {}
    for variant_name, variant_cfg in eval_cfg["variants"].items():
        print(f"\n[VisualFeatureEval] Loading {variant_name}")
        cfg, model, tokenizer, gloss_vocab, text_vocab, ckpt = load_model(variant_name, variant_cfg, device)
        results[variant_name] = {}

        for split in eval_cfg["eval"].get("splits", ["dev", "test"]):
            print(f"[VisualFeatureEval] {variant_name} split={split}")
            loader = build_loader(cfg, split, tokenizer, gloss_vocab, text_vocab, eval_cfg["eval"])
            metrics, rows = evaluate_split(
                variant_name,
                cfg,
                model,
                tokenizer,
                loader,
                device,
                use_amp=bool(eval_cfg["eval"].get("use_amp", True)),
            )
            results[variant_name][split] = metrics
            write_predictions(out_dir / f"predictions_{variant_name}_{split}.txt", rows)
            print(
                f"  BLEU-4={metrics['bleu4']:.2f} | ROUGE-L={metrics['rougeL_pct']:.2f} | "
                f"METEOR={metrics['meteor']:.4f}"
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_summary(out_dir, results)
    print(f"\n[VisualFeatureEval] Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
