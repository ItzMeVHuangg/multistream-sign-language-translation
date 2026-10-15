import argparse
import copy
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

from ablations.baseline_s1_slt.train_baseline_s1_slt import (  # noqa: E402
    Stream1BaselineMBartSLT,
    load_config as load_s1_config,
    progress_miniters,
)
from ablations.baseline_s3_slt.train_baseline_s3_slt import (  # noqa: E402
    Stream3BaselineMBartSLT,
    load_config as load_s3_config,
)
from ablations.main_pipeline.train_main_pipeline import (  # noqa: E402
    RoverMainMBartSLT,
    load_config as load_main_config,
)
from data.dataset import PhoenixDataset, build_transforms, build_vocabularies  # noqa: E402
from models.translator_llm import load_mbart50_tokenizer  # noqa: E402
from training.train_slt import PhoenixSLTWithKeypoints, build_mbart_collate  # noqa: E402
from utils.metrics import compute_bleu, compute_rouge  # noqa: E402


MODEL_LOADERS = {
    "main_pipeline": (load_main_config, RoverMainMBartSLT),
    "baseline_s1": (load_s1_config, Stream1BaselineMBartSLT),
    "baseline_s3": (load_s3_config, Stream3BaselineMBartSLT),
}


def load_eval_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def disable_backbone_pretraining(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for key in ("stream1", "stream3"):
        if key in cfg:
            cfg[key]["pretrained"] = False
    return cfg


def build_loader(
    cfg: dict,
    split: str,
    tokenizer,
    gloss_vocab,
    text_vocab,
    eval_cfg: dict,
) -> DataLoader:
    paths = cfg.get("paths", {})
    dc = cfg["data"]
    annot_root = Path(paths.get("annot_root")) if paths.get("annot_root") else None
    frames_root = Path(paths.get("frames_root")) if paths.get("frames_root") else None
    kpts_root = Path(paths["mediapipe_kpts_root"])

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
        cache_root=Path(paths["frame_cache_root"]) if paths.get("frame_cache_root") else None,
        cache_read=True,
        cache_write=False,
        frames_root=frames_root,
        annot_root=annot_root,
    )
    ds = PhoenixSLTWithKeypoints(
        base,
        kpts_root,
        dc["max_frames"],
        cfg["stream2"]["keypoint_dim"],
    )
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


def load_model(model_name: str, model_cfg: dict, device: torch.device):
    config_loader, model_cls = MODEL_LOADERS[model_name]
    cfg = config_loader(model_cfg["config"])
    model_cfg_for_init = disable_backbone_pretraining(cfg)

    paths = cfg.get("paths", {})
    annot_root = Path(paths.get("annot_root")) if paths.get("annot_root") else None
    gloss_vocab, text_vocab = build_vocabularies(annot_root=annot_root)

    tokenizer = load_mbart50_tokenizer(cfg["llm"].get("model_name", "facebook/mbart-large-50"))
    tokenizer.src_lang = cfg["llm"].get("src_lang", "de_DE")
    tokenizer.tgt_lang = cfg["llm"].get("tgt_lang", "de_DE")

    if model_name == "main_pipeline":
        model = model_cls(
            model_cfg_for_init,
            len(gloss_vocab),
            gloss_vocab=gloss_vocab,
            tokenizer=tokenizer,
        ).to(device)
    else:
        model = model_cls(model_cfg_for_init, len(gloss_vocab)).to(device)
    ckpt_path = Path(model_cfg["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{model_name} checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    return cfg, model, tokenizer, gloss_vocab, text_vocab, ckpt


def decode_settings(model_name: str, cfg: dict) -> Tuple[bool, int]:
    if model_name == "main_pipeline":
        beam_width = int(cfg["main_pipeline"].get("slt_beam_width", cfg["llm"].get("num_beams", 4)))
    elif model_name == "baseline_s1":
        beam_width = int(cfg["baseline_s1"].get("beam_width", 1))
    elif model_name == "baseline_s3":
        beam_width = int(cfg["baseline_s3"].get("beam_width", 1))
    else:
        beam_width = 1
    return beam_width > 1, beam_width


@torch.no_grad()
def evaluate_split(
    model_name: str,
    cfg: dict,
    model,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[dict, List[dict]]:
    bos_idx = tokenizer.lang_code_to_id.get(tokenizer.tgt_lang, tokenizer.eos_token_id)
    eos_idx = tokenizer.eos_token_id
    use_beam, beam_width = decode_settings(model_name, cfg)

    hypotheses: List[str] = []
    references: List[str] = []
    rows: List[dict] = []

    for batch in tqdm(loader, desc=f"[Eval {model_name}]", leave=False, miniters=progress_miniters(loader)):
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
        hypotheses.extend(hyps)
        references.extend(refs)
        for hyp, ref in zip(hyps, refs):
            rows.append({"hyp": hyp, "ref": ref})

    bleu = compute_bleu(hypotheses, references)
    rouge = compute_rouge(hypotheses, references)
    metrics = {
        "num_samples": len(hypotheses),
        "beam_width": beam_width,
        "bleu1": bleu["bleu1"],
        "bleu2": bleu["bleu2"],
        "bleu3": bleu["bleu3"],
        "bleu4": bleu["bleu4"],
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "rouge1_pct": rouge["rouge1"] * 100.0,
        "rouge2_pct": rouge["rouge2"] * 100.0,
        "rougeL_pct": rouge["rougeL"] * 100.0,
    }
    return metrics, rows


def write_predictions(path: Path, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            f.write(f"[{i}] hyp: {row['hyp']}\n")
            f.write(f"[{i}] ref: {row['ref']}\n\n")


def iter_summary_rows(results: Dict[str, Dict[str, dict]]) -> List[dict]:
    rows = []
    for model_name, by_split in results.items():
        for split, metrics in by_split.items():
            row = {"model": model_name, "split": split}
            row.update(metrics)
            rows.append(row)
    return rows


def write_summary(out_dir: Path, results: Dict[str, Dict[str, dict]]):
    rows = iter_summary_rows(results)

    json_path = out_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    txt_path = out_dir / "metrics.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("SLT Evaluation Metrics\n")
        f.write("======================\n\n")
        f.write(
            f"{'Model':<16} {'Split':<6} {'N':>5} {'Beam':>4} "
            f"{'BLEU-1':>8} {'BLEU-2':>8} {'BLEU-3':>8} {'BLEU-4':>8} "
            f"{'ROUGE-1':>9} {'ROUGE-2':>9} {'ROUGE-L':>9}\n"
        )
        f.write("-" * 105 + "\n")
        for row in rows:
            f.write(
                f"{row['model']:<16} {row['split']:<6} {row['num_samples']:>5} {row['beam_width']:>4} "
                f"{row['bleu1']:>8.2f} {row['bleu2']:>8.2f} {row['bleu3']:>8.2f} {row['bleu4']:>8.2f} "
                f"{row['rouge1_pct']:>9.2f} {row['rouge2_pct']:>9.2f} {row['rougeL_pct']:>9.2f}\n"
            )

        f.write("\nDetailed by model\n")
        f.write("-----------------\n")
        for model_name, by_split in results.items():
            f.write(f"\n{model_name}\n")
            f.write("-" * len(model_name) + "\n")
            for split, m in by_split.items():
                f.write(
                    f"{split}: BLEU-1={m['bleu1']:.2f} BLEU-2={m['bleu2']:.2f} "
                    f"BLEU-3={m['bleu3']:.2f} BLEU-4={m['bleu4']:.2f} | "
                    f"ROUGE-1={m['rouge1_pct']:.2f} ROUGE-2={m['rouge2_pct']:.2f} "
                    f"ROUGE-L={m['rougeL_pct']:.2f}\n"
                )


def main():
    parser = argparse.ArgumentParser(description="Evaluate SLT checkpoints on dev/test")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_eval_slt.yaml")))
    args = parser.parse_args()

    eval_cfg = load_eval_config(args.config)
    out_dir = Path(eval_cfg.get("output_dir", "output/eval_slt"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EvalSLT] Device: {device}")
    if device.type == "cuda":
        print(f"[EvalSLT] GPU: {torch.cuda.get_device_name(0)}")

    results: Dict[str, Dict[str, dict]] = {}
    for model_name, model_cfg in eval_cfg["models"].items():
        if not model_cfg.get("enabled", True):
            continue
        print(f"\n[EvalSLT] Loading {model_name}")
        cfg, model, tokenizer, gloss_vocab, text_vocab, ckpt = load_model(model_name, model_cfg, device)
        results[model_name] = {}

        for split in eval_cfg["eval"].get("splits", ["dev", "test"]):
            print(f"[EvalSLT] {model_name} split={split}")
            loader = build_loader(cfg, split, tokenizer, gloss_vocab, text_vocab, eval_cfg["eval"])
            metrics, rows = evaluate_split(
                model_name,
                cfg,
                model,
                tokenizer,
                loader,
                device,
                use_amp=bool(eval_cfg["eval"].get("use_amp", True)),
            )
            results[model_name][split] = metrics
            write_predictions(out_dir / f"predictions_{model_name}_{split}.txt", rows)
            print(
                f"  BLEU-1={metrics['bleu1']:.2f} BLEU-2={metrics['bleu2']:.2f} "
                f"BLEU-3={metrics['bleu3']:.2f} BLEU-4={metrics['bleu4']:.2f} | "
                f"ROUGE-1={metrics['rouge1_pct']:.2f} ROUGE-2={metrics['rouge2_pct']:.2f} "
                f"ROUGE-L={metrics['rougeL_pct']:.2f}"
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_summary(out_dir, results)
    print(f"\n[EvalSLT] Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
