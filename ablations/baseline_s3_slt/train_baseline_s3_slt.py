import argparse
import copy
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import yaml

os.environ.setdefault("TORCH_SHM_DIR", str(Path(tempfile.gettempdir()) / "torch_shm"))

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing
from torch.optim import AdamW
from torch.utils.data import DataLoader
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except (RuntimeError, ValueError):
    pass

from data.dataset import PhoenixDataset, build_transforms, build_vocabularies
from models.late_fusion import GlossEmbedding, LateFusion
from models.stream3_crossattn import Stream3_CrossAttn
from models.translator_llm import SLTTransformerLLM, load_mbart50_tokenizer
from training.train_slt import (
    PhoenixSLTWithKeypoints,
    build_mbart_collate,
    get_cosine_schedule_with_warmup,
)
from ablations.metrics_fallback import safe_translation_metrics
from utils.ctc_decoder import batch_ctc_decode


def deep_update(base: dict, updates: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str) -> dict:
    cfg_path = Path(path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        overlay = yaml.safe_load(f)
    base_ref = overlay.pop("base_config", None)
    if not base_ref:
        return overlay
    base_path = (cfg_path.parent / base_ref).resolve()
    with open(base_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    return deep_update(base_cfg, overlay)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def progress_miniters(loader):
    try:
        return max(1, len(loader) // 20)
    except TypeError:
        return 1


def mbart_lengths(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1).clamp(min=1)


def pad_gloss_sequences(seqs: List[List[int]], pad_idx: int, device: torch.device) -> torch.Tensor:
    max_len = max(1, max((len(seq) for seq in seqs), default=0))
    out = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        if seq:
            out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return out


class Stream3BaselineMBartSLT(nn.Module):
    def __init__(self, cfg: dict, gloss_vocab_size: int):
        super().__init__()
        s3 = cfg["stream3"]
        fc = cfg["fusion"]
        lc = cfg["llm"]

        self.stream3 = Stream3_CrossAttn(
            keypoint_dim=s3.get("keypoint_dim", 225),
            d_model=s3.get("d_model", 512),
            nhead=s3.get("nhead", 8),
            bilstm_hidden=s3.get("bilstm_hidden", 512),
            bilstm_layers=s3.get("bilstm_layers", 2),
            num_classes=gloss_vocab_size,
            attn_dropout=s3.get("attn_dropout", 0.1),
            bilstm_dropout=s3.get("bilstm_dropout", 0.3),
            pretrained=s3.get("pretrained", True),
            use_checkpoint=s3.get("use_checkpoint", True),
        )
        self.gloss_embed = GlossEmbedding(
            vocab_size=gloss_vocab_size,
            embed_dim=fc["gloss_embed_dim"],
            padding_idx=1,
        )
        self.late_fusion = LateFusion(
            visual_dim=s3.get("d_model", 512),
            gloss_embed_dim=fc["gloss_embed_dim"],
            fused_dim=fc["fused_dim"],
            mode=fc["mode"],
            dropout=fc["dropout"],
            nhead=fc.get("nhead", 8),
        )
        self.translator = SLTTransformerLLM(
            src_dim=fc["fused_dim"],
            model_name=lc.get("model_name", "facebook/mbart-large-50"),
            freeze_encoder=lc.get("freeze_encoder", True),
            visual_prefix_len=lc.get("visual_prefix_len", 32),
            adapter_hidden=lc.get("adapter_hidden", 768),
            max_gen_length=lc.get("max_gen_length", cfg["slt"].get("max_seq_len", 128)),
            num_beams=lc.get("num_beams", 4),
            label_smoothing=cfg["slt"].get("label_smoothing", 0.1),
        )

    def load_stream3_checkpoint(self, path: str, device: torch.device):
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Stream 3 checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        self.stream3.load_state_dict(state, strict=True)
        print(
            f"[BaselineS3] Loaded Stream 3 from {ckpt_path} "
            f"(epoch={ckpt.get('epoch', '?')}, WER={ckpt.get('wer', ckpt.get('best_wer', '?'))})"
        )

    def freeze_stream3(self):
        self.stream3.eval()
        for p in self.stream3.parameters():
            p.requires_grad = False

    def _encode(self, frames, keypoints, frame_lens, gloss_ids, gloss_source: str, cfg: dict):
        with torch.no_grad():
            log_probs, visual_feats, src_lens = self.stream3(frames, keypoints, frame_lens)
        src_lens = src_lens.clamp(max=visual_feats.size(1))

        if gloss_source == "predicted":
            base = cfg["baseline_s3"]
            pred = batch_ctc_decode(
                log_probs,
                src_lens.clamp(max=log_probs.shape[0]),
                blank_idx=cfg["cslr"]["ctc_blank_idx"],
                mode=base.get("decode_mode", "greedy"),
                beam_width=base.get("beam_width", 1),
            )
            gloss_ids = pad_gloss_sequences(pred, pad_idx=1, device=frames.device)
        elif gloss_source != "ground_truth":
            raise ValueError(f"Unknown gloss_source={gloss_source!r}")

        gloss_feats = self.gloss_embed(gloss_ids)
        fused = self.late_fusion(visual_feats, gloss_feats)
        return fused, src_lens

    def forward(self, frames, keypoints, frame_lens, gloss_ids, decoder_input_ids, decoder_lengths, cfg: dict):
        gloss_source = cfg["baseline_s3"].get("train_gloss_source", "ground_truth")
        fused, src_lens = self._encode(frames, keypoints, frame_lens, gloss_ids, gloss_source, cfg)
        return self.translator(
            fused,
            decoder_input_ids,
            src_lengths=src_lens,
            tgt_lengths=decoder_lengths,
        )

    @torch.no_grad()
    def generate(self, frames, keypoints, frame_lens, gloss_ids, bos_idx: int, eos_idx: int, use_beam: bool, cfg: dict):
        gloss_source = cfg["baseline_s3"].get("eval_gloss_source", "predicted")
        fused, src_lens = self._encode(frames, keypoints, frame_lens, gloss_ids, gloss_source, cfg)
        if use_beam:
            return self.translator.beam_decode(fused, bos_idx, eos_idx, src_lengths=src_lens)
        return self.translator.greedy_decode(fused, bos_idx, eos_idx, src_lengths=src_lens)


def train_baseline_s3_slt(cfg: dict, resume_path: str | None = None) -> float:
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BaselineS3] Device: {device}")
    if device.type == "cuda":
        print(f"[BaselineS3] GPU: {torch.cuda.get_device_name(0)}")

    paths = cfg.get("paths", {})
    annot_root = Path(paths.get("annot_root")) if paths.get("annot_root") else None
    frames_root = Path(paths.get("frames_root")) if paths.get("frames_root") else None
    gloss_vocab, text_vocab = build_vocabularies(annot_root=annot_root)
    print(f"[Vocab] Gloss: {len(gloss_vocab)}")

    lc = cfg["llm"]
    tokenizer = load_mbart50_tokenizer(lc.get("model_name", "facebook/mbart-large-50"))
    tokenizer.src_lang = lc.get("src_lang", "de_DE")
    tokenizer.tgt_lang = lc.get("tgt_lang", "de_DE")
    bos_idx = tokenizer.lang_code_to_id.get(tokenizer.tgt_lang, tokenizer.eos_token_id)
    eos_idx = tokenizer.eos_token_id
    pad_idx = tokenizer.pad_token_id

    dc = cfg["data"]
    kpts_root = Path(paths["mediapipe_kpts_root"])
    max_frames = dc["max_frames"]
    keypoint_dim = cfg["stream2"]["keypoint_dim"]
    cache_root = paths.get("frame_cache_root") or os.environ.get("PHOENIX_FRAME_CACHE")
    cache_root = Path(cache_root) if cache_root else None

    def make_ds(split: str):
        base = PhoenixDataset(
            split=split,
            gloss_vocab=gloss_vocab,
            text_vocab=text_vocab,
            max_frames=max_frames,
            temporal_stride=dc["temporal_stride"],
            transform=build_transforms(split, dc["img_height"], dc["img_width"]),
            return_translation=True,
            img_h=dc["img_height"],
            img_w=dc["img_width"],
            cache_root=cache_root,
            cache_read=True,
            cache_write=True,
            frames_root=frames_root,
            annot_root=annot_root,
        )
        return PhoenixSLTWithKeypoints(base, kpts_root, max_frames, keypoint_dim)

    train_ds = make_ds("train")
    dev_ds = make_ds("dev")

    sc = cfg["slt"]
    collate = build_mbart_collate(tokenizer, sc.get("max_seq_len", lc.get("max_gen_length", 128)))
    g = torch.Generator()
    g.manual_seed(seed)

    def build_loaders(num_workers: int):
        print(f"[DataLoader] Using {num_workers} workers" if num_workers > 0 else "[DataLoader] Using num_workers=0")
        common = {
            "num_workers": num_workers,
            "pin_memory": dc.get("pin_memory", True),
            "prefetch_factor": dc.get("prefetch_factor", 2) if num_workers > 0 else None,
            "collate_fn": collate,
            "worker_init_fn": worker_init_fn,
            "persistent_workers": (num_workers > 0 and dc.get("persistent_workers", True)),
        }
        train_loader = DataLoader(
            train_ds,
            batch_size=sc["batch_size"],
            shuffle=True,
            drop_last=True,
            generator=g,
            **common,
        )
        dev_loader = DataLoader(
            dev_ds,
            batch_size=sc["batch_size"],
            shuffle=False,
            **common,
        )
        return train_loader, dev_loader

    train_loader, dev_loader = build_loaders(dc.get("num_workers", 0))

    model = Stream3BaselineMBartSLT(cfg, len(gloss_vocab)).to(device)
    model.load_stream3_checkpoint(cfg["baseline_s3"]["stream3_ckpt"], device)
    model.freeze_stream3()

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Params total    : {total_p:,}")
    print(f"[Model] Params trainable: {trainable_p:,}")
    print(f"[Model] Frozen encoder  : Stream 3 Cross-Attn")
    print(
        "[BaselineS3] "
        f"train_gloss_source={cfg['baseline_s3'].get('train_gloss_source')} | "
        f"eval_gloss_source={cfg['baseline_s3'].get('eval_gloss_source')}"
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        label_smoothing=sc.get("label_smoothing", lc.get("label_smoothing", 0.1)),
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=sc["learning_rate"], weight_decay=sc["weight_decay"])

    num_epochs = sc["num_epochs"]
    grad_acc = sc.get("grad_accumulation_steps", 4)
    steps_per_ep = len(train_loader)
    total_steps = num_epochs * math.ceil(steps_per_ep / max(1, grad_acc))
    eta_min_ratio = sc.get("eta_min", 1e-7) / sc["learning_rate"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        sc.get("warmup_steps", 1000),
        total_steps,
        eta_min_ratio,
    )
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and sc.get("use_amp", True)) else None

    out_dir = Path(cfg["baseline_s3"].get("output_dir", "checkpoints/ablations/baseline_s3_slt"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best_baseline_s3_slt.pth"
    newest_ckpt_path = out_dir / "newest_baseline_s3_slt.pth"
    log_dir = Path(cfg["baseline_s3"].get("log_dir", "logs/ablations/baseline_s3_slt"))
    writer = SummaryWriter(log_dir=str(log_dir))

    es_cfg = sc.get("early_stopping", {})
    do_es = es_cfg.get("enabled", True)
    patience = es_cfg.get("patience", 8)
    no_improve = 0
    best_bleu = -1.0
    global_step = 0
    start_epoch = 0
    use_beam = cfg["baseline_s3"].get("beam_width", 1) > 1

    resume_target = Path(resume_path) if resume_path else (newest_ckpt_path if newest_ckpt_path.exists() else None)
    if resume_target:
        print(f"[BaselineS3] Resume from {resume_target}")
        ckpt = torch.load(resume_target, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_bleu = float(ckpt.get("best_bleu", ckpt.get("bleu", best_bleu)))
        no_improve = int(ckpt.get("no_improve", 0))
        global_step = int(ckpt.get("global_step", 0))
        print(f"[BaselineS3] Resuming at epoch {start_epoch + 1} | best BLEU-4={best_bleu:.2f}")

    print(
        f"\n[BaselineS3] Training {num_epochs} epochs | eff. batch={sc['batch_size'] * grad_acc} | "
        f"mBART={lc.get('model_name', 'facebook/mbart-large-50')} | feature=s3_feats"
    )

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        model.train()
        model.stream3.eval()
        optimizer.zero_grad()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"[BaselineS3 Train] Epoch {epoch + 1}",
            leave=False,
            miniters=progress_miniters(train_loader),
        )
        for step, batch in enumerate(pbar):
            frames = batch["frames"].to(device)
            keypoints = batch["keypoints"].to(device)
            frame_lens = batch["frame_lens"].to(device)
            gloss = batch["gloss"].to(device)
            input_ids = batch["mbart_input_ids"].to(device)
            attn_mask = batch["mbart_attention_mask"].to(device)

            decoder_in = input_ids[:, :-1].contiguous()
            labels = input_ids[:, 1:].contiguous()
            decoder_lens = (mbart_lengths(attn_mask) - 1).clamp(min=1)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits = model(frames, keypoints, frame_lens, gloss, decoder_in, decoder_lens, cfg)
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss_scaled = loss / grad_acc

            if scaler is not None:
                scaler.scale(loss_scaled).backward()
                if (step + 1) % grad_acc == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, sc["gradient_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss_scaled.backward()
                if (step + 1) % grad_acc == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, sc["gradient_clip"])
                    optimizer.step()
                    optimizer.zero_grad()

            if (step + 1) % grad_acc == 0:
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            if writer and global_step % 20 == 0:
                writer.add_scalar("baseline_s3/train_loss", loss.item(), global_step)
                writer.add_scalar("baseline_s3/lr", optimizer.param_groups[0]["lr"], global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(num_batches, 1)

        model.eval()
        hyp_sents, ref_sents = [], []
        with torch.no_grad():
            for batch in tqdm(
                dev_loader,
                desc="[BaselineS3 Eval]",
                leave=False,
                miniters=progress_miniters(dev_loader),
            ):
                frames = batch["frames"].to(device)
                keypoints = batch["keypoints"].to(device)
                frame_lens = batch["frame_lens"].to(device)
                gloss = batch["gloss"].to(device)
                pred_ids = model.generate(frames, keypoints, frame_lens, gloss, bos_idx, eos_idx, use_beam=use_beam, cfg=cfg)
                hyp_sents.extend(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
                ref_sents.extend(batch["translation_texts"])

        bleu, rouge, meteor = safe_translation_metrics(hyp_sents, ref_sents)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch + 1:3d}/{num_epochs}] loss={avg_train_loss:.4f} | "
            f"BLEU-4={bleu['bleu']:.2f} | ROUGE-L={rouge['rougeL']:.4f} | "
            f"METEOR={meteor:.4f} | lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.0f}s"
        )
        sample_n = int(cfg["baseline_s3"].get("sample_predictions", 0))
        for i in range(min(sample_n, len(hyp_sents))):
            print(f"  [sample {i+1}] hyp: {hyp_sents[i]}")
            print(f"  [sample {i+1}] ref: {ref_sents[i]}")
        writer.add_scalar("baseline_s3/bleu4", bleu["bleu"], epoch)
        writer.add_scalar("baseline_s3/bleu1", bleu["bleu1"], epoch)
        writer.add_scalar("baseline_s3/rougeL", rouge["rougeL"], epoch)
        writer.add_scalar("baseline_s3/meteor", meteor, epoch)

        if bleu["bleu"] > best_bleu:
            best_bleu = bleu["bleu"]
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "bleu": best_bleu,
                    "bleu_scores": bleu,
                    "rouge": rouge,
                    "meteor": meteor,
                    "gloss_vocab": gloss_vocab,
                    "text_vocab": text_vocab,
                    "cfg": cfg,
                    "stream3_ckpt": cfg["baseline_s3"]["stream3_ckpt"],
                },
                best_ckpt_path,
            )
            print(f"  [ckpt] Best baseline saved (BLEU-4={best_bleu:.2f})")
        else:
            no_improve += 1
            print(f"  [no improve] {no_improve}/{patience}")

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "bleu": bleu["bleu"],
                "best_bleu": best_bleu,
                "bleu_scores": bleu,
                "rouge": rouge,
                "meteor": meteor,
                "gloss_vocab": gloss_vocab,
                "text_vocab": text_vocab,
                "cfg": cfg,
                "stream3_ckpt": cfg["baseline_s3"]["stream3_ckpt"],
                "no_improve": no_improve,
                "global_step": global_step,
            },
            newest_ckpt_path,
        )
        print(f"  [ckpt] Newest S3 baseline saved (epoch={epoch + 1})")

        if do_es and no_improve >= patience:
            print(f"\n[BaselineS3] Early stopping at epoch {epoch + 1}.")
            break

    writer.close()
    print(f"\n[BaselineS3] Done. Best BLEU-4 = {best_bleu:.2f}")
    return best_bleu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S3-only CSLR -> SLT baseline ablation")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_baseline_s3_slt.yaml")))
    parser.add_argument("--resume", default=None, help="Optional checkpoint path. Defaults to newest checkpoint if present.")
    args = parser.parse_args()
    train_baseline_s3_slt(load_config(args.config), args.resume)
