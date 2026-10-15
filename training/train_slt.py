import argparse
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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
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
    torch.multiprocessing.set_sharing_strategy("file_system")
except (RuntimeError, ValueError):
    pass

from data.dataset import PhoenixDataset, build_transforms
from models.late_fusion import GlossEmbedding, LateFusion
from models.rover_ensemble import ThreeStreamCSLR
from models.translator_llm import SLTTransformerLLM, load_mbart50_tokenizer
from training.train_cslr import PhoenixWithKeypoints, collate_fn_kpts
from utils.metrics import compute_bleu, compute_meteor, compute_rouge


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _progress_miniters(loader):
    try:
        return max(1, len(loader) // 20)
    except TypeError:
        return 1


class PhoenixSLTWithKeypoints(Dataset):
    """Adds MediaPipe keypoints and raw translation text for mBART tokenization."""

    def __init__(self, base_dataset: PhoenixDataset, kpts_root: Path, max_frames: int, keypoint_dim: int):
        self.with_kpts = PhoenixWithKeypoints(
            base_dataset=base_dataset,
            kpts_root=kpts_root,
            max_frames=max_frames,
            keypoint_dim=keypoint_dim,
            load_frames=True,
            load_kpts=True,
        )
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.with_kpts[idx]
        raw_tokens = self.base.samples[idx]["translation"]
        if raw_tokens and raw_tokens[0] == self.base.text_vocab.BOS:
            raw_tokens = raw_tokens[1:]
        if raw_tokens and raw_tokens[-1] == self.base.text_vocab.EOS:
            raw_tokens = raw_tokens[:-1]
        item["translation_text"] = " ".join(raw_tokens)
        return item


def build_mbart_collate(tokenizer, max_length: int):
    def _collate(batch: List[dict]) -> dict:
        out = collate_fn_kpts(batch)
        encoded = tokenizer(
            [b["translation_text"] for b in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out["mbart_input_ids"] = encoded["input_ids"]
        out["mbart_attention_mask"] = encoded["attention_mask"]
        out["translation_texts"] = [b["translation_text"] for b in batch]
        return out

    return _collate


class ThreeStreamMBartSLT(nn.Module):
    def __init__(self, cfg: dict, gloss_vocab_size: int):
        super().__init__()
        tc = cfg.get("three_stream", {})
        fc = cfg["fusion"]
        lc = cfg["llm"]

        self.cslr = ThreeStreamCSLR(
            num_classes=gloss_vocab_size,
            cfg=cfg,
            share_swin=tc.get("share_swin", True),
        )
        self.gloss_embed = GlossEmbedding(
            vocab_size=gloss_vocab_size,
            embed_dim=fc["gloss_embed_dim"],
        )
        self.late_fusion = LateFusion(
            visual_dim=cfg.get("stream3", {}).get("d_model", 512),
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

    def freeze_cslr(self):
        self.cslr.eval()
        for p in self.cslr.parameters():
            p.requires_grad = False

    def _fuse(self, frames, keypoints, frame_lens, gloss_ids):
        with torch.no_grad():
            out = self.cslr(frames, keypoints, frame_lens)
        visual_feats = out["s3_feats"]
        src_lens = out["s3_lens"].clamp(max=visual_feats.size(1))
        gloss_feats = self.gloss_embed(gloss_ids)
        fused = self.late_fusion(visual_feats, gloss_feats)
        return fused, src_lens

    def forward(self, frames, keypoints, frame_lens, gloss_ids, decoder_input_ids, decoder_lengths):
        fused, src_lens = self._fuse(frames, keypoints, frame_lens, gloss_ids)
        return self.translator(
            fused,
            decoder_input_ids,
            src_lengths=src_lens,
            tgt_lengths=decoder_lengths,
        )

    @torch.no_grad()
    def generate(self, frames, keypoints, frame_lens, gloss_ids, bos_idx: int, eos_idx: int, use_beam: bool):
        fused, src_lens = self._fuse(frames, keypoints, frame_lens, gloss_ids)
        if use_beam:
            return self.translator.beam_decode(fused, bos_idx, eos_idx, src_lengths=src_lens)
        return self.translator.greedy_decode(fused, bos_idx, eos_idx, src_lengths=src_lens)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    eta_min_ratio: float = 0.0,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(eta_min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def _mbart_lengths(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1).clamp(min=1)


def train_slt(cfg: dict, cslr_ckpt_path: str) -> float:
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SLT] Device: {device}")

    ckpt = torch.load(cslr_ckpt_path, map_location="cpu", weights_only=False)
    gloss_vocab = ckpt["gloss_vocab"]
    text_vocab = ckpt.get("text_vocab")
    print(f"[Vocab] Gloss: {len(gloss_vocab)}")

    lc = cfg["llm"]
    tokenizer = load_mbart50_tokenizer(lc.get("model_name", "facebook/mbart-large-50"))
    tokenizer.src_lang = lc.get("src_lang", "de_DE")
    tokenizer.tgt_lang = lc.get("tgt_lang", "de_DE")
    bos_idx = tokenizer.lang_code_to_id.get(tokenizer.tgt_lang, tokenizer.eos_token_id)
    eos_idx = tokenizer.eos_token_id
    pad_idx = tokenizer.pad_token_id

    dc = cfg["data"]
    paths = cfg.get("paths", {})
    kpts_root = Path(paths["mediapipe_kpts_root"])
    frames_root = Path(paths.get("frames_root")) if paths.get("frames_root") else None
    annot_root = Path(paths.get("annot_root")) if paths.get("annot_root") else None
    max_frames = dc["max_frames"]
    keypoint_dim = cfg["stream2"]["keypoint_dim"]

    cache_root = cfg.get("paths", {}).get("frame_cache_root") or os.environ.get("PHOENIX_FRAME_CACHE")
    cache_root = Path(cache_root) if cache_root else None
    if cache_root:
        print(f"[Cache] Frame cache root: {cache_root}")

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
        train_loader = DataLoader(
            train_ds,
            batch_size=sc["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=dc.get("pin_memory", True),
            prefetch_factor=dc.get("prefetch_factor", 2) if num_workers > 0 else None,
            collate_fn=collate,
            drop_last=True,
            generator=g,
            worker_init_fn=_worker_init_fn,
            persistent_workers=(num_workers > 0),
        )
        dev_loader = DataLoader(
            dev_ds,
            batch_size=sc["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=dc.get("pin_memory", True),
            prefetch_factor=dc.get("prefetch_factor", 2) if num_workers > 0 else None,
            collate_fn=collate,
            worker_init_fn=_worker_init_fn,
            persistent_workers=(num_workers > 0),
        )
        return train_loader, dev_loader

    num_workers = dc["num_workers"]
    train_loader, dev_loader = build_loaders(num_workers)

    model = ThreeStreamMBartSLT(cfg, len(gloss_vocab)).to(device)
    model.cslr.load_state_dict(ckpt["model"])
    model.freeze_cslr()

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Params total    : {total_p:,}")
    print(f"[Model] Params trainable: {trainable_p:,}")
    print(f"[Model] Params frozen   : {total_p - trainable_p:,}  (ThreeStreamCSLR backbone)")

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

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"]) / "slt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_slt.pth"
    writer = SummaryWriter(log_dir=str(Path(cfg["paths"]["log_dir"]) / "slt"))

    es_cfg = sc.get("early_stopping", {})
    do_es = es_cfg.get("enabled", True)
    patience = es_cfg.get("patience", 8)
    no_improve = 0
    best_bleu = -1.0
    global_step = 0
    use_beam = cfg.get("eval", {}).get("beam_width", 1) > 1

    print(
        f"\n[SLT] Training {num_epochs} epochs | eff. batch={sc['batch_size'] * grad_acc} | "
        f"mBART={lc.get('model_name', 'facebook/mbart-large-50')} | feature=s3_feats"
    )

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        model.cslr.eval()
        optimizer.zero_grad()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"[SLT Train] Epoch {epoch + 1}", leave=False, miniters=_progress_miniters(train_loader))
        for step, batch in enumerate(pbar):
            frames = batch["frames"].to(device)
            keypoints = batch["keypoints"].to(device)
            frame_lens = batch["frame_lens"].to(device)
            gloss = batch["gloss"].to(device)
            input_ids = batch["mbart_input_ids"].to(device)
            attn_mask = batch["mbart_attention_mask"].to(device)

            decoder_in = input_ids[:, :-1].contiguous()
            labels = input_ids[:, 1:].contiguous()
            decoder_lens = (_mbart_lengths(attn_mask) - 1).clamp(min=1)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits = model(frames, keypoints, frame_lens, gloss, decoder_in, decoder_lens)
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
                writer.add_scalar("slt/train_loss", loss.item(), global_step)
                writer.add_scalar("slt/lr", optimizer.param_groups[0]["lr"], global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(num_batches, 1)

        model.eval()
        hyp_sents, ref_sents = [], []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="[SLT Eval]", leave=False, miniters=_progress_miniters(dev_loader)):
                frames = batch["frames"].to(device)
                keypoints = batch["keypoints"].to(device)
                frame_lens = batch["frame_lens"].to(device)
                gloss = batch["gloss"].to(device)
                pred_ids = model.generate(frames, keypoints, frame_lens, gloss, bos_idx, eos_idx, use_beam=use_beam)
                hyp_sents.extend(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
                ref_sents.extend(batch["translation_texts"])

        bleu = compute_bleu(hyp_sents, ref_sents)
        rouge = compute_rouge(hyp_sents, ref_sents)
        meteor = compute_meteor(hyp_sents, ref_sents)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch + 1:3d}/{num_epochs}] loss={avg_train_loss:.4f} | "
            f"BLEU-4={bleu['bleu']:.2f} | ROUGE-L={rouge['rougeL']:.4f} | "
            f"METEOR={meteor:.4f} | lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.0f}s"
        )
        writer.add_scalar("slt/bleu4", bleu["bleu"], epoch)
        writer.add_scalar("slt/bleu1", bleu["bleu1"], epoch)
        writer.add_scalar("slt/rougeL", rouge["rougeL"], epoch)
        writer.add_scalar("slt/meteor", meteor, epoch)

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
                    "mbart_model_name": lc.get("model_name", "facebook/mbart-large-50"),
                    "cfg": cfg,
                    "seed": seed,
                },
                best_ckpt_path,
            )
            print(f"  [ckpt] Best saved (BLEU-4={best_bleu:.2f})")
        else:
            no_improve += 1
            print(f"  [no improve] {no_improve}/{patience}")

        if do_es and no_improve >= patience:
            print(f"\n[SLT] Early stopping at epoch {epoch + 1}.")
            break

    writer.close()
    print(f"\n[SLT] Done. Best BLEU-4 = {best_bleu:.2f}")
    return best_bleu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SLT stage: ThreeStreamCSLR s3_feats + mBART-50")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--cslr_ckpt", default="checkpoints/three_stream/best_three_stream.pth")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_slt(cfg, args.cslr_ckpt)
