import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import yaml

os.environ.setdefault("TORCH_SHM_DIR", str(Path(tempfile.gettempdir()) / "torch_shm"))

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from ablations.baseline_s3_slt.train_baseline_s3_slt import (
    deep_update,
    set_seed,
    worker_init_fn,
    progress_miniters,
    mbart_lengths,
    pad_gloss_sequences,
)
from data.dataset import PhoenixDataset, build_transforms, build_vocabularies
from models.late_fusion import GlossEmbedding, LateFusion
from models.rover_ensemble import ThreeStreamCSLR
from models.translator_llm import SLTTransformerLLM, load_mbart50_tokenizer
from training.train_slt import (
    PhoenixSLTWithKeypoints,
    build_mbart_collate,
    get_cosine_schedule_with_warmup,
)
from ablations.metrics_fallback import safe_translation_metrics


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


def load_stream_weights(module: nn.Module, ckpt_path: str, device: torch.device, label: str):
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    module.load_state_dict(state, strict=True)
    print(
        f"[MainPipeline] Loaded {label} from {path} "
        f"(epoch={ckpt.get('epoch', '?')}, WER={ckpt.get('wer', ckpt.get('best_wer', '?'))})"
    )


class RoverMainMBartSLT(nn.Module):
    def __init__(self, cfg: dict, gloss_vocab_size: int, gloss_vocab=None, tokenizer=None):
        super().__init__()
        tc = cfg.get("three_stream", {})
        fc = cfg["fusion"]
        lc = cfg["llm"]
        mp = cfg.get("main_pipeline", {})

        self.cslr = ThreeStreamCSLR(
            num_classes=gloss_vocab_size,
            cfg=cfg,
            share_swin=tc.get("share_swin", False),
        )
        self.visual_sources = mp.get("visual_sources", ["s1", "s2", "s3"])
        self.visual_target = mp.get("visual_target", "s3")
        self.gloss_vocab = gloss_vocab
        self.tokenizer = tokenizer
        self.use_gloss_text_encoder = bool(mp.get("use_gloss_text_encoder", False))
        self.gloss_text_lower = bool(mp.get("gloss_text_lower", True))
        self.max_gloss_text_tokens = int(mp.get("max_gloss_text_tokens", lc.get("max_gen_length", 128)))
        if self.visual_target not in self.visual_sources:
            self.visual_target = self.visual_sources[-1]
        base_visual_weights = cfg.get("ensemble", {}).get("stream_weights", [1.0, 0.3, 1.0])
        visual_weights = mp.get("visual_weights", base_visual_weights)
        source_idx = {"s1": 0, "s2": 1, "s3": 2}
        self.visual_weights = [
            float(visual_weights[source_idx[src]]) if len(visual_weights) == 3 else float(visual_weights[i])
            for i, src in enumerate(self.visual_sources)
        ]
        self.visual_dim = cfg.get("stream3", {}).get("d_model", 512)
        stream_dims = {
            "s1": cfg.get("stream1", {}).get("d_model", 512),
            "s2": cfg.get("stream2", {}).get("d_model", 512),
            "s3": cfg.get("stream3", {}).get("d_model", 512),
        }
        mixer_in_dim = sum(stream_dims[src] for src in self.visual_sources)
        if mixer_in_dim == self.visual_dim:
            self.visual_mixer = nn.Identity()
        else:
            self.visual_mixer = nn.Sequential(
                nn.Linear(mixer_in_dim, self.visual_dim),
                nn.GELU(),
                nn.Dropout(fc.get("dropout", 0.2)),
                nn.LayerNorm(self.visual_dim),
            )
        self.gloss_embed = GlossEmbedding(
            vocab_size=gloss_vocab_size,
            embed_dim=fc["gloss_embed_dim"],
            padding_idx=1,
        )
        self.late_fusion = LateFusion(
            visual_dim=self.visual_dim,
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
            trainable_modules=cfg["slt"].get("llm_trainable_modules"),
            no_repeat_ngram_size=lc.get("no_repeat_ngram_size", 0),
            repetition_penalty=lc.get("repetition_penalty", 1.0),
            length_penalty=lc.get("length_penalty", 1.0),
        )

    def load_stream_checkpoints(self, cfg: dict, device: torch.device):
        mp = cfg["main_pipeline"]
        load_stream_weights(self.cslr.stream1, mp["stream1_ckpt"], device, "Stream 1")
        load_stream_weights(self.cslr.stream2, mp["stream2_ckpt"], device, "Stream 2")
        load_stream_weights(self.cslr.stream3, mp["stream3_ckpt"], device, "Stream 3")

    def freeze_cslr(self):
        self.cslr.eval()
        for p in self.cslr.parameters():
            p.requires_grad = False

    def _rover_gloss(self, out: dict, cfg: dict):
        seqs = self.cslr.rover(
            [out["s1_log_probs"], out["s2_log_probs"], out["s3_log_probs"]],
            [out["s1_lens"], out["s2_lens"], out["s3_lens"]],
        )
        return seqs

    @staticmethod
    def _resize_time(feats: torch.Tensor, target_len: int) -> torch.Tensor:
        if feats.size(1) == target_len:
            return feats
        return F.interpolate(
            feats.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    @staticmethod
    def _gloss_mask(gloss_ids: torch.Tensor) -> torch.Tensor:
        mask = gloss_ids.eq(0) | gloss_ids.eq(1)
        all_pad = mask.all(dim=1)
        if all_pad.any():
            mask = mask.clone()
            mask[all_pad, 0] = False
        return mask

    def _gloss_text_inputs(self, gloss_ids: torch.Tensor):
        if not self.use_gloss_text_encoder or self.gloss_vocab is None or self.tokenizer is None:
            return None, None
        texts = []
        for seq in gloss_ids.detach().cpu().tolist():
            tokens = self.gloss_vocab.decode([int(idx) for idx in seq], skip_special=True)
            text = " ".join(tokens)
            if self.gloss_text_lower:
                text = text.lower()
            texts.append(text)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_gloss_text_tokens,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"].to(gloss_ids.device),
            encoded["attention_mask"].to(gloss_ids.device),
        )

    def _mixed_visual(self, out: dict) -> tuple[torch.Tensor, torch.Tensor]:
        target_feats = out[f"{self.visual_target}_feats"]
        target_len = target_feats.size(1)
        parts = []
        for src, weight in zip(self.visual_sources, self.visual_weights):
            feats = self._resize_time(out[f"{src}_feats"], target_len)
            parts.append(feats * weight)
        visual_feats = self.visual_mixer(torch.cat(parts, dim=-1))
        src_lens = out[f"{self.visual_target}_lens"].clamp(max=visual_feats.size(1))
        return visual_feats, src_lens

    def _encode(self, frames, keypoints, frame_lens, gloss_ids, gloss_source: str, cfg: dict):
        with torch.no_grad():
            out = self.cslr(frames, keypoints, frame_lens)

        visual_feats, src_lens = self._mixed_visual(out)

        if gloss_source == "rover":
            gloss_ids = pad_gloss_sequences(self._rover_gloss(out, cfg), pad_idx=1, device=frames.device)
        elif gloss_source != "ground_truth":
            raise ValueError(f"Unknown gloss_source={gloss_source!r}")

        source_input_ids, source_attention_mask = self._gloss_text_inputs(gloss_ids)
        gloss_feats = self.gloss_embed(gloss_ids)
        fused = self.late_fusion(visual_feats, gloss_feats, gloss_mask=self._gloss_mask(gloss_ids))
        return fused, src_lens, source_input_ids, source_attention_mask

    def forward(self, frames, keypoints, frame_lens, gloss_ids, decoder_input_ids, decoder_lengths, cfg: dict):
        gloss_source = cfg["main_pipeline"].get("train_gloss_source", "ground_truth")
        fused, src_lens, source_input_ids, source_attention_mask = self._encode(
            frames, keypoints, frame_lens, gloss_ids, gloss_source, cfg
        )
        return self.translator(
            fused,
            decoder_input_ids,
            src_lengths=src_lens,
            tgt_lengths=decoder_lengths,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
        )

    @torch.no_grad()
    def generate(self, frames, keypoints, frame_lens, gloss_ids, bos_idx: int, eos_idx: int, use_beam: bool, cfg: dict):
        gloss_source = cfg["main_pipeline"].get("eval_gloss_source", "rover")
        fused, src_lens, source_input_ids, source_attention_mask = self._encode(
            frames, keypoints, frame_lens, gloss_ids, gloss_source, cfg
        )
        if use_beam:
            num_beams = int(cfg["main_pipeline"].get("slt_beam_width", cfg["llm"].get("num_beams", 4)))
            return self.translator.beam_decode(
                fused,
                bos_idx,
                eos_idx,
                src_lengths=src_lens,
                num_beams=num_beams,
                source_input_ids=source_input_ids,
                source_attention_mask=source_attention_mask,
            )
        return self.translator.greedy_decode(
            fused,
            bos_idx,
            eos_idx,
            src_lengths=src_lens,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
        )


def train_main_pipeline(
    cfg: dict,
    resume_path: str | None = None,
    init_model_path: str | None = None,
) -> float:
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MainPipeline] Device: {device}")
    if device.type == "cuda":
        print(f"[MainPipeline] GPU: {torch.cuda.get_device_name(0)}")

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

    model = RoverMainMBartSLT(cfg, len(gloss_vocab), gloss_vocab=gloss_vocab, tokenizer=tokenizer).to(device)
    model.load_stream_checkpoints(cfg, device)
    model.freeze_cslr()

    init_model_path = init_model_path or cfg["main_pipeline"].get("init_model")
    if resume_path and init_model_path:
        raise ValueError("Use either --resume for full training-state resume or --init-model for fresh optimizer init, not both.")
    if init_model_path:
        init_path = Path(init_model_path)
        if not init_path.exists():
            raise FileNotFoundError(f"Initial model checkpoint not found: {init_path}")
        init_ckpt = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(init_ckpt.get("model", init_ckpt), strict=True)
        print(f"[MainPipeline] Initialized model weights from {init_path} with fresh optimizer/scheduler.")

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Params total    : {total_p:,}")
    print(f"[Model] Params trainable: {trainable_p:,}")
    print(f"[Model] Frozen encoder  : Three independent CSLR streams + weighted ROVER")
    print(
        "[MainPipeline] "
        f"train_gloss_source={cfg['main_pipeline'].get('train_gloss_source')} | "
        f"eval_gloss_source={cfg['main_pipeline'].get('eval_gloss_source')} | "
        f"stream_weights={cfg['ensemble'].get('stream_weights')}"
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
    total_steps = num_epochs * max(1, (steps_per_ep + grad_acc - 1) // grad_acc)
    eta_min_ratio = sc.get("eta_min", 1e-7) / sc["learning_rate"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        sc.get("warmup_steps", 1000),
        total_steps,
        eta_min_ratio,
    )
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and sc.get("use_amp", True)) else None

    out_dir = Path(cfg["main_pipeline"].get("output_dir", "checkpoints/ablations/main_pipeline"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best_main_pipeline_slt.pth"
    newest_ckpt_path = out_dir / "newest_main_pipeline_slt.pth"
    log_dir = Path(cfg["main_pipeline"].get("log_dir", "logs/ablations/main_pipeline"))
    writer = SummaryWriter(log_dir=str(log_dir))

    es_cfg = sc.get("early_stopping", {})
    do_es = es_cfg.get("enabled", True)
    patience = es_cfg.get("patience", 8)
    no_improve = 0
    best_bleu = -1.0
    global_step = 0
    start_epoch = 0
    slt_beam_width = int(cfg["main_pipeline"].get("slt_beam_width", lc.get("num_beams", 4)))
    use_beam = slt_beam_width > 1

    resume_target = Path(resume_path) if resume_path else (
        None if init_model_path else (newest_ckpt_path if newest_ckpt_path.exists() else None)
    )
    if resume_target:
        print(f"[MainPipeline] Resume from {resume_target}")
        try:
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
            print(f"[MainPipeline] Resuming at epoch {start_epoch + 1} | best BLEU-4={best_bleu:.2f}")
        except RuntimeError as exc:
            if resume_path:
                raise
            print(f"[MainPipeline] Auto-resume skipped: checkpoint is incompatible with current model ({exc})")
            print("[MainPipeline] Starting a fresh run.")

    print(
        f"\n[MainPipeline] Training {num_epochs} epochs | eff. batch={sc['batch_size'] * grad_acc} | "
        f"mBART={lc.get('model_name', 'facebook/mbart-large-50')} | "
        f"gloss train/eval={cfg['main_pipeline'].get('train_gloss_source')}/{cfg['main_pipeline'].get('eval_gloss_source')} | "
        f"gloss_text_encoder={model.use_gloss_text_encoder} | "
        f"visual={model.visual_sources} weights={model.visual_weights} target={model.visual_target} | "
        f"slt_beam={slt_beam_width}"
    )

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        model.train()
        model.cslr.eval()
        optimizer.zero_grad()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"[MainPipeline Train] Epoch {epoch + 1}",
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
                writer.add_scalar("main_pipeline/train_loss", loss.item(), global_step)
                writer.add_scalar("main_pipeline/lr", optimizer.param_groups[0]["lr"], global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(num_batches, 1)

        model.eval()
        hyp_sents, ref_sents = [], []
        with torch.no_grad():
            for batch in tqdm(
                dev_loader,
                desc="[MainPipeline Eval]",
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
        sample_n = int(cfg["main_pipeline"].get("sample_predictions", 0))
        for i in range(min(sample_n, len(hyp_sents))):
            print(f"  [sample {i+1}] hyp: {hyp_sents[i]}")
            print(f"  [sample {i+1}] ref: {ref_sents[i]}")
        writer.add_scalar("main_pipeline/bleu4", bleu["bleu"], epoch)
        writer.add_scalar("main_pipeline/bleu1", bleu["bleu1"], epoch)
        writer.add_scalar("main_pipeline/rougeL", rouge["rougeL"], epoch)
        writer.add_scalar("main_pipeline/meteor", meteor, epoch)

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
                },
                best_ckpt_path,
            )
            print(f"  [ckpt] Best main pipeline saved (BLEU-4={best_bleu:.2f})")
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
                "no_improve": no_improve,
                "global_step": global_step,
            },
            newest_ckpt_path,
        )
        print(f"  [ckpt] Newest main pipeline saved (epoch={epoch + 1})")

        if do_es and no_improve >= patience:
            print(f"\n[MainPipeline] Early stopping at epoch {epoch + 1}.")
            break

    writer.close()
    print(f"\n[MainPipeline] Done. Best BLEU-4 = {best_bleu:.2f}")
    return best_bleu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROVER-main CSLR -> SLT pipeline")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_main_pipeline.yaml")))
    parser.add_argument("--resume", default=None, help="Optional checkpoint path. Defaults to newest checkpoint if present.")
    parser.add_argument("--init-model", default=None, help="Load model weights only and start a fresh optimizer/scheduler.")
    args = parser.parse_args()
    train_main_pipeline(load_config(args.config), args.resume, args.init_model)
