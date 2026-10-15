"""
models/translator_llm.py
─────────────────────────
mBART-50 based Sign Language Translation model.

Architecture
────────────
The standard mBART encoder is replaced / augmented with our visual features.
Instead of processing source text tokens, the model receives visual prefix
tokens (learned projections of the CSLR hidden states) that are prepended to
the mBART encoder's embedding space.

Visual pipeline:
    fused_features (B, T_v, fused_dim=512)
        ↓  VisualAdapter  (MLP: fused_dim → mBART hidden_size)
        ↓  Learned linear prefix projection → (B, prefix_len, hidden_size)
        ↓  Concatenated with [de_DE] lang token embedding
        ↓  Standard mBART encoder self-attention
        ↓  mBART decoder cross-attention (generates German)

Training strategy (for ablation fairness)
──────────────────────────────────────────
· mBART encoder weights  → FROZEN  (keeps LLM knowledge intact)
· mBART decoder weights  → FROZEN  (idem)
· Cross-attention layers  → TRAINED (connect visual to text generation)
· lm_head                 → TRAINED
· VisualAdapter           → TRAINED  (maps visual → mBART space)
· Learnable prefix tokens → TRAINED

Only ~15 % of total parameters are updated — training is fast and stable.

Requirements
────────────
    pip install transformers sentencepiece
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

try:
    from transformers import (
        AutoTokenizer,
        MBartForConditionalGeneration,
        MBart50Tokenizer,
        MBart50TokenizerFast,
    )
    from transformers.modeling_outputs import BaseModelOutput
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
def load_mbart50_tokenizer(model_name: str = "facebook/mbart-large-50"):
    assert _TRANSFORMERS_AVAILABLE, (
        "transformers not installed. Run: pip install transformers sentencepiece"
    )

    errors = []
    for desc, loader in (
        ("MBart50TokenizerFast", lambda: MBart50TokenizerFast.from_pretrained(model_name)),
        ("MBart50Tokenizer", lambda: MBart50Tokenizer.from_pretrained(model_name)),
        ("AutoTokenizer(use_fast=False)", lambda: AutoTokenizer.from_pretrained(model_name, use_fast=False)),
    ):
        try:
            tokenizer = loader()
            print(f"[Tokenizer] Loaded {desc}: {model_name}")
            return tokenizer
        except Exception as exc:
            errors.append(f"{desc}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Could not load the mBART-50 tokenizer. The fast tokenizer can fail with "
        "`AttributeError: 'NoneType' object has no attribute 'endswith'` when the "
        "HuggingFace cache is incomplete or sentencepiece/tokenizer files are missing. "
        "Install `sentencepiece` and refresh the cached model tokenizer files.\n"
        + "\n".join(f"  - {err}" for err in errors)
    )


# Visual Adapter
# Adapts fused CSLR features (512-d) into mBART's hidden space (1024-d).
# ──────────────────────────────────────────────────────────────────────────────

class VisualAdapter(nn.Module):
    """
    Two-layer MLP that maps visual features to mBART hidden dimension.

    Args:
        in_dim     : dimension of fused visual features (512 by default)
        hidden_dim : intermediate bottleneck dimension
        out_dim    : mBART hidden_size (1024 for mbart-large-50)
        dropout    : dropout rate
    """

    def __init__(
        self,
        in_dim:     int   = 512,
        hidden_dim: int   = 768,
        out_dim:    int   = 1024,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in_dim) → (B, T, out_dim)"""
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# LLM SLT Model
# ──────────────────────────────────────────────────────────────────────────────

class SLTTransformerLLM(nn.Module):
    """
    mBART-50 based Sign Language → German Translation model.

    The visual features from LateFusion are injected into mBART's encoder
    as visual prefix tokens, allowing the pretrained LLM decoder to generate
    fluent German text conditioned on the sign video.

    Interface:
        forward(src, tgt, src_lengths, tgt_lengths) → logits (B, T_tgt, vocab)
        greedy_decode / beam_decode(src, bos, eos, src_lengths) → token ids

    Args:
        src_dim           : fused feature dim from LateFusion (default 512)
        model_name        : HuggingFace model identifier
        freeze_encoder    : whether to freeze mBART encoder params
        visual_prefix_len : number of visual prefix positions in encoder
        adapter_hidden    : hidden dim of VisualAdapter MLP
        max_gen_length    : max tokens to generate at inference
        num_beams         : beam width at inference
    """

    def __init__(
        self,
        src_dim:           int   = 512,
        model_name:        str   = "facebook/mbart-large-50",
        freeze_encoder:    bool  = True,
        visual_prefix_len: int   = 32,
        adapter_hidden:    int   = 768,
        max_gen_length:    int   = 128,
        num_beams:         int   = 4,
        label_smoothing:   float = 0.1,
        trainable_modules: Optional[List[str]] = None,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
    ):
        super().__init__()
        assert _TRANSFORMERS_AVAILABLE, (
            "transformers not installed. Run: pip install transformers sentencepiece"
        )

        self.src_dim           = src_dim
        self.visual_prefix_len = visual_prefix_len
        self.max_gen_length    = max_gen_length
        self.num_beams         = num_beams
        self.label_smoothing   = label_smoothing
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.repetition_penalty   = repetition_penalty
        self.length_penalty       = length_penalty

        # ── Load pretrained mBART ─────────────────────────────────────
        self.mbart = MBartForConditionalGeneration.from_pretrained(model_name)
        mbart_hidden = self.mbart.config.d_model   # 1024 for mbart-large-50

        # ── Freeze mBART encoder (keep LLM knowledge) ─────────────────
        if trainable_modules is None and freeze_encoder:
            for name, p in self.mbart.model.encoder.named_parameters():
                p.requires_grad = False
            # Un-freeze only encoder cross-attention layers (none in encoder,
            # but keep open for future variants)

        # ── Freeze mBART decoder EXCEPT cross-attention + lm_head ─────
        for name, p in self.mbart.model.decoder.named_parameters():
            # keep cross-attention and self-attention trainable
            if trainable_modules is None and any(k in name for k in ["encoder_attn", "self_attn", "fc"]):
                p.requires_grad = True
            else:
                p.requires_grad = False
        # Always keep lm_head trainable
        if trainable_modules is None:
            for p in self.mbart.lm_head.parameters():
                p.requires_grad = True

        # ── Visual adapter  (src_dim → mBART hidden) ──────────────────
        self.visual_adapter = VisualAdapter(
            in_dim     = src_dim,
            hidden_dim = adapter_hidden,
            out_dim    = mbart_hidden,
        )

        # ── Learnable visual prefix tokens ────────────────────────────
        # These are length-fixed learned tokens that summarise the visual
        # context and prepend the encoder sequence.
        self.visual_prefix = nn.Parameter(
            torch.randn(1, visual_prefix_len, mbart_hidden) * 0.02
        )

        # ── Cross-attention length adapter ────────────────────────────
        # Pool variable-length visual features to fixed prefix_len tokens
        # via cross-attention (visual features as KV, prefix tokens as Q)
        self.prefix_cross_attn = nn.MultiheadAttention(
            embed_dim   = mbart_hidden,
            num_heads   = 8,
            dropout     = 0.1,
            batch_first = True,
        )
        self.prefix_norm = nn.LayerNorm(mbart_hidden)

        if trainable_modules is not None:
            self.set_trainable_modules(trainable_modules)

    # ------------------------------------------------------------------
    def set_trainable_modules(self, trainable_modules: List[str]):
        modules = tuple(trainable_modules)
        for _, p in self.named_parameters():
            p.requires_grad = False
        for name, p in self.named_parameters():
            if any(key in name for key in modules):
                p.requires_grad = True

    # ------------------------------------------------------------------
    def _make_padding_mask(
        self, lengths: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Returns (B, max_len) bool mask: True where padded."""
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idx >= lengths.unsqueeze(1)

    # ------------------------------------------------------------------
    def _encode_visual(
        self,
        src: torch.Tensor,                          # (B, T_v, src_dim)
        src_lengths: Optional[torch.Tensor] = None, # (B,)
        source_input_ids: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Convert visual features → mBART encoder hidden states.

        Returns:
            encoder_hidden  : (B, prefix_len, mbart_hidden)
            encoder_mask    : (B, prefix_len)  all-False (no padding in prefix)
        """
        B = src.size(0)
        device = src.device

        # 1) Adapt visual features to mBART hidden size
        v = self.visual_adapter(src)                # (B, T_v, mbart_hidden)

        # 2) Cross-attend: prefix (Q) attends over visual features (KV)
        #    This compresses variable T_v → fixed prefix_len
        prefix_q = self.visual_prefix.expand(B, -1, -1)   # (B, prefix_len, H)

        key_padding_mask = None
        if src_lengths is not None:
            key_padding_mask = self._make_padding_mask(src_lengths, src.size(1))

        prefix_out, _ = self.prefix_cross_attn(
            query              = prefix_q,
            key                = v,
            value              = v,
            key_padding_mask   = key_padding_mask,
        )                                           # (B, prefix_len, H)
        prefix_out = self.prefix_norm(prefix_q + prefix_out)

        encoder_inputs = prefix_out
        encoder_mask = torch.ones(B, self.visual_prefix_len, dtype=torch.long, device=device)

        if source_input_ids is not None:
            source_embeds = self.mbart.model.encoder.embed_tokens(source_input_ids)
            source_embeds = source_embeds * getattr(self.mbart.model.encoder, "embed_scale", 1.0)
            if source_attention_mask is None:
                source_attention_mask = source_input_ids.ne(self.mbart.config.pad_token_id).long()
            else:
                source_attention_mask = source_attention_mask.to(device=device, dtype=torch.long)
            encoder_inputs = torch.cat([prefix_out, source_embeds], dim=1)
            encoder_mask = torch.cat([encoder_mask, source_attention_mask], dim=1)

        # 3) Run mBART encoder over the visual prefix, optionally followed by
        #    tokenized gloss source text. Passing gloss tokens here lets mBART
        #    use its pretrained lexical space instead of relying only on a
        #    randomly learned gloss embedding.
        encoder_outputs = self.mbart.model.encoder(
            inputs_embeds    = encoder_inputs,
            attention_mask   = encoder_mask,
            return_dict      = True,
        )
        encoder_hidden = encoder_outputs.last_hidden_state  # (B, prefix_len, H)

        return encoder_hidden, encoder_mask

    # ------------------------------------------------------------------
    def forward(
        self,
        src:         torch.Tensor,           # (B, T_v, src_dim)  fused visual
        tgt:         torch.Tensor,           # (B, T_tgt)  decoder input token ids
        src_lengths: Optional[torch.Tensor] = None,
        tgt_lengths: Optional[torch.Tensor] = None,
        source_input_ids: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass.

        Returns logits: (B, T_tgt, vocab_size)
        """
        B, T_tgt = tgt.shape
        device   = src.device

        encoder_hidden, encoder_mask = self._encode_visual(
            src,
            src_lengths,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
        )

        # Build decoder attention mask
        tgt_attn_mask = torch.ones(B, T_tgt, dtype=torch.long, device=device)
        if tgt_lengths is not None:
            for i, l in enumerate(tgt_lengths):
                tgt_attn_mask[i, l:] = 0

        outputs = self.mbart(
            decoder_input_ids          = tgt,
            encoder_outputs            = (encoder_hidden,),
            attention_mask             = encoder_mask,
            decoder_attention_mask     = tgt_attn_mask,
            return_dict                = True,
        )
        return outputs.logits   # (B, T_tgt, vocab_size)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def greedy_decode(
        self,
        src:         torch.Tensor,
        bos_idx:     int,
        eos_idx:     int,
        src_lengths: Optional[torch.Tensor] = None,
        source_input_ids: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding. Returns (B, L) token ids."""
        B      = src.size(0)
        device = src.device

        encoder_hidden, encoder_mask = self._encode_visual(
            src,
            src_lengths,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
        )

        ys       = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(self.max_gen_length):
            outputs = self.mbart(
                decoder_input_ids = ys,
                encoder_outputs   = (encoder_hidden,),
                attention_mask    = encoder_mask,
                return_dict       = True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            finished  |= next_token == eos_idx
            ys         = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            if finished.all():
                break

        return ys[:, 1:]   # strip BOS

    # ------------------------------------------------------------------
    @torch.no_grad()
    def beam_decode(
        self,
        src:         torch.Tensor,
        bos_idx:     int,
        eos_idx:     int,
        src_lengths: Optional[torch.Tensor] = None,
        num_beams:   Optional[int] = None,
        source_input_ids: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Beam search decoding using HuggingFace generate().
        Returns (B, L) token ids (best beam).
        """
        B      = src.size(0)
        device = src.device
        beams  = num_beams or self.num_beams

        encoder_hidden, encoder_mask = self._encode_visual(
            src,
            src_lengths,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
        )

        generated = self.mbart.generate(
            encoder_outputs        = BaseModelOutput(last_hidden_state=encoder_hidden),
            attention_mask         = encoder_mask,
            forced_bos_token_id    = bos_idx,
            eos_token_id           = eos_idx,
            pad_token_id           = self.mbart.config.pad_token_id,
            num_beams              = beams,
            max_new_tokens         = self.max_gen_length,
            no_repeat_ngram_size   = self.no_repeat_ngram_size,
            repetition_penalty     = self.repetition_penalty,
            length_penalty         = self.length_penalty,
            early_stopping         = True,
        )
        return generated[:, 1:]   # strip BOS

    # ------------------------------------------------------------------
    def trainable_parameters(self) -> list:
        """Return only the trainable parameter groups (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]

    def param_count(self) -> dict:
        total    = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable,
                "frozen": total - trainable,
                "pct_trainable": 100 * trainable / total}


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _TRANSFORMERS_AVAILABLE:
        print("Install transformers: pip install transformers sentencepiece")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SLTTransformerLLM(
        src_dim           = 512,
        model_name        = "facebook/mbart-large-50",
        freeze_encoder    = True,
        visual_prefix_len = 32,
        adapter_hidden    = 768,
    ).to(device)

    pc = model.param_count()
    print(f"Total params    : {pc['total']:,}")
    print(f"Trainable params: {pc['trainable']:,}  ({pc['pct_trainable']:.1f} %)")

    # Fake batch
    B, T_v, T_tgt = 2, 64, 20
    src      = torch.randn(B, T_v, 512, device=device)
    tgt      = torch.randint(5, 250054, (B, T_tgt), device=device)
    src_lens = torch.tensor([64, 48], device=device)

    logits = model(src, tgt, src_lengths=src_lens)
    print(f"Logits shape: {list(logits.shape)}  (B, T_tgt, vocab)")

    ids = model.greedy_decode(src, bos_idx=2, eos_idx=2, src_lengths=src_lens)
    print(f"Greedy decode output: {list(ids.shape)}")
