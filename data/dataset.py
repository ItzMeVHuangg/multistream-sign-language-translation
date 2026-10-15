
import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image


BASE_PATH    = Path("/datastore/cndt_phungdtm/KLTN_HoangBinh/Dataset")
PHOENIX_ROOT = BASE_PATH / "PHOENIX-2014-T"
FRAMES_ROOT  = Path(os.environ.get(
    "PHOENIX_FRAMES_ROOT",
    str(PHOENIX_ROOT / "features" / "fullFrame-210x260px"),
))
ANNOT_ROOT   = Path(os.environ.get(
    "PHOENIX_ANNOT_ROOT",
    str(PHOENIX_ROOT / "annotations" / "manual"),
))

class Vocabulary:
 
    BLANK = "<blank>"
    PAD   = "<pad>"
    UNK   = "<unk>"
    BOS   = "<bos>"
    EOS   = "<eos>"

    def __init__(self):
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        for tok in [self.BLANK, self.PAD, self.UNK, self.BOS, self.EOS]:
            self._add(tok)

    def _add(self, token: str) -> int:
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx]   = token
        return self.token2idx[token]

    def build_from_sequences(self, sequences: List[List[str]]):
        for seq in sequences:
            for tok in seq:
                self._add(tok)

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.token2idx[self.UNK]
        return [self.token2idx.get(t, unk) for t in tokens]

    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        special = {self.BLANK, self.PAD, self.UNK, self.BOS, self.EOS}
        tokens  = [self.idx2token.get(i, self.UNK) for i in indices]
        if skip_special:
            tokens = [t for t in tokens if t not in special]
        return tokens

    def __len__(self) -> int:
        return len(self.token2idx)

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)})"


def _resolve_frames_root(frames_root: Optional[Path]) -> Path:
    root = Path(frames_root) if frames_root else FRAMES_ROOT
    if root.exists():
        return root
    if root.name == "fullFrame-256x256px":
        fallback = root.parent / "fullFrame-210x260px"
        if fallback.exists():
            print(f"[PhoenixDataset] WARNING: {root} not found; falling back to {fallback}")
            return fallback
    return root


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Frame transforms
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_transforms(split: str, img_h: int = 224, img_w: int = 224) -> T.Compose:
    """
    Per-frame transform pipeline.
      Train : Resize 256x256 --- ToTensor --- Normalize ONLY.
              RandomCrop and RandomHorizontalFlip are applied at clip level
              (see PhoenixDataset._load_frames) so all frames in a clip share
              the same spatial parameters --- temporal consistency is preserved.
      Dev/Test: Resize 256x256 --- CenterCrop 224x224 --- ToTensor --- Normalize.
    """
    normalize = T.Normalize(mean=[0.45, 0.45, 0.45],
                            std=[0.225, 0.225, 0.225])
    if split == "train":
        # No RandomCrop / RandomHFlip here --- applied per-clip in _load_frames
        return T.Compose([
            T.Resize((img_h + 32, img_w + 32)),   # 256x256
            T.ToTensor(),
            normalize,
        ])
    else:                                          # dev / test: center crop only
        return T.Compose([
            T.Resize((img_h + 32, img_w + 32)),   # 256x256
            T.CenterCrop((img_h, img_w)),          # 224x224
            T.ToTensor(),
            normalize,
        ])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PHOENIX-2014-T Dataset
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class PhoenixDataset(Dataset):
    """
    PHOENIX-2014-T dataset loader.

    Paths default to the module-level constants, but training scripts can pass
    frames_root/annot_root from config for reproducible dataset protocols.

    Args:
        split             : "train" | "dev" | "test"
        gloss_vocab       : Vocabulary for gloss tokens  (used by CSLR)
        text_vocab        : Vocabulary for German tokens (used by SLT)
        max_frames        : fixed clip length; shorter clips are zero-padded,
                            longer clips are truncated from the end
        temporal_stride   : keep every N-th frame  (1 = keep all frames)
        transform         : torchvision transform applied to each frame PIL image
        return_translation: if True, also return tokenised German translation
    """

    def __init__(
        self,
        split: str,
        gloss_vocab: Vocabulary,
        text_vocab: Vocabulary,
        max_frames: int = 256,
        temporal_stride: int = 1,
        transform: Optional[T.Compose] = None,
        return_translation: bool = True,
        temporal_scale_range: Optional[Tuple[float, float]] = None,
        clip_aug_crop: Optional[Tuple[int, int]] = None,
        img_h: Optional[int] = None,
        img_w: Optional[int] = None,
        cache_root: Optional[Path] = None,
        cache_read: bool = True,
        cache_write: bool = True,
        cache_version: str = "ustm_v1",
        frames_root: Optional[Path] = None,
        annot_root: Optional[Path] = None,
    ):
        assert split in ("train", "dev", "test"), \
            f"split must be 'train', 'dev', or 'test' -- got '{split}'"

        self.split                = split
        self.gloss_vocab          = gloss_vocab
        self.text_vocab           = text_vocab
        self.max_frames           = max_frames
        self.temporal_stride      = temporal_stride
        self.transform            = transform
        self.return_translation   = return_translation
        self.img_h                = img_h
        self.img_w                = img_w
        # Paper: temporal augmentation --20% for train only
        self.temporal_scale_range = temporal_scale_range if split == "train" else None
        # Clip-level spatial aug: same crop/flip params for all frames in clip
        self.clip_aug_crop        = clip_aug_crop if split == "train" else None

        self.cache_root    = Path(cache_root) if cache_root else None
        self.cache_read    = cache_read
        self.cache_write   = cache_write
        self.cache_version = cache_version
        self.frames_root   = _resolve_frames_root(frames_root)
        self.annot_root    = Path(annot_root) if annot_root else ANNOT_ROOT

        self.cache_enabled = bool(self.cache_root) and self.cache_read
        if self.cache_root and (self.temporal_scale_range is not None or self.clip_aug_crop is not None):
            self.cache_enabled = False
            print("[PhoenixDataset] Cache disabled: random temporal/spatial augmentation enabled.")
        if self.cache_root and self.cache_write:
            (self.cache_root / self.split).mkdir(parents=True, exist_ok=True)

        # Validate that paths exist before doing anything else
        self._check_paths()

        self.samples = self._load_annotations()
        print(
            f"[PhoenixDataset] split={split:5s} | "
            f"{len(self.samples):4d} samples | "
            f"frames: {self.frames_root / split}"
        )

    # ------------------------------------------------------------------
    def _check_paths(self):
        for p, label in [(self.frames_root, "FRAMES_ROOT"), (self.annot_root, "ANNOT_ROOT")]:
            if not p.exists():
                raise FileNotFoundError(
                    f"{label} not found: {p}\n"
                    f"Check paths.frames_root / paths.annot_root in config.yaml."
                )
        split_dir = self.frames_root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Frame split directory not found: {split_dir}"
            )

    # ------------------------------------------------------------------
    def _load_annotations(self) -> List[Dict]:
        """
        Parse PHOENIX-2014-T.<split>.corpus.csv.

        Each row becomes a sample dict with keys:
          video_id, frame_dir, gloss (list of str), translation (list of str)
        """
        csv_path = self.annot_root / f"PHOENIX-2014-T.{self.split}.corpus.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")

        samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                video_id     = row["name"].strip()
                gloss_tokens = row["orth"].strip().split()
                # Wrap translation with BOS / EOS for the decoder
                trans_tokens = (
                    [self.text_vocab.BOS]
                    + row["translation"].strip().split()
                    + [self.text_vocab.EOS]
                )
                # Frames live in: frames_root/<split>/<video_id>/*.png
                frame_dir = self.frames_root / self.split / video_id

                samples.append({
                    "video_id":    video_id,
                    "frame_dir":   frame_dir,
                    "gloss":       gloss_tokens,
                    "translation": trans_tokens,
                })
        return samples

    # ------------------------------------------------------------------
    def _load_frames(self, frame_dir: Path) -> Tuple[torch.Tensor, int]:
        cache_path = self._cache_path(frame_dir.name)
        if self.cache_enabled and cache_path and cache_path.exists():
            cached = torch.load(cache_path, map_location='cpu', weights_only=False)
            return cached["frames"], int(cached["frame_len"])

        max_retries = 5
        for attempt in range(max_retries):
            frame_files = sorted(frame_dir.glob("*.png"))
            if not frame_files:
                frame_files = sorted(frame_dir.glob("*.jpg"))
            
            if frame_files:
                break
                
            import time
            print(f"[Warning] No frames found in {frame_dir}. Attempt {attempt + 1}/{max_retries}. Retrying in 2 seconds...")
            time.sleep(2)
            
        if not frame_files:
            raise FileNotFoundError(f"No frame images found in: {frame_dir} after {max_retries} retries. Check file system stability.")

        # Temporal subsampling
        frame_files = frame_files[:: self.temporal_stride]

        # Paper: temporal scale augmentation --20% (train only)
        if self.temporal_scale_range is not None:
            lo, hi  = self.temporal_scale_range
            scale   = random.uniform(lo, hi)
            n_total = len(frame_files)
            n_keep  = max(1, int(round(n_total * scale)))
            indices = (
                [int(round(i * (n_total - 1) / (n_keep - 1))) for i in range(n_keep)]
                if n_keep > 1 else [0]
            )
            frame_files = [frame_files[i] for i in indices]

        # Per-frame: Resize --- ToTensor --- Normalize (no random spatial aug yet)
        frames = []
        for fp in frame_files:
            with Image.open(fp) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)      # -> tensor (C, H_resize, W_resize)
                frames.append(img)

        frames   = torch.stack(frames)         # (T, C, H_resize, W_resize)
        T_actual = frames.shape[0]

        # Truncate before spatial aug (saves compute on padded frames)
        if T_actual > self.max_frames:
            frames   = frames[: self.max_frames]
            T_actual = self.max_frames

        # [FIX] Clip-level spatial augmentation: same RandomCrop + RandomHFlip
        # for ALL frames --- eliminates frame-to-frame spatial jitter.
        if self.clip_aug_crop is not None:
            crop_h, crop_w = self.clip_aug_crop
            H, W = frames.shape[2], frames.shape[3]
            top  = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)
            frames = frames[:, :, top:top + crop_h, left:left + crop_w]
            if random.random() < 0.5:
                frames = frames.flip(-1)       # horizontal flip, consistent

        # Zero-pad to max_frames
        if T_actual < self.max_frames:
            pad    = torch.zeros(
                self.max_frames - T_actual, *frames.shape[1:],
                dtype=frames.dtype
            )
            frames = torch.cat([frames, pad], dim=0)

        if self.cache_root and self.cache_write and cache_path:
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            torch.save({"frames": frames, "frame_len": int(T_actual)}, tmp_path)
            os.replace(tmp_path, cache_path)

        return frames, T_actual

    def _cache_path(self, video_id: str) -> Optional[Path]:
        if not self.cache_root or self.img_h is None or self.img_w is None:
            return None
        safe_id = video_id.replace("/", "_")
        frame_source = self.frames_root.name.replace("/", "_").replace("\\", "_")
        fname = (
            f"{safe_id}_{frame_source}_mf{self.max_frames}_ts{self.temporal_stride}_"
            f"h{self.img_h}_w{self.img_w}_{self.cache_version}.pt"
        )
        return self.cache_root / self.split / fname

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        frames, frame_len = self._load_frames(sample["frame_dir"])  # (T, C, H, W)

        gloss_ids = torch.tensor(
            self.gloss_vocab.encode(sample["gloss"]), dtype=torch.long
        )

        item = {
            "video_id":  sample["video_id"],
            "frames":    frames,                                  # (max_frames, C, H, W)
            "frame_len": torch.tensor(frame_len, dtype=torch.long),
            "gloss":     gloss_ids,                              # (G,)
            "gloss_len": torch.tensor(len(sample["gloss"]), dtype=torch.long),
        }

        if self.return_translation:
            trans_ids = torch.tensor(
                self.text_vocab.encode(sample["translation"]), dtype=torch.long
            )
            item["translation"]     = trans_ids                  # (S,)
            item["translation_len"] = torch.tensor(
                len(sample["translation"]), dtype=torch.long
            )

        return item


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Collate function
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collates a list of sample dicts into batched tensors.
    Frames are already padded to max_frames, so they stack directly.
    Gloss and translation sequences are padded to the longest in the batch.
    """
    video_ids  = [b["video_id"]  for b in batch]
    frames     = torch.stack([b["frames"]    for b in batch])   # (B, T, C, H, W)
    frame_lens = torch.stack([b["frame_len"] for b in batch])   # (B,)

    # Pad gloss sequences
    max_gloss  = max(b["gloss_len"].item() for b in batch)
    gloss_pad  = torch.zeros(len(batch), max_gloss, dtype=torch.long)
    gloss_lens = []
    for i, b in enumerate(batch):
        g = b["gloss"]
        gloss_pad[i, : len(g)] = g
        gloss_lens.append(b["gloss_len"])

    out = {
        "video_ids":  video_ids,
        "frames":     frames,
        "frame_lens": frame_lens,
        "gloss":      gloss_pad,
        "gloss_lens": torch.stack(gloss_lens),
    }

    # Pad translation sequences (SLT only)
    if "translation" in batch[0]:
        max_trans  = max(b["translation_len"].item() for b in batch)
        trans_pad  = torch.zeros(len(batch), max_trans, dtype=torch.long)
        trans_lens = []
        for i, b in enumerate(batch):
            t = b["translation"]
            trans_pad[i, : len(t)] = t
            trans_lens.append(b["translation_len"])
        out["translation"]      = trans_pad
        out["translation_lens"] = torch.stack(trans_lens)

    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Vocabulary builder  (call once before training)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_vocabularies(annot_root: Optional[Path] = None) -> Tuple[Vocabulary, Vocabulary]:
    """
    Build gloss and German text vocabularies from the train split only
    (as required by the PHOENIX-2014-T benchmark protocol).

    Uses annot_root if provided; otherwise ANNOT_ROOT.

    Returns:
        gloss_vocab : Vocabulary for sign gloss tokens
        text_vocab  : Vocabulary for German translation tokens
    """
    root = Path(annot_root) if annot_root else ANNOT_ROOT
    csv_path = root / "PHOENIX-2014-T.train.corpus.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {csv_path}")

    gloss_vocab = Vocabulary()
    text_vocab  = Vocabulary()
    all_gloss, all_trans = [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            all_gloss.append(row["orth"].strip().split())
            all_trans.append(row["translation"].strip().split())

    gloss_vocab.build_from_sequences(all_gloss)
    text_vocab.build_from_sequences(all_trans)

    print(f"[Vocab] Gloss vocab size : {len(gloss_vocab):5d}  (from {csv_path.name})")
    print(f"[Vocab] Text  vocab size : {len(text_vocab):5d}  (from {csv_path.name})")
    return gloss_vocab, text_vocab
