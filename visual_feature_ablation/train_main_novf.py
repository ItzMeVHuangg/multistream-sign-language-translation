import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch

os.environ.setdefault("TORCH_SHM_DIR", str(Path(tempfile.gettempdir()) / "torch_shm"))

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablations.main_pipeline import train_main_pipeline as main_train  # noqa: E402


class RoverMainNoVisualMBartSLT(main_train.RoverMainMBartSLT):
    """Main pipeline SLT ablation that uses predicted ROVER gloss but no visual features."""

    def _encode(self, frames, keypoints, frame_lens, gloss_ids, gloss_source: str, cfg: dict):
        with torch.no_grad():
            out = self.cslr(frames, keypoints, frame_lens)

        if gloss_source == "rover":
            gloss_ids = main_train.pad_gloss_sequences(
                self._rover_gloss(out, cfg),
                pad_idx=1,
                device=frames.device,
            )
        elif gloss_source != "ground_truth":
            raise ValueError(f"Unknown gloss_source={gloss_source!r}")

        gloss_mask = self._gloss_mask(gloss_ids)
        src_lens = (~gloss_mask).sum(dim=1).clamp(min=1)
        visual_len = max(1, gloss_ids.size(1))
        visual_dtype = out[f"{self.visual_target}_feats"].dtype
        visual_feats = torch.zeros(
            gloss_ids.size(0),
            visual_len,
            self.visual_dim,
            device=frames.device,
            dtype=visual_dtype,
        )

        gloss_feats = self.gloss_embed(gloss_ids)
        fused = self.late_fusion(visual_feats, gloss_feats, gloss_mask=gloss_mask)
        return fused, src_lens


def train_main_novf(cfg: dict, resume_path: str | None = None) -> float:
    original_model_cls = main_train.RoverMainMBartSLT
    main_train.RoverMainMBartSLT = RoverMainNoVisualMBartSLT
    try:
        print("[MainNoVF] Training main pipeline ablation with visual features disabled.")
        return main_train.train_main_pipeline(cfg, resume_path)
    finally:
        main_train.RoverMainMBartSLT = original_model_cls


def main():
    parser = argparse.ArgumentParser(description="Train main pipeline with no visual features.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_no_visual.yaml")))
    parser.add_argument("--resume", default=None, help="Optional checkpoint path.")
    args = parser.parse_args()

    train_main_novf(main_train.load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
