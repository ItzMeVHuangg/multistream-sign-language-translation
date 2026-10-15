import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TORCH_SHM_DIR", str(Path(tempfile.gettempdir()) / "torch_shm"))

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablations.main_pipeline.train_main_pipeline import load_config  # noqa: E402
from visual_feature_ablation.train_main_novf import train_main_novf  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train main pipeline with visual features disabled.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_no_visual.yaml")))
    parser.add_argument("--resume", default=None, help="Optional checkpoint path.")
    args = parser.parse_args()

    train_main_novf(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
