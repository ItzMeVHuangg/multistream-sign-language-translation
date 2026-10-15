import argparse
import copy
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.eval_three_stream import main as eval_three_stream_main


def deep_update(base: dict, updates: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_overlay_config(path: str) -> dict:
    cfg_path = Path(path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        overlay = yaml.safe_load(f)
    base_ref = overlay.pop("base_config")
    base_path = (cfg_path.parent / base_ref).resolve()
    with open(base_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    return deep_update(base_cfg, overlay)


def main():
    parser = argparse.ArgumentParser(description="ROVER stream-weight ablation runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--save_hyps", action="store_true")
    args = parser.parse_args()

    cfg = load_overlay_config(args.config)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        merged_config = f.name

    eval_args = [
        "eval_three_stream",
        "--config", merged_config,
        "--ckpt1", "checkpoints/stream1_only/best_stream1.pth",
        "--ckpt2", "checkpoints/stream2_only/best_stream2.pth",
        "--ckpt3", "checkpoints/stream3_only/best_stream3.pth",
        "--split", args.split,
        "--decode", "greedy",
        "--beam_width", "1",
        "--batch_size", str(args.batch_size),
    ]
    if args.save_hyps:
        eval_args.append("--save_hyps")

    import sys
    old_argv = sys.argv
    try:
        sys.argv = eval_args
        eval_three_stream_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
