import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def _write_ctm(path: Path, video_ids: Sequence[str], hypotheses: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for video_id, hyp in zip(video_ids, hypotheses):
            if not hyp:
                continue
            for idx, word in enumerate(hyp):
                f.write(f"{video_id} 1 {idx / 100:.2f} {(idx + 1) / 100:.2f} {word}\n")


def _write_merged_ctm(path: Path, video_ids: Sequence[str], hypotheses: Sequence[Sequence[str]]) -> None:
    """Write a complete CTM with [EMPTY] for blank utterances.

    USTM's official Python WER expects every STM id to exist in the CTM dict.
    This fallback keeps official scoring usable when preprocess.sh rejects an
    empty or early-training hypothesis file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for video_id, hyp in zip(video_ids, hypotheses):
        if hyp:
            for idx, word in enumerate(hyp):
                rows.append((video_id, f"{video_id} 1 {idx / 100:.2f} {(idx + 1) / 100:.2f} {word}\n"))
        else:
            rows.append((video_id, f"{video_id} 1 0.00 0.03 [EMPTY]\n"))
    with path.open("w", encoding="utf-8") as f:
        for _, row in sorted(rows, key=lambda item: item[0]):
            f.write(row)


def _load_python_wer(evaluate_dir: Path):
    module_path = evaluate_dir / "python_wer_evaluation.py"
    if not module_path.exists():
        raise FileNotFoundError(f"python_wer_evaluation.py not found in {evaluate_dir}")
    spec = importlib.util.spec_from_file_location("ustm_python_wer_evaluation", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.wer_calculation


def _find_eval_dir(configured: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if configured:
        candidates.append(Path(configured))
    for root in [
        Path.cwd() / "USTM",
        Path.cwd() / "USTM" / "USTM",
        Path.cwd().parent / "USTM",
        Path.cwd().parent / "USTM" / "USTM",
        Path("/datastore/cndt_phungdtm/KLTN_HoangBinh/3Stream/USTM"),
        Path("/datastore/cndt_phungdtm/KLTN_HoangBinh/3Stream/USTM/USTM"),
    ]:
        candidates.append(root / "evaluation" / "slr_eval")
        candidates.append(root / "slr_eval")
    for path in candidates:
        if (path / "python_wer_evaluation.py").exists() and (path / "phoenix2014-T-groundtruth-dev.stm").exists():
            return path
    return None


def compute_official_wer(
    video_ids: Sequence[str],
    hypotheses: Sequence[Sequence[str]],
    split: str,
    work_dir: Path,
    evaluate_dir: Optional[str] = None,
    evaluate_prefix: str = "phoenix2014-T-groundtruth",
    use_preprocess: bool = True,
) -> float:
    """Return PHOENIX/USTM-style WER as a fraction, e.g. 0.189.

    The CTM/STM path mirrors USTM's Python evaluation. On Linux with bash, it
    first tries USTM's preprocess.sh. If that shell helper rejects an early
    blank-heavy hypothesis file, this function still writes a complete CTM with
    [EMPTY] placeholders and scores it with USTM's Python WER implementation.
    """
    eval_dir = _find_eval_dir(evaluate_dir)
    if eval_dir is None:
        raise FileNotFoundError("Cannot find USTM/PHOENIX slr_eval directory")

    stm_path = eval_dir / f"{evaluate_prefix}-{split}.stm"
    if not stm_path.exists():
        raise FileNotFoundError(f"Groundtruth STM not found: {stm_path}")

    work_dir.mkdir(parents=True, exist_ok=True)
    raw_ctm = work_dir / f"output-hypothesis-{split}.ctm"
    out_ctm = work_dir / f"out.output-hypothesis-{split}.ctm"

    _write_ctm(raw_ctm, video_ids, hypotheses)
    _write_merged_ctm(out_ctm, video_ids, hypotheses)

    preprocessed = False
    preprocess_sh = eval_dir / "preprocess.sh"
    if use_preprocess and preprocess_sh.exists():
        tmp_ctm = work_dir / "tmp.ctm"
        tmp2_ctm = work_dir / "tmp2.ctm"
        proc = subprocess.run(
            ["bash", str(preprocess_sh), str(raw_ctm), str(tmp_ctm), str(tmp2_ctm)],
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0 and tmp2_ctm.exists():
            # Keep USTM's token simplifications, but add [EMPTY] for utterances
            # that have no prediction so python_wer_evaluation has all keys.
            pred_by_id = {}
            for line in tmp2_ctm.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[4] != "[EMPTY]":
                    pred_by_id.setdefault(parts[0], []).append(parts[4])
            _write_merged_ctm(out_ctm, video_ids, [pred_by_id.get(v, []) for v in video_ids])
            preprocessed = True
        else:
            print(
                "[Official WER] preprocess.sh failed; using Python CTM fallback. "
                f"returncode={proc.returncode} stderr={proc.stderr.strip()[:300]}"
            )

    wer_calculation = _load_python_wer(eval_dir)
    wer_percent = wer_calculation(str(stm_path), str(out_ctm))
    print(f"[Official WER] split={split} dir={eval_dir} preprocessed={preprocessed} WER={wer_percent:.2f}%")
    return wer_percent / 100.0
