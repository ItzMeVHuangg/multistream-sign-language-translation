import math
from collections import Counter
from typing import List, Tuple

from utils.metrics import compute_bleu, compute_meteor, compute_rouge


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1)))


def _fallback_bleu(hypotheses: List[str], references: List[str]) -> dict:
    clipped = [0, 0, 0, 0]
    total = [0, 0, 0, 0]
    hyp_len = 0
    ref_len = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_toks = hyp.split()
        ref_toks = ref.split()
        hyp_len += len(hyp_toks)
        ref_len += len(ref_toks)
        for n in range(1, 5):
            hyp_ng = _ngrams(hyp_toks, n)
            ref_ng = _ngrams(ref_toks, n)
            clipped[n - 1] += sum(min(count, ref_ng[ng]) for ng, count in hyp_ng.items())
            total[n - 1] += max(0, len(hyp_toks) - n + 1)

    precisions = [
        100.0 * (clipped[i] + 1.0) / (total[i] + 1.0)
        for i in range(4)
    ]
    if hyp_len == 0:
        scores = [0.0, 0.0, 0.0, 0.0]
    else:
        bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(1, hyp_len))
        scores = [
            100.0 * bp * math.exp(
                sum(math.log(max(precisions[i], 1e-9) / 100.0) for i in range(order)) / order
            )
            for order in range(1, 5)
        ]
    return {
        "bleu": scores[3],
        "bleu1": scores[0],
        "bleu2": scores[1],
        "bleu3": scores[2],
        "bleu4": scores[3],
    }


def _lcs_len(a: List[str], b: List[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _f1(overlap: int, hyp_len: int, ref_len: int) -> float:
    if overlap == 0 or hyp_len == 0 or ref_len == 0:
        return 0.0
    precision = overlap / hyp_len
    recall = overlap / ref_len
    return 2.0 * precision * recall / (precision + recall)


def _fallback_rouge(hypotheses: List[str], references: List[str]) -> dict:
    scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n_items = max(1, len(hypotheses))
    for hyp, ref in zip(hypotheses, references):
        hyp_toks = hyp.split()
        ref_toks = ref.split()
        for n, key in [(1, "rouge1"), (2, "rouge2")]:
            hyp_ng = _ngrams(hyp_toks, n)
            ref_ng = _ngrams(ref_toks, n)
            overlap = sum(min(count, ref_ng[ng]) for ng, count in hyp_ng.items())
            scores[key] += _f1(overlap, sum(hyp_ng.values()), sum(ref_ng.values()))
        scores["rougeL"] += _f1(_lcs_len(hyp_toks, ref_toks), len(hyp_toks), len(ref_toks))
    return {key: value / n_items for key, value in scores.items()}


def _fallback_meteor(hypotheses: List[str], references: List[str]) -> float:
    scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_counts = Counter(hyp.split())
        ref_counts = Counter(ref.split())
        overlap = sum(min(count, ref_counts[token]) for token, count in hyp_counts.items())
        scores.append(_f1(overlap, sum(hyp_counts.values()), sum(ref_counts.values())))
    return sum(scores) / max(1, len(scores))


def safe_translation_metrics(hypotheses: List[str], references: List[str]) -> Tuple[dict, dict, float]:
    try:
        bleu = compute_bleu(hypotheses, references)
    except Exception as e:
        print(f"[Metrics] {e}; using internal approximate BLEU fallback.")
        bleu = _fallback_bleu(hypotheses, references)

    try:
        rouge = compute_rouge(hypotheses, references)
    except Exception as e:
        print(f"[Metrics] {e}; using internal approximate ROUGE fallback.")
        rouge = _fallback_rouge(hypotheses, references)

    try:
        meteor = compute_meteor(hypotheses, references)
    except Exception as e:
        print(f"[Metrics] {e}; using internal approximate METEOR fallback.")
        meteor = _fallback_meteor(hypotheses, references)

    return bleu, rouge, meteor
