

from typing import List, Tuple


def levenshtein_distance(s1, s2):

    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_wer(
    hypotheses: List[List[str]],
    references: List[List[str]],
) -> float:
    """
    Compute corpus-level WER.

    WER = (S + D + I) / N
    where N = total reference tokens.

    Args:
        hypotheses: list of predicted token lists
        references : list of ground-truth token lists

    Returns:
        WER as a float (0---1 range, lower is better)
    """
    total_errors = 0
    total_ref_len = 0
    for hyp, ref in zip(hypotheses, references):
        total_errors  += levenshtein_distance(hyp, ref)
        total_ref_len += len(ref)
    if total_ref_len == 0:
        return 0.0
    return total_errors / total_ref_len


def compute_wer_strings(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Convenience wrapper accepting space-joined strings."""
    hyp_lists = [h.strip().split() for h in hypotheses]
    ref_lists = [r.strip().split() for r in references]
    return compute_wer(hyp_lists, ref_lists)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# BLEU
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_bleu(
    hypotheses: List[str],
    references: List[str],
    max_order: int = 4,
) -> dict:
    """
    Compute cumulative corpus BLEU-n scores using sacrebleu.

    BLEU-1..BLEU-4 are computed with the same corpus-level BLEU
    definition, changing only max_ngram_order. This matches the SLT
    reporting convention used by papers such as CV-SLT/MSKA, where
    BLEU-1 >= BLEU-2 >= BLEU-3 >= BLEU-4 in normal cases.

    Args:
        hypotheses: list of predicted sentences (strings)
        references : list of reference sentences (strings)

    Returns:
        dict with keys "bleu1" .. "bleu4" and "bleu" (BLEU-4)
    """
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        raise ImportError("Install sacrebleu: pip install sacrebleu")

    scores = {}
    for order in range(1, max_order + 1):
        metric = BLEU(max_ngram_order=order)
        scores[f"bleu{order}"] = metric.corpus_score(hypotheses, [references]).score
    scores["bleu"] = scores[f"bleu{max_order}"]
    return scores


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ROUGE
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_rouge(
    hypotheses: List[str],
    references: List[str],
) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    Returns:
        dict with keys "rouge1", "rouge2", "rougeL"
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("Install rouge-score: pip install rouge-score")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        for k in agg:
            agg[k] += scores[k].fmeasure
    n = max(len(hypotheses), 1)
    return {k: v / n for k, v in agg.items()}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# METEOR
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_meteor(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """
    Compute corpus-level METEOR score using nltk.

    Returns:
        METEOR score as float (0---1)
    """
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4",  quiet=True)
    except ImportError:
        raise ImportError("Install nltk: pip install nltk")

    scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.split()
        ref_tokens = ref.split()
        scores.append(meteor_score([ref_tokens], hyp_tokens))
    return sum(scores) / max(len(scores), 1)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# All-in-one evaluator
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def evaluate_all(
    hyp_glosses: List[List[str]],
    ref_glosses: List[List[str]],
    hyp_sentences: List[str],
    ref_sentences: List[str],
) -> dict:
    """
    Run all metrics and return a unified dict.

    Returns:
        {
          "wer":    float,
          "bleu":   float,
          "bleu1":  float,
          "bleu2":  float,
          "bleu3":  float,
          "bleu4":  float,
          "rouge1": float,
          "rouge2": float,
          "rougeL": float,
          "meteor": float,
        }
    """
    results = {}

    # WER (CSLR)
    results["wer"] = compute_wer(hyp_glosses, ref_glosses)

    # Translation metrics (SLT)
    bleu_scores   = compute_bleu(hyp_sentences, ref_sentences)
    rouge_scores  = compute_rouge(hyp_sentences, ref_sentences)
    meteor        = compute_meteor(hyp_sentences, ref_sentences)

    results.update(bleu_scores)
    results.update(rouge_scores)
    results["meteor"] = meteor

    return results
