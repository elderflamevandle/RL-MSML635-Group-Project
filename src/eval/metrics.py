"""Evaluation metrics. Separate from reward metrics.

Reward metrics (ROUGE-L in src/rewards/metrics.py) are used for shaping the
training signal. These eval metrics (EM / abstention / Balanced Accuracy)
are used for scoring runs against the GRACE baseline and comparing R0/R1/R2.
"""
import re
import string

_PUNCT_RE = re.compile(r"[" + re.escape(string.punctuation) + r"]")

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text

def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0

def abstention_accuracy(predictions: list[tuple[str, bool]]) -> float:
    """predictions: list of (path, is_answerable) tuples.
    Returns the True-Negative Rate -- fraction of unanswerable samples
    that were correctly routed to the <llm> path."""
    unans = [p for p in predictions if not p[1]]
    if not unans:
        return 0.0
    correct = sum(1 for p, _ in unans if p == "llm")
    return correct / len(unans)

def balanced_accuracy(em_answerable: float, abstention_unanswerable: float) -> float:
    """Primary metric: arithmetic mean of answerable EM and unanswerable abstention.
    Penalises policies that collapse into a single path."""
    return 0.5 * em_answerable + 0.5 * abstention_unanswerable
