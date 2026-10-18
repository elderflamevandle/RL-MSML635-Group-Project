"""Reward-component metrics. ROUGE-L here matches what GRACE uses."""
from rouge_score import rouge_scorer

_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(prediction: str, reference: str | list[str]) -> float:
    """ROUGE-L F1. If reference is a list, returns the max over references."""
    if isinstance(reference, str):
        references = [reference]
    else:
        references = reference
    if not prediction or not references:
        return 0.0
    scores = [
        _SCORER.score(ref, prediction)["rougeL"].fmeasure for ref in references
    ]
    return max(scores)


# Used by R1/R2 reward functions, NOT by R0. Do not remove as "unused".
def quote_in_passages(quote: str, passages: list[str]) -> bool:
    """True if `quote` appears as a substring in any passage (case-sensitive)."""
    return any(quote.strip() in p for p in passages)
