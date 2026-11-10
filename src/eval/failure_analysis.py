"""Classify why each prediction failed.

Takes a predictions.jsonl (as produced by `src.eval.evaluate` or
`scripts/local_sanity.py`), joined to its source split by sample_id,
and tags every non-correct prediction with one of:

    FORMAT_INVALID       - no valid <evidence>/<llm> + <answer> structure
    WRONG_PATH           - took <llm> on answerable OR <evidence> on unanswerable
    FABRICATED_EVIDENCE  - evidence-path, quote NOT in retrieved_passages
    WRONG_CONTENT        - evidence-path, quote grounded, but answer differs from gold
    ANSWER_FORMAT        - evidence-path, grounded quote, right answer semantically but EM miss
    CORRECT              - EM match on answerable OR correct <llm>-path on unanswerable

Why we need this: the aggregate metrics (BA, EM, TNR) tell us *how often* the
policy is wrong; this module tells us *why*. Required reading before iterating
on reward hyperparameters or deciding whether R2 is "too strict" vs "just right".

The classifier is a pure function: give it predictions + samples and it hands
back categories without side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from src.common.parse import parse_response
from src.common.types import Sample
from src.eval.metrics import exact_match


class FailureCategory(str, Enum):
    CORRECT = "correct"
    FORMAT_INVALID = "format_invalid"
    WRONG_PATH = "wrong_path"
    FABRICATED_EVIDENCE = "fabricated_evidence"
    WRONG_CONTENT = "wrong_content"
    ANSWER_FORMAT = "answer_format"  # semantic match but EM miss


@dataclass(frozen=True)
class ClassifiedPrediction:
    sample_id: str
    is_answerable: bool
    path: str
    category: FailureCategory
    detail: str  # one-line human-readable rationale


def _quote_in(quote: str, passages: list[str]) -> bool:
    """True if `quote` appears verbatim (case-insensitive, whitespace-collapsed)
    in any of `passages`. Mirrors the gate inside R1/R2."""
    def _norm(s: str) -> str:
        return " ".join(s.lower().split())
    nq = _norm(quote)
    if not nq:
        return False
    return any(nq in _norm(p) for p in passages)


def _looks_semantically_right(pred: str, gold: str) -> bool:
    """Rough proxy: lowercased stripped equality OR gold-is-substring-of-pred.
    Not meant to replace BERTScore / LJ; used only to split 'wrong_content'
    from 'answer_format' so humans can prioritize fixes."""
    p = pred.lower().strip().rstrip(".,")
    g = gold.lower().strip().rstrip(".,")
    if not p or not g:
        return False
    return p == g or g in p or p in g


def classify(prediction: dict, sample: Sample) -> ClassifiedPrediction:
    """Classify a single (prediction, sample) pair.

    `prediction` is a row from predictions.jsonl (needs: sample_id, raw_response,
    is_answerable). Everything else is re-derived from the raw_response via
    `parse_response` so the classifier is robust to minor prediction-file schema
    drift.
    """
    raw = prediction.get("raw_response", "")
    parsed = parse_response(raw)
    is_ans = sample.is_answerable

    # 1. Format
    if parsed.path == "invalid":
        return ClassifiedPrediction(
            sample_id=sample.sample_id,
            is_answerable=is_ans,
            path="invalid",
            category=FailureCategory.FORMAT_INVALID,
            detail="no valid <evidence>/<llm> + <answer> structure",
        )

    # 2. Correct abstention on unanswerable
    if not is_ans and parsed.path == "llm":
        return ClassifiedPrediction(
            sample_id=sample.sample_id,
            is_answerable=is_ans,
            path="llm",
            category=FailureCategory.CORRECT,
            detail="correctly abstained on unanswerable",
        )

    # 3. Path routing
    if is_ans and parsed.path == "llm":
        return ClassifiedPrediction(
            sample_id=sample.sample_id,
            is_answerable=is_ans,
            path="llm",
            category=FailureCategory.WRONG_PATH,
            detail="abstained on an answerable query",
        )
    if not is_ans and parsed.path == "evidence":
        return ClassifiedPrediction(
            sample_id=sample.sample_id,
            is_answerable=is_ans,
            path="evidence",
            category=FailureCategory.WRONG_PATH,
            detail="answered on an unanswerable query (hallucination)",
        )

    # 4. Evidence path on an answerable — most interesting bucket
    if parsed.path == "evidence" and is_ans:
        all_quotes_grounded = all(
            _quote_in(q, sample.retrieved_passages) for q in parsed.quoted_evidence
        )
        if not all_quotes_grounded:
            return ClassifiedPrediction(
                sample_id=sample.sample_id,
                is_answerable=is_ans,
                path="evidence",
                category=FailureCategory.FABRICATED_EVIDENCE,
                detail="at least one quoted evidence span is not in retrieved_passages",
            )
        em = exact_match(parsed.answer, sample.gold_answer)
        if em == 1.0:
            return ClassifiedPrediction(
                sample_id=sample.sample_id,
                is_answerable=is_ans,
                path="evidence",
                category=FailureCategory.CORRECT,
                detail="EM=1 on answerable",
            )
        if _looks_semantically_right(parsed.answer, sample.gold_answer):
            return ClassifiedPrediction(
                sample_id=sample.sample_id,
                is_answerable=is_ans,
                path="evidence",
                category=FailureCategory.ANSWER_FORMAT,
                detail=f"semantic match but EM miss "
                       f"(pred={parsed.answer!r} gold={sample.gold_answer!r})",
            )
        return ClassifiedPrediction(
            sample_id=sample.sample_id,
            is_answerable=is_ans,
            path="evidence",
            category=FailureCategory.WRONG_CONTENT,
            detail=f"grounded quote but wrong answer "
                   f"(pred={parsed.answer!r} gold={sample.gold_answer!r})",
        )

    # Fallback — shouldn't be reached given the exhaustive branches above
    return ClassifiedPrediction(
        sample_id=sample.sample_id,
        is_answerable=is_ans,
        path=parsed.path,
        category=FailureCategory.WRONG_PATH,
        detail="unclassified path/answerable combination",
    )


def classify_all(
    predictions: Iterable[dict], samples: dict[str, Sample]
) -> list[ClassifiedPrediction]:
    """Classify an entire predictions.jsonl against a sample_id → Sample map.
    Predictions whose sample_id is unknown are silently skipped."""
    out: list[ClassifiedPrediction] = []
    for p in predictions:
        s = samples.get(p.get("sample_id"))
        if s is None:
            continue
        out.append(classify(p, s))
    return out


def summarize(classifications: list[ClassifiedPrediction]) -> dict:
    """Produce category counts + a few derived rates useful for the audit table."""
    n = len(classifications)
    counts: dict[str, int] = {cat.value: 0 for cat in FailureCategory}
    for c in classifications:
        counts[c.category.value] += 1
    rates = {k: (v / n if n else 0.0) for k, v in counts.items()}

    n_ans = sum(1 for c in classifications if c.is_answerable)
    n_unans = sum(1 for c in classifications if not c.is_answerable)

    # Rate of each category *conditional on the subset where it can occur*:
    n_evidence_on_ans = sum(
        1 for c in classifications if c.is_answerable and c.path == "evidence"
    )
    fabricated_rate = (
        sum(1 for c in classifications
            if c.category == FailureCategory.FABRICATED_EVIDENCE)
        / n_evidence_on_ans
        if n_evidence_on_ans else 0.0
    )
    hallucination_rate = (
        sum(1 for c in classifications
            if not c.is_answerable and c.path == "evidence")
        / n_unans
        if n_unans else 0.0
    )
    abstention_on_ans_rate = (
        sum(1 for c in classifications
            if c.is_answerable and c.path == "llm")
        / n_ans
        if n_ans else 0.0
    )

    return {
        "n": n,
        "n_answerable": n_ans,
        "n_unanswerable": n_unans,
        "counts": counts,
        "rates": rates,
        "derived": {
            "fabricated_evidence_rate_on_evidence_path_answerable": fabricated_rate,
            "hallucination_rate_on_unanswerable": hallucination_rate,
            "abstention_rate_on_answerable": abstention_on_ans_rate,
        },
    }
