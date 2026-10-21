"""Unit tests for src.eval.evaluate._score.

We exercise the scoring+aggregation layer with fake responses so no model
is loaded. BERTScore and the Groq LJ are monkey-patched to avoid network / heavy
deps.
"""
from __future__ import annotations

from src.common.types import Sample
from src.eval import evaluate as E


def _samples() -> list[Sample]:
    return [
        Sample(
            sample_id="a",
            question="Who directed Big Stone Gap?",
            gold_passages=["Big Stone Gap was directed by Adriana Trigiani."],
            distractor_passages=[],
            gold_answer="Adriana Trigiani",
            gold_evidence=["Big Stone Gap was directed by Adriana Trigiani."],
            retrieved_passages=["Big Stone Gap was directed by Adriana Trigiani."],
            is_answerable=True,
        ),
        Sample(
            sample_id="b",
            question="What color is a dragon fruit's flesh?",
            gold_passages=[],
            distractor_passages=["Lorem ipsum dolor sit amet."],
            gold_answer="pink",
            gold_evidence=[],
            retrieved_passages=["Lorem ipsum dolor sit amet."],
            is_answerable=False,
        ),
    ]


def test_score_core_metrics_without_optional_scorers():
    samples = _samples()
    responses = [
        "<evidence>Big Stone Gap was directed by Adriana Trigiani.</evidence><answer>Adriana Trigiani</answer>",
        "<llm>I don't know.</llm><answer>unknown</answer>",
    ]
    rows, metrics = E._score(samples, responses)

    # Row-level
    assert rows[0]["em"] == 1.0
    assert rows[0]["is_answerable"] is True
    assert rows[0]["path"] == "evidence"
    assert rows[1]["path"] == "llm"
    assert rows[1]["is_answerable"] is False
    # Nothing optional should appear
    assert "bertscore_f1" not in rows[0]
    assert "judge_score" not in rows[0]

    # Aggregate
    assert metrics["n"] == 2
    assert metrics["n_answerable"] == 1
    assert metrics["em_answerable"] == 1.0
    assert metrics["abstention_unanswerable"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["path_shares"]["evidence"] == 0.5
    assert metrics["path_shares"]["llm"] == 0.5
    assert "bertscore_f1_answerable" not in metrics
    assert "judge_score_answerable" not in metrics


def test_score_with_bertscore(monkeypatch):
    """BERTScore is monkey-patched to avoid loading the big model."""
    monkeypatch.setattr(E, "_bertscore_f1", lambda preds, refs: [0.92] * len(preds))
    samples = _samples()
    responses = [
        "<evidence>...</evidence><answer>Adriana Trigiani</answer>",
        "<llm>I don't know.</llm><answer>unknown</answer>",
    ]
    rows, metrics = E._score(samples, responses, with_bertscore=True)
    assert rows[0]["bertscore_f1"] == 0.92
    assert rows[1]["bertscore_f1"] is None  # unanswerable
    assert metrics["bertscore_f1_answerable"] == 0.92


def test_score_with_judge(monkeypatch):
    """Groq judge is monkey-patched; verifies plumbing + aggregation."""
    captured: list[tuple[str, str, str]] = []

    def fake_scores(questions, preds, refs, cache_dir):
        captured.extend(zip(questions, preds, refs))
        return [1.0] * len(preds)

    monkeypatch.setattr(E, "_judge_scores", fake_scores)

    samples = _samples()
    responses = [
        "<evidence>...</evidence><answer>Adriana Trigiani</answer>",
        "<llm>I don't know.</llm><answer>unknown</answer>",
    ]
    rows, metrics = E._score(samples, responses, with_judge=True)
    # Only the answerable sample should have been judged
    assert len(captured) == 1
    assert captured[0][0] == "Who directed Big Stone Gap?"
    assert rows[0]["judge_score"] == 1.0
    assert rows[1]["judge_score"] is None
    assert metrics["judge_score_answerable"] == 1.0


def test_score_with_both_scorers(monkeypatch):
    monkeypatch.setattr(E, "_bertscore_f1", lambda preds, refs: [0.80, 0.90][:len(preds)])
    monkeypatch.setattr(E, "_judge_scores",
                        lambda questions, preds, refs, cache_dir: [1.0] * len(preds))
    samples = _samples()
    responses = [
        "<evidence>...</evidence><answer>Adriana Trigiani</answer>",
        "<llm>...</llm><answer>unknown</answer>",
    ]
    _, metrics = E._score(samples, responses, with_bertscore=True, with_judge=True)
    assert "bertscore_f1_answerable" in metrics
    assert "judge_score_answerable" in metrics
    assert metrics["judge_score_answerable"] == 1.0
