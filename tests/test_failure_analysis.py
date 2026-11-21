"""Unit tests for src/eval/failure_analysis.py."""
from src.common.types import Sample
from src.eval.failure_analysis import (
    FailureCategory,
    classify,
    classify_all,
    summarize,
)


def _ans_sample(sid: str = "A") -> Sample:
    return Sample(
        sample_id=sid,
        question="Who directed Big Stone Gap?",
        gold_passages=["Big Stone Gap was directed by Adriana Trigiani."],
        distractor_passages=["Lorem ipsum."],
        gold_answer="Adriana Trigiani",
        gold_evidence=["Big Stone Gap was directed by Adriana Trigiani."],
        retrieved_passages=[
            "Big Stone Gap was directed by Adriana Trigiani.",
            "Lorem ipsum.",
        ],
        is_answerable=True,
    )


def _unans_sample(sid: str = "U") -> Sample:
    return Sample(
        sample_id=sid,
        question="What color is dragon fruit's flesh?",
        gold_passages=[],
        distractor_passages=["Something irrelevant."],
        gold_answer="pink",
        gold_evidence=[],
        retrieved_passages=["Something irrelevant.", "More irrelevant text."],
        is_answerable=False,
    )


def _pred(raw: str, sid: str = "A") -> dict:
    return {"sample_id": sid, "raw_response": raw}


def test_correct_evidence_path_on_answerable():
    s = _ans_sample()
    p = _pred(
        "<evidence>Big Stone Gap was directed by Adriana Trigiani.</evidence>"
        "<answer>Adriana Trigiani</answer>"
    )
    c = classify(p, s)
    assert c.category == FailureCategory.CORRECT
    assert c.is_answerable is True


def test_correct_abstention_on_unanswerable():
    s = _unans_sample()
    p = _pred(
        "<llm>I cannot determine this from the retrieved context.</llm>"
        "<answer>unknown</answer>",
        sid="U",
    )
    c = classify(p, s)
    assert c.category == FailureCategory.CORRECT
    assert c.is_answerable is False


def test_format_invalid():
    s = _ans_sample()
    p = _pred("just some text with no tags")
    c = classify(p, s)
    assert c.category == FailureCategory.FORMAT_INVALID


def test_wrong_path_llm_on_answerable():
    s = _ans_sample()
    p = _pred("<llm>abstaining</llm><answer>unknown</answer>")
    c = classify(p, s)
    assert c.category == FailureCategory.WRONG_PATH
    assert "abstained on an answerable" in c.detail


def test_wrong_path_evidence_on_unanswerable_is_hallucination():
    s = _unans_sample()
    p = _pred(
        "<evidence>Something irrelevant.</evidence>"
        "<answer>purple</answer>",
        sid="U",
    )
    c = classify(p, s)
    assert c.category == FailureCategory.WRONG_PATH
    assert "hallucination" in c.detail.lower() or "unanswerable" in c.detail.lower()


def test_fabricated_evidence():
    s = _ans_sample()
    p = _pred(
        "<evidence>This exact sentence does not appear in retrieved passages.</evidence>"
        "<answer>Adriana Trigiani</answer>"
    )
    c = classify(p, s)
    assert c.category == FailureCategory.FABRICATED_EVIDENCE


def test_wrong_content_grounded_but_wrong_answer():
    s = _ans_sample()
    p = _pred(
        "<evidence>Big Stone Gap was directed by Adriana Trigiani.</evidence>"
        "<answer>Steven Spielberg</answer>"
    )
    c = classify(p, s)
    assert c.category == FailureCategory.WRONG_CONTENT


def test_answer_format_semantic_match():
    s = _ans_sample()
    p = _pred(
        "<evidence>Big Stone Gap was directed by Adriana Trigiani.</evidence>"
        "<answer>Adriana Trigiani is the director.</answer>"
    )
    c = classify(p, s)
    # Pred string contains gold string, so proxy says 'semantic match'
    assert c.category == FailureCategory.ANSWER_FORMAT


def test_classify_all_skips_unknown_sample_ids():
    samples = {"A": _ans_sample("A")}
    preds = [
        {"sample_id": "A", "raw_response": "<evidence>Big Stone Gap was directed by Adriana Trigiani.</evidence><answer>Adriana Trigiani</answer>"},
        {"sample_id": "GHOST", "raw_response": "x"},
    ]
    out = classify_all(preds, samples)
    assert len(out) == 1
    assert out[0].sample_id == "A"


def test_summarize_produces_rates_and_derived():
    sA = _ans_sample("A")
    sB = _ans_sample("B")
    sU = _unans_sample("U")
    samples = {"A": sA, "B": sB, "U": sU}
    preds = [
        {"sample_id": "A", "raw_response":
            "<evidence>Big Stone Gap was directed by Adriana Trigiani.</evidence><answer>Adriana Trigiani</answer>"},
        {"sample_id": "B", "raw_response":
            "<llm>no idea</llm><answer>unknown</answer>"},  # wrong path (ans → llm)
        {"sample_id": "U", "raw_response":
            "<evidence>Something irrelevant.</evidence><answer>purple</answer>"},  # hallucination
    ]
    classifications = classify_all(preds, samples)
    s = summarize(classifications)
    assert s["n"] == 3
    assert s["n_answerable"] == 2
    assert s["n_unanswerable"] == 1
    # 1 correct, 1 wrong-path, 1 hallucination (also wrong-path by our taxonomy)
    assert s["counts"]["correct"] == 1
    assert s["counts"]["wrong_path"] == 2
    # Derived
    assert s["derived"]["hallucination_rate_on_unanswerable"] == 1.0
    assert s["derived"]["abstention_rate_on_answerable"] == 0.5
