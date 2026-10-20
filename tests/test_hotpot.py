from pathlib import Path
import pytest
from src.data.hotpot import _hotpot_row_to_sample, load_hotpot_tiny

TINY = Path("data/raw/hotpot_tiny.jsonl")


def _make_row(
    *,
    supporting_titles=("A",),
    supporting_sent_ids=(0,),
    titles=("A", "B"),
    sentences=(("Alpha-0.", "Alpha-1."), ("Beta-0.",)),
    question="q?",
    answer="ans",
    row_id="row-1",
):
    return {
        "id": row_id,
        "question": question,
        "answer": answer,
        "supporting_facts": {
            "title": list(supporting_titles),
            "sent_id": list(supporting_sent_ids),
        },
        "context": {
            "title": list(titles),
            "sentences": [list(s) for s in sentences],
        },
    }


def test_row_to_sample_splits_gold_vs_distractor():
    row = _make_row()
    s = _hotpot_row_to_sample(row)
    assert s.sample_id == "row-1"
    assert s.question == "q?"
    assert s.gold_answer == "ans"
    assert s.gold_passages == ["Alpha-0. Alpha-1."]
    assert s.distractor_passages == ["Beta-0."]
    assert s.retrieved_passages == s.gold_passages + s.distractor_passages
    assert s.is_answerable is True


def test_row_to_sample_extracts_gold_evidence_by_sent_id():
    row = _make_row(
        supporting_titles=("A", "A"),
        supporting_sent_ids=(0, 1),
    )
    s = _hotpot_row_to_sample(row)
    assert s.gold_evidence == ["Alpha-0.", "Alpha-1."]


def test_row_to_sample_fail_loud_on_empty_gold():
    # supporting_fact title doesn't appear in context titles -> no gold passages
    row = _make_row(supporting_titles=("Z",))
    with pytest.raises(AssertionError, match="No gold passages"):
        _hotpot_row_to_sample(row)


def test_row_to_sample_skips_out_of_range_sent_id():
    # sent_id past the end of the paragraph should not crash; it's just dropped
    row = _make_row(
        supporting_titles=("A",),
        supporting_sent_ids=(99,),
    )
    s = _hotpot_row_to_sample(row)
    assert s.gold_evidence == []  # out-of-range dropped silently
    assert s.gold_passages == ["Alpha-0. Alpha-1."]  # passage still gold


@pytest.mark.skipif(not TINY.exists(), reason="data/raw/hotpot_tiny.jsonl not generated yet")
def test_load_hotpot_tiny_returns_samples():
    samples = load_hotpot_tiny(TINY)
    assert len(samples) == 50
    s = samples[0]
    assert s.sample_id
    assert s.question
    assert len(s.gold_passages) > 0
    assert s.gold_answer
    assert isinstance(s.is_answerable, bool)
