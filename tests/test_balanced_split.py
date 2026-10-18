from src.common.types import Sample
from src.data.balanced_split import build_balanced_split

def _mk(sid, is_ans=True):
    return Sample(
        sample_id=sid, question="q",
        gold_passages=["gold"] if is_ans else [],
        distractor_passages=["d1", "d2", "d3", "d4", "d5"],
        gold_answer="a", gold_evidence=["gold"],
        retrieved_passages=[], is_answerable=True,
    )

def test_balanced_split_marks_unanswerable_when_gold_missing_from_retrieved():
    samples = [_mk(f"s{i}") for i in range(10)]
    def fake_retrieve(q, k):
        return samples[0].distractor_passages[:k]
    out = build_balanced_split(samples, retrieve_fn=fake_retrieve, k=5)
    assert all(s.is_answerable is False for s in out)
    assert all(len(s.retrieved_passages) == 5 for s in out)

def test_balanced_split_marks_answerable_when_gold_in_retrieved():
    samples = [_mk(f"s{i}") for i in range(10)]
    def fake_retrieve(q, k):
        return samples[0].gold_passages + samples[0].distractor_passages[:k-1]
    out = build_balanced_split(samples, retrieve_fn=fake_retrieve, k=5)
    assert all(s.is_answerable is True for s in out)
