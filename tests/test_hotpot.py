from pathlib import Path
import pytest
from src.data.hotpot import load_hotpot_tiny

TINY = Path("data/raw/hotpot_tiny.jsonl")


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
