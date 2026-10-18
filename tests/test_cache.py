from pathlib import Path
from src.data.cache import RetrievalCache

def test_cache_round_trip(tmp_path):
    c = RetrievalCache(tmp_path)
    assert c.get("bm25", "What?", 5) is None
    c.set("bm25", "What?", 5, ["p1", "p2"])
    assert c.get("bm25", "What?", 5) == ["p1", "p2"]

def test_cache_different_keys_isolated(tmp_path):
    c = RetrievalCache(tmp_path)
    c.set("bm25", "q", 3, ["a"])
    c.set("dense", "q", 3, ["b"])
    assert c.get("bm25", "q", 3) == ["a"]
    assert c.get("dense", "q", 3) == ["b"]
