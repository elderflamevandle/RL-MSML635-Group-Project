import pytest
from src.data.retrievers import BM25Retriever, DenseRetriever

CORPUS = [
    "Christopher Nolan directed Inception in 2010.",
    "Stanley Kubrick directed 2001: A Space Odyssey.",
    "Steven Spielberg directed Jaws in 1975.",
    "Quentin Tarantino directed Pulp Fiction.",
    "Martin Scorsese directed Goodfellas.",
]

def test_bm25_returns_top_k_in_order():
    r = BM25Retriever(CORPUS)
    hits = r.retrieve("Who directed Inception?", k=2)
    assert len(hits) == 2
    assert "Nolan" in hits[0]

def test_bm25_k_capped_at_corpus_size():
    r = BM25Retriever(CORPUS[:3])
    hits = r.retrieve("film", k=10)
    assert len(hits) == 3

@pytest.mark.slow
def test_dense_returns_top_k():
    r = DenseRetriever(CORPUS, model_name="Qwen/Qwen3-Embedding-0.6B")
    hits = r.retrieve("Who directed Inception?", k=2)
    assert len(hits) == 2
    assert "Nolan" in hits[0]
