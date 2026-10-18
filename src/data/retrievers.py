"""BM25 and dense retrievers. Both expose the same `retrieve(query, k)` interface."""
import re
from typing import Protocol
import numpy as np
from rank_bm25 import BM25Okapi

class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[str]: ...

_TOKEN_RE = re.compile(r"\w+")

def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())

class BM25Retriever:
    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        self._bm25 = BM25Okapi([_tokenize(c) for c in corpus])

    def retrieve(self, query: str, k: int) -> list[str]:
        scores = self._bm25.get_scores(_tokenize(query))
        scores = np.nan_to_num(scores, nan=-np.inf)
        idx = np.argsort(-scores, kind="stable")[:k]
        return [self.corpus[i] for i in idx]

class DenseRetriever:
    def __init__(self, corpus: list[str], model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        from sentence_transformers import SentenceTransformer
        self.corpus = corpus
        self._model = SentenceTransformer(model_name)
        self._embs = self._model.encode(
            corpus, convert_to_numpy=True, normalize_embeddings=True
        )

    def retrieve(self, query: str, k: int) -> list[str]:
        q_emb = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = self._embs @ q_emb
        sims = np.nan_to_num(sims, nan=-np.inf)
        idx = np.argsort(-sims, kind="stable")[:k]
        return [self.corpus[i] for i in idx]
