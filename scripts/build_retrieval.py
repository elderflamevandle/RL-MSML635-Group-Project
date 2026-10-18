"""Build retrieval + balanced split for train/val with BM25 and dense retrievers.

Writes one file per (retriever, split) in data/cache/.

Each output row is a dict matching Sample's __dict__ plus a `prompt` field
built by `src.prompts.templates.build_prompt(question, retrieved_passages)`
so verl can read it directly.
"""
from pathlib import Path
import json
from src.data.hotpot import load_hotpot_tiny  # works on full files too (same format)
from src.data.retrievers import BM25Retriever, DenseRetriever
from src.data.balanced_split import build_balanced_split
from src.data.cache import RetrievalCache
from src.prompts.templates import build_prompt

K = 5
SPLITS = [
    ("train", Path("data/raw/hotpot_train.jsonl")),
    ("val",   Path("data/raw/hotpot_val.jsonl")),
]

def all_passages(samples) -> list[str]:
    seen, out = set(), []
    for s in samples:
        for p in s.gold_passages + s.distractor_passages:
            if p not in seen:
                seen.add(p); out.append(p)
    return out

def dump(samples, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            row = {
                "sample_id": s.sample_id,
                "question": s.question,
                "gold_passages": s.gold_passages,
                "distractor_passages": s.distractor_passages,
                "gold_answer": s.gold_answer,
                "gold_evidence": s.gold_evidence,
                "retrieved_passages": s.retrieved_passages,
                "is_answerable": s.is_answerable,
                "prompt": build_prompt(s.question, s.retrieved_passages),
            }
            f.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    cache = RetrievalCache(Path("data/cache/retrieval"))

    def cached_retrieve(name: str, R):
        def _fn(q: str, k: int) -> list[str]:
            hit = cache.get(name, q, k)
            if hit is not None:
                return hit
            passages = R.retrieve(q, k)
            cache.set(name, q, k, passages)
            return passages
        return _fn

    for split, raw in SPLITS:
        samples = load_hotpot_tiny(raw)
        corpus = all_passages(samples)
        for name, R in [("bm25", BM25Retriever(corpus)),
                        ("dense", DenseRetriever(corpus))]:
            split_samples = build_balanced_split(samples, cached_retrieve(name, R), k=K)
            n_ans = sum(s.is_answerable for s in split_samples)
            print(f"{name}/{split}: n={len(split_samples)}, ans={n_ans}, unans={len(split_samples)-n_ans}")
            ans_rate = n_ans / len(split_samples) if split_samples else 0.0
            assert 0.1 < ans_rate < 0.95, (
                f"Suspicious answerable rate {ans_rate:.2f} for {name}/{split} — "
                f"gold/retrieved string normalization likely mismatched. "
                f"Expected roughly 0.3-0.7 for a balanced split."
            )
            dump(split_samples, Path(f"data/cache/{name}_{split}.jsonl"))
