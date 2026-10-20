"""Build retrieval + balanced split for train/val with BM25 and dense retrievers.

Writes one file per (retriever, split) in data/cache/.

Each output row is a dict matching Sample's __dict__ plus a `prompt` field
built by `src.prompts.templates.build_prompt(question, retrieved_passages)`
so verl can read it directly.

Usage:
    python scripts/build_retrieval.py                    # auto device, bs=32
    python scripts/build_retrieval.py --dense-device cpu # force CPU for dense
    python scripts/build_retrieval.py --dense-batch 8    # lower bs for small-VRAM GPUs
    python scripts/build_retrieval.py --no-dense         # BM25 only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib
import random

from src.common.types import Sample
from src.data.hotpot import load_hotpot_tiny  # works on full files too (same format)
from src.data.retrievers import BM25Retriever, DenseRetriever
from src.data.balanced_split import build_balanced_split
from src.data.cache import RetrievalCache
from src.prompts.templates import build_prompt

K = 5


def _should_be_unanswerable(sample_id: str, target_unans_rate: float = 0.45) -> bool:
    """Deterministic distractor-swap decision for a sample.

    Hash the sample_id to [0,1); if below the target unans rate, this sample
    is forced unanswerable (gold stripped from retrieved_passages).
    """
    h = int(hashlib.sha1(sample_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < target_unans_rate


def distractor_swap(samples: list[Sample], corpus: list[str], k: int, seed: int = 0) -> list[Sample]:
    """Force ~target_unans_rate of samples to be unanswerable by replacing their
    retrieved_passages with k passages drawn from `corpus` that exclude any of
    the sample's gold_passages.

    Only mutates samples flagged by `_should_be_unanswerable`. Deterministic for
    a given seed + sample_id set.
    """
    rng = random.Random(seed)
    gold_free_pool = corpus  # we filter per-sample below
    out: list[Sample] = []
    for s in samples:
        if not _should_be_unanswerable(s.sample_id):
            out.append(s)
            continue
        candidates = [p for p in gold_free_pool if p not in s.gold_passages]
        if len(candidates) < k:
            # Degenerate case: corpus too small; leave as-is
            out.append(s)
            continue
        swapped = rng.sample(candidates, k)
        out.append(Sample(
            sample_id=s.sample_id,
            question=s.question,
            gold_passages=s.gold_passages,
            distractor_passages=s.distractor_passages,
            gold_answer=s.gold_answer,
            gold_evidence=s.gold_evidence,
            retrieved_passages=swapped,
            is_answerable=False,
        ))
    return out
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-device", default=None,
                    help="Device for the dense retriever. 'cpu'|'cuda'|None (auto).")
    ap.add_argument("--dense-batch", type=int, default=32,
                    help="Encoding batch size for the dense retriever.")
    ap.add_argument("--no-dense", action="store_true",
                    help="Skip the dense retriever entirely (BM25 only).")
    args = ap.parse_args()

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
        retrievers = [("bm25", BM25Retriever(corpus))]
        if not args.no_dense:
            retrievers.append((
                "dense",
                DenseRetriever(corpus, device=args.dense_device, batch_size=args.dense_batch),
            ))
        for name, R in retrievers:
            split_samples = build_balanced_split(samples, cached_retrieve(name, R), k=K)
            # Distractor swap: force ~45% of samples unanswerable by replacing
            # retrieved_passages with k gold-free passages. GRACE-style balance.
            split_samples = distractor_swap(split_samples, corpus, k=K, seed=0)
            n_ans = sum(s.is_answerable for s in split_samples)
            print(f"{name}/{split}: n={len(split_samples)}, ans={n_ans}, unans={len(split_samples)-n_ans}")
            ans_rate = n_ans / len(split_samples) if split_samples else 0.0
            assert 0.1 < ans_rate < 0.95, (
                f"Suspicious answerable rate {ans_rate:.2f} for {name}/{split} — "
                f"gold/retrieved string normalization likely mismatched. "
                f"Expected roughly 0.3-0.7 for a balanced split."
            )
            dump(split_samples, Path(f"data/cache/{name}_{split}.jsonl"))


if __name__ == "__main__":
    main()
