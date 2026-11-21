"""Convert our retrieval-cache JSONL into the parquet layout verl's RLHFDataset reads.

verl @ 4aa02fe wants parquet (not jsonl) with these columns:
    data_source   str                      -- picks the reward fn
    prompt        list[{role, content}]    -- chat messages, NOT a plain string
    reward_model  {ground_truth: str}
    extra_info    {index: int, sample_json: str}

Usage:
    python scripts/build_verl_data.py                       # all data/cache/*.jsonl
    python scripts/build_verl_data.py data/cache/bm25_val.jsonl --limit 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prompts.templates import build_prompt  # noqa: E402

DATA_SOURCE = "hotpot_rag"


def convert(src: Path, dst: Path, limit: int | None = None) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    with src.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            prompt = row.pop("prompt", None) or build_prompt(
                row["question"], row["retrieved_passages"]
            )
            rows.append(
                {
                    "data_source": DATA_SOURCE,
                    "prompt": [{"role": "user", "content": prompt}],
                    "reward_model": {"ground_truth": row["gold_answer"]},
                    "extra_info": {
                        "index": i,
                        "sample_json": json.dumps(row, ensure_ascii=False),
                    },
                }
            )
    if not rows:
        raise SystemExit(f"{src} is empty")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), dst)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path,
                    help="JSONL files (default: every data/cache/*_{train,val}.jsonl)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--suffix", default="",
                    help="appended to the output stem, e.g. --suffix _tiny")
    args = ap.parse_args()

    files = args.files or sorted(Path("data/cache").glob("*_train.jsonl")) + sorted(
        Path("data/cache").glob("*_val.jsonl")
    )
    if not files:
        raise SystemExit("no JSONL found under data/cache -- run scripts/build_retrieval.py first")
    for src in files:
        dst = src.with_name(src.stem + args.suffix + ".parquet")
        n = convert(src, dst, args.limit)
        print(f"{src} -> {dst}  ({n} rows)")


if __name__ == "__main__":
    main()
