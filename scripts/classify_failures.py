"""CLI: classify failures in a predictions.jsonl file.

Writes two files to --out:
    classifications.jsonl   # one classified row per prediction
    failure_summary.json    # counts + derived rates

Usage:
    python scripts/classify_failures.py \\
        results/local_sanity/20261020-194948/predictions.jsonl \\
        data/raw/hotpot_tiny.jsonl \\
        --out results/local_sanity/20261020-194948/failures
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.hotpot import load_hotpot_tiny
from src.eval.failure_analysis import classify_all, summarize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("split", type=Path,
                    help="Source split .jsonl carrying sample_id → gold fields.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    samples = {s.sample_id: s for s in load_hotpot_tiny(args.split)}
    preds = [
        json.loads(l) for l in args.predictions.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    classifications = classify_all(preds, samples)
    summary = summarize(classifications)

    (args.out / "classifications.jsonl").write_text(
        "\n".join(json.dumps(asdict(c) | {"category": c.category.value})
                  for c in classifications),
        encoding="utf-8",
    )
    (args.out / "failure_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"n={summary['n']}  answerable={summary['n_answerable']}  unanswerable={summary['n_unanswerable']}")
    print("category rates:")
    for cat, rate in sorted(summary["rates"].items(), key=lambda kv: -kv[1]):
        print(f"  {cat:22s} {rate:5.2%}  (n={summary['counts'][cat]})")
    print("derived:")
    for k, v in summary["derived"].items():
        print(f"  {k:50s} {v:5.2%}")
    print(f"wrote {args.out}/classifications.jsonl + failure_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
