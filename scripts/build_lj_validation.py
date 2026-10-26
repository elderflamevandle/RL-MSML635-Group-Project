"""Materialize a 50-example LJ validation template from HotpotQA tiny.

Each row has (question, gold_answer, prediction, human_label) where
`prediction` is a slot the labeler fills with a realistic candidate
answer and `human_label` is the 0/1 agreement decision.

We seed each row with a plausible synthetic prediction so the labeler
can see the schema at a glance, but they are expected to overwrite
`prediction` with answers taken from a real training run's
predictions.jsonl once Groq LJ runs are available.

Produces:
    data/lj_validation/50_questions.jsonl

Usage:
    python scripts/build_lj_validation.py                   # 50 rows from tiny file
    python scripts/build_lj_validation.py --n 100 --seed 7  # bigger / different sample

Each row schema:
    {
      "sample_id": str,
      "question": str,
      "gold_answer": str,
      "prediction": str,         # LABELER FILLS THIS from a real run
      "human_label": null | 0 | 1,
      "notes": ""
    }
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.hotpot import load_hotpot_tiny


def _seed_prediction(gold: str) -> str:
    """Return a plausible candidate so labelers see the shape; MUST be overwritten."""
    return f"<FILL FROM REAL RUN — e.g. '{gold}' or a wrong candidate>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny-file", type=Path,
                    default=Path("data/raw/hotpot_tiny.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/lj_validation/50_questions.jsonl"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples = load_hotpot_tiny(args.tiny_file)
    rng = random.Random(args.seed)
    picks = rng.sample(samples, min(args.n, len(samples)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "sample_id": s.sample_id,
            "question": s.question,
            "gold_answer": s.gold_answer,
            "prediction": _seed_prediction(s.gold_answer),
            "human_label": None,
            "notes": "",
        }
        for s in picks
    ]
    args.out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.out}")
    print("Next: open the file, replace each 'prediction' with a real candidate from a run,")
    print("and set 'human_label' to 0 or 1. Then run scripts/validate_lj.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
