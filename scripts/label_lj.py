"""Interactive labeler for data/lj_validation/50_questions.jsonl.

For every row with `human_label is None`, show the question, gold answer, and
prediction and ask: is the prediction semantically equivalent to the gold?
(The judge's own criterion — see src/eval/judge.PROMPT.)

Keys:
    y / 1   label = 1 (equivalent)
    n / 0   label = 0 (not equivalent)
    s       skip (leave this row unlabeled, move on)
    b       back to the previous row (edit its label)
    q       quit, keep progress
    ?       show the rubric again

The file is rewritten atomically after every keypress, so Ctrl-C never loses
work. Resume by re-running — labeled rows are skipped.

Usage:
    python scripts/label_lj.py
    python scripts/label_lj.py --file data/lj_validation/50_questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RUBRIC = """
Rubric (matches the Groq judge):
  Label 1 if the prediction is semantically equivalent to the gold answer.
  Label 0 otherwise.
  - Empty / 'unknown' / 'I don't know' predictions on an answerable → 0
  - Minor casing / punctuation differences → 1
  - Right entity but extra/missing words that don't change meaning → 1
  - Related but factually different → 0
"""


def _atomic_write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def _render(row: dict, idx: int, total: int, labeled: int) -> None:
    bar = f"[{labeled}/{total} labeled | row {idx + 1}/{total}]"
    print()
    print("=" * 72)
    print(bar)
    print(f"sample_id : {row['sample_id']}")
    path = row.get("_path", "")
    if path:
        print(f"path      : {path}")
    print(f"question  : {row['question']}")
    print(f"gold      : {row['gold_answer']!r}")
    pred = row.get("prediction", "")
    pred_disp = repr(pred) if pred.strip() else "(empty — model emitted no valid answer)"
    print(f"prediction: {pred_disp}")
    existing = row.get("human_label")
    if existing is not None:
        print(f"current label: {existing}")
    print("-" * 72)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path,
                    default=Path("data/lj_validation/50_questions.jsonl"))
    args = ap.parse_args()

    if not args.file.exists():
        print(f"ERROR: {args.file} not found. "
              f"Run scripts/build_lj_validation.py then populate_lj_predictions.py first.",
              file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in args.file.read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(rows)

    def n_labeled() -> int:
        return sum(1 for r in rows if r.get("human_label") in (0, 1))

    print(RUBRIC)
    print(f"File: {args.file}")
    print(f"Starting with {n_labeled()}/{total} labeled.")

    i = 0
    while i < total:
        row = rows[i]
        if row.get("human_label") in (0, 1):
            i += 1
            continue
        _render(row, i, total, n_labeled())
        try:
            raw = input("label (y/n/s/b/q/?): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nquit requested — progress saved.")
            return 0

        if raw in ("y", "1"):
            row["human_label"] = 1
            _atomic_write(args.file, rows)
            i += 1
        elif raw in ("n", "0"):
            row["human_label"] = 0
            _atomic_write(args.file, rows)
            i += 1
        elif raw == "s":
            i += 1
        elif raw == "b":
            if i == 0:
                print("already at the first row.")
                continue
            i -= 1
            rows[i]["human_label"] = None
            _atomic_write(args.file, rows)
        elif raw == "q":
            print("quit — progress saved.")
            return 0
        elif raw == "?":
            print(RUBRIC)
        else:
            print("unrecognized input. use y / n / s / b / q / ?")

    print()
    print(f"DONE — {n_labeled()}/{total} labeled.")
    print(f"Next: GROQ_API_KEY=... python scripts/validate_lj.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
