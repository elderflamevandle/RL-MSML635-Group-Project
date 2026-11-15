"""Fill the LJ-validation template's `prediction` slot from a real predictions.jsonl.

Joins by sample_id and copies `answer` → `prediction` for any row that still
holds the "<FILL FROM REAL RUN" placeholder (so re-runs are idempotent and
won't clobber hand-edited predictions).

Usage:
    python scripts/populate_lj_predictions.py \\
        results/local_sanity/20261020-194948/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PLACEHOLDER_MARKER = "<FILL FROM REAL RUN"


def _atomic_write(path: Path, lines: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path,
                    help="predictions.jsonl from a run (needs sample_id + answer)")
    ap.add_argument("--template", type=Path,
                    default=Path("data/lj_validation/50_questions.jsonl"))
    args = ap.parse_args()

    if not args.template.exists():
        print(f"ERROR: {args.template} not found. "
              f"Run scripts/build_lj_validation.py first.", file=sys.stderr)
        return 2
    if not args.predictions.exists():
        print(f"ERROR: {args.predictions} not found.", file=sys.stderr)
        return 3

    pred_by_id = {
        p["sample_id"]: p
        for p in (json.loads(l) for l in args.predictions.read_text(encoding="utf-8").splitlines() if l.strip())
    }

    rows = [json.loads(l) for l in args.template.read_text(encoding="utf-8").splitlines() if l.strip()]
    filled, skipped_already, skipped_missing = 0, 0, 0
    for r in rows:
        if not str(r.get("prediction", "")).startswith(PLACEHOLDER_MARKER):
            skipped_already += 1
            continue
        p = pred_by_id.get(r["sample_id"])
        if p is None:
            skipped_missing += 1
            continue
        r["prediction"] = p.get("answer", "")
        r["_path"] = p.get("path", "")
        filled += 1

    _atomic_write(args.template, [json.dumps(r) for r in rows])
    print(f"filled {filled}  already-set {skipped_already}  missing-in-predictions {skipped_missing}")
    print(f"wrote {args.template}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
