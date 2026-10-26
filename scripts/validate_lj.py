"""Validate the Groq LLM-as-Judge against hand-labeled rows.

Reads a .jsonl file (default: data/lj_validation/50_questions.jsonl) where
each row has `question`, `gold_answer`, `prediction`, `human_label` (0 or 1).
Calls GroqJudge on each row, compares to the human label, and reports
agreement + per-row disagreements.

Gate from the plan: agreement >= 0.85 before using LJ in reported metrics.

Usage:
    GROQ_API_KEY=sk-... python scripts/validate_lj.py
    GROQ_API_KEY=sk-... python scripts/validate_lj.py --file data/lj_validation/50_questions.jsonl --out results/lj_validation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.judge import GroqJudge

AGREEMENT_GATE = 0.85


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path,
                    default=Path("data/lj_validation/50_questions.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/lj_validation"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache/judge"))
    args = ap.parse_args()

    if not args.file.exists():
        print(f"ERROR: {args.file} not found. Run scripts/build_lj_validation.py first.",
              file=sys.stderr)
        return 2
    if "GROQ_API_KEY" not in os.environ:
        print("ERROR: GROQ_API_KEY not set in environment.", file=sys.stderr)
        return 3

    rows = [json.loads(l) for l in args.file.read_text(encoding="utf-8").splitlines() if l.strip()]
    labelable = [r for r in rows if r.get("human_label") in (0, 1)]
    skipped = len(rows) - len(labelable)
    if not labelable:
        print(f"ERROR: no rows have human_label set in {args.file}", file=sys.stderr)
        print("Open the file and fill `prediction` + `human_label` on each row.", file=sys.stderr)
        return 4

    args.out.mkdir(parents=True, exist_ok=True)
    judge = GroqJudge(cache_dir=args.cache_dir)

    agree = 0
    disagreements = []
    scored_rows = []
    for r in labelable:
        try:
            lj = judge.score(r["question"], r["prediction"], r["gold_answer"])
        except Exception as exc:  # noqa: BLE001 — surface anything upstream
            print(f"judge error on {r['sample_id']}: {exc}", file=sys.stderr)
            continue
        human = int(r["human_label"])
        row_out = {**r, "lj_score": lj}
        scored_rows.append(row_out)
        if int(lj) == human:
            agree += 1
        else:
            disagreements.append({
                "sample_id": r["sample_id"],
                "question": r["question"],
                "prediction": r["prediction"],
                "gold_answer": r["gold_answer"],
                "human_label": human,
                "lj_score": lj,
            })

    n = len(scored_rows)
    rate = agree / n if n else 0.0
    passed = rate >= AGREEMENT_GATE

    (args.out / "lj_vs_human.jsonl").write_text(
        "\n".join(json.dumps(r) for r in scored_rows), encoding="utf-8"
    )
    (args.out / "disagreements.jsonl").write_text(
        "\n".join(json.dumps(r) for r in disagreements), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(json.dumps({
        "n": n,
        "n_rows_in_file": len(rows),
        "n_rows_unlabeled_skipped": skipped,
        "agree": agree,
        "disagree": len(disagreements),
        "agreement_rate": rate,
        "gate": AGREEMENT_GATE,
        "passed": passed,
    }, indent=2), encoding="utf-8")

    print(f"n={n}  agree={agree}  disagree={len(disagreements)}  agreement_rate={rate:.3f}")
    print(f"gate={AGREEMENT_GATE}  passed={passed}")
    if skipped:
        print(f"(skipped {skipped} rows with no human_label)")
    print(f"wrote {args.out}/summary.json + lj_vs_human.jsonl + disagreements.jsonl")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
