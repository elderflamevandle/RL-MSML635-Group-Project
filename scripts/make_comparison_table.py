"""Dump a markdown + CSV comparison table of all runs in `results/`.

Usage:
    python scripts/make_comparison_table.py

Writes:
    docs/results-table.csv
    docs/results-table.md
"""
from pathlib import Path
import csv
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.aggregate import collect_metrics  # noqa: E402

rows = collect_metrics(Path("results"))
rows.sort(key=lambda r: (r["reward"], r["optimizer"]))

if not rows:
    print("No runs with metrics.json found in results/.")
    raise SystemExit(1)

cols = [
    "reward", "optimizer", "balanced_accuracy", "em_answerable",
    "abstention_unanswerable", "bert_f1_answerable", "lj_answerable",
    "path_shares",
]

out_csv = Path("docs/results-table.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        row = {c: r.get(c, "") for c in cols}
        if isinstance(row.get("path_shares"), dict):
            row["path_shares"] = ";".join(
                f"{k}={v:.2f}" for k, v in row["path_shares"].items()
            )
        w.writerow(row)

out_md = Path("docs/results-table.md")
lines = ["# Results comparison\n\n"]
lines.append("| " + " | ".join(cols) + " |\n")
lines.append("|" + "|".join(["---"] * len(cols)) + "|\n")
for r in rows:
    cells = []
    for c in cols:
        v = r.get(c, "")
        if isinstance(v, dict):
            v = "; ".join(f"{k}={vv:.2f}" for k, vv in v.items())
        elif isinstance(v, float):
            v = f"{v:.3f}"
        cells.append(str(v))
    lines.append("| " + " | ".join(cells) + " |\n")
out_md.write_text("".join(lines), encoding="utf-8")

print(f"Wrote {out_csv} and {out_md}")
