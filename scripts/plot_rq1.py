"""RQ1 — Reward comparison bar chart.

Reads all metrics.json files under --results-dir, groups by (reward, optimizer),
and emits a grouped bar chart over R0 / R1 / R2 with GRPO and DAPO side-by-side.

Usage:
    python scripts/plot_rq1.py                                   # defaults
    python scripts/plot_rq1.py --results-dir results/            # explicit
    python scripts/plot_rq1.py --metric em_answerable            # not balanced_accuracy
    python scripts/plot_rq1.py --out docs/figures/rq1_em.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.aggregate import collect_metrics
from src.eval.plots import plot_rq1_rewards, save_fig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--metric", default="balanced_accuracy",
                    help="Metric key in metrics.json (e.g. balanced_accuracy, em_answerable, abstention_unanswerable).")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/rq1_rewards.png"))
    args = ap.parse_args()

    rows = collect_metrics(args.results_dir)
    if not rows:
        print(f"No metrics.json files found under {args.results_dir}", file=sys.stderr)
        return 2
    fig = plot_rq1_rewards(rows, metric=args.metric)
    out = save_fig(fig, args.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
