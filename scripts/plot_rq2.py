"""RQ2 — Optimizer comparison bar chart.

For each reward (R0 / R1 / R2), overlays GRPO vs DAPO on the chosen metric.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.aggregate import collect_metrics
from src.eval.plots import plot_rq2_optimizers, save_fig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--metric", default="balanced_accuracy")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/rq2_optimizers.png"))
    args = ap.parse_args()

    rows = collect_metrics(args.results_dir)
    if not rows:
        print(f"No metrics.json files found under {args.results_dir}", file=sys.stderr)
        return 2
    fig = plot_rq2_optimizers(rows, metric=args.metric)
    out = save_fig(fig, args.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
