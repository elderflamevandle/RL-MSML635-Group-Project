"""Score a predictions.jsonl file against R0, R1, and R2 reward functions.

Each prediction row is joined to its original Sample by sample_id so the
reward functions see the real gold_passages / gold_evidence / gold_answer.
Produces:

    <out-dir>/rewards.json          # per-reward summary (mean, std, hist bins)
    <out-dir>/rewards_per_sample.jsonl  # one row per sample with all three scores
    <out-dir>/reward_hist.png       # overlaid histograms

Usage:
    python scripts/score_predictions.py \\
        results/local_sanity/20261020-194948/predictions.jsonl \\
        data/raw/hotpot_tiny.jsonl \\
        --out results/local_sanity/20261020-194948/reward_analysis

Works on any predictions.jsonl + source split that both carry sample_id.
The split loader (`load_hotpot_tiny`) accepts raw-hotpot or balanced-split
files, so this tool is useful on RL run outputs too.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.hotpot import load_hotpot_tiny
from src.rewards.r0_grace import r0_grace
from src.rewards.r1_cost_sensitive import r1_cost_sensitive
from src.rewards.r2_hard_gated import r2_hard_gated


REWARDS = {
    "r0": r0_grace,
    "r1": r1_cost_sensitive,
    "r2": r2_hard_gated,
}


def _summary(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(scores),
        "mean": statistics.mean(scores),
        "std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
    }


def score_predictions(
    predictions_path: Path, split_path: Path, out_dir: Path
) -> dict:
    """Core logic. Returns the summary dict; also writes files to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = {s.sample_id: s for s in load_hotpot_tiny(split_path)}
    predictions = [
        json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    per_sample: list[dict] = []
    all_scores: dict[str, list[float]] = {k: [] for k in REWARDS}

    missed = 0
    for pred in predictions:
        s = samples.get(pred["sample_id"])
        if s is None:
            missed += 1
            continue
        resp = pred["raw_response"]
        row: dict = {
            "sample_id": s.sample_id,
            "is_answerable": s.is_answerable,
            "path": pred.get("path"),
        }
        for name, fn in REWARDS.items():
            score, parts = fn(resp, s)
            all_scores[name].append(score)
            row[name] = score
            row[f"{name}_parts"] = parts
        per_sample.append(row)

    summary = {
        "n_predictions": len(predictions),
        "n_scored": len(per_sample),
        "n_missing_sample": missed,
        "rewards": {name: _summary(all_scores[name]) for name in REWARDS},
        # Discrimination = range of scores. Higher = reward distinguishes
        # outputs better; a reward that returns the same score for every
        # output is useless for policy gradients.
        "discrimination": {
            name: max(all_scores[name]) - min(all_scores[name])
            if all_scores[name] else 0.0
            for name in REWARDS
        },
    }

    # Breakdown by (is_answerable, path) — the richest view
    def _slice_mean(filt) -> dict:
        buckets: dict[tuple, dict[str, list[float]]] = {}
        for r in per_sample:
            key = (r["is_answerable"], r["path"])
            if not filt(r):
                continue
            buckets.setdefault(key, {k: [] for k in REWARDS})
            for name in REWARDS:
                buckets[key][name].append(r[name])
        return {
            f"ans={ans}, path={path}": {
                name: {"n": len(v), "mean": statistics.mean(v) if v else 0.0}
                for name, v in reward_lists.items()
            }
            for (ans, path), reward_lists in buckets.items()
        }

    summary["by_bucket"] = _slice_mean(lambda r: True)

    (out_dir / "rewards.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "rewards_per_sample.jsonl").write_text(
        "\n".join(json.dumps(r) for r in per_sample), encoding="utf-8"
    )

    # Overlaid histogram
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = {"r0": "#4C72B0", "r1": "#DD8452", "r2": "#55A868"}
    for name in REWARDS:
        if all_scores[name]:
            ax.hist(all_scores[name], bins=15, alpha=0.55,
                    label=f"{name.upper()} (μ={statistics.mean(all_scores[name]):.2f})",
                    color=colors[name])
    ax.set_xlabel("reward score")
    ax.set_ylabel("count")
    ax.set_title(f"Reward distribution over {len(per_sample)} predictions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reward_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path,
                    help="Path to a predictions.jsonl produced by src.eval.evaluate or scripts/local_sanity.py.")
    ap.add_argument("split", type=Path,
                    help="Path to the source split .jsonl (e.g. data/raw/hotpot_tiny.jsonl) that carries sample_id → gold fields.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for rewards.json + rewards_per_sample.jsonl + reward_hist.png.")
    args = ap.parse_args()

    summary = score_predictions(args.predictions, args.split, args.out)
    # Short console summary — easy to paste into a log file.
    print(f"predictions: {summary['n_predictions']}  scored: {summary['n_scored']}  missing_sample: {summary['n_missing_sample']}")
    for name in REWARDS:
        s = summary["rewards"][name]
        print(f"  {name.upper()}: mean={s['mean']:+.3f}  std={s['std']:.3f}  range=[{s['min']:+.2f}, {s['max']:+.2f}]")
    print("by_bucket:")
    for bucket, scores in summary["by_bucket"].items():
        line = ", ".join(f"{n}={scores[n]['mean']:+.3f}" for n in REWARDS)
        print(f"  {bucket}  (n={next(iter(scores.values()))['n']})  {line}")
    print(f"wrote {args.out}/rewards.json + rewards_per_sample.jsonl + reward_hist.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
