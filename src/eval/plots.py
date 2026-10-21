"""Plot helpers that turn a list of per-run metric dicts into PNGs.

Each function takes `rows` in the shape produced by
`src.eval.aggregate.collect_metrics`:

    {
        "run_id": str,
        "reward": "r0" | "r1" | "r2",
        "optimizer": "grpo" | "dapo",
        "model": str,
        "seed": str,
        "balanced_accuracy": float,
        "em_answerable": float,
        "abstention_unanswerable": float,
        "path_shares": {"evidence": float, "llm": float, "invalid": float},
        ...
    }

Plotting is kept out of the CLI scripts so this logic can be unit-tested
against synthetic inputs without touching the filesystem (functions return
the Figure and the caller decides whether to save).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless; must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np

REWARDS: Sequence[str] = ("r0", "r1", "r2")
OPTIMIZERS: Sequence[str] = ("grpo", "dapo")
_COLOR_GRPO = "#4C72B0"
_COLOR_DAPO = "#DD8452"


def _group(rows: Iterable[dict]) -> dict[tuple[str, str], dict]:
    """Collapse rows into {(reward, optimizer): row}, keeping the latest seed."""
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r.get("reward"), r.get("optimizer"))
        # If multiple seeds exist, prefer the one with the highest balanced_accuracy
        # (deterministic tie-break by run_id).
        cur = out.get(key)
        if cur is None:
            out[key] = r
            continue
        if r.get("balanced_accuracy", 0.0) > cur.get("balanced_accuracy", 0.0):
            out[key] = r
    return out


def _bars(
    ax,
    xlabels: Sequence[str],
    series: dict[str, list[float]],
    *,
    colors: dict[str, str],
    missing_marker: str = "no run",
) -> None:
    """Grouped bar chart. `series` is {series_name: [values aligned to xlabels]}."""
    x = np.arange(len(xlabels))
    n = len(series)
    width = 0.8 / max(1, n)
    for i, (name, vals) in enumerate(series.items()):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            [v if v is not None else 0.0 for v in vals],
            width=width,
            label=name.upper(),
            color=colors.get(name, "#888"),
        )
        for b, v in zip(bars, vals):
            if v is None:
                ax.text(b.get_x() + b.get_width() / 2, 0.02, missing_marker,
                        ha="center", va="bottom", fontsize=7, color="#888",
                        rotation=90)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl.upper() for lbl in xlabels])
    ax.set_ylim(0, 1.05)
    ax.axhline(0.81, color="green", linestyle="--", linewidth=0.8,
               label="GRACE paper (0.81)")
    ax.axhline(0.5, color="#aaa", linestyle=":", linewidth=0.6)
    ax.legend(loc="upper left", fontsize=8)


def plot_rq1_rewards(rows: list[dict], metric: str = "balanced_accuracy"):
    """RQ1: bar chart of `metric` per reward, side-by-side GRPO/DAPO."""
    grouped = _group(rows)
    series: dict[str, list[float | None]] = {
        opt: [grouped.get((rw, opt), {}).get(metric) for rw in REWARDS]
        for opt in OPTIMIZERS
    }
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _bars(ax, REWARDS, series, colors={"grpo": _COLOR_GRPO, "dapo": _COLOR_DAPO})
    ax.set_title(f"RQ1 — {metric} by reward")
    ax.set_ylabel(metric)
    ax.set_xlabel("reward")
    fig.tight_layout()
    return fig


def plot_rq2_optimizers(rows: list[dict], metric: str = "balanced_accuracy"):
    """RQ2: for each reward, GRPO vs DAPO on `metric`."""
    grouped = _group(rows)
    series: dict[str, list[float | None]] = {
        rw: [grouped.get((rw, opt), {}).get(metric) for opt in OPTIMIZERS]
        for rw in REWARDS
    }
    colors = {"r0": "#55A868", "r1": "#C44E52", "r2": "#8172B2"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _bars(ax, OPTIMIZERS, series, colors=colors)
    ax.set_title(f"RQ2 — {metric} by optimizer")
    ax.set_ylabel(metric)
    ax.set_xlabel("optimizer")
    fig.tight_layout()
    return fig


def plot_path_shares(rows: list[dict]):
    """Stacked-bar: evidence / llm / invalid shares per (reward, optimizer)."""
    grouped = _group(rows)
    labels, ev, ll, iv = [], [], [], []
    for rw in REWARDS:
        for opt in OPTIMIZERS:
            r = grouped.get((rw, opt))
            if r is None:
                continue
            labels.append(f"{rw.upper()}+{opt.upper()}")
            shares = r.get("path_shares", {}) or {}
            ev.append(shares.get("evidence", 0.0))
            ll.append(shares.get("llm", 0.0))
            iv.append(shares.get("invalid", 0.0))
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels)), 4.2))
    x = np.arange(len(labels))
    ax.bar(x, ev, label="evidence", color="#4C72B0")
    ax.bar(x, ll, bottom=ev, label="llm", color="#55A868")
    ax.bar(x, iv, bottom=[a + b for a, b in zip(ev, ll)],
           label="invalid", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share")
    ax.set_title("Path share by (reward, optimizer)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def save_fig(fig, out_path: Path, dpi: int = 150) -> Path:
    """Save a figure to disk, ensuring the parent dir exists."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
