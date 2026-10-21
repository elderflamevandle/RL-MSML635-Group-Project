"""Unit tests for src/eval/plots.py.

We exercise the functions with synthetic metric rows and verify:
  * a Figure is returned
  * the axis-label / title plumbing is wired
  * missing (reward, optimizer) combinations don't crash
  * save_fig writes a non-empty PNG to disk
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from src.eval.plots import (
    plot_rq1_rewards,
    plot_rq2_optimizers,
    plot_path_shares,
    save_fig,
)


FULL_GRID = [
    {"run_id": "r0_grpo_x_s0_t", "reward": "r0", "optimizer": "grpo",
     "balanced_accuracy": 0.70, "em_answerable": 0.65, "abstention_unanswerable": 0.75,
     "path_shares": {"evidence": 0.55, "llm": 0.30, "invalid": 0.15}},
    {"run_id": "r0_dapo_x_s0_t", "reward": "r0", "optimizer": "dapo",
     "balanced_accuracy": 0.74, "em_answerable": 0.70, "abstention_unanswerable": 0.78,
     "path_shares": {"evidence": 0.56, "llm": 0.34, "invalid": 0.10}},
    {"run_id": "r1_grpo_x_s0_t", "reward": "r1", "optimizer": "grpo",
     "balanced_accuracy": 0.78, "em_answerable": 0.72, "abstention_unanswerable": 0.84,
     "path_shares": {"evidence": 0.52, "llm": 0.40, "invalid": 0.08}},
    {"run_id": "r1_dapo_x_s0_t", "reward": "r1", "optimizer": "dapo",
     "balanced_accuracy": 0.80, "em_answerable": 0.75, "abstention_unanswerable": 0.85,
     "path_shares": {"evidence": 0.53, "llm": 0.41, "invalid": 0.06}},
    {"run_id": "r2_grpo_x_s0_t", "reward": "r2", "optimizer": "grpo",
     "balanced_accuracy": 0.76, "em_answerable": 0.60, "abstention_unanswerable": 0.92,
     "path_shares": {"evidence": 0.40, "llm": 0.55, "invalid": 0.05}},
    {"run_id": "r2_dapo_x_s0_t", "reward": "r2", "optimizer": "dapo",
     "balanced_accuracy": 0.77, "em_answerable": 0.62, "abstention_unanswerable": 0.92,
     "path_shares": {"evidence": 0.41, "llm": 0.55, "invalid": 0.04}},
]


def test_plot_rq1_returns_figure():
    fig = plot_rq1_rewards(FULL_GRID)
    assert isinstance(fig, Figure)
    assert "RQ1" in fig.axes[0].get_title()
    # Both optimizer legend entries + GRACE reference line = at least 3 handles
    handles, labels = fig.axes[0].get_legend_handles_labels()
    assert "GRPO" in labels
    assert "DAPO" in labels


def test_plot_rq2_returns_figure():
    fig = plot_rq2_optimizers(FULL_GRID)
    assert isinstance(fig, Figure)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert "GRPO" in labels
    assert "DAPO" in labels


def test_plot_path_shares_returns_figure():
    fig = plot_path_shares(FULL_GRID)
    assert isinstance(fig, Figure)
    _, labels = fig.axes[0].get_legend_handles_labels()
    assert set(labels) == {"evidence", "llm", "invalid"}
    # 6 (reward, optimizer) tuples → 6 x-ticks
    assert len(fig.axes[0].get_xticklabels()) == 6


def test_plot_rq1_handles_partial_grid():
    """Only R0+GRPO exists — the other 5 cells should render as 'no run' bars without crashing."""
    partial = [FULL_GRID[0]]
    fig = plot_rq1_rewards(partial)
    assert isinstance(fig, Figure)


def test_plot_rq2_metric_switch():
    """Confirm an alternate metric key is honored."""
    fig = plot_rq2_optimizers(FULL_GRID, metric="em_answerable")
    assert "em_answerable" in fig.axes[0].get_title()


def test_save_fig_writes_png(tmp_path: Path):
    fig = plot_rq1_rewards(FULL_GRID)
    out = tmp_path / "sub" / "fig.png"
    saved = save_fig(fig, out)
    assert saved == out
    assert out.exists()
    assert out.stat().st_size > 1000  # a real PNG is at least a few KB


def test_plot_handles_missing_path_shares():
    """path_shares missing entirely in a row should default to zeros without crashing."""
    rows = [
        {"run_id": "r0_grpo_x_s0_t", "reward": "r0", "optimizer": "grpo",
         "balanced_accuracy": 0.5},
    ]
    fig = plot_path_shares(rows)
    assert isinstance(fig, Figure)
