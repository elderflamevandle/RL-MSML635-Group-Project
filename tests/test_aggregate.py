import json
from src.eval.aggregate import collect_metrics


def test_collect_metrics_reads_all_runs(tmp_path):
    for name, metrics in [
        ("r0_grpo_x_s0_20261102-120000", {
            "balanced_accuracy": 0.75,
            "em_answerable": 0.6,
            "abstention_unanswerable": 0.9,
        }),
        ("r1_grpo_x_s0_20261103-120000", {
            "balanced_accuracy": 0.78,
            "em_answerable": 0.65,
            "abstention_unanswerable": 0.91,
        }),
    ]:
        d = tmp_path / name
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps(metrics))

    rows = collect_metrics(tmp_path)
    assert len(rows) == 2
    assert {r["reward"] for r in rows} == {"r0", "r1"}
    assert {r["optimizer"] for r in rows} == {"grpo"}
    # Metrics content is merged into each row
    r0 = next(r for r in rows if r["reward"] == "r0")
    assert r0["balanced_accuracy"] == 0.75
    assert r0["run_id"] == "r0_grpo_x_s0_20261102-120000"


def test_collect_metrics_skips_dirs_without_metrics_json(tmp_path):
    (tmp_path / "r0_grpo_x_s0_ok").mkdir()
    (tmp_path / "r0_grpo_x_s0_ok" / "metrics.json").write_text(json.dumps({"balanced_accuracy": 0.5}))
    (tmp_path / "empty_dir").mkdir()
    (tmp_path / "stray_file.txt").write_text("ignore me")

    rows = collect_metrics(tmp_path)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "r0_grpo_x_s0_ok"


def test_collect_metrics_skips_dirs_with_malformed_name(tmp_path):
    # Only 2 parts, can't split into reward/optimizer/model/seed
    (tmp_path / "junk_dir").mkdir()
    (tmp_path / "junk_dir" / "metrics.json").write_text(json.dumps({"balanced_accuracy": 0.5}))

    rows = collect_metrics(tmp_path)
    assert rows == []
