"""Run reward-hacking audit on all completed runs' predictions.jsonl.

Reads each `results/<run>/predictions.jsonl`, joins against the eval split
referenced in `results/<run>/config.yaml` to recover `retrieved_passages` +
`question`, and writes `results/<run>/audit.json`.

Usage:
    python scripts/run_audits.py
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
from src.rewards.audit import audit_batch  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_val_file(cfg: dict) -> Path:
    # The config's data.val_files may be a str or a single-element list.
    data = cfg.get("data", {})
    v = data.get("val_files") or data.get("val_file")
    if isinstance(v, list):
        v = v[0]
    if not v:
        raise KeyError("config.yaml missing data.val_files / data.val_file")
    return Path(v)


def audit_run(run_dir: Path) -> dict | None:
    pred = run_dir / "predictions.jsonl"
    cfg_path = run_dir / "config.yaml"
    if not pred.exists() or not cfg_path.exists():
        return None

    rows = _load_jsonl(pred)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    val_file = _resolve_val_file(cfg)
    split_rows = {r["sample_id"]: r for r in _load_jsonl(val_file)}

    pairs: list[tuple[str, str]] = []
    retrieved: list[list[str]] = []
    for r in rows:
        s = split_rows.get(r["sample_id"])
        if s is None:
            # Prediction for a sample not in the split — skip rather than crash.
            continue
        pairs.append((r.get("raw_response", ""), s["question"]))
        retrieved.append(s["retrieved_passages"])

    if not pairs:
        return None

    report = audit_batch(pairs, retrieved)
    (run_dir / "audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    results = Path("results")
    if not results.exists():
        print("No results/ directory.")
        return
    audited = 0
    for run_dir in sorted(results.iterdir()):
        if not run_dir.is_dir():
            continue
        report = audit_run(run_dir)
        if report is None:
            continue
        audited += 1
        print(
            f"{run_dir.name}: path_shares={report['path_shares']}, "
            f"fab_quote={report['fabricated_quote_rate']:.2f}, "
            f"q_copy={report['question_copy_rate']:.2f}"
        )
    if audited == 0:
        print("No runs had predictions.jsonl + config.yaml yet.")


if __name__ == "__main__":
    main()
