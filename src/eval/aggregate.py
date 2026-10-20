"""Collect metrics across runs into a tidy list of dicts.

Each run directory in `results/` is expected to follow the run-id convention
from `src.training.run_naming`:
    <reward>_<optimizer>_<model>_s<seed>_<YYYYMMDD-HHMMSS>

and to contain a `metrics.json` produced by `src.eval.evaluate`. Any directory
not matching the convention or missing `metrics.json` is skipped.
"""
from pathlib import Path
import json


def collect_metrics(results_dir: Path | str) -> list[dict]:
    rows = []
    for run_dir in Path(results_dir).iterdir():
        if not run_dir.is_dir():
            continue
        mf = run_dir / "metrics.json"
        if not mf.exists():
            continue
        parts = run_dir.name.split("_")
        if len(parts) < 4:
            continue
        reward, optimizer, model, seed = parts[0], parts[1], parts[2], parts[3]
        m = json.loads(mf.read_text(encoding="utf-8"))
        rows.append({
            "run_id": run_dir.name,
            "reward": reward,
            "optimizer": optimizer,
            "model": model,
            "seed": seed,
            **m,
        })
    return rows
