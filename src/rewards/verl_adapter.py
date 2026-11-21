"""Bridge between our reward functions and verl's `custom_reward_function` hook.

verl calls   compute_score(data_source, solution_str, ground_truth, extra_info)
we implement r(response: str, sample: Sample) -> (float, parts)

`scripts/build_verl_data.py` stores the whole Sample as a JSON string in
`extra_info["sample_json"]`, so the reward sees exactly the same Sample the
eval harness does. Point a config at this file with:

    custom_reward_function:
      path: src/rewards/verl_adapter.py
      name: r0_score        # or r1_score / r2_score

verl loads this file by path (not as a package), so make the repo root
importable before touching `src.*`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.common.types import Sample  # noqa: E402
from src.rewards.r0_grace import r0_grace  # noqa: E402
from src.rewards.r1_cost_sensitive import r1_cost_sensitive  # noqa: E402
from src.rewards.r2_hard_gated import r2_hard_gated  # noqa: E402

REWARDS = {"r0": r0_grace, "r1": r1_cost_sensitive, "r2": r2_hard_gated}


def sample_from_extra_info(extra_info) -> Sample:
    if extra_info is None or "sample_json" not in extra_info:
        raise KeyError(
            "extra_info['sample_json'] missing -- rebuild the parquet files with "
            "scripts/build_verl_data.py"
        )
    return Sample(**json.loads(extra_info["sample_json"]))


def _numeric_parts(parts: dict) -> dict:
    """verl averages every extra key, so only numbers may go back."""
    out = {}
    for k, v in parts.items():
        if isinstance(v, bool) or v is None:
            out[k] = float(bool(v))
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    path = parts.get("path")
    out["path_evidence"] = float(path == "evidence")
    out["path_llm"] = float(path == "llm")
    out["path_invalid"] = float(path == "invalid")
    return out


def _score(name: str, solution_str: str, extra_info) -> dict:
    reward, parts = REWARDS[name](solution_str, sample_from_extra_info(extra_info))
    return {"score": float(reward), **_numeric_parts(parts)}


def r0_score(data_source, solution_str, ground_truth, extra_info=None, **_):
    return _score("r0", solution_str, extra_info)


def r1_score(data_source, solution_str, ground_truth, extra_info=None, **_):
    return _score("r1", solution_str, extra_info)


def r2_score(data_source, solution_str, ground_truth, extra_info=None, **_):
    return _score("r2", solution_str, extra_info)
