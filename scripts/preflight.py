"""Lint a verl training config locally before shipping to the server.

Catches:
  * missing required top-level sections (_meta, trainer, data, algorithm, reward_fn)
  * missing _meta keys that scripts/launch_run.sh needs for run-id naming
  * reward_fn.path that cannot be imported in this venv (typo, wrong module path)
  * data.train_files / data.val_files that don't exist on this machine
  * algorithm.name that doesn't match a known optimizer (grpo | dapo)

Usage:
    python scripts/preflight.py configs/r0_grpo.yaml
    python scripts/preflight.py configs/*.yaml       # multi-file, one line per file

Exits 0 if all configs pass; exits 1 otherwise. Does NOT verify that verl
accepts every key — that iteration happens on the server. Preflight catches
the mistakes you can catch *before* burning server time.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# Repo root on sys.path so `import src...` works when preflight runs directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

REQUIRED_TOP = ["_meta", "trainer", "data", "algorithm", "reward_fn"]
REQUIRED_META = ["reward", "optimizer", "model"]
REQUIRED_DATA = ["train_files", "val_files", "prompt_key"]
KNOWN_OPTIMIZERS = {"grpo", "dapo", "ppo"}


def _check_import(dotted: str) -> str | None:
    """Return None if ok, else an error message."""
    if ":" not in dotted:
        return f"reward_fn.path should be 'module.path:function_name', got {dotted!r}"
    mod_path, fn_name = dotted.split(":", 1)
    try:
        mod = importlib.import_module(mod_path)
    except Exception as exc:  # noqa: BLE001 — surface any import failure to user
        return f"cannot import module {mod_path!r}: {type(exc).__name__}: {exc}"
    if not hasattr(mod, fn_name):
        return f"module {mod_path!r} has no attribute {fn_name!r}"
    return None


def is_grid_manifest(cfg: dict) -> bool:
    """A grid manifest has a top-level 'runs' list but no training sections."""
    return isinstance(cfg, dict) and "runs" in cfg and not any(k in cfg for k in REQUIRED_TOP)


def preflight(cfg_path: Path) -> list[str]:
    """Return a list of problem messages. Empty list == all good.

    'File does not exist on this machine' is a note, not an error, so callers
    can still tell the two apart via the 'does not exist' substring check.
    """
    problems: list[str] = []
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"YAML parse error: {exc}"]
    if not isinstance(cfg, dict):
        return [f"top-level must be a mapping; got {type(cfg).__name__}"]

    if is_grid_manifest(cfg):
        # Grid manifests have a different schema; preflight them as a list of
        # child-config paths. Those get checked by their own invocation.
        return []

    for k in REQUIRED_TOP:
        if k not in cfg:
            problems.append(f"missing top-level section: {k!r}")

    meta = cfg.get("_meta", {}) or {}
    for k in REQUIRED_META:
        if k not in meta:
            problems.append(f"_meta missing key: {k!r}")

    data = cfg.get("data", {}) or {}
    for k in REQUIRED_DATA:
        if k not in data:
            problems.append(f"data missing key: {k!r}")

    for file_key in ("train_files", "val_files"):
        p = data.get(file_key)
        if p and not Path(p).exists():
            problems.append(
                f"data.{file_key} points at {p!r} which does not exist on this machine "
                f"(fine if intended to run on the server; "
                f"otherwise run scripts/build_retrieval.py)"
            )

    alg = cfg.get("algorithm", {}) or {}
    alg_name = alg.get("name")
    if alg_name is None:
        problems.append("algorithm.name is missing")
    elif alg_name not in KNOWN_OPTIMIZERS:
        problems.append(
            f"algorithm.name={alg_name!r} is unknown; expected one of {sorted(KNOWN_OPTIMIZERS)}"
        )
    if alg_name == "dapo":
        for req_key in ("clip_ratio_low", "clip_ratio_high"):
            if req_key not in alg:
                problems.append(
                    f"algorithm.{req_key} is required when algorithm.name='dapo'"
                )

    reward = cfg.get("reward_fn", {}) or {}
    path = reward.get("path")
    if not path:
        problems.append("reward_fn.path is missing")
    else:
        err = _check_import(path)
        if err:
            problems.append(f"reward_fn.path: {err}")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", type=Path, help="One or more YAML config paths.")
    args = ap.parse_args()

    rc = 0
    for cfg in args.configs:
        if not cfg.exists():
            print(f"{cfg}: MISSING FILE")
            rc = 1
            continue
        try:
            raw = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception:
            raw = {}
        if isinstance(raw, dict) and is_grid_manifest(raw):
            print(f"{cfg}: ok (grid manifest — skipping schema check)")
            continue
        problems = preflight(cfg)
        notes = [p for p in problems if "does not exist on this machine" in p]
        errors = [p for p in problems if p not in notes]
        if not problems:
            print(f"{cfg}: ok")
            continue
        if errors:
            rc = 1
            label = f"{len(errors)} error(s)"
            if notes:
                label += f" + {len(notes)} note(s)"
            print(f"{cfg}: {label}")
        else:
            print(f"{cfg}: ok ({len(notes)} note(s) — files not present locally)")
        for p in errors:
            print(f"  ERROR: {p}")
        for p in notes:
            print(f"  note:  {p}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
