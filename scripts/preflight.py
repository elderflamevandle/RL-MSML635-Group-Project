"""Lint a verl training config locally before shipping to the server.

Field names are checked against real verl @ 4aa02fe (verl/trainer/config/*.yaml
in that checkout), not guessed -- see configs/skeleton_t4.yaml for the list of
renames from the original (GRACE-paper-reference-config-shaped) draft.

Catches:
  * missing required top-level sections (_meta, trainer, data, algorithm,
    custom_reward_function)
  * missing _meta keys that scripts/launch_run.sh needs for run-id naming
  * custom_reward_function.{path,name} that cannot be imported/found in this venv
  * data.train_files / data.val_files that don't exist on this machine, or that
    are still .jsonl (verl's RLHFDataset reads parquet; see scripts/build_verl_data.py)
  * algorithm.adv_estimator that doesn't match a known optimizer (grpo | dapo | gae)

Usage:
    python scripts/preflight.py configs/r0_grpo_t4.yaml
    python scripts/preflight.py configs/*.yaml       # multi-file, one line per file

Exits 0 if all configs pass; exits 1 otherwise. Does NOT verify that verl
accepts every key — that iteration happens on the server. Preflight catches
the mistakes you can catch *before* burning server time.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# Repo root on sys.path so `import src...` works when preflight runs directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

REQUIRED_TOP = ["_meta", "trainer", "data", "algorithm", "custom_reward_function"]
REQUIRED_META = ["reward", "optimizer", "model"]
REQUIRED_DATA = ["train_files", "val_files", "prompt_key"]
KNOWN_OPTIMIZERS = {"grpo", "dapo", "gae", "reinforce_plus_plus"}


def _check_reward_fn(path: str, name: str) -> str | None:
    """Return None if ok, else an error message.

    verl's `get_custom_reward_fn` loads this as a bare file (importlib from
    a path), not as a package -- so we mirror that exactly rather than
    `import_module`, which would resolve relative to sys.path instead.
    """
    file_path = Path(path)
    if not file_path.exists():
        return f"custom_reward_function.path {path!r} does not exist"
    spec = importlib.util.spec_from_file_location("preflight_check", file_path)
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 — surface any import failure to user
        return f"cannot load {path!r}: {type(exc).__name__}: {exc}"
    if not hasattr(mod, name):
        return f"{path!r} has no function named {name!r}"
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
        if not p:
            continue
        if Path(p).suffix != ".parquet":
            problems.append(
                f"data.{file_key}={p!r} is not .parquet -- verl's RLHFDataset reads "
                f"parquet, not jsonl. Run scripts/build_verl_data.py to convert."
            )
        elif not Path(p).exists():
            problems.append(
                f"data.{file_key} points at {p!r} which does not exist on this machine "
                f"(fine if intended to run on the server; "
                f"otherwise run scripts/build_retrieval.py then scripts/build_verl_data.py)"
            )

    alg = cfg.get("algorithm", {}) or {}
    adv_estimator = alg.get("adv_estimator")
    if adv_estimator is None:
        problems.append("algorithm.adv_estimator is missing")
    elif adv_estimator not in KNOWN_OPTIMIZERS:
        problems.append(
            f"algorithm.adv_estimator={adv_estimator!r} is unknown; "
            f"expected one of {sorted(KNOWN_OPTIMIZERS)}"
        )
    if adv_estimator == "dapo":
        for req_key in ("clip_ratio_low", "clip_ratio_high"):
            if req_key not in alg:
                problems.append(
                    f"algorithm.{req_key} is required when algorithm.adv_estimator='dapo'"
                )

    reward = cfg.get("custom_reward_function", {}) or {}
    path = reward.get("path")
    name = reward.get("name")
    if not path:
        problems.append("custom_reward_function.path is missing")
    elif not name:
        problems.append("custom_reward_function.name is missing")
    else:
        err = _check_reward_fn(path, name)
        if err:
            problems.append(f"custom_reward_function: {err}")

    actor = (cfg.get("actor_rollout_ref", {}) or {}).get("actor", {}) or {}
    if "kl_coef" in (cfg.get("algorithm", {}) or {}):
        problems.append(
            "algorithm.kl_coef is not a real verl field -- GRPO's KL penalty is "
            "actor_rollout_ref.actor.use_kl_loss + .kl_loss_coef"
        )
    if (cfg.get("actor_rollout_ref", {}) or {}).get("model", {}).get("enable_lora"):
        problems.append(
            "actor_rollout_ref.model.enable_lora is not a real verl field -- "
            "LoRA is switched on by setting model.lora_rank > 0"
        )
    if actor.get("use_kl_loss") and "kl_loss_coef" not in actor:
        problems.append("actor_rollout_ref.actor.kl_loss_coef is required when use_kl_loss is true")

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
