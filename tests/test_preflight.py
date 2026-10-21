"""Unit tests for scripts/preflight.py.

Confirms we catch the common config mistakes that would otherwise manifest
as cryptic verl errors after launching training on the server.
"""
import importlib.util
import sys
from pathlib import Path

# Dynamically load scripts/preflight.py since scripts/ isn't a package.
_root = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "preflight_mod", _root / "scripts" / "preflight.py"
)
preflight_mod = importlib.util.module_from_spec(_spec)
sys.modules["preflight_mod"] = preflight_mod
_spec.loader.exec_module(preflight_mod)
preflight = preflight_mod.preflight


_MINIMAL_OK = """
_meta:
  reward: r0
  optimizer: grpo
  model: qwen3-0.6b
trainer:
  project_name: rl-rag
data:
  train_files: /nonexistent/train.jsonl  # OK: just surfaces a note
  val_files: /nonexistent/val.jsonl
  prompt_key: prompt
algorithm:
  name: grpo
reward_fn:
  path: src.rewards.r0_grace:r0_grace
"""


def test_valid_config_passes_import_check(tmp_path: Path):
    cfg = tmp_path / "ok.yaml"
    cfg.write_text(_MINIMAL_OK)
    problems = preflight(cfg)
    # Only the two 'file does not exist on this machine' notes expected
    assert len(problems) == 2
    assert all("does not exist" in p for p in problems)


def test_missing_meta_section_is_caught(tmp_path: Path):
    cfg = tmp_path / "no_meta.yaml"
    cfg.write_text(_MINIMAL_OK.replace("_meta:\n  reward: r0\n  optimizer: grpo\n  model: qwen3-0.6b\n", ""))
    problems = preflight(cfg)
    assert any("_meta" in p for p in problems)


def test_bad_reward_fn_import_is_caught(tmp_path: Path):
    cfg = tmp_path / "bad_reward.yaml"
    cfg.write_text(_MINIMAL_OK.replace(
        "path: src.rewards.r0_grace:r0_grace",
        "path: src.rewards.r0_grace:not_a_real_function",
    ))
    problems = preflight(cfg)
    assert any("has no attribute" in p for p in problems)


def test_wrong_module_path_is_caught(tmp_path: Path):
    cfg = tmp_path / "wrong_module.yaml"
    cfg.write_text(_MINIMAL_OK.replace(
        "path: src.rewards.r0_grace:r0_grace",
        "path: src.rewards.r9_imaginary:r9_imaginary",
    ))
    problems = preflight(cfg)
    assert any("cannot import module" in p for p in problems)


def test_unknown_optimizer_is_caught(tmp_path: Path):
    cfg = tmp_path / "bad_opt.yaml"
    cfg.write_text(_MINIMAL_OK.replace("name: grpo", "name: sgd"))
    problems = preflight(cfg)
    assert any("is unknown" in p for p in problems)


def test_dapo_requires_clip_ratios(tmp_path: Path):
    cfg = tmp_path / "dapo_missing_clip.yaml"
    cfg.write_text(_MINIMAL_OK.replace("name: grpo", "name: dapo"))
    problems = preflight(cfg)
    msgs = " ".join(problems)
    assert "clip_ratio_low" in msgs
    assert "clip_ratio_high" in msgs


def test_yaml_parse_error_is_reported(tmp_path: Path):
    cfg = tmp_path / "broken.yaml"
    cfg.write_text("this: is: invalid: yaml:\n  - [")
    problems = preflight(cfg)
    assert any("YAML parse error" in p for p in problems)


def test_all_shipped_configs_pass_locally():
    """Every YAML config checked into configs/ should at least pass preflight
    (modulo the data-file 'not on this machine' notes, which don't count)."""
    root = Path(__file__).resolve().parent.parent
    configs = sorted((root / "configs").glob("*.yaml"))
    # grid.yaml is a grid manifest, not a run config — skip it.
    configs = [c for c in configs if c.name != "grid.yaml"]
    assert configs, "no configs/*.yaml found — did the file layout change?"
    for cfg in configs:
        problems = preflight(cfg)
        hard = [p for p in problems if "does not exist on this machine" not in p]
        assert not hard, f"{cfg.name} failed preflight:\n  " + "\n  ".join(hard)
