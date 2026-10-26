"""Unit tests for the LJ validation builder + validator scripts.

Both scripts live in scripts/, so we load them dynamically.
"""
import importlib.util
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, _root / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_lj_validation_schema(tmp_path, monkeypatch):
    """Builder produces 50 rows with the expected keys + null human_label."""
    build_mod = _load("build_lj_validation")
    out = tmp_path / "lj.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "build_lj_validation.py",
        "--tiny-file", "data/raw/hotpot_tiny.jsonl",
        "--out", str(out),
        "--n", "10",
        "--seed", "0",
    ])
    rc = build_mod.main()
    assert rc == 0
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 10
    for r in rows:
        assert set(r.keys()) >= {"sample_id", "question", "gold_answer",
                                  "prediction", "human_label", "notes"}
        assert r["human_label"] is None
        # Builder seeds a placeholder prediction that labelers must overwrite
        assert "FILL FROM REAL RUN" in r["prediction"]


def test_build_lj_validation_is_deterministic(tmp_path, monkeypatch):
    """Same seed → same sample IDs. Guards against seed drift."""
    build_mod = _load("build_lj_validation")

    def _build(path: Path, seed: int):
        monkeypatch.setattr(sys, "argv", [
            "build_lj_validation.py",
            "--tiny-file", "data/raw/hotpot_tiny.jsonl",
            "--out", str(path),
            "--n", "10",
            "--seed", str(seed),
        ])
        build_mod.main()
        return [json.loads(l)["sample_id"]
                for l in path.read_text(encoding="utf-8").splitlines()]

    a = _build(tmp_path / "a.jsonl", seed=0)
    b = _build(tmp_path / "b.jsonl", seed=0)
    c = _build(tmp_path / "c.jsonl", seed=1)
    assert a == b
    assert a != c
