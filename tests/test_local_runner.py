"""Tests for the local HF runner — mock the heavy transformers/torch imports.

Running the real model is expensive; we only verify the control flow:
device selection, lazy loading, batch iteration, prompt-stripping.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---- lightweight torch stand-in ----------------------------------------

class _FakeTensor:
    def __init__(self, data):
        self.data = data
        self.shape = (len(data), max(len(r) for r in data) if data else 0)

    def to(self, *_a, **_k):
        return self

    def __getitem__(self, idx):
        return _FakeTensor([self.data[idx]])


class _FakeTorchNoGrad:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _FakeTorch:
    float16 = "float16-sentinel"
    float32 = "float32-sentinel"
    bfloat16 = "bfloat16-sentinel"

    @staticmethod
    def no_grad():
        return _FakeTorchNoGrad()

    class cuda:
        @staticmethod
        def is_available():
            return False


# ---- fake transformers --------------------------------------------------

def _fake_tokenizer_output(prompts):
    """Return something with .shape[1] and .to(device) semantics."""
    lens = [len(p.split()) for p in prompts]
    ids = _FakeTensor([list(range(L)) for L in lens])
    out = MagicMock()
    out.__getitem__.side_effect = {"input_ids": ids}.__getitem__
    out.to.return_value = out
    out.__iter__ = lambda self: iter([("input_ids", ids)])
    return out, ids


def _make_fake_tokenizer():
    t = MagicMock()
    t.pad_token_id = 0
    t.pad_token = "<pad>"
    t.eos_token = "<eos>"

    def _call(prompts, **kw):
        enc, ids = _fake_tokenizer_output(prompts)

        class _Enc(dict):
            def to(self, *_a, **_k):
                return self

        d = _Enc(input_ids=ids)
        return d

    t.side_effect = _call
    t.decode.side_effect = lambda row, **kw: f"GEN<{len(row.data[0])}>"
    return t


def _make_fake_model():
    m = MagicMock()
    m.eval.return_value = None
    m.to.return_value = m

    def _generate(**kw):
        # Return something with the right shape — 1 row per input, prompt_len + 5 tokens.
        in_ids = kw["input_ids"]
        prompt_len = in_ids.shape[1]
        rows = [list(range(prompt_len)) + [99, 98, 97, 96, 95] for _ in in_ids.data]
        return _FakeTensor(rows)

    m.generate.side_effect = _generate
    return m


# ---- actual tests -------------------------------------------------------

def test_local_runner_defers_torch_import():
    """Importing the module must not require torch to be importable."""
    import importlib
    import src.eval.local_runner as mod
    importlib.reload(mod)
    assert hasattr(mod, "LocalRunner")


def test_resolve_device_called_at_construction():
    from src.eval import local_runner as lr
    r = lr.LocalRunner("fake-model", device="cpu")
    assert r.device == "cpu"


def test_generate_strips_prompt_and_decodes_each_row():
    from src.eval import local_runner as lr

    fake_torch = _FakeTorch()
    fake_tok = _make_fake_tokenizer()
    fake_model = _make_fake_model()

    class _AutoTok:
        @staticmethod
        def from_pretrained(path, **kw):
            return fake_tok

    class _AutoModel:
        @staticmethod
        def from_pretrained(path, **kw):
            return fake_model

    with patch.dict(sys.modules, {
        "torch": fake_torch,
        "transformers": SimpleNamespace(
            AutoModelForCausalLM=_AutoModel,
            AutoTokenizer=_AutoTok,
        ),
    }):
        r = lr.LocalRunner("fake-model", device="cpu")
        out = r.generate(
            ["one two three", "a b"],
            cfg=lr.GenConfig(max_new_tokens=5, batch_size=1),
        )

    assert len(out) == 2
    # Fake decode returns "GEN<N>" where N = number of generated tokens kept.
    assert out[0].startswith("GEN<")


def test_generate_progress_callback_fires():
    from src.eval import local_runner as lr

    seen = []

    class _AutoTok:
        @staticmethod
        def from_pretrained(path, **kw):
            return _make_fake_tokenizer()

    class _AutoModel:
        @staticmethod
        def from_pretrained(path, **kw):
            return _make_fake_model()

    with patch.dict(sys.modules, {
        "torch": _FakeTorch(),
        "transformers": SimpleNamespace(
            AutoModelForCausalLM=_AutoModel,
            AutoTokenizer=_AutoTok,
        ),
    }):
        r = lr.LocalRunner("fake-model", device="cpu")
        r.generate(
            ["p1", "p2", "p3", "p4"],
            cfg=lr.GenConfig(batch_size=2),
            progress=lambda d, t: seen.append((d, t)),
        )

    assert seen == [(2, 4), (4, 4)]


def test_gen_config_defaults_are_deterministic():
    from src.eval.local_runner import GenConfig
    cfg = GenConfig()
    assert cfg.temperature == 0.0  # greedy
    assert cfg.max_new_tokens == 512
    assert cfg.batch_size == 1
