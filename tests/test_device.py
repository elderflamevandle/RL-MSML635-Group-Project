"""Tests for the device-selection switch.

We don't want tests to depend on whether CUDA is really present on the box
running them (laptop vs CI vs server), so we drive `resolve_device` with
synthetic `DeviceInfo` objects.
"""
from __future__ import annotations

import os
import pytest

from src.common.device import DeviceInfo, resolve_device, summary


def _info(cuda: bool) -> DeviceInfo:
    if cuda:
        return DeviceInfo(
            cuda_available=True,
            cuda_device_count=1,
            cuda_device_name="FakeGPU",
            cuda_total_memory_gb=6.0,
            torch_version="2.11.0",
            torch_cuda_version="12.1",
        )
    return DeviceInfo(False, 0, None, None, "2.11.0+cpu", None)


def test_auto_picks_cuda_when_available():
    assert resolve_device("auto", info=_info(True)) == "cuda"


def test_auto_falls_back_to_cpu_when_no_cuda():
    assert resolve_device("auto", info=_info(False)) == "cpu"


def test_none_is_same_as_auto():
    assert resolve_device(None, info=_info(True)) == "cuda"
    assert resolve_device(None, info=_info(False)) == "cpu"


def test_cpu_always_returns_cpu():
    assert resolve_device("cpu", info=_info(True)) == "cpu"
    assert resolve_device("cpu", info=_info(False)) == "cpu"


def test_cuda_raises_when_unavailable():
    with pytest.raises(RuntimeError, match="cuda"):
        resolve_device("cuda", info=_info(False))


def test_cuda_returns_cuda_when_available():
    assert resolve_device("cuda", info=_info(True)) == "cuda"


def test_env_var_overrides_preference(monkeypatch):
    monkeypatch.setenv("RL_RAG_DEVICE", "cpu")
    # Even if caller asks for cuda, env forces cpu.
    assert resolve_device("cuda", info=_info(True)) == "cpu"


def test_env_var_cuda_still_honored(monkeypatch):
    monkeypatch.setenv("RL_RAG_DEVICE", "cuda")
    assert resolve_device("cpu", info=_info(True)) == "cuda"


def test_invalid_preference_raises():
    with pytest.raises(ValueError):
        resolve_device("gpu", info=_info(True))


def test_summary_shapes_without_torch():
    info = DeviceInfo(False, 0, None, None, None, None)
    assert "cpu" in summary(info).lower()
    assert "torch" in summary(info).lower()


def test_summary_shapes_with_cuda():
    s = summary(_info(True))
    assert "FakeGPU" in s
    assert "6.0GB" in s


def test_summary_shapes_cpu_build():
    s = summary(_info(False))
    assert "cpu" in s
    assert "2.11.0+cpu" in s
