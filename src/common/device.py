"""Device selection with a laptop <-> server-friendly switch.

Usage:
    from src.common.device import resolve_device
    dev = resolve_device("auto")      # cuda if available, else cpu
    dev = resolve_device("cuda")      # cuda; raises if unavailable
    dev = resolve_device("cpu")       # cpu, always
    dev = resolve_device(None)        # same as "auto"

Return value is a plain string ("cuda" or "cpu") so callers that live
outside the training/eval stack (unit tests, scripts) do not need to import
torch just to talk about devices.

The actual torch import is lazy — this module is safe to import on a laptop
that only has `torch==*+cpu`, or even no torch at all (it will just always
resolve to "cpu" with a warning).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger(__name__)

DevicePreference = Literal["auto", "cpu", "cuda"] | None
ResolvedDevice = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class DeviceInfo:
    """Snapshot of what's available on this machine."""
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    cuda_total_memory_gb: float | None
    torch_version: str | None
    torch_cuda_version: str | None


def probe() -> DeviceInfo:
    """Best-effort probe of torch + CUDA state. Never raises."""
    try:
        import torch  # type: ignore
    except ImportError:
        return DeviceInfo(False, 0, None, None, None, None)

    cuda_ok = torch.cuda.is_available()
    if not cuda_ok:
        return DeviceInfo(False, 0, None, None, torch.__version__, None)

    name = torch.cuda.get_device_name(0)
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    return DeviceInfo(
        cuda_available=True,
        cuda_device_count=torch.cuda.device_count(),
        cuda_device_name=name,
        cuda_total_memory_gb=total_bytes / (1024 ** 3),
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
    )


def resolve_device(
    preference: DevicePreference = "auto",
    *,
    info: DeviceInfo | None = None,
) -> ResolvedDevice:
    """Resolve a user preference to a concrete device string.

    - "auto" or None → cuda if available, else cpu
    - "cuda"         → cuda; RuntimeError if unavailable
    - "cpu"          → cpu
    """
    pref = (preference or "auto").lower()
    if pref not in ("auto", "cpu", "cuda"):
        raise ValueError(f"preference must be auto|cpu|cuda, got {preference!r}")

    env_override = os.environ.get("RL_RAG_DEVICE")
    if env_override:
        pref = env_override.lower()
        if pref not in ("auto", "cpu", "cuda"):
            raise ValueError(
                f"RL_RAG_DEVICE must be auto|cpu|cuda, got {env_override!r}"
            )

    if pref == "cpu":
        return "cpu"

    info = info or probe()
    if pref == "cuda":
        if not info.cuda_available:
            raise RuntimeError(
                "preference='cuda' but torch.cuda.is_available() is False. "
                "Install CUDA-enabled torch (see scripts/install_cuda_torch.sh)."
            )
        return "cuda"

    # auto
    if info.cuda_available:
        return "cuda"
    log.info("CUDA unavailable; falling back to CPU.")
    return "cpu"


def summary(info: DeviceInfo | None = None) -> str:
    """One-liner describing the current device state (for logs/CLIs)."""
    info = info or probe()
    if info.torch_version is None:
        return "torch not installed; device = cpu"
    if not info.cuda_available:
        return f"torch={info.torch_version} (cpu build); device = cpu"
    mem = f"{info.cuda_total_memory_gb:.1f}GB" if info.cuda_total_memory_gb else "?"
    return (
        f"torch={info.torch_version} cuda={info.torch_cuda_version} "
        f"device={info.cuda_device_name} ({mem} VRAM)"
    )
