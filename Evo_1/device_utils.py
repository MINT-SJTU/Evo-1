"""Device-agnostic helpers for the XPU / CUDA / CPU backends.

Every place that used to ask ``torch.cuda`` directly goes through this module, so
inference, serving and training share one code path across accelerators.

Two rules this module encodes:

* A missing CUDA build must never be a crash. ``torch.cuda.memory_allocated()``
  and friends raise on a wheel built without CUDA, which is what an XPU or
  CPU-only install has. Callers use the helpers below, which are no-ops (or
  return zero) when the requested backend is unavailable.
* Detection order is XPU, then CUDA, then CPU: a host can expose an Intel GPU
  next to an NVIDIA one. An explicit ``device=`` argument always wins over
  detection and is never swapped for another backend.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import torch

logger = logging.getLogger(__name__)

DeviceLike = Union[str, torch.device, None]
_ACCELERATORS = ("xpu", "cuda")
_WARNED: set[str] = set()


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def available_device_types() -> list[str]:
    """Accelerator types present on this host, in detection order, ending with ``cpu``."""
    return [name for name in _ACCELERATORS if is_available(name)] + ["cpu"]


def default_device_type() -> str:
    """Best available backend: ``xpu`` > ``cuda`` > ``cpu``."""
    return available_device_types()[0]


def resolve_device(device: DeviceLike = None) -> torch.device:
    """Resolve ``None`` / str / :class:`torch.device` to a concrete device.

    ``None`` selects the best available backend without an index, so the current
    device of the process (for example set by a distributed launcher) is used.
    Explicit values are honoured verbatim.
    """
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device(default_device_type())
    return torch.device(device)


def device_type(device: Any) -> str:
    """Backend type (``"xpu"`` / ``"cuda"`` / ``"cpu"``) of a device, string or tensor."""
    if isinstance(device, torch.Tensor):
        return device.device.type
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def _resolved_type(device: DeviceLike) -> str:
    return device_type(device) if device is not None else default_device_type()


def _backend(device: DeviceLike):
    """The ``torch.xpu`` or ``torch.cuda`` module for ``device``, or ``None`` on CPU or when unavailable."""
    resolved = _resolved_type(device)
    if resolved in _ACCELERATORS and is_available(resolved):
        return getattr(torch, resolved)
    return None


def device_count(backend: Optional[str] = None) -> int:
    """Number of visible devices of a backend; 0 when the backend is unavailable."""
    resolved = backend or default_device_type()
    if resolved in _ACCELERATORS and is_available(resolved):
        return getattr(torch, resolved).device_count()
    return 0


def synchronize(device: DeviceLike = None) -> None:
    """Synchronize the accelerator for ``device``; no-op on CPU."""
    module = _backend(device)
    if module is not None:
        module.synchronize()


def empty_cache(device: DeviceLike = None) -> None:
    """Release cached allocator blocks for ``device``; no-op on CPU."""
    module = _backend(device)
    if module is not None:
        module.empty_cache()


def memory_allocated(device: DeviceLike = None) -> int:
    """Bytes allocated on the accelerator, 0 on CPU."""
    module = _backend(device)
    return module.memory_allocated() if module is not None else 0


def max_memory_allocated(device: DeviceLike = None) -> int:
    """Peak bytes allocated since the last reset, 0 on CPU."""
    module = _backend(device)
    return module.max_memory_allocated() if module is not None else 0


def memory_reserved(device: DeviceLike = None) -> int:
    """Bytes reserved by the caching allocator, 0 on CPU."""
    module = _backend(device)
    return module.memory_reserved() if module is not None else 0


def max_memory_reserved(device: DeviceLike = None) -> int:
    """Peak bytes reserved since the last reset, 0 on CPU."""
    module = _backend(device)
    return module.max_memory_reserved() if module is not None else 0


def reset_peak_memory_stats(device: DeviceLike = None) -> None:
    """Reset the peak memory counters; no-op on CPU."""
    module = _backend(device)
    if module is not None:
        module.reset_peak_memory_stats()


def flash_attn_installed() -> bool:
    """Whether the ``flash_attn`` package can be imported."""
    try:
        import flash_attn  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def use_flash_attn(device: DeviceLike = None) -> bool:
    """Whether flash-attn should be used for ``device``.

    flash-attn only runs on CUDA. On every other backend the standard attention
    path is used, and the reason is logged once.
    """
    resolved = _resolved_type(device)
    if resolved != "cuda":
        _warn_once(
            f"flash_attn_{resolved}",
            f"flash_attn is CUDA only; using standard attention on the {resolved} backend.",
        )
        return False
    if not flash_attn_installed():
        _warn_once(
            "flash_attn_missing",
            "flash_attn is not installed. Falling back to standard attention.",
        )
        return False
    return True


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message)
