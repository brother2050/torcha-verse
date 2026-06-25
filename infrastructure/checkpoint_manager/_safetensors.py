"""Safetensors soft-dependency shim.

The checkpoint manager prefers the ``safetensors`` format for weight
files because it is a small, mmap-safe, pickle-free container.
The dependency is optional -- the manager transparently falls back
to ``torch.save`` / ``torch.load(weights_only=True)`` when
``safetensors`` is not installed.

This module centralises the soft-dependency check + thin wrappers
so the rest of the checkpoint sub-package can call
:func:`is_available` / :func:`save` / :func:`load` without
scattering ``try / except ImportError`` blocks.
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = ["is_available", "save", "load"]


try:
    from safetensors.torch import load_file as _st_load
    from safetensors.torch import save_file as _st_save

    _AVAILABLE: bool = True
except Exception:  # pragma: no cover - safetensors not installed
    _st_load = None
    _st_save = None
    _AVAILABLE = False


def is_available() -> bool:
    """Return ``True`` iff ``safetensors`` is importable."""
    return _AVAILABLE


def save(state_dict: Dict[str, Any], path: str) -> None:
    """Save a state dict via ``safetensors``.

    Args:
        state_dict: Already cleaned, CPU-contiguous tensors.
        path:       Target file path.

    Raises:
        RuntimeError: ``safetensors`` is not installed.
    """
    if not _AVAILABLE:
        raise RuntimeError(
            "safetensors is not installed; install it with "
            "`pip install safetensors` to enable safetensors support."
        )
    _st_save(state_dict, path)  # type: ignore[misc]


def load(path: str) -> Dict[str, Any]:
    """Load a safetensors file and return the state dict.

    Raises:
        RuntimeError: ``safetensors`` is not installed.
    """
    if not _AVAILABLE:
        raise RuntimeError(
            "safetensors is not installed; install it with "
            "`pip install safetensors` to enable safetensors support."
        )
    return _st_load(path)  # type: ignore[misc]
