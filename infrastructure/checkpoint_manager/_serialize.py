"""Tensor-state serialisation helpers for :class:`CheckpointManager`.

Pure functions that turn a model's ``state_dict`` into something
safe to hand to :mod:`torch.save` or :mod:`safetensors`, and
vice-versa.  Keeping them as free functions (not methods on the
manager) makes them trivially unit-testable and lets future
serialisation formats plug in without touching the manager class.

Public surface:

* :func:`extract_state_dict` -- return a detached, CPU-side copy
  of a model's state dict.
* :func:`to_cpu_contiguous` -- move tensors to CPU and make them
  contiguous (a ``safetensors`` requirement).
* :func:`save_weights` -- save a state dict under a checkpoint
  directory, picking the format based on
  :mod:`._safetensors` availability and the
  ``use_safetensors`` flag.
* :func:`load_weights_file` -- load a weights file, auto-detecting
  the format.  Legacy ``.pt`` / ``.bin`` files are loaded with
  ``weights_only=True`` to prevent pickle RCE; the caller must
  opt in to unsafe loading explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from . import _safetensors
from ._protocols import WEIGHTS_FILE, WEIGHTS_FILE_FALLBACK

__all__ = [
    "extract_state_dict",
    "to_cpu_contiguous",
    "save_weights",
    "load_weights_file",
]


def extract_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a detached copy of the model's state dict (CPU)."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def to_cpu_contiguous(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Move tensors to CPU and make them contiguous (safetensors req.)."""
    cleaned: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        tensor = tensor.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        cleaned[key] = tensor
    return cleaned


def save_weights(
    state_dict: Dict[str, torch.Tensor],
    ckpt_dir: Path,
    use_safetensors: bool,
) -> Path:
    """Save weights inside a checkpoint directory.

    Returns:
        The path of the file that was written.
    """
    if use_safetensors and _safetensors.is_available():
        target = ckpt_dir / WEIGHTS_FILE
        _safetensors.save(to_cpu_contiguous(state_dict), str(target))
        return target
    target = ckpt_dir / WEIGHTS_FILE_FALLBACK
    torch.save(state_dict, target)
    return target


def save_weights_to_path(
    state_dict: Dict[str, torch.Tensor],
    target: Path,
    use_safetensors: bool,
) -> Path:
    """Save weights to an explicit path (for ``save_weights_only``).

    If ``target`` does not end in ``.pt`` / ``.bin`` / ``.pth`` and
    we are falling back to ``torch.save``, ``.pt`` is appended.
    """
    if use_safetensors and _safetensors.is_available():
        if target.suffix != ".safetensors":
            target = target.with_suffix(".safetensors")
        target.parent.mkdir(parents=True, exist_ok=True)
        _safetensors.save(to_cpu_contiguous(state_dict), str(target))
        return target
    if target.suffix not in (".pt", ".bin", ".pth"):
        target = target.with_suffix(".pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, target)
    return target


def load_weights_file(
    path: Path,
    map_location: Optional[Union[str, torch.device]],
) -> Dict[str, torch.Tensor]:
    """Load a weights file, auto-detecting the format.

    ``safetensors`` is preferred when available.  For legacy
    ``.pt`` / ``.bin`` files we attempt a secure
    ``weights_only=True`` load.  Failures are propagated to the
    caller; callers must opt in to unsafe loading explicitly via
    :meth:`CheckpointManager.load_checkpoint(allow_unsafe_pickle=True)`.
    """
    if path.suffix == ".safetensors" and _safetensors.is_available():
        return _safetensors.load(str(path))
    return torch.load(path, map_location=map_location, weights_only=True)
