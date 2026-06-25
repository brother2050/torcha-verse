"""Checkpoint loading logic for :class:`CheckpointManager`.

A *checkpoint* is either:

1. A directory containing ``model.safetensors`` /
   ``model.pt`` (weights), ``training_state.pt`` (optimizer +
   scheduler + RNG) and ``metadata.json``.
2. A single file produced by a legacy / weights-only workflow.

This module contains the four small helpers that the manager
calls for both cases:

* :func:`find_weights_file` -- locate the weights file inside a
  checkpoint directory.
* :func:`build_metadata` -- assemble the metadata dictionary
  written next to a checkpoint.
* :func:`load_from_directory` -- load a full directory checkpoint
  into ``model`` / ``optimizer`` / ``scheduler``.
* :func:`load_from_file` -- load a single-file (legacy /
  weights-only) checkpoint into ``model`` / ``optimizer``.

Pickle is **always** gated behind ``allow_unsafe_pickle``; the
manager never silently downgrades to ``weights_only=False`` on
errors.  See the v0.4.0 release notes for the full rationale.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from ._protocols import META_FILE, STATE_FILE, WEIGHTS_FILE, WEIGHTS_FILE_FALLBACK
from ._serialize import load_weights_file
from ._state import restore_rng_states

__all__ = [
    "find_weights_file",
    "build_metadata",
    "load_from_directory",
    "load_from_file",
]


def find_weights_file(ckpt_dir: Path) -> Path:
    """Locate the weights file inside a checkpoint directory."""
    for name in (WEIGHTS_FILE, WEIGHTS_FILE_FALLBACK):
        candidate = ckpt_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No weights file found in checkpoint directory {ckpt_dir}."
    )


def build_metadata(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    metadata: Optional[Dict[str, Any]],
    use_safetensors: bool,
) -> Dict[str, Any]:
    """Assemble the metadata dictionary written next to the checkpoint."""
    meta: Dict[str, Any] = {
        "step": step,
        "timestamp": time.time(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "TorchaVerse",
        "model_class": type(model).__name__,
        "num_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "optimizer_class": (
            type(optimizer).__name__ if optimizer is not None else None
        ),
        "format": "safetensors" if use_safetensors else "torch",
    }
    if metadata:
        meta["user_metadata"] = metadata
        meta.update(metadata)
    return meta


def load_from_directory(
    ckpt_dir: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    map_location: Optional[Union[str, torch.device]],
    strict: bool,
    allow_unsafe_pickle: bool = False,
    logger=None,
) -> Dict[str, Any]:
    """Load a full checkpoint from a directory.

    Returns:
        The metadata dictionary read from ``metadata.json`` (may
        include ``step``).
    """
    weights_path = find_weights_file(ckpt_dir)
    state_dict = load_weights_file(weights_path, map_location)
    model.load_state_dict(state_dict, strict=strict)

    metadata: Dict[str, Any] = {}
    meta_path = ckpt_dir / META_FILE
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    state_path = ckpt_dir / STATE_FILE
    if state_path.exists():
        # Training state contains optimizer / scheduler state and
        # RNG states (including numpy / python) which require
        # full unpickling.  Pickle is unsafe: only allow it when
        # the caller has explicitly opted in via
        # ``allow_unsafe_pickle=True`` (and even then the path
        # should be trusted, e.g. self-produced checkpoints).
        if not allow_unsafe_pickle:
            raise RuntimeError(
                "Refusing to unpickle training state without explicit "
                "opt-in: re-call load_checkpoint(allow_unsafe_pickle=True) "
                "only when loading a self-produced checkpoint from a "
                "trusted location. The pickle format allows arbitrary "
                "code execution."
            )
        training_state = torch.load(
            state_path, map_location=map_location, weights_only=False
        )
        if optimizer is not None and training_state.get("optimizer") is not None:
            optimizer.load_state_dict(training_state["optimizer"])
        if scheduler is not None and training_state.get("scheduler") is not None:
            scheduler.load_state_dict(training_state["scheduler"])
        restore_rng_states(training_state.get("rng_states"))
        metadata.setdefault("step", training_state.get("step", 0))

    if logger is not None:
        logger.info("Loaded checkpoint from %s.", ckpt_dir)
    return metadata


def load_from_file(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    map_location: Optional[Union[str, torch.device]],
    strict: bool,
    allow_unsafe_pickle: bool = False,
) -> Dict[str, Any]:
    """Load a single-file checkpoint (legacy or weights-only).

    A legacy single-file checkpoint may be a dict containing
    everything, including optimizer state, so full unpickling is
    required.  Pickle is unsafe: gate behind ``allow_unsafe_pickle``.
    """
    if not allow_unsafe_pickle:
        raise RuntimeError(
            "Refusing to load a single-file pickle checkpoint without "
            "explicit opt-in: re-call load_checkpoint(allow_unsafe_pickle=True) "
            "only when the file is a self-produced checkpoint from a "
            "trusted location."
        )
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(payload, dict) and "state_dict" in payload:
        model.load_state_dict(payload["state_dict"], strict=strict)
        if optimizer is not None and payload.get("optimizer_state_dict"):
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return payload.get("metadata", {})

    # Otherwise treat it as a plain state dict.
    model.load_state_dict(payload, strict=strict)
    return {}
