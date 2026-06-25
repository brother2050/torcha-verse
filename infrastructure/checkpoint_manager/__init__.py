"""Checkpoint lifecycle management for TorchaVerse.

The :mod:`infrastructure.checkpoint_manager` sub-package
provides a unified API for serialising model weights and
training state, and a pluggable storage-backend Protocol so that
downstream code can swap in a remote (S3, HuggingFace Hub) or
memory-mapped storage backend without changing the manager's
surface area.

The v0.6.x refactor splits the previous single-file
``infrastructure/checkpoint_manager.py`` (631 lines) into seven
focused modules:

* :mod:`infrastructure.checkpoint_manager._protocols` -- the
  :class:`CheckpointBackend` Protocol +
  :class:`LocalCheckpointBackend` + filename conventions.
* :mod:`infrastructure.checkpoint_manager._safetensors` -- the
  ``safetensors`` soft-dependency shim (save / load /
  is_available).
* :mod:`infrastructure.checkpoint_manager._serialize` -- state-
  dict extraction / cleaning / weight-only file IO.
* :mod:`infrastructure.checkpoint_manager._load` -- full-
  directory + single-file loading + metadata building.
* :mod:`infrastructure.checkpoint_manager._state` -- RNG state
  capture / restore for reproducible resume.
* :mod:`infrastructure.checkpoint_manager._prune` -- checkpoint
  pruning policy.
* :mod:`infrastructure.checkpoint_manager._manager` -- the
  :class:`CheckpointManager` core class.

The public API is unchanged -- ``from infrastructure.checkpoint_manager
import CheckpointManager`` keeps working.

Supported features:

* Full checkpoints (model + optimizer + scheduler + RNG state)
  for resumable training.
* Weights-only checkpoints (e.g. for inference export).
* Automatic versioning and disk-space reclamation via
  ``save_total_limit``.
* The ``safetensors`` format for weights, with a transparent
  fallback to ``torch.save`` when the library is unavailable.
* A pluggable :class:`CheckpointBackend` Protocol.
"""

from __future__ import annotations

from ._manager import CheckpointManager
from ._protocols import (
    CheckpointBackend,
    LocalCheckpointBackend,
    META_FILE,
    STATE_FILE,
    WEIGHTS_FILE,
    WEIGHTS_FILE_FALLBACK,
)

__all__ = [
    "CheckpointManager",
    "CheckpointBackend",
    "LocalCheckpointBackend",
    "WEIGHTS_FILE",
    "WEIGHTS_FILE_FALLBACK",
    "STATE_FILE",
    "META_FILE",
]
