"""The :class:`CheckpointManager` core.

The v0.6.x refactor of
:mod:`infrastructure.checkpoint_manager` splits the previous
single-file implementation (631 lines) into seven focused
modules.  The :class:`CheckpointManager` class itself lives in
this file and is reduced to the public surface:

* :meth:`save_checkpoint` / :meth:`load_checkpoint` -- full
  checkpoint lifecycle.
* :meth:`save_weights_only` / :meth:`load_weights` -- weights-only
  variant (e.g. for inference export).
* :meth:`list_checkpoints` / :meth:`get_latest_checkpoint` --
  enumeration helpers.
* :meth:`resume` -- convenience wrapper that loads a checkpoint
  and returns the step counter.

All serialisation, RNG, pruning and metadata building is
delegated to the sibling modules:

* :mod:`._safetensors` -- safetensors soft-dependency shim.
* :mod:`._protocols` -- :class:`CheckpointBackend` Protocol +
  :class:`LocalCheckpointBackend` + filename constants.
* :mod:`._serialize` -- state-dict extraction / cleaning /
  loading.
* :mod:`._load` -- directory / file loading + metadata
  building.
* :mod:`._state` -- RNG state capture / restore.
* :mod:`._prune` -- checkpoint pruning policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from ..logger import get_logger
from . import _safetensors
from ._load import (
    build_metadata,
    load_from_directory,
    load_from_file,
)
from ._prune import list_checkpoints, prune_checkpoints
from ._protocols import (
    CheckpointBackend,
    LocalCheckpointBackend,
    META_FILE,
    STATE_FILE,
    WEIGHTS_FILE,
    WEIGHTS_FILE_FALLBACK,
)
from ._serialize import (
    extract_state_dict,
    load_weights_file,
    save_weights,
    save_weights_to_path,
)
from ._state import capture_rng_states

__all__ = ["CheckpointManager"]


class CheckpointManager:
    """Manage the full lifecycle of model checkpoints.

    Each checkpoint is stored as a directory containing the model
    weights (``safetensors`` when available, otherwise a ``.pt``
    file), the training state (optimizer / scheduler / RNG), and
    a human-readable metadata file.  Older checkpoints are
    pruned automatically once ``save_total_limit`` is exceeded.

    Args:
        save_dir: Root directory in which versioned checkpoints
            are stored.
        save_total_limit: Maximum number of checkpoints to keep.
            Older checkpoints are deleted when this limit is
            exceeded.  ``None`` or ``0`` disables pruning.
        use_safetensors: When ``True`` (default) the manager
            prefers ``safetensors`` for the weights file; the
            flag is silently downgraded to ``False`` if the
            library is not installed.
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        save_total_limit: Optional[int] = 5,
        use_safetensors: bool = True,
    ) -> None:
        self.save_dir: Path = Path(save_dir).expanduser().resolve()
        self.save_total_limit: Optional[int] = save_total_limit
        self.use_safetensors: bool = use_safetensors and _safetensors.is_available()
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API: full checkpoints
    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        path: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        step: int = 0,
        scheduler: Optional[Any] = None,
    ) -> Path:
        """Save a full training checkpoint.

        Args:
            model:     The model whose ``state_dict`` is saved.
            optimizer: The optimizer whose state is saved.  May
                be ``None``.
            path:      Directory in which to create the versioned
                checkpoint.  When ``None`` the manager's
                ``save_dir`` is used.
            metadata:  Arbitrary user metadata merged into the
                saved metadata.
            step:      Training step / global iteration counter.
            scheduler: Optional learning-rate scheduler.

        Returns:
            The path to the created checkpoint directory.
        """
        root = Path(path).expanduser().resolve() if path else self.save_dir
        root.mkdir(parents=True, exist_ok=True)

        ckpt_dir = root / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Persist model weights.
        state_dict = extract_state_dict(model)
        save_weights(state_dict, ckpt_dir, self.use_safetensors)

        # 2. Persist training state (optimizer, scheduler, RNG).
        training_state: Dict[str, Any] = {
            "step": step,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "rng_states": capture_rng_states(),
        }
        torch.save(training_state, ckpt_dir / STATE_FILE)

        # 3. Persist metadata.
        meta = build_metadata(
            model, optimizer, step, metadata, self.use_safetensors,
        )
        with open(ckpt_dir / META_FILE, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False, default=str)

        self.logger.info(
            "Saved checkpoint to %s (step=%d, format=%s).",
            ckpt_dir, step,
            "safetensors" if self.use_safetensors else "torch",
        )

        # 4. Reclaim disk space if needed.
        prune_checkpoints(
            root, self.save_total_limit,
            on_prune=lambda p: self.logger.info("Pruned old checkpoint %s.", p),
        )

        return ckpt_dir

    def load_checkpoint(
        self,
        path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
        allow_unsafe_pickle: bool = False,
    ) -> Dict[str, Any]:
        """Load a training checkpoint.

        Args:
            path: Either a directory created by
                :meth:`save_checkpoint` or a single-file
                (legacy) checkpoint.
            model: The model to load weights into.
            optimizer: Optional optimizer to restore.
            scheduler: Optional scheduler to restore.
            map_location: Optional ``map_location`` argument for
                ``torch.load``.
            strict: ``strict`` argument for
                :meth:`Module.load_state_dict`.
            allow_unsafe_pickle: When ``True`` the manager is
                allowed to unpickle the training-state file.
                Defaults to ``False`` for safety.

        Returns:
            The metadata dictionary read from the checkpoint.
        """
        path = Path(path).expanduser().resolve()
        if path.is_dir():
            return load_from_directory(
                path, model, optimizer, scheduler,
                map_location, strict,
                allow_unsafe_pickle=allow_unsafe_pickle,
                logger=self.logger,
            )
        return load_from_file(
            path, model, optimizer,
            map_location, strict,
            allow_unsafe_pickle=allow_unsafe_pickle,
        )

    # ------------------------------------------------------------------
    # Public API: weights-only
    # ------------------------------------------------------------------
    def save_weights_only(
        self,
        model: nn.Module,
        path: Union[str, Path],
    ) -> Path:
        """Save a weights-only checkpoint (no optimizer / scheduler / RNG)."""
        state_dict = extract_state_dict(model)
        target = Path(path).expanduser().resolve()
        written = save_weights_to_path(
            state_dict, target, self.use_safetensors,
        )
        self.logger.info(
            "Saved weights-only checkpoint to %s (format=%s).",
            written,
            "safetensors" if self.use_safetensors else "torch",
        )
        return written

    def load_weights(
        self,
        path: Union[str, Path],
        model: nn.Module,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> nn.Module:
        """Load a weights file into ``model`` and return ``model``."""
        path = Path(path).expanduser().resolve()
        state_dict = load_weights_file(path, map_location)
        model.load_state_dict(state_dict, strict=strict)
        self.logger.info("Loaded weights from %s.", path)
        return model

    # ------------------------------------------------------------------
    # Enumeration / convenience
    # ------------------------------------------------------------------
    def list_checkpoints(
        self,
        root: Optional[Union[str, Path]] = None,
    ) -> List[Path]:
        """List the checkpoint directories under ``root`` (defaults to
        :attr:`save_dir`) in age order."""
        target = Path(root).expanduser().resolve() if root else self.save_dir
        return list_checkpoints(target)

    def get_latest_checkpoint(
        self,
        root: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """Return the most recent checkpoint directory, or ``None``."""
        cks = self.list_checkpoints(root)
        return cks[-1] if cks else None

    def resume(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
        allow_unsafe_pickle: bool = False,
    ) -> Dict[str, Any]:
        """Resume from the most recent checkpoint in :attr:`save_dir`.

        If no checkpoint exists the call is a no-op and the
        returned dict is empty.  Otherwise the call delegates to
        :meth:`load_checkpoint` and returns its metadata.
        """
        latest = self.get_latest_checkpoint()
        if latest is None:
            self.logger.info("No checkpoint found to resume from.")
            return {}
        return self.load_checkpoint(
            latest, model, optimizer, scheduler,
            map_location, strict, allow_unsafe_pickle=allow_unsafe_pickle,
        )
