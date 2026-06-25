"""Checkpoint lifecycle management for TorchaVerse.

This module provides :class:`CheckpointManager`, a unified API for
serialising model weights and training state.  It supports:

* Full checkpoints (model + optimizer + scheduler + RNG state) for
  resumable training.
* Weights-only checkpoints (e.g. for inference export).
* Automatic versioning and disk-space reclamation via ``save_total_limit``.
* The ``safetensors`` format for weights, with a transparent fallback to
  ``torch.save`` when the library is unavailable.
* A pluggable :class:`CheckpointBackend` Protocol so that downstream
  code can swap in a remote (S3, HuggingFace Hub) or memory-mapped
  storage backend without changing the manager's surface area.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import torch
import torch.nn as nn

from .logger import get_logger

__all__ = ["CheckpointManager", "CheckpointBackend", "LocalCheckpointBackend"]

# Optional safetensors dependency.
try:  # pragma: no cover - import guard
    from safetensors.torch import load_file as _safetensors_load
    from safetensors.torch import save_file as _safetensors_save

    _HAS_SAFETENSORS: bool = True
except Exception:  # pragma: no cover - safetensors not installed
    _HAS_SAFETENSORS = False


# File names used inside a checkpoint directory.
_WEIGHTS_FILE = "model.safetensors"
_WEIGHTS_FILE_FALLBACK = "model.pt"
_STATE_FILE = "training_state.pt"
_META_FILE = "metadata.json"


# ---------------------------------------------------------------------------
# Backend protocol + default implementation
# ---------------------------------------------------------------------------
@runtime_checkable
class CheckpointBackend(Protocol):
    """Pluggable storage backend for checkpoint payloads.

    A backend is responsible for moving bytes between an in-memory
    representation and a durable location.  The :class:`CheckpointManager`
    calls :meth:`write` to persist a payload (a ``bytes`` blob plus a
    logical ``key``) and :meth:`read` to retrieve it.

    The default :class:`LocalCheckpointBackend` writes to a local
    directory.  Custom backends can target S3, the HuggingFace Hub, an
    in-memory cache, or anything else without changing the manager.
    """

    def write(self, key: str, data: bytes) -> str:
        """Persist ``data`` under ``key``; return the resolved URI."""
        ...

    def read(self, key: str) -> bytes:
        """Return the bytes previously stored under ``key``."""
        ...

    def exists(self, key: str) -> bool:
        """Return ``True`` iff ``key`` has been previously written."""
        ...


class LocalCheckpointBackend:
    """Default :class:`CheckpointBackend` that writes to a local directory.

    Keys are translated to relative paths under ``root`` (a
    :class:`pathlib.Path`).  ``read`` raises :class:`FileNotFoundError`
    when the file is missing.
    """

    def __init__(self, root: Union[str, Path]) -> None:
        self.root: Path = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, key: str, data: bytes) -> str:
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            handle.write(data)
        return str(target)

    def read(self, key: str) -> bytes:
        target = self.root / key
        if not target.exists():
            raise FileNotFoundError(f"Checkpoint not found: {target}")
        with open(target, "rb") as handle:
            return handle.read()

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()


class CheckpointManager:
    """Manage the full lifecycle of model checkpoints.

    Each checkpoint is stored as a directory containing the model weights
    (``safetensors`` when available, otherwise a ``.pt`` file), the training
    state (optimizer / scheduler / RNG), and a human-readable metadata
    file.  Older checkpoints are pruned automatically once
    ``save_total_limit`` is exceeded.

    Args:
        save_dir: Root directory in which versioned checkpoints are stored.
        save_total_limit: Maximum number of checkpoints to keep.  Older
            checkpoints are deleted when this limit is exceeded.  ``None``
            or ``0`` disables pruning.
        use_safetensors: Prefer the ``safetensors`` format for weights.
            Falls back to ``torch.save`` automatically when the library is
            not installed.
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        save_total_limit: Optional[int] = 5,
        use_safetensors: bool = True,
    ) -> None:
        self.save_dir: Path = Path(save_dir).expanduser().resolve()
        self.save_total_limit: Optional[int] = save_total_limit
        self.use_safetensors: bool = use_safetensors and _HAS_SAFETENSORS
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
            model: The model whose ``state_dict`` is saved.
            optimizer: The optimizer whose state is saved.  May be ``None``.
            path: Directory in which to create the versioned checkpoint.  When
                ``None`` the manager's ``save_dir`` is used.
            metadata: Arbitrary user metadata merged into the saved metadata.
            step: Training step / global iteration counter.
            scheduler: Optional learning-rate scheduler.

        Returns:
            The path to the created checkpoint directory.
        """
        root = Path(path).expanduser().resolve() if path else self.save_dir
        root.mkdir(parents=True, exist_ok=True)

        ckpt_dir = root / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Persist model weights.
        state_dict = self._extract_state_dict(model)
        self._save_weights(state_dict, ckpt_dir)

        # 2. Persist training state (optimizer, scheduler, RNG).
        training_state: Dict[str, Any] = {
            "step": step,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "rng_states": self._capture_rng_states(),
        }
        torch.save(training_state, ckpt_dir / _STATE_FILE)

        # 3. Persist metadata.
        meta = self._build_metadata(model, optimizer, step, metadata)
        with open(ckpt_dir / _META_FILE, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False, default=str)

        self.logger.info(
            "Saved checkpoint to %s (step=%d, format=%s).",
            ckpt_dir,
            step,
            "safetensors" if self.use_safetensors else "torch",
        )

        # 4. Reclaim disk space if needed.
        self._prune(root)

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
        """Load a checkpoint and restore model / optimizer state.

        Args:
            path: Path to a checkpoint directory (created by
                :meth:`save_checkpoint`) or a single weights file.
            model: Model to load the weights into.
            optimizer: Optional optimizer to restore state into.
            scheduler: Optional scheduler to restore state into.
            map_location: Device to map tensors to when loading.
            strict: Forwarded to ``load_state_dict`` for weight loading.
            allow_unsafe_pickle: When ``True`` the legacy
                ``weights_only=False`` path is used for pickle
                checkpoints.  This is **insecure** (arbitrary code
                execution) and is only intended for loading
                self-produced checkpoints from a trusted location.
                Defaults to ``False`` so that pickle RCE is impossible
                by default.

        Returns:
            The metadata dictionary associated with the checkpoint.
        """
        path = Path(path).expanduser().resolve()

        if path.is_dir():
            return self._load_from_directory(
                path,
                model,
                optimizer,
                scheduler,
                map_location,
                strict,
                allow_unsafe_pickle=allow_unsafe_pickle,
            )
        if path.is_file():
            return self._load_from_file(
                path,
                model,
                optimizer,
                map_location,
                strict,
                allow_unsafe_pickle=allow_unsafe_pickle,
            )

        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # ------------------------------------------------------------------
    # Public API: weights only
    # ------------------------------------------------------------------
    def save_weights_only(
        self,
        model: nn.Module,
        path: Union[str, Path],
    ) -> Path:
        """Save only the model weights (no optimizer/training state).

        Args:
            model: The model to save.
            path: Target file path.  The appropriate extension
                (``.safetensors`` or ``.pt``) is appended automatically when
                none is provided.

        Returns:
            The final path of the saved weights file.
        """
        state_dict = self._extract_state_dict(model)
        target = Path(path).expanduser().resolve()

        if self.use_safetensors:
            if target.suffix not in (".safetensors", ".st"):
                target = target.with_suffix(".safetensors")
            target.parent.mkdir(parents=True, exist_ok=True)
            _safetensors_save(self._to_cpu_contiguous(state_dict), str(target))
        else:
            if target.suffix not in (".pt", ".bin", ".pth"):
                target = target.with_suffix(".pt")
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, target)

        self.logger.info(
            "Saved weights-only checkpoint to %s (format=%s).",
            target,
            "safetensors" if self.use_safetensors else "torch",
        )
        return target

    def load_weights(
        self,
        path: Union[str, Path],
        model: nn.Module,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> nn.Module:
        """Load weights from a file into ``model``.

        Args:
            path: Path to a ``.safetensors`` or ``.pt`` weights file.
            model: Model to load the weights into.
            map_location: Device to map tensors to.
            strict: Whether to enforce exact key matching.

        Returns:
            The model with loaded weights (modified in place).
        """
        path = Path(path).expanduser().resolve()
        state_dict = self._load_weights_file(path, map_location)
        model.load_state_dict(state_dict, strict=strict)
        self.logger.info("Loaded weights from %s into %s.", path, type(model).__name__)
        return model

    # ------------------------------------------------------------------
    # Versioning / discovery
    # ------------------------------------------------------------------
    def list_checkpoints(
        self, root: Optional[Union[str, Path]] = None
    ) -> List[Path]:
        """List all checkpoint directories under ``root`` sorted by step."""
        root = Path(root).expanduser().resolve() if root else self.save_dir
        if not root.exists():
            return []
        checkpoints = [
            p
            for p in root.iterdir()
            if p.is_dir() and p.name.startswith("checkpoint-")
        ]

        def _step(p: Path) -> int:
            try:
                return int(p.name.split("-", 1)[1])
            except (IndexError, ValueError):
                return -1

        return sorted(checkpoints, key=_step)

    def get_latest_checkpoint(
        self, root: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """Return the path of the most recent checkpoint, or ``None``."""
        checkpoints = self.list_checkpoints(root)
        return checkpoints[-1] if checkpoints else None

    def resume(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        root: Optional[Union[str, Path]] = None,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Resume training from the latest checkpoint.

        Args:
            model: Model to restore weights into.
            optimizer: Optional optimizer to restore state into.
            scheduler: Optional scheduler to restore state into.
            root: Checkpoint root directory.  Defaults to ``save_dir``.
            map_location: Device to map tensors to.

        Returns:
            A tuple ``(step, metadata)`` where ``step`` is the training step
            stored in the checkpoint (``0`` when nothing was found).
        """
        ckpt = self.get_latest_checkpoint(root)
        if ckpt is None:
            self.logger.info("No checkpoint found; starting from scratch.")
            return 0, {}

        metadata = self.load_checkpoint(
            ckpt, model, optimizer, scheduler, map_location=map_location
        )
        step = int(metadata.get("step", 0))
        self.logger.info("Resumed from %s at step %d.", ckpt, step)
        return step, metadata

    # ------------------------------------------------------------------
    # Internals: weights I/O
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
        """Return a detached copy of the model's state dict (CPU)."""
        # ``detach`` avoids saving the autograd graph; ``state_dict`` already
        # returns references so we clone to be safe.
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    @staticmethod
    def _to_cpu_contiguous(
        state_dict: Dict[str, torch.Tensor]
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

    def _save_weights(
        self, state_dict: Dict[str, torch.Tensor], ckpt_dir: Path
    ) -> None:
        """Save weights inside a checkpoint directory."""
        if self.use_safetensors:
            _safetensors_save(
                self._to_cpu_contiguous(state_dict),
                str(ckpt_dir / _WEIGHTS_FILE),
            )
        else:
            torch.save(state_dict, ckpt_dir / _WEIGHTS_FILE_FALLBACK)

    def _load_weights_file(
        self, path: Path, map_location: Optional[Union[str, torch.device]]
    ) -> Dict[str, torch.Tensor]:
        """Load a weights file, auto-detecting the format.

        ``safetensors`` is preferred when available.  For legacy ``.pt`` /
        ``.bin`` files we attempt a secure ``weights_only=True`` load.  The
        previous implementation silently downgraded to ``weights_only=False``
        on any exception, which allowed pickle RCE.  We now propagate the
        error instead, and require callers to opt in to unsafe loading
        explicitly via :meth:`load_checkpoint(allow_unsafe_pickle=True)`.
        """
        if path.suffix == ".safetensors" and _HAS_SAFETENSORS:
            return _safetensors_load(str(path))
        # Use the safe path; failures must be surfaced so the caller can
        # either retry, use safetensors, or opt in to unsafe loading.
        return torch.load(path, map_location=map_location, weights_only=True)

    # ------------------------------------------------------------------
    # Internals: directory / file loading
    # ------------------------------------------------------------------
    def _load_from_directory(
        self,
        ckpt_dir: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        map_location: Optional[Union[str, torch.device]],
        strict: bool,
        allow_unsafe_pickle: bool = False,
    ) -> Dict[str, Any]:
        """Load a full checkpoint from a directory."""
        weights_path = self._find_weights_file(ckpt_dir)
        state_dict = self._load_weights_file(weights_path, map_location)
        model.load_state_dict(state_dict, strict=strict)

        metadata: Dict[str, Any] = {}
        meta_path = ckpt_dir / _META_FILE
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)

        state_path = ckpt_dir / _STATE_FILE
        if state_path.exists():
            # Training state contains optimizer/scheduler state and RNG
            # states (including numpy/python) which require full unpickling.
            # Pickle is unsafe: only allow it when the caller has explicitly
            # opted in via ``allow_unsafe_pickle=True`` (and even then the
            # path should be trusted, e.g. self-produced checkpoints).
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
            self._restore_rng_states(training_state.get("rng_states"))
            metadata.setdefault("step", training_state.get("step", 0))

        self.logger.info("Loaded checkpoint from %s.", ckpt_dir)
        return metadata

    def _load_from_file(
        self,
        path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        map_location: Optional[Union[str, torch.device]],
        strict: bool,
        allow_unsafe_pickle: bool = False,
    ) -> Dict[str, Any]:
        """Load a single-file checkpoint (legacy or weights-only)."""
        # A legacy single-file checkpoint may be a dict containing
        # everything, including optimizer state, so full unpickling is
        # required.  Pickle is unsafe: gate behind ``allow_unsafe_pickle``.
        if not allow_unsafe_pickle:
            raise RuntimeError(
                "Refusing to load a single-file pickle checkpoint without "
                "explicit opt-in: re-call load_checkpoint(allow_unsafe_pickle=True) "
                "only when the file is a self-produced checkpoint from a "
                "trusted location."
            )
        payload = torch.load(
            path, map_location=map_location, weights_only=False
        )

        if isinstance(payload, dict) and "state_dict" in payload:
            model.load_state_dict(payload["state_dict"], strict=strict)
            if optimizer is not None and payload.get("optimizer_state_dict"):
                optimizer.load_state_dict(payload["optimizer_state_dict"])
            return payload.get("metadata", {})

        # Otherwise treat it as a plain state dict.
        model.load_state_dict(payload, strict=strict)
        return {}

    @staticmethod
    def _find_weights_file(ckpt_dir: Path) -> Path:
        """Locate the weights file inside a checkpoint directory."""
        for name in (_WEIGHTS_FILE, _WEIGHTS_FILE_FALLBACK):
            candidate = ckpt_dir / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No weights file found in checkpoint directory {ckpt_dir}."
        )

    # ------------------------------------------------------------------
    # Internals: metadata & RNG
    # ------------------------------------------------------------------
    def _build_metadata(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        step: int,
        metadata: Optional[Dict[str, Any]],
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
            "format": "safetensors" if self.use_safetensors else "torch",
        }
        if metadata:
            meta["user_metadata"] = metadata
            meta.update(metadata)
        return meta

    @staticmethod
    def _capture_rng_states() -> Dict[str, Any]:
        """Capture RNG states for reproducible resume."""
        states: Dict[str, Any] = {
            "python": random.getstate(),
        }
        try:
            import numpy as np

            states["numpy"] = np.random.get_state()
        except Exception:  # pragma: no cover - numpy optional
            pass
        states["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            states["torch_cuda"] = torch.cuda.get_rng_state_all()
        return states

    @staticmethod
    def _restore_rng_states(states: Optional[Dict[str, Any]]) -> None:
        """Restore previously captured RNG states."""
        if not states:
            return
        if "python" in states:
            random.setstate(states["python"])
        if "numpy" in states:
            try:
                import numpy as np

                np.random.set_state(states["numpy"])
            except Exception:  # pragma: no cover
                pass
        if "torch" in states:
            torch.set_rng_state(states["torch"])
        if "torch_cuda" in states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["torch_cuda"])

    # ------------------------------------------------------------------
    # Internals: pruning
    # ------------------------------------------------------------------
    def _prune(self, root: Path) -> None:
        """Delete oldest checkpoints beyond ``save_total_limit``."""
        if not self.save_total_limit or self.save_total_limit <= 0:
            return

        checkpoints = self.list_checkpoints(root)
        excess = len(checkpoints) - self.save_total_limit
        if excess <= 0:
            return

        for ckpt in checkpoints[:excess]:
            try:
                shutil.rmtree(ckpt)
                self.logger.info("Pruned old checkpoint %s.", ckpt)
            except OSError as exc:
                self.logger.warning(
                    "Failed to remove old checkpoint %s: %s", ckpt, exc
                )
