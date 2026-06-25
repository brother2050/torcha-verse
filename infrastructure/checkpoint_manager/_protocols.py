"""Storage backends + filename conventions for the checkpoint sub-package.

Two related concerns live in this module:

* The :class:`CheckpointBackend` Protocol -- the pluggable
  storage surface that :class:`~CheckpointManager` can delegate
  to.  The default in-tree implementation is
  :class:`LocalCheckpointBackend`, which writes to a local
  directory.
* The on-disk filename conventions used by every checkpoint
  directory: ``model.safetensors`` (or ``model.pt`` as a
  fallback), ``training_state.pt``, ``metadata.json``.

The Protocol is intentionally tiny (``write`` / ``read`` /
``exists``); custom backends for S3, the HuggingFace Hub, or
in-memory caches can be plugged in by implementing the three
methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Union, runtime_checkable

__all__ = [
    "CheckpointBackend",
    "LocalCheckpointBackend",
    "WEIGHTS_FILE",
    "WEIGHTS_FILE_FALLBACK",
    "STATE_FILE",
    "META_FILE",
]


#: Canonical weights filename (when ``safetensors`` is available).
WEIGHTS_FILE: str = "model.safetensors"
#: Fallback weights filename (when ``safetensors`` is not available).
WEIGHTS_FILE_FALLBACK: str = "model.pt"
#: Training-state filename (optimizer / scheduler / RNG).
STATE_FILE: str = "training_state.pt"
#: Human-readable metadata filename.
META_FILE: str = "metadata.json"


@runtime_checkable
class CheckpointBackend(Protocol):
    """Pluggable storage backend for checkpoint payloads.

    A backend is responsible for moving bytes between an in-memory
    representation and a durable location.  The
    :class:`~CheckpointManager` calls :meth:`write` to persist a
    payload (a ``bytes`` blob plus a logical ``key``) and
    :meth:`read` to retrieve it.

    The default :class:`LocalCheckpointBackend` writes to a local
    directory.  Custom backends can target S3, the HuggingFace
    Hub, an in-memory cache, or anything else without changing
    the manager.
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
        with open(target, "wb") as fh:
            fh.write(data)
        return str(target)

    def read(self, key: str) -> bytes:
        path = self.root / key
        if not path.is_file():
            raise FileNotFoundError(f"Key {key!r} not found in {self.root}")
        with open(path, "rb") as fh:
            return fh.read()

    def exists(self, key: str) -> bool:
        return (self.root / key).is_file()
