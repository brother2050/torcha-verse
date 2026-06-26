"""Warm-tier (content-addressed object store) helpers (v0.6.x).

The warm tier stores blobs on the local filesystem, named by their
sha256 digest and sharded two hex characters deep.  This module
hosts the file-system helpers used by :class:`AssetStore`:

* :func:`content_path` -- resolve a content hash to an on-disk path.
* :func:`copy_to_staging` -- copy a source file to a staging path
  with an ``fsync`` so the staged file is durable before the atomic
  rename.
* :func:`cleanup_staging` -- best-effort removal of a staging file.
* :func:`hash_file` -- return the sha256 hex digest of a file.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Union

from ._schema import CHUNK_SIZE

__all__ = [
    "content_path",
    "copy_to_staging",
    "cleanup_staging",
    "hash_file",
]


def content_path(objects_dir: Union[Path, str], content_hash: str) -> Path:
    """Return the on-disk path for a content hash (sharded 2 deep).

    The shard prefix is the first two hex characters of the hash,
    so a single directory never accumulates more than 1/256th of
    the total blobs.
    """
    return Path(objects_dir) / content_hash[:2] / content_hash


def copy_to_staging(src: Path, staging_path: Path) -> None:
    """Copy ``src`` to a staging path with an ``fsync``.

    Writes the source content to ``staging_path`` in chunked fashion
    with an ``fsync`` so that the staged file is durable before the
    atomic rename in the hot / warm path.  This function performs
    *no* locking and may run concurrently with other store
    operations.
    """
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fsrc, open(staging_path, "wb") as fdst:
        while True:
            chunk = fsrc.read(CHUNK_SIZE)
            if not chunk:
                break
            fdst.write(chunk)
        fdst.flush()
        os.fsync(fdst.fileno())


def cleanup_staging(staging_path: Path) -> None:
    """Remove a staging file if it still exists (best-effort)."""
    try:
        staging_path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        # Staging cleanup is best-effort; never raise.
        return


def hash_file(path: Path) -> str:
    """Return the sha256 hex digest of the file at ``path``."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()
