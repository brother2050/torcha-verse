"""Checkpoint-directory pruning policy.

The :class:`CheckpointManager` keeps at most
``save_total_limit`` checkpoints on disk; older ones are deleted
when a new save is performed.  The pruning algorithm is
factored out into this module so the manager does not have to
inline it.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

__all__ = ["prune_checkpoints", "list_checkpoints"]


def list_checkpoints(root: Path) -> List[Path]:
    """List the checkpoint directories under ``root`` in age order.

    A "checkpoint directory" is identified by the
    ``checkpoint-<step>`` naming convention.  The list is sorted
    by the integer ``<step>`` suffix in ascending order so
    ``checkpoints[:excess]`` is the oldest slice to delete.
    """
    if not root.is_dir():
        return []
    out: List[Path] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("checkpoint-"):
            continue
        try:
            int(name.split("-", 1)[1])
        except ValueError:
            continue
        out.append(entry)
    out.sort(key=lambda p: int(p.name.split("-", 1)[1]))
    return out


def prune_checkpoints(
    root: Path,
    save_total_limit: Optional[int],
    on_prune=None,
) -> List[Path]:
    """Delete the oldest checkpoints beyond ``save_total_limit``.

    Args:
        root: The checkpoint root directory.
        save_total_limit: Maximum number of checkpoints to keep;
            ``None`` or ``0`` disables pruning.
        on_prune: Optional callback invoked with each pruned path,
            typically used for logging.

    Returns:
        The list of pruned paths (empty when ``save_total_limit``
        is unset, no excess, or no checkpoints exist).
    """
    if not save_total_limit or save_total_limit <= 0:
        return []
    checkpoints = list_checkpoints(root)
    excess = len(checkpoints) - save_total_limit
    if excess <= 0:
        return []
    pruned: List[Path] = []
    for ckpt in checkpoints[:excess]:
        try:
            shutil.rmtree(ckpt)
            pruned.append(ckpt)
            if on_prune is not None:
                on_prune(ckpt)
        except OSError:
            # Best-effort: an OS error must not abort the new
            # save, only log via the caller-provided callback.
            continue
    return pruned
