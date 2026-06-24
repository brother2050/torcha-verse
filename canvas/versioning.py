"""Canvas version control for the TorchaVerse v0.3.0 architecture (L5).

This module provides a Git-like versioning system for canvases.  Every
:meth:`~canvas.canvas.Canvas.snapshot` of a canvas can be *committed* as an
immutable :class:`CanvasVersion`, forming a linear history (with optional
branching).  Versions can be inspected, compared, checked out and reverted.

The versioning layer is *torch-free* and depends only on the canvas core
(:mod:`canvas.canvas`) and the Python standard library.

Public surface
--------------

* :class:`CanvasVersion` -- an immutable snapshot of a canvas at a point in
  time (version id, timestamp, author, message, state, parent id).
* :class:`CanvasHistory` -- a thread-safe history manager that tracks a
  :class:`~canvas.canvas.Canvas`, supports commit / log / checkout / diff /
  revert / branch / merge / tag operations.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .canvas import Canvas, CanvasState

__all__ = ["CanvasVersion", "CanvasHistory"]

# ---------------------------------------------------------------------------
# Module-level logger.
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger("canvas.versioning")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Default author name when none is provided.
_DEFAULT_AUTHOR: str = "user"
#: Prefix used for branch canvas names.
_BRANCH_NAME_PREFIX: str = "branch:"


# ---------------------------------------------------------------------------
# CanvasVersion
# ---------------------------------------------------------------------------
@dataclass
class CanvasVersion:
    """An immutable snapshot of a canvas at a point in time.

    A version captures the full :class:`CanvasState` along with metadata
    (who committed it, when, and why).  Versions form a linear chain via
    the ``parent_id`` field, enabling history traversal and diffing.

    Attributes:
        version_id: Unique identifier for this version.
        timestamp: POSIX timestamp at which the version was committed.
        author: Name of the user who committed the version.
        message: Human-readable commit message.
        state: The :class:`CanvasState` captured by this version.
        parent_id: Id of the parent version (``None`` for the initial commit).
    """

    version_id: str
    timestamp: float
    author: str
    message: str
    state: CanvasState
    parent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this version to a JSON-serialisable dictionary."""
        return {
            "version_id": self.version_id,
            "timestamp": self.timestamp,
            "author": self.author,
            "message": self.message,
            "state": self.state.to_dict(),
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CanvasVersion":
        """Reconstruct a :class:`CanvasVersion` from a serialised dict.

        Args:
            d: Dictionary produced by :meth:`to_dict`.

        Returns:
            A new :class:`CanvasVersion` instance.
        """
        return cls(
            version_id=d["version_id"],
            timestamp=float(d["timestamp"]),
            author=d.get("author", _DEFAULT_AUTHOR),
            message=d.get("message", ""),
            state=CanvasState.from_dict(d.get("state") or {}),
            parent_id=d.get("parent_id"),
        )

    def __repr__(self) -> str:
        return "CanvasVersion(id={!r}, message={!r}, parent={!r})".format(
            self.version_id, self.message, self.parent_id
        )


# ---------------------------------------------------------------------------
# CanvasHistory
# ---------------------------------------------------------------------------
class CanvasHistory:
    """A thread-safe version-history manager for a :class:`Canvas`.

    A :class:`CanvasHistory` tracks a single canvas and records immutable
    :class:`CanvasVersion` snapshots each time :meth:`commit` is called.
    It supports the standard version-control operations: log, checkout,
    diff, revert, branch, merge and tag.

    The history is stored in-memory and is not persisted.  All operations
    are guarded by a re-entrant lock for thread safety.

    Args:
        canvas: The :class:`Canvas` to track.  The history holds a reference
            to this canvas and updates its state on revert / merge.
    """

    def __init__(self, canvas: Canvas) -> None:
        if not isinstance(canvas, Canvas):
            raise TypeError("canvas must be a Canvas instance.")
        self._canvas: Canvas = canvas
        self._versions: List[CanvasVersion] = []
        self._tags: Dict[str, str] = {}  # tag_name -> version_id
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def canvas(self) -> Canvas:
        """The tracked :class:`Canvas`."""
        return self._canvas

    @property
    def version_count(self) -> int:
        """The number of committed versions."""
        with self._lock:
            return len(self._versions)

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------
    def commit(
        self, message: str, author: str = _DEFAULT_AUTHOR
    ) -> CanvasVersion:
        """Commit the current canvas state as a new version.

        Args:
            message: Human-readable commit message.
            author: Name of the committer.

        Returns:
            The newly created :class:`CanvasVersion`.
        """
        with self._lock:
            parent_id = (
                self._versions[-1].version_id if self._versions else None
            )
            state = CanvasState.from_dict(self._canvas.state.to_dict())
            version = CanvasVersion(
                version_id=str(uuid4()),
                timestamp=time.time(),
                author=author,
                message=message,
                state=state,
                parent_id=parent_id,
            )
            self._versions.append(version)
            _logger.debug(
                "Committed version %r (message=%r).",
                version.version_id, message,
            )
            return version

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------
    def log(self) -> List[CanvasVersion]:
        """Return the full version history (oldest first).

        Returns:
            A list of :class:`CanvasVersion` in chronological order.
        """
        with self._lock:
            return list(self._versions)

    # ------------------------------------------------------------------
    # Checkout
    # ------------------------------------------------------------------
    def checkout(self, version_id: str) -> CanvasState:
        """Return the state of a specific version (without modifying the canvas).

        Args:
            version_id: The id of the version to check out.

        Returns:
            A copy of the version's :class:`CanvasState`.

        Raises:
            KeyError: If no version with ``version_id`` exists.
        """
        with self._lock:
            version = self._find_version(version_id)
            if version is None:
                raise KeyError(
                    "No version with id {!r}.".format(version_id)
                )
            return CanvasState.from_dict(version.state.to_dict())

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------
    def diff(self, v1_id: str, v2_id: str) -> Dict[str, Any]:
        """Compare two versions and return their differences.

        The diff reports added / removed / modified nodes and added /
        removed connections between the two versions.

        Args:
            v1_id: The id of the first (older) version.
            v2_id: The id of the second (newer) version.

        Returns:
            A dictionary with keys ``added_nodes``, ``removed_nodes``,
            ``modified_nodes``, ``added_connections`` and
            ``removed_connections``.

        Raises:
            KeyError: If either version id is not found.
        """
        with self._lock:
            v1 = self._find_version(v1_id)
            v2 = self._find_version(v2_id)
            if v1 is None:
                raise KeyError("No version with id {!r}.".format(v1_id))
            if v2 is None:
                raise KeyError("No version with id {!r}.".format(v2_id))

            v1_nodes = {n.id: n for n in v1.state.nodes}
            v2_nodes = {n.id: n for n in v2.state.nodes}

            added_nodes = [
                n.to_dict()
                for nid, n in v2_nodes.items()
                if nid not in v1_nodes
            ]
            removed_nodes = [
                n.to_dict()
                for nid, n in v1_nodes.items()
                if nid not in v2_nodes
            ]
            modified_nodes: List[Dict[str, Any]] = []
            for nid, n2 in v2_nodes.items():
                if nid in v1_nodes:
                    n1 = v1_nodes[nid]
                    if (
                        n1.type != n2.type
                        or n1.inputs != n2.inputs
                    ):
                        modified_nodes.append(
                            {
                                "id": nid,
                                "from": n1.to_dict(),
                                "to": n2.to_dict(),
                            }
                        )

            v1_conns = {
                (c.from_node, c.from_port, c.to_node, c.to_port): c
                for c in v1.state.connections
            }
            v2_conns = {
                (c.from_node, c.from_port, c.to_node, c.to_port): c
                for c in v2.state.connections
            }

            added_conns = [
                c.to_dict()
                for key, c in v2_conns.items()
                if key not in v1_conns
            ]
            removed_conns = [
                c.to_dict()
                for key, c in v1_conns.items()
                if key not in v2_conns
            ]

            return {
                "added_nodes": added_nodes,
                "removed_nodes": removed_nodes,
                "modified_nodes": modified_nodes,
                "added_connections": added_conns,
                "removed_connections": removed_conns,
            }

    # ------------------------------------------------------------------
    # Revert
    # ------------------------------------------------------------------
    def revert(self, version_id: str) -> None:
        """Revert the tracked canvas to a historical version.

        This replaces the canvas's current state with a copy of the
        specified version's state.  A new commit is *not* automatically
        created; call :meth:`commit` afterwards to record the revert.

        Args:
            version_id: The id of the version to revert to.

        Raises:
            KeyError: If no version with ``version_id`` exists.
        """
        with self._lock:
            version = self._find_version(version_id)
            if version is None:
                raise KeyError(
                    "No version with id {!r}.".format(version_id)
                )
            new_state = CanvasState.from_dict(
                version.state.to_dict()
            )
            self._canvas._replace_state(new_state)
            _logger.info(
                "Reverted canvas %r to version %r.",
                self._canvas.name, version_id,
            )

    # ------------------------------------------------------------------
    # Branch
    # ------------------------------------------------------------------
    def branch(self, name: str) -> Canvas:
        """Create a branch (a forked copy) of the current canvas state.

        The branch is an independent :class:`Canvas` whose initial state
        is a deep copy of the tracked canvas's current state.  Changes to
        the branch do not affect the tracked canvas until
        :meth:`merge` is called.

        Args:
            name: Name for the branch canvas.

        Returns:
            A new :class:`Canvas` representing the branch.
        """
        with self._lock:
            return self._canvas.fork(_BRANCH_NAME_PREFIX + name)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    def merge(self, branch: Canvas) -> None:
        """Merge a branch canvas back into the tracked canvas.

        The branch's nodes are added to the tracked canvas with a prefix
        to avoid id collisions, and its connections are rewired
        accordingly.  After the merge, the tracked canvas contains nodes
        and connections from both itself and the branch.

        Args:
            branch: The branch :class:`Canvas` to merge in.
        """
        with self._lock:
            merged = self._canvas.merge(branch)
            new_state = CanvasState.from_dict(
                merged.state.to_dict()
            )
            self._canvas._replace_state(new_state)
            _logger.info(
                "Merged branch into canvas %r.", self._canvas.name
            )

    # ------------------------------------------------------------------
    # Tag
    # ------------------------------------------------------------------
    def tag(self, version_id: str, name: str) -> None:
        """Tag a version with a human-readable name.

        Tags provide a stable, memorable alias for a version id.  A tag
        can later be used to look up the version (though this class does
        not expose a tag-to-version lookup method; callers can inspect
        the ``tags`` mapping via :attr:`tags` if needed).

        Args:
            version_id: The id of the version to tag.
            name: The tag name.

        Raises:
            KeyError: If no version with ``version_id`` exists.
        """
        with self._lock:
            version = self._find_version(version_id)
            if version is None:
                raise KeyError(
                    "No version with id {!r}.".format(version_id)
                )
            self._tags[name] = version_id
            _logger.debug(
                "Tagged version %r as %r.", version_id, name
            )

    @property
    def tags(self) -> Dict[str, str]:
        """Return a copy of the tag-name -> version-id mapping."""
        with self._lock:
            return dict(self._tags)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _find_version(self, version_id: str) -> Optional[CanvasVersion]:
        """Return the version with ``version_id`` or ``None``.

        The caller must already hold ``self._lock``.
        """
        for version in self._versions:
            if version.version_id == version_id:
                return version
        return None

    def __repr__(self) -> str:
        with self._lock:
            return "CanvasHistory(canvas={!r}, versions={})".format(
                self._canvas.name, len(self._versions)
            )
