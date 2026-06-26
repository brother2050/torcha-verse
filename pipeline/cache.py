"""Fingerprint cache for the v0.9.0 incremental recompute path.

The v0.6.x / v0.8.x :class:`infrastructure.cache_store.CacheStore`
is a plain LRU+TTL store.  The v0.9.0 incremental-recompute
acceptance target in :doc:`/docs/V0.8_UPGRADE_PLAN.md` §5.2
requires a higher-level structure:

* per-input fingerprint (SHA-256 of the node args / kwargs / parent
  fingerprints)
* two LRU buckets -- ``outputs`` (per-DAG-node result) and
  ``objects`` (e.g. loaded model weights, VAE)
* a hierarchical invalidation rule: when a node's fingerprint
  changes, all of its descendants are invalidated as well

This module is intentionally small (no diffusers, no torch dep
on the hash side) and is meant to be plugged into the existing
:class:`pipeline.dag.DAG` and :class:`pipeline.composer.Pipeline`
via :func:`compute_fingerprint` + :class:`HierarchicalCache`.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = [
    "compute_fingerprint",
    "HierarchicalCache",
    "CacheStats",
]


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------
def _stable(obj: Any) -> Any:
    """Normalise ``obj`` into something JSON-serialisable with
    stable semantics.

    * ``torch.Tensor`` -> ``{"__tensor__": list(shape) + [dtype, str(data)]}``
    * ``bytes`` -> ``"sha256:..."``
    * ``set`` / ``frozenset`` -> sorted list
    * ``dict`` -> sorted tuple list
    * other -> ``obj`` (assumed JSON-safe)
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_stable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_stable(v) for v in obj)
    if isinstance(obj, dict):
        return sorted((_stable(k), _stable(v)) for k, v in obj.items())
    if isinstance(obj, bytes):
        return "sha256:" + hashlib.sha256(obj).hexdigest()
    # Fall back to repr (works for UUID, Path, dataclass).
    return f"repr:{obj!r}"


def compute_fingerprint(
    node_id: str,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    parent_fingerprints: Iterable[str] = (),
) -> str:
    """Return the SHA-256 fingerprint of a node's "logical input".

    The fingerprint is stable across process restarts as long as
    the args / kwargs are JSON-serialisable.  ``parent_fingerprints``
    is folded in so a child's fingerprint naturally invalidates
    when any parent changes (hierarchical cache).
    """
    payload = {
        "node_id": node_id,
        "args": _stable(args),
        "kwargs": _stable(kwargs or {}),
        "parents": sorted(parent_fingerprints),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Hierarchical cache
# ---------------------------------------------------------------------------
class CacheStats:
    """A snapshot of cache utilisation -- returned by
    :meth:`HierarchicalCache.stats`.
    """

    def __init__(self) -> None:
        self.hits: int = 0
        self.misses: int = 0
        self.invalidations: int = 0
        self.size_outputs: int = 0
        self.size_objects: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "invalidations": self.invalidations,
            "size_outputs": self.size_outputs,
            "size_objects": self.size_objects,
        }


class HierarchicalCache:
    """A two-bucket LRU cache with hierarchical invalidation.

    The cache holds two kinds of entries:

    * **outputs** -- the result of a DAG node, keyed by the
      node's fingerprint.  Hierarchical: when a parent node's
      fingerprint changes, the child's fingerprint changes too
      (because :func:`compute_fingerprint` folds the parent
      fingerprints in), so the cache automatically invalidates
      every descendant of the changed node.
    * **objects** -- heavier shared objects (e.g. loaded
      checkpoints, VAE), keyed by an explicit ``name`` plus the
      object's own fingerprint.  Invalidation here is opt-in
      via :meth:`invalidate_object`.

    Args:
        capacity_outputs: Max number of output entries to keep.
        capacity_objects: Max number of object entries to keep.
    """

    def __init__(
        self, capacity_outputs: int = 256, capacity_objects: int = 16,
    ) -> None:
        if capacity_outputs <= 0 or capacity_objects <= 0:
            raise ValueError("capacity must be positive")
        self._capacity_outputs = capacity_outputs
        self._capacity_objects = capacity_objects
        self._outputs: "OrderedDict[str, Any]" = OrderedDict()
        self._objects: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.RLock()
        self.stats = CacheStats()

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def get_output(self, fingerprint: str) -> Optional[Any]:
        with self._lock:
            if fingerprint not in self._outputs:
                self.stats.misses += 1
                return None
            self._outputs.move_to_end(fingerprint)
            self.stats.hits += 1
            return self._outputs[fingerprint]

    def put_output(self, fingerprint: str, value: Any) -> None:
        with self._lock:
            if fingerprint in self._outputs:
                self._outputs.move_to_end(fingerprint)
            self._outputs[fingerprint] = value
            while len(self._outputs) > self._capacity_outputs:
                self._outputs.popitem(last=False)

    # ------------------------------------------------------------------
    # Objects
    # ------------------------------------------------------------------
    def get_object(self, name: str) -> Optional[Any]:
        with self._lock:
            if name not in self._objects:
                self.stats.misses += 1
                return None
            self._objects.move_to_end(name)
            self.stats.hits += 1
            return self._objects[name]

    def put_object(self, name: str, value: Any) -> None:
        with self._lock:
            if name in self._objects:
                self._objects.move_to_end(name)
            self._objects[name] = value
            while len(self._objects) > self._capacity_objects:
                self._objects.popitem(last=False)

    def invalidate_object(self, name: str) -> None:
        with self._lock:
            if name in self._objects:
                del self._objects[name]
                self.stats.invalidations += 1

    def invalidate_all(self) -> None:
        with self._lock:
            self._outputs.clear()
            self._objects.clear()
            self.stats.invalidations += 1

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats_snapshot(self) -> CacheStats:
        with self._lock:
            snap = CacheStats()
            snap.hits = self.stats.hits
            snap.misses = self.stats.misses
            snap.invalidations = self.stats.invalidations
            snap.size_outputs = len(self._outputs)
            snap.size_objects = len(self._objects)
            return snap
