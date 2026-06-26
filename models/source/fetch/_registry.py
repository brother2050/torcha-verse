"""Source-registry for the unified model fetcher (v0.6.x).

This module is an internal sub-module of :mod:`models.source.fetch`;
importing it directly is supported but the canonical entry point
remains :func:`models.source.fetch.fetch`.

The :class:`SourceRegistry` maps a friendly source name (e.g.
``"huggingface"`` / ``"hf"``) to a concrete adapter instance.  The
default registry is built by :meth:`SourceRegistry.default` and
contains a :class:`~models.source.huggingface.HuggingFaceSource` and
a :class:`~models.source.civitai.CivitaiSource` -- both backed by
the stdlib :class:`~models.source.huggingface.UrllibTransport`.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

from infrastructure.logger import get_logger

from ..civitai import CivitaiSource
from ..huggingface import HuggingFaceSource

__all__ = [
    "SourceRegistry",
    "SOURCE_ALIASES",
]


#: Module-level logger.
_logger = get_logger("models.source.fetch")


#: Mapping of public source name -> canonical source id used in the
#: cache directory layout.
SOURCE_ALIASES: Dict[str, str] = {
    "huggingface": "huggingface",
    "hf": "huggingface",
    "civitai": "civitai",
    "cv": "civitai",
}


class SourceRegistry:
    """A tiny registry that maps source names to adapter instances.

    The default registry contains a :class:`HuggingFaceSource` and a
    :class:`CivitaiSource` constructed with the default
    :class:`UrllibTransport`.  Tests can build a registry with
    custom adapters (e.g. backed by a fake transport) and pass it to
    :class:`ModelFetcher` via the ``registry=`` keyword.
    """

    def __init__(self) -> None:
        self._sources: Dict[str, Any] = {}
        self._lock: threading.Lock = threading.Lock()

    def register(self, name: str, adapter: Any) -> None:
        """Register an adapter under ``name`` (overwrites if present)."""
        if not name or not name.strip():
            raise ValueError("`name` must be non-empty")
        with self._lock:
            self._sources[name.strip()] = adapter

    def get(self, name: str) -> Any:
        """Return the adapter registered under ``name``.

        Looks up the *canonical* source id first, then falls back to
        the alias table.  Raises :class:`KeyError` when no adapter
        is registered.
        """
        canonical = SOURCE_ALIASES.get(name.strip(), name.strip())
        with self._lock:
            if canonical in self._sources:
                return self._sources[canonical]
            if name.strip() in self._sources:
                return self._sources[name.strip()]
        raise KeyError("no source adapter registered for {!r}".format(name))

    def available(self) -> List[str]:
        """Return the list of registered source names."""
        with self._lock:
            return sorted(self._sources.keys())

    @classmethod
    def default(cls) -> "SourceRegistry":
        """Build a registry with the standard HuggingFace + Civitai adapters."""
        reg = cls()
        reg.register("huggingface", HuggingFaceSource())
        reg.register("civitai", CivitaiSource())
        return reg
