"""Paper registry singleton for the TorchaVerse paper integration system.

This module provides :class:`PaperRegistry`, a thread-safe singleton
registry -- modelled on :class:`core.module_bus.ModuleBus` -- that holds
the catalogue of :class:`~papers.spec.PaperSpec` records known to the
framework.  Papers are registered in-memory and/or loaded from YAML
files on disk via :meth:`PaperRegistry.load_from_dir`.

Layering: ``papers`` sits alongside the L4 ``nodes`` package and depends
only on the dependency-free :mod:`papers.spec` module plus
:mod:`yaml` (PyYAML, already a framework dependency).  It does **not**
import ``torch`` or any L1/L2/L3 module, so it is importable in any
environment, including minimal CI sandboxes.

Public surface
--------------
* :class:`PaperRegistry` -- thread-safe singleton registry of papers.
* :class:`PaperNotFoundError` -- raised when a paper cannot be resolved.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .spec import PaperSpec

__all__ = [
    "PaperRegistry",
    "PaperNotFoundError",
]


# ---------------------------------------------------------------------------
# Module-level logger (stdlib only -- no torch dependency).
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger("PaperRegistry")


#: Supported YAML file suffixes.
_YAML_SUFFIXES: tuple[str, ...] = (".yaml", ".yml")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class PaperNotFoundError(KeyError):
    """Raised when a paper cannot be resolved by :class:`PaperRegistry`.

    Subclass of :class:`KeyError` so callers may catch it together with
    ordinary lookup failures.

    Args:
        name: The paper name that was requested.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        message = "No paper registered for name={!r}.".format(name)
        super().__init__(message)

    def __str__(self) -> str:
        return "PaperNotFoundError: name={!r}".format(self.name)


# ---------------------------------------------------------------------------
# PaperRegistry
# ---------------------------------------------------------------------------
class PaperRegistry:
    """Thread-safe singleton registry of :class:`PaperSpec` records.

    The registry is the discovery surface for the paper integration
    system.  Like :class:`core.module_bus.ModuleBus` it is a process-wide
    singleton (``PaperRegistry()`` always returns the same instance) with
    a :meth:`reset` classmethod for testing.

    Papers are keyed by their ``name``.  :meth:`search` performs a
    case-insensitive substring match across the bibliographic and
    integration fields.  :meth:`load_from_dir` bulk-loads every
    ``*.yaml`` / ``*.yml`` file in a directory, and :meth:`load_bundled`
    loads the YAML files shipped inside this package.

    Example::

        from papers import PaperRegistry
        registry = PaperRegistry()
        registry.load_bundled()           # load shipped papers
        for spec in registry.list():
            print(spec.name, spec.title)
        musetalk = registry.get("musetalk")
        hits = registry.search("lip sync")
    """

    _instance: Optional["PaperRegistry"] = None
    _initialized: bool = False
    _singleton_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton plumbing
    # ------------------------------------------------------------------
    def __new__(cls, *args: Any, **kwargs: Any) -> "PaperRegistry":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:  # double-check
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._papers: Dict[str, PaperSpec] = {}
        self._lock: threading.RLock = threading.RLock()
        self._logger: logging.Logger = _logger

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(value: str) -> str:
        """Strip surrounding whitespace from a name token."""
        if not isinstance(value, str):
            raise TypeError(
                "Expected str, got {}.".format(type(value).__name__)
            )
        return value.strip()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, spec: PaperSpec) -> None:
        """Register a :class:`PaperSpec`.

        Re-registering an existing name replaces the previous entry.

        Args:
            spec: The paper specification to register.

        Raises:
            ValueError: If ``spec.name`` is empty.
            TypeError: If ``spec`` is not a :class:`PaperSpec`.
        """
        if not isinstance(spec, PaperSpec):
            raise TypeError(
                "spec must be a PaperSpec, got {!r}.".format(spec)
            )
        name = self._normalize(spec.name)
        if not name:
            raise ValueError("PaperSpec.name must be a non-empty string.")

        with self._lock:
            self._papers[name] = spec
        self._logger.debug("Registered paper name=%s.", name)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def get(self, name: str) -> PaperSpec:
        """Return the :class:`PaperSpec` registered for ``name``.

        Args:
            name: The paper name.

        Returns:
            The :class:`PaperSpec`.

        Raises:
            PaperNotFoundError: If no paper is registered for ``name``.
        """
        key = self._normalize(name)
        with self._lock:
            spec = self._papers.get(key)
        if spec is None:
            raise PaperNotFoundError(name)
        return spec

    def has(self, name: str) -> bool:
        """Return ``True`` if a paper is registered for ``name``."""
        return self._normalize(name) in self._papers

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def list(self) -> List[PaperSpec]:
        """Return every registered :class:`PaperSpec`, sorted by name.

        Returns:
            A list of :class:`PaperSpec` sorted alphabetically by name.
        """
        with self._lock:
            specs = list(self._papers.values())
        specs.sort(key=lambda s: s.name)
        return specs

    def search(self, query: str) -> List[PaperSpec]:
        """Fuzzy-search papers by substring.

        The match is case-insensitive and matches any paper whose name,
        title, authors, arxiv id, method, category or node_type
        *contains* the query substring.  An empty query returns every
        paper (same as :meth:`list`).

        Args:
            query: Substring to search for.

        Returns:
            A list of matching :class:`PaperSpec` sorted by name.
        """
        needle = (query or "").strip().lower()
        if not needle:
            return self.list()

        results: List[PaperSpec] = []
        for spec in self.list():
            haystack = " ".join(
                [
                    spec.name,
                    spec.title,
                    " ".join(spec.authors),
                    spec.arxiv_id,
                    spec.method,
                    spec.category,
                    spec.node_type,
                    spec.integration_type,
                ]
            ).lower()
            if needle in haystack:
                results.append(spec)
        return results

    # ------------------------------------------------------------------
    # Bulk loading
    # ------------------------------------------------------------------
    def load_from_dir(self, dir: Union[str, Path]) -> int:
        """Load every YAML paper spec from a directory.

        Each ``*.yaml`` / ``*.yml`` file is parsed and registered under
        its ``paper.name``.  Files that fail to parse are skipped with a
        warning so a single malformed file does not abort the whole load.

        Args:
            dir: Path to the directory containing paper YAML files.

        Returns:
            The number of papers successfully loaded.

        Raises:
            FileNotFoundError: If ``dir`` does not exist.
            NotADirectoryError: If ``dir`` is not a directory.
        """
        directory = Path(dir)
        if not directory.exists():
            raise FileNotFoundError(
                "Paper directory does not exist: {}".format(directory)
            )
        if not directory.is_dir():
            raise NotADirectoryError(
                "Not a directory: {}".format(directory)
            )

        loaded = 0
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in _YAML_SUFFIXES:
                continue
            try:
                spec = self._load_file(path)
            except Exception as exc:  # noqa: BLE001 - log & continue
                self._logger.warning(
                    "Skipping paper file %s: %s", path, exc
                )
                continue
            if spec is not None:
                self.register(spec)
                loaded += 1
        self._logger.debug(
            "Loaded %d paper(s) from %s.", loaded, directory
        )
        return loaded

    def load_bundled(self) -> int:
        """Load the YAML paper specs shipped inside this package.

        Returns:
            The number of bundled papers loaded.
        """
        bundled_dir = Path(__file__).resolve().parent
        return self.load_from_dir(bundled_dir)

    @staticmethod
    def _load_file(path: Path) -> Optional[PaperSpec]:
        """Parse a single YAML file into a :class:`PaperSpec`.

        Args:
            path: Path to the YAML file.

        Returns:
            The parsed :class:`PaperSpec`, or ``None`` if the file is
            empty / contains no ``paper`` section.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            return None
        if not data.get("paper"):
            return None
        return PaperSpec.from_dict(data)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def count(self) -> int:
        """Return the number of registered papers."""
        with self._lock:
            return len(self._papers)

    def clear(self) -> None:
        """Remove every registered paper (modules remain loadable)."""
        with self._lock:
            n = len(self._papers)
            self._papers.clear()
        if n:
            self._logger.debug("Cleared %d paper(s).", n)

    def __repr__(self) -> str:
        with self._lock:
            return "PaperRegistry(papers={})".format(len(self._papers))

    # ------------------------------------------------------------------
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing).

        After calling this, ``PaperRegistry()`` returns a fresh, empty
        registry.
        """
        with cls._singleton_lock:
            cls._instance = None
            cls._initialized = False
