"""Module assembly bus for TorchaVerse v0.3.0.

This module provides :class:`ModuleBus`, a thread-safe singleton registry
that unifies the discovery, instantiation, and caching of every pluggable
component in the framework.  It is intended to replace the scattered
singletons (``ModelRegistry``, ``TokenizerHub``, ``KVCacheManager`` …)
with a single, namespace-aware assembly point.

Components are registered as *factories* -- zero-argument callables that
return the component instance -- under a ``(kind, name)`` pair.  ``kind``
uses dot-separated namespaces so that related modules can be grouped and
discovered hierarchically, for example::

    "model.text"        text models
    "model.image"       image models
    "node"              capability nodes (flat kind; the node type is the name)
    "tokenizer.text"    text tokenizers
    "lora"              LoRA adapters
    "character"         character/persona definitions

Key features:

* :class:`ModuleSpec` -- description record of a registered module.
* :class:`ModuleBus` -- thread-safe singleton registry with factory
  caching.  The factory for a ``(kind, name, version)`` triple is invoked
  at most once until the cache is invalidated.
* :func:`register_module` -- decorator for convenient registration.
* :class:`ModuleNotFoundError` -- raised when a module cannot be resolved.

Design notes
------------
:class:`ModuleBus` sits at the very base of the dependency graph and is
therefore implemented with **no third-party dependencies** -- it only
uses the standard library.  In particular it does *not* import
:mod:`infrastructure.logger` (which would transitively pull in
``torch``); the standard :mod:`logging` module is used instead.  This
keeps the bus importable in any environment, including minimal CI
sandboxes.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "ModuleSpec",
    "ModuleBus",
    "ModuleNotFoundError",
    "register_module",
]


# ---------------------------------------------------------------------------
# Module-level logger (stdlib only -- see module docstring).
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger("ModuleBus")


#: Type alias for a module factory callable.
ModuleFactory = Callable[..., Any]

#: Type alias for a registry key ``(kind, name)``.
_RegistryKey = Tuple[str, str]

#: Type alias for a cache key ``(kind, name, version)``.
_CacheKey = Tuple[str, str, str]

#: Sentinel used to distinguish a cached ``None`` from an absent entry.
_MISSING: Any = object()

#: Default version applied when none is supplied.
_DEFAULT_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ModuleNotFoundError(KeyError):
    """Raised when a module cannot be resolved by :class:`ModuleBus`.

    This is a framework-level error (unrelated to Python's built-in
    ``ModuleNotFoundError`` for failed ``import`` statements).  It is a
    subclass of :class:`KeyError` so that callers may catch it together
    with ordinary lookup failures.

    Args:
        kind: The module kind that was requested.
        name: The module name that was requested.
    """

    def __init__(self, kind: str, name: str) -> None:
        self.kind: str = kind
        self.name: str = name
        message = "No module registered for kind={!r} name={!r}.".format(kind, name)
        super().__init__(message)

    def __str__(self) -> str:
        return "ModuleNotFoundError: kind={!r} name={!r}".format(self.kind, self.name)


# ---------------------------------------------------------------------------
# ModuleSpec
# ---------------------------------------------------------------------------
@dataclass
class ModuleSpec:
    """Description record of a registered module.

    Instances should be treated as immutable once published to the bus.

    Attributes:
        kind: Dot-separated namespace of the module, e.g.
            ``"model.text"`` or ``"lora"``.
        name: Unique name of the module within its kind.
        version: Semantic version string of the module.
        factory: Zero-argument callable that constructs the module
            instance.
        description: Human-readable description.
        tags: Optional list of free-form tags.
    """

    kind: str
    name: str
    version: str
    factory: ModuleFactory
    description: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def key(self) -> _RegistryKey:
        """Return the ``(kind, name)`` registry key for this spec."""
        return (self.kind, self.name)

    def __repr__(self) -> str:
        return (
            "ModuleSpec(kind={!r}, name={!r}, version={!r}, tags={!r})".format(
                self.kind, self.name, self.version, self.tags
            )
        )


# ---------------------------------------------------------------------------
# ModuleBus
# ---------------------------------------------------------------------------
class ModuleBus:
    """Thread-safe singleton registry and factory cache.

    The bus replaces the collection of scattered singletons
    (``ModelRegistry``, ``TokenizerHub``, ``KVCacheManager`` …) with a
    single, namespace-aware assembly point.  Modules are registered as
    factories under a ``(kind, name)`` pair where ``kind`` is a
    dot-separated namespace (e.g. ``"model.text"``,
    ``"node"``, ``"lora"``, ``"character"``).

    Resolving a module invokes its factory at most once per
    ``(kind, name, version)`` triple; subsequent resolves return the
    cached instance until the cache is invalidated.

    Thread safety
    -------------
    A re-entrant lock (:class:`threading.RLock`) guards the registry and
    cache dictionaries.  Factory invocation, however, happens *outside*
    the global lock and is serialised per cache key with a dedicated
    :class:`threading.Lock`.  This guarantees that a (potentially
    expensive) factory runs at most once per key while still allowing
    unrelated resolves to proceed concurrently.

    Example:
        >>> bus = ModuleBus()
        >>> bus.register("model.text", "llama", lambda: {"weights": 1})
        >>> model = bus.resolve("model.text", "llama")
        >>> bus.has("model.text", "llama")
        True
        >>> [s.name for s in bus.list("model")]
        ['llama']
    """

    _instance: Optional["ModuleBus"] = None
    _initialized: bool = False
    _singleton_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton plumbing
    # ------------------------------------------------------------------
    def __new__(cls, *args: Any, **kwargs: Any) -> "ModuleBus":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:  # double-check
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Fast path: already initialised -- avoid the lock entirely.
        if self._initialized:
            return
        # Run the whole initialisation under ``_singleton_lock`` (the same
        # lock used by ``__new__``) so that two concurrent ``ModuleBus()``
        # calls cannot both pass the ``_initialized`` check (TOCTOU).
        with self._singleton_lock:
            if self._initialized:
                return
            self._initialized = True

            self._registry: Dict[_RegistryKey, ModuleSpec] = {}
            self._cache: Dict[_CacheKey, Any] = {}
            self._factory_locks: Dict[_CacheKey, threading.Lock] = {}
            self._lock: threading.RLock = threading.RLock()
            self._logger: logging.Logger = _logger

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(value: str) -> str:
        """Strip surrounding whitespace from a kind/name token.

        Args:
            value: The token to normalise.

        Returns:
            The stripped token.

        Raises:
            TypeError: If ``value`` is not a string.
        """
        if not isinstance(value, str):
            raise TypeError(
                "Expected str, got {}.".format(type(value).__name__)
            )
        return value.strip()

    def _key(self, kind: str, name: str) -> _RegistryKey:
        """Build a normalised ``(kind, name)`` registry key."""
        return (self._normalize(kind), self._normalize(name))

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(
        self,
        kind: str,
        name: str,
        factory: Callable[..., Any],
        version: str = _DEFAULT_VERSION,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Register a module factory.

        Re-registering an existing ``(kind, name)`` pair replaces the
        previous spec and invalidates any cached instances for that pair
        so the new factory is observed immediately.

        Args:
            kind: Dot-separated namespace, e.g. ``"model.text"``.
            name: Unique name within the kind.
            factory: Zero-argument callable returning the module
                instance.
            version: Semantic version string.
            description: Human-readable description.
            tags: Optional list of free-form tags.

        Raises:
            ValueError: If ``kind``, ``name`` or ``version`` is empty.
            TypeError: If ``factory`` is not callable.
        """
        nkind = self._normalize(kind)
        nname = self._normalize(name)
        if not nkind:
            raise ValueError("Module kind must be a non-empty string.")
        if not nname:
            raise ValueError("Module name must be a non-empty string.")
        version_str = str(version).strip() if version is not None else ""
        if not version_str:
            raise ValueError("Module version must be a non-empty string.")
        if not callable(factory):
            raise TypeError("factory must be callable.")

        spec = ModuleSpec(
            kind=nkind,
            name=nname,
            version=version_str,
            factory=factory,
            description=description or "",
            tags=list(tags) if tags else [],
        )

        with self._lock:
            existing = self._registry.get((nkind, nname))
            self._registry[(nkind, nname)] = spec
            if existing is not None:
                # Drop stale cached instances so a re-registration with a
                # new factory/version is observed immediately.
                self._invalidate_locked(nkind, nname)
                self._logger.debug(
                    "Re-registered module kind=%s name=%s (was v%s).",
                    nkind, nname, existing.version,
                )
            self._logger.debug(
                "Registered module kind=%s name=%s version=%s.",
                nkind, nname, spec.version,
            )

    # ------------------------------------------------------------------
    # Unregistration
    # ------------------------------------------------------------------
    def unregister(self, kind: str, name: str) -> bool:
        """Remove a module from the registry and drop its cached instances.

        This is the inverse of :meth:`register`.  After the call the
        ``(kind, name)`` pair is no longer resolvable and any cached
        instances produced by its factory are discarded.  It is used by
        the plugin system to unload a plugin's nodes.

        Args:
            kind: Dot-separated namespace, e.g. ``"node"``.
            name: Unique name within the kind.

        Returns:
            ``True`` if a module was removed, ``False`` if nothing was
            registered for ``(kind, name)``.
        """
        nkind = self._normalize(kind)
        nname = self._normalize(name)
        with self._lock:
            existed = (nkind, nname) in self._registry
            if existed:
                del self._registry[(nkind, nname)]
            # Drop cached instances / per-key locks for this pair.
            self._invalidate_locked(nkind, nname)
        if existed:
            self._logger.debug(
                "Unregistered module kind=%s name=%s.", nkind, nname
            )
        return existed

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def resolve(
        self,
        kind: str,
        name: str,
        version: Optional[str] = None,
    ) -> Any:
        """Resolve and return a module instance (cached).

        The factory for ``(kind, name, version)`` is invoked at most
        once; subsequent calls return the cached instance until
        :meth:`invalidate` or :meth:`invalidate_all` is called.

        Args:
            kind: Dot-separated namespace.
            name: Unique name within the kind.
            version: Optional explicit version.  When ``None`` the
                version registered with the spec is used.

        Returns:
            The module instance produced by the registered factory.

        Raises:
            ModuleNotFoundError: If no module is registered for
                ``(kind, name)``.
        """
        nkind = self._normalize(kind)
        nname = self._normalize(name)
        key: _RegistryKey = (nkind, nname)

        with self._lock:
            spec = self._registry.get(key)
            if spec is None:
                raise ModuleNotFoundError(kind, name)
            effective_version = (
                str(version).strip() if version is not None else spec.version
            )
            cache_key: _CacheKey = (nkind, nname, effective_version)
            cached = self._cache.get(cache_key, _MISSING)
            if cached is not _MISSING:
                return cached
            # Ensure a per-key lock exists for serialised factory
            # invocation.  ``setdefault`` is atomic under the global lock.
            factory_lock = self._factory_locks.setdefault(
                cache_key, threading.Lock()
            )

        # Invoke the factory outside the global lock to avoid blocking
        # unrelated resolves, but under a per-key lock so the factory
        # runs at most once per (kind, name, version).
        with factory_lock:
            # Double-check after acquiring the per-key lock: another
            # thread may have produced the instance already.
            with self._lock:
                cached = self._cache.get(cache_key, _MISSING)
                if cached is not _MISSING:
                    return cached
            self._logger.debug(
                "Instantiating module kind=%s name=%s version=%s.",
                nkind, nname, effective_version,
            )
            instance = spec.factory()
            with self._lock:
                existing = self._cache.get(cache_key, _MISSING)
                if existing is not _MISSING:
                    # Another thread won the race; prefer its instance.
                    return existing
                self._cache[cache_key] = instance
            return instance

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def has(self, kind: str, name: str) -> bool:
        """Return ``True`` if a module is registered for ``(kind, name)``."""
        with self._lock:
            return self._key(kind, name) in self._registry

    def get_spec(self, kind: str, name: str) -> ModuleSpec:
        """Return the :class:`ModuleSpec` registered for ``(kind, name)``.

        Args:
            kind: Dot-separated namespace.
            name: Unique name within the kind.

        Returns:
            The :class:`ModuleSpec` for the requested module.

        Raises:
            ModuleNotFoundError: If no module is registered.
        """
        nkind = self._normalize(kind)
        nname = self._normalize(name)
        with self._lock:
            spec = self._registry.get((nkind, nname))
        if spec is None:
            raise ModuleNotFoundError(kind, name)
        return spec

    def list(self, kind: Optional[str] = None) -> List[ModuleSpec]:
        """List registered module specs.

        Args:
            kind: When given, filter by exact kind or by namespace
                prefix.  For example ``list("model")`` returns specs
                whose kind is ``"model"`` or starts with ``"model."``,
                i.e. it descends one namespace level.  When ``None``
                all specs are returned.

        Returns:
            A list of :class:`ModuleSpec` sorted by ``(kind, name)``.
        """
        with self._lock:
            specs = list(self._registry.values())
        if kind is not None:
            prefix = self._normalize(kind)
            specs = [
                spec
                for spec in specs
                if spec.kind == prefix or spec.kind.startswith(prefix + ".")
            ]
        specs.sort(key=lambda spec: (spec.kind, spec.name))
        return specs

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------
    def invalidate(self, kind: str, name: str) -> None:
        """Invalidate cached instances for ``(kind, name)`` (all versions).

        The module remains registered; only cached instances are
        dropped so the next :meth:`resolve` re-invokes the factory.

        Args:
            kind: Dot-separated namespace.
            name: Unique name within the kind.
        """
        nkind = self._normalize(kind)
        nname = self._normalize(name)
        with self._lock:
            self._invalidate_locked(nkind, nname)

    def _invalidate_locked(self, nkind: str, nname: str) -> None:
        """Drop cache entries for ``(nkind, nname)`` (caller holds lock)."""
        stale = [
            key for key in self._cache
            if key[0] == nkind and key[1] == nname
        ]
        for key in stale:
            del self._cache[key]
            self._factory_locks.pop(key, None)
        if stale:
            self._logger.debug(
                "Invalidated %d cached instance(s) for kind=%s name=%s.",
                len(stale), nkind, nname,
            )

    def invalidate_all(self) -> None:
        """Clear every cached instance.

        Modules remain registered; only cached instances are dropped.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._factory_locks.clear()
        if count:
            self._logger.debug("Invalidated all %d cached instance(s).", count)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def count(self) -> int:
        """Return the number of registered modules."""
        with self._lock:
            return len(self._registry)

    def __repr__(self) -> str:
        with self._lock:
            return (
                "ModuleBus(modules={}, cached={})".format(
                    len(self._registry), len(self._cache)
                )
            )

    # ------------------------------------------------------------------
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing).

        After calling this, ``ModuleBus()`` returns a fresh, empty bus.
        """
        with cls._singleton_lock:
            cls._instance = None
            cls._initialized = False


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------
def register_module(
    kind: str,
    name: str,
    *,
    version: str = _DEFAULT_VERSION,
    description: str = "",
    tags: Optional[List[str]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a factory with the global :class:`ModuleBus`.

    Supports both positional and keyword invocation forms::

        @register_module("model.text", "llama")
        def make_llama():
            ...

        @register_module(kind="model.text", name="llama")
        def make_llama():
            ...

    The decorated callable is returned unchanged so it can still be
    called directly if desired.

    Args:
        kind: Dot-separated namespace.
        name: Unique name within the kind.
        version: Semantic version string.
        description: Human-readable description.
        tags: Optional list of free-form tags.

    Returns:
        A decorator that registers the factory and returns it unchanged.
    """

    def _decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        ModuleBus().register(
            kind,
            name,
            factory,
            version=version,
            description=description,
            tags=tags,
        )
        return factory

    return _decorator
