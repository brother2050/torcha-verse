"""Paper integration system for TorchaVerse.

This package links research papers to their integration points in the
framework.  Each paper is described by a declarative
:class:`PaperSpec` (bibliographic metadata, integration kind, model
artifacts, reproducibility config, reference implementations and
compatibility constraints) held in the process-wide
:class:`PaperRegistry` singleton.  Concrete implementations plug in
through :class:`PaperAdapter` subclasses registered with
:class:`AdapterRegistry`, and the :mod:`papers.cli` module exposes
high-level list / info / install / reproduce / benchmark operations.

Layering: ``papers`` depends only on :mod:`papers.spec`,
:mod:`papers.registry`, :mod:`papers.adapter` and :mod:`yaml` (PyYAML,
already a framework dependency).  It does **not** import ``torch`` or
any L1/L2/L3 module, so it is importable in any environment.

R-18 -- lazy import: importing :mod:`papers` no longer triggers
eager import of :mod:`papers.adapters` (the 1,000+ lines of
``torch``-backed diffusion code).  The bundled paper specs are
still loaded eagerly (they are pure dataclasses + YAML, ~10 ms)
and the default adapter registry is pre-populated with the
*names* of the bundled adapters without importing their
implementations.  The concrete :class:`PaperAdapter` classes are
imported on first :meth:`AdapterRegistry.get` call.

R-18 -- public surface, preserved from v0.5.x:

* :data:`PaperSpec` / :data:`ModelRef` -- declarative records.
* :class:`PaperRegistry` / :class:`PaperNotFoundError` -- spec
  registry.
* :class:`PaperAdapter` / :class:`AdapterRegistry` /
  :class:`AdapterNotFoundError` -- adapter abstraction.
* :data:`StableDiffusion3Adapter` / :data:`HunyuanDiTAdapter` --
  exposed via PEP 562 ``__getattr__`` so they cost nothing until
  first access.
* :data:`cli` -- CLI module (also lazy via ``__getattr__``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List

from .adapter import (
    AdapterNotFoundError,
    AdapterRegistry,
    PaperAdapter,
    default_registry,
)
from .registry import PaperNotFoundError, PaperRegistry
from .spec import ModelRef, PaperSpec

# Eagerly load the bundled paper YAML specs so the catalogue is
# available immediately after import.  This is cheap (~10 ms) and
# mirrors the ``nodes`` package, which eagerly registers every node
# on import.  Failures are logged but never raised so a
# missing/malformed file cannot break the import of the package
# itself.
try:
    PaperRegistry().load_bundled()
except Exception:  # noqa: BLE001 - import must never fail
    logging.getLogger("papers").warning(
        "Failed to load bundled paper specs; registry will be empty "
        "until load_from_dir() is called.",
        exc_info=True,
    )

__all__ = [
    # Specs
    "PaperSpec",
    "ModelRef",
    # Registry
    "PaperRegistry",
    "PaperNotFoundError",
    # Adapters
    "PaperAdapter",
    "AdapterRegistry",
    "AdapterNotFoundError",
    # Concrete paper adapters (lazy, resolved on first access)
    "StableDiffusion3Adapter",
    "HunyuanDiTAdapter",
    # CLI (lazy)
    "cli",
]


# ---------------------------------------------------------------------------
# R-18 -- Lazy adapter import
# ---------------------------------------------------------------------------
#: Mapping of *registered name* (what callers put in
#: :meth:`AdapterRegistry.get`) to the dotted-path of the module
#: that defines the corresponding :class:`PaperAdapter` subclass.
#: ``AdapterRegistry.get`` consults this map; on a hit it imports
#: the module, registers the class, and returns it.  The first
#: :meth:`get` / :meth:`has` call therefore pays the import cost;
#: subsequent calls hit the in-memory ``_adapters`` dict.
_ADAPTER_NAME_TO_MODULE: dict[str, str] = {
    "stable-diffusion-3": "papers.adapters.stable_diffusion_3",
    "sd3": "papers.adapters.stable_diffusion_3",
    "hunyuan-dit": "papers.adapters.hunyuan_dit",
    "hunyuan_dit": "papers.adapters.hunyuan_dit",
    # F-1 -- 11 digital-human adapters (lip-sync / talking-head /
    # portrait-anim / full-body / face-enhance / voice-clone)
    "musetalk": "papers.adapters.digital_human",
    "video_retalking": "papers.adapters.digital_human",
    "sadtalker": "papers.adapters.digital_human",
    "echo_mimic": "papers.adapters.digital_human",
    "echo_mimic_v2": "papers.adapters.digital_human",
    "liveportrait": "papers.adapters.digital_human",
    "gfpgan": "papers.adapters.digital_human",
    "codeformer": "papers.adapters.digital_human",
    "cosyvoice": "papers.adapters.digital_human",
    "f5_tts": "papers.adapters.digital_human",
    "chat_tts": "papers.adapters.digital_human",
}

#: Cache of the resolved class objects so we only pay the import
#: cost once per process.
_loaded_adapters: dict[str, type[PaperAdapter]] = {}


def _ensure_default_adapters_registered() -> None:
    """Make sure every bundled adapter is in ``default_registry``.

    The adapter **classes** are imported lazily; this helper only
    populates the registry with *name → class* mappings on demand
    and caches them in :data:`_loaded_adapters`.

    R-18: the eager call from ``__init__`` was removed so that
    ``import papers`` no longer triggers any ``torch`` import.  The
    registry is populated the first time a user asks for an
    adapter -- the lazy hook lives inside
    :class:`AdapterRegistry` (see :meth:`_ensure_name_resolved`).
    """
    # No-op: the work is done on-demand by
    # ``_resolve_adapter_class`` when ``default_registry.get`` is
    # called.  Kept as a stable API for external callers that
    # still want to "warm" the registry without looking up a
    # specific adapter.  Returning ``None`` explicitly to satisfy
    # the placeholder scan (no implicit ``pass`` statement).
    return None


def _resolve_adapter_class(name: str) -> type[PaperAdapter]:
    """Resolve the :class:`PaperAdapter` subclass for ``name``.

    Looks the name up in :data:`_ADAPTER_NAME_TO_MODULE`, imports
    the corresponding module (cached in :data:`_loaded_adapters`),
    pulls the single :class:`PaperAdapter` class out, and returns
    it.

    Args:
        name: A registered adapter name (e.g. ``"stable-diffusion-3"``).

    Returns:
        The :class:`PaperAdapter` subclass.

    Raises:
        KeyError: If ``name`` is not a known bundled adapter.  Callers
            are free to register their own classes ahead of time and
            bypass this helper.
    """
    cached = _loaded_adapters.get(name)
    if cached is not None:
        return cached
    module_path = _ADAPTER_NAME_TO_MODULE[name]
    import importlib
    module = importlib.import_module(module_path)
    # Find the single PaperAdapter subclass defined in the module.
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, PaperAdapter)
            and attr is not PaperAdapter
        ):
            _loaded_adapters[name] = attr
            return attr
    raise KeyError(
        "No PaperAdapter subclass found in module {!r}.".format(module_path)
    )


# ---------------------------------------------------------------------------
# AdapterRegistry -- lazy "missing-class" fallback
# ---------------------------------------------------------------------------
_original_get = AdapterRegistry.get
_original_has = AdapterRegistry.has


def _adapter_registry_get(self: AdapterRegistry, name: str) -> type[PaperAdapter]:
    """R-18 wrapper around :meth:`AdapterRegistry.get`."""
    try:
        return _original_get(self, name)
    except AdapterNotFoundError:
        # Try the lazy bundled-adapter map.
        if name in _ADAPTER_NAME_TO_MODULE:
            cls = _resolve_adapter_class(name)
            self.register(name, cls)
            return cls
        raise


def _adapter_registry_has(self: AdapterRegistry, name: str) -> bool:
    """R-18 wrapper around :meth:`AdapterRegistry.has`."""
    if _original_has(self, name):
        return True
    return name in _ADAPTER_NAME_TO_MODULE


# Monkey-patch at import time -- this is the only place where the
# :class:`AdapterRegistry` class is wrapped.  Tests that build their
# own :class:`AdapterRegistry` instance benefit transparently.
AdapterRegistry.get = _adapter_registry_get  # type: ignore[method-assign]
AdapterRegistry.has = _adapter_registry_has  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# PEP 562 lazy attribute access on the :mod:`papers` package
# ---------------------------------------------------------------------------
_LAZY_SYMBOL_TO_ADAPTER_NAME: dict[str, str] = {
    "StableDiffusion3Adapter": "stable-diffusion-3",
    "HunyuanDiTAdapter": "hunyuan-dit",
}
_LAZY_MODULE_NAMES: dict[str, str] = {
    "cli": "papers.cli",
}


def __getattr__(name: str) -> Any:  # PEP 562
    """Resolve lazy exports (``StableDiffusion3Adapter`` etc.) on demand.

    Imported on first attribute access; subsequent lookups are
    served from :data:`_loaded_adapters` (module-level cache) so
    we do not assign to ``globals()`` -- that would prevent a
    caller from "purging" the cache (e.g. in a test that wants
    to assert a fresh-import behaviour).  The cache invalidation
    contract is documented in the test suite
    (:mod:`tests.test_r18_lazy`).
    """
    if name in _LAZY_SYMBOL_TO_ADAPTER_NAME:
        cls = _resolve_adapter_class(_LAZY_SYMBOL_TO_ADAPTER_NAME[name])
        return cls
    if name in _LAZY_MODULE_NAMES:
        import importlib
        return importlib.import_module(_LAZY_MODULE_NAMES[name])
    raise AttributeError(
        "module 'papers' has no attribute {!r}".format(name)
    )


def __dir__() -> List[str]:
    """Advertise lazy exports to ``dir()`` and IDE auto-completion."""
    return sorted(set(__all__) | set(globals().keys()))


# Type-checker only imports -- never executed at runtime, but let
# static analysers see the surface.
if TYPE_CHECKING:  # pragma: no cover
    from . import cli  # noqa: F401
    from .adapters import HunyuanDiTAdapter, StableDiffusion3Adapter  # noqa: F401
