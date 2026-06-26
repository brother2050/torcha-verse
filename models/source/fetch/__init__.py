"""Unified model fetcher for the TorchaVerse model fetcher (v0.6.x).

This sub-package ties together :mod:`models.source.license_check`,
:mod:`models.source.cache`, :mod:`models.source.huggingface` and
:mod:`models.source.civitai` behind a single ``fetch(...)`` entry
point.  The public function (and the :class:`ModelFetcher` class
that owns the state) implements the v0.4.0 minimum-viable contract:

* resolve a source by name (``"huggingface"`` / ``"hf"`` /
  ``"civitai"``);
* query the source for the model license;
* verify the license against the caller's allow-list
  (default: :data:`models.source.license_check.DEFAULT_ALLOW_LICENSE`);
* if the cache already has a valid manifest, short-circuit the
  network and return the cached location;
* otherwise download, write to cache atomically, verify the
  integrity of the on-disk files, and return the cache location.

Sub-modules
-----------

* :mod:`._registry` -- :class:`SourceRegistry` and the
  ``"hf"``/``"cv"`` alias table.
* :mod:`._result` -- the :class:`FetchResult` dataclass.
* :mod:`._download_helpers` -- license / download / pin helpers
  (kept module-level for testability).
* :mod:`._fetcher` -- the :class:`ModelFetcher` class itself.
* :mod:`._entry` -- the :func:`fetch` free function and the
  process-level :class:`ModelFetcher` singleton.

The :func:`fetch` free function and the process-level
:class:`ModelFetcher` singleton live in *this* module so that
``sys.modules["models.source.fetch"]._default_fetcher`` is the
canonical place to monkey-patch the singleton in tests (this
matches the v0.4.x / v0.5.x public contract).

The function is intentionally small.  All policy (license
verification, allow-list, atomic write, sha256 check) is delegated
to the dedicated modules, and the source adapters are stateless
beyond their transport instance.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``models.source`` (this sub-package) -- facade.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Optional, Sequence

# Re-export the public API at the sub-package level so that
# ``from models.source.fetch import fetch, ModelFetcher, ...`` and
# ``import models.source.fetch`` both work (the latter being the
# shape that test code uses to monkey-patch the singleton).
from ..auth import ChecksumMismatch
from ..mirrors import MirrorSet
from ._fetcher import ModelFetcher
from ._inner import fetch_inner as _inner_fetch  # noqa: F401  (re-export for tests)
from ._registry import SOURCE_ALIASES, SourceRegistry
from ._result import FetchResult

# ---------------------------------------------------------------------------
# Process-level singleton
# ---------------------------------------------------------------------------
# The default fetcher is lazy-initialised on the first call to
# :func:`fetch`.  Tests can swap the singleton by patching this
# module attribute (the v0.4.x / v0.5.x public contract):
#
#     import sys
#     monkeypatch.setattr(sys.modules["models.source.fetch"],
#                         "_default_fetcher", my_fetcher)
#
# The wrapper below reads ``_default_fetcher`` at *call* time,
# so the patch is honoured.

_default_fetcher: Optional[ModelFetcher] = None
_default_fetcher_lock: threading.Lock = threading.Lock()


def _get_default_fetcher() -> ModelFetcher:
    """Return the process-level singleton fetcher (lazy-initialised)."""
    global _default_fetcher
    if _default_fetcher is None:
        with _default_fetcher_lock:
            if _default_fetcher is None:
                _default_fetcher = ModelFetcher()
    return _default_fetcher


def fetch(
    repo_id: str,
    source: str = "huggingface",
    revision: str = "",
    allow_license: Optional[Sequence[str]] = None,
    verify_cache: bool = True,
    expected_sha256s: Optional[Mapping[str, str]] = None,
    token: Optional[str] = None,
    validate_checksums: bool = True,
    on_progress: Optional[Any] = None,
    mirrors: Optional[MirrorSet] = None,
) -> FetchResult:
    """Fetch a model from a registered source.

    Convenience wrapper around :meth:`ModelFetcher.fetch` that uses
    a process-level singleton :class:`ModelFetcher`.  Callers that
    want custom cache roots or custom allow-lists should construct
    their own :class:`ModelFetcher` and call :meth:`fetch` on it
    directly.

    Args:
        repo_id: Repository / version id.
        source: Source name (default ``"huggingface"``).
        revision: Source revision (default ``""`` == HEAD / latest).
        allow_license: Optional explicit allow-list.
        verify_cache: Whether to re-hash cached files on lookup.
        expected_sha256s: Optional ``{file_name: sha256_hex}`` map;
            see :meth:`ModelFetcher.fetch` for the contract.
        token: Optional per-call API token; see
            :meth:`ModelFetcher.fetch` for the contract.
        validate_checksums: Whether to enforce ``expected_sha256s``.
        on_progress: Optional download progress callback (HF only).
        mirrors: Optional per-call mirror set (HF only).

    Returns:
        A :class:`FetchResult`.
    """
    return _get_default_fetcher().fetch(
        source=source,
        repo_id=repo_id,
        revision=revision,
        allow_license=allow_license,
        verify_cache=verify_cache,
        expected_sha256s=expected_sha256s,
        token=token,
        validate_checksums=validate_checksums,
        on_progress=on_progress,
        mirrors=mirrors,
    )


__all__ = [
    "fetch",
    "ModelFetcher",
    "FetchResult",
    "SourceRegistry",
    "SOURCE_ALIASES",
    "ChecksumMismatch",
    # Re-exported internal hooks (kept for tests that
    # ``monkeypatch.setattr`` on the module).
    "_default_fetcher",
    "_default_fetcher_lock",
    "_get_default_fetcher",
]
