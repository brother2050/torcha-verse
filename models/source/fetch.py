"""Unified model fetcher for the TorchaVerse model fetcher (v0.4.0).

This module ties together :mod:`models.source.license_check`,
:mod:`models.source.cache`, :mod:`models.source.huggingface` and
:mod:`models.source.civitai` behind a single ``fetch(...)`` entry
point.  The public function (and the :class:`ModelFetcher` class
that owns the state) implements the v0.4.0 minimum-viable contract:

* resolve a source by name (``"huggingface"`` / ``"hf"`` /
  ``"civitai"``);
* query the source for the model license;
* verify the license against the caller's allow-list
  (default: :data:`license_check.DEFAULT_ALLOW_LICENSE`);
* if the cache already has a valid manifest, short-circuit the
  network and return the cached location;
* otherwise download, write to cache atomically, verify the
  integrity of the on-disk files, and return the cache location.

The function is intentionally small.  All policy (license
verification, allow-list, atomic write, sha256 check) is delegated
to the dedicated modules, and the source adapters are stateless
beyond their transport instance.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``models.source`` (this module) -- facade.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from infrastructure.logger import get_logger

from .cache import CacheLocation, CachedModel, ModelCache, default_cache_root
from .civitai import CivitaiSource
from .huggingface import (
    FileDownload,
    HttpTransport,
    HuggingFaceSource,
    UrllibTransport,
)
from .license_check import (
    DEFAULT_ALLOW_LICENSE,
    LicenseCheckResult,
    check_license,
    normalise_spdx,
)

__all__ = [
    "fetch",
    "ModelFetcher",
    "FetchResult",
    "SourceRegistry",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Module-level logger.
_logger = get_logger("models.source.fetch")

#: Mapping of public source name -> canonical source id used in the
#: cache directory layout.
_SOURCE_ALIASES: Dict[str, str] = {
    "huggingface": "huggingface",
    "hf": "huggingface",
    "civitai": "civitai",
    "cv": "civitai",
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class FetchResult:
    """Outcome of a :func:`fetch` call.

    Attributes:
        location: The :class:`CacheLocation` of the fetched model.
        manifest: The :class:`CachedModel` manifest that was loaded
            from the cache (either pre-existing or freshly written).
        source: The canonical source id used in the cache.
        license_check: The :class:`LicenseCheckResult` from the
            license whitelist verification.
        from_cache: ``True`` when the model was already cached and
            no network call was needed; ``False`` when a fresh
            download happened.
    """

    location: CacheLocation
    manifest: CachedModel
    source: str
    license_check: LicenseCheckResult
    from_cache: bool

    @property
    def accepted(self) -> bool:
        """``True`` when the license was accepted by the whitelist."""
        return self.license_check.accepted

    def as_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "location": self.location.as_dict(),
            "manifest": self.manifest.as_dict(),
            "source": self.source,
            "license_check": {
                "accepted": self.license_check.accepted,
                "reason": self.license_check.reason,
                "license_id": self.license_check.license_id,
            },
            "from_cache": self.from_cache,
        }

    def __repr__(self) -> str:
        return (
            "FetchResult(source={!r}, repo_id={!r}, license={!r}, "
            "from_cache={})".format(
                self.source,
                self.manifest.repo_id,
                self.license_check.license_id,
                self.from_cache,
            )
        )


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
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
        canonical = _SOURCE_ALIASES.get(name.strip(), name.strip())
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


# ---------------------------------------------------------------------------
# ModelFetcher
# ---------------------------------------------------------------------------
class ModelFetcher:
    """Stateful unified fetcher.

    Owns a :class:`ModelCache`, a :class:`SourceRegistry`, and the
    active allow-list.  A single instance is safe to share across
    threads.

    Args:
        cache: Optional :class:`ModelCache`.  When ``None`` a cache
            rooted at :func:`default_cache_root` is used.
        registry: Optional :class:`SourceRegistry`.  When ``None``
            the default registry is used.
        allow_license: Optional default allow-list.  When ``None``
            :data:`license_check.DEFAULT_ALLOW_LICENSE` is used.
            The per-call ``allow_license=`` argument (if any) takes
            precedence.
    """

    def __init__(
        self,
        cache: Optional[ModelCache] = None,
        registry: Optional[SourceRegistry] = None,
        allow_license: Optional[Sequence[str]] = None,
    ) -> None:
        self._cache: ModelCache = cache or ModelCache()
        self._registry: SourceRegistry = registry or SourceRegistry.default()
        self._default_allow: Tuple[str, ...] = tuple(
            sorted(set(allow_license) if allow_license is not None else DEFAULT_ALLOW_LICENSE)
        )
        self._lock: threading.RLock = threading.RLock()
        self._logger = _logger

    @property
    def cache(self) -> ModelCache:
        """The :class:`ModelCache` owned by this fetcher."""
        return self._cache

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------
    def fetch(
        self,
        source: str,
        repo_id: str,
        revision: str = "",
        allow_license: Optional[Sequence[str]] = None,
        verify_cache: bool = True,
    ) -> FetchResult:
        """Fetch a model, with caching and license verification.

        Args:
            source: Source name (``"huggingface"`` / ``"hf"`` /
                ``"civitai"`` / ``"cv"``).
            repo_id: Repository / version id, depending on the
                source (``"Qwen/Qwen2.5-0.5B-Instruct"`` for HF,
                ``"12345"`` for Civitai).
            revision: Source revision (``"main"``, tag, commit
                hash, ...).  Empty string means "HEAD / latest".
            allow_license: Optional explicit allow-list for this
                call.  Overrides the fetcher's default.
            verify_cache: When ``True`` (default) re-hash every
                cached file before returning; on mismatch the cache
                entry is wiped and the model is re-fetched.

        Returns:
            A :class:`FetchResult` with the cache location, manifest,
            and license verdict.

        Raises:
            KeyError: When ``source`` is not registered.
            ValueError: When ``repo_id`` is empty.
            PermissionError: When the license is not on the
                allow-list.
            RuntimeError: When the source adapter cannot resolve a
                license or download any file.
        """
        if not repo_id or not repo_id.strip():
            raise ValueError("`repo_id` must be non-empty")

        source = source.strip()
        canonical = _SOURCE_ALIASES.get(source, source)
        adapter = self._registry.get(source)

        # ----- license check ------------------------------------------------
        license_id = self._resolve_license_id(adapter, repo_id, canonical)
        effective_allow = (
            list(allow_license) if allow_license is not None
            else list(self._default_allow)
        )
        check = check_license(license_id, allow_license=effective_allow)
        if not check.accepted:
            # Persist the rejection in the log; do NOT cache anything.
            self._logger.warning(
                "License rejected for %s/%s@%s: %s",
                canonical, repo_id, revision, check.reason,
            )
            raise PermissionError(check.reason)

        # ----- cache lookup -------------------------------------------------
        loc = self._cache.location_for(canonical, repo_id, revision)
        if self._cache.has(canonical, repo_id, revision):
            if verify_cache and not self._cache.verify(canonical, repo_id, revision):
                self._logger.warning(
                    "Cache verification failed for %s/%s@%s; re-fetching",
                    canonical, repo_id, revision,
                )
                self._cache.clear(canonical, repo_id, revision)
            else:
                manifest = self._cache.load_manifest(canonical, repo_id, revision)
                self._logger.info(
                    "Cache hit for %s/%s@%s (license=%s)",
                    canonical, repo_id, revision, check.license_id,
                )
                return FetchResult(
                    location=loc,
                    manifest=manifest,
                    source=canonical,
                    license_check=check,
                    from_cache=True,
                )

        # ----- download + atomic write --------------------------------------
        downloads = self._download_default_artifacts(adapter, repo_id, revision, canonical)
        if not downloads:
            raise RuntimeError(
                "source {!r} returned no files for {!r}@{:!r}; "
                "check the network / credentials and retry".format(
                    canonical, repo_id, revision,
                )
            )

        files_spec: List[Dict[str, Any]] = [
            {"name": d.name, "data": d.data, "sha256": d.sha256}
            for d in downloads
        ]
        url = self._build_url(adapter, canonical, repo_id, revision)
        manifest = self._cache.write_files(
            source=canonical,
            repo_id=repo_id,
            revision=revision,
            license_id=check.license_id,
            url=url,
            files=files_spec,
        )
        # The just-written files are already sha256-verified by
        # ``write_files``; call :meth:`verify` to cross-check the
        # manifest's per-file digests.
        if not self._cache.verify(canonical, repo_id, revision):
            self._cache.clear(canonical, repo_id, revision)
            raise RuntimeError(
                "post-write integrity check failed for "
                "{}/{}/{}".format(canonical, repo_id, revision)
            )
        return FetchResult(
            location=loc,
            manifest=manifest,
            source=canonical,
            license_check=check,
            from_cache=False,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_license_id(
        self, adapter: Any, repo_id: str, canonical: str
    ) -> str:
        """Call the adapter's ``resolve_license`` and normalise the result."""
        try:
            raw = adapter.resolve_license(repo_id)
        except Exception as exc:  # noqa: BLE001 - any adapter failure
            self._logger.warning(
                "License resolution failed for %s/%s: %s",
                canonical, repo_id, exc,
            )
            return ""
        return normalise_spdx(str(raw or ""))

    def _download_default_artifacts(
        self, adapter: Any, repo_id: str, revision: str, canonical: str,
    ) -> List[FileDownload]:
        """Call the adapter's ``download_default_artifacts`` if present."""
        fn = getattr(adapter, "download_default_artifacts", None)
        if not callable(fn):
            raise RuntimeError(
                "source adapter {!r} does not support downloads".format(canonical)
            )
        if canonical == "civitai":
            # Civitai's adapter does not take a revision.
            return fn(repo_id)  # type: ignore[arg-type]
        return fn(repo_id, revision or "main")  # type: ignore[arg-type]

    @staticmethod
    def _build_url(
        adapter: Any, canonical: str, repo_id: str, revision: str
    ) -> str:
        """Build a human-readable URL describing where the model came from."""
        if canonical == "huggingface":
            base = getattr(adapter, "_api_base", "https://huggingface.co")
            base = base.rstrip("/")
            if revision:
                return "{}/{}/tree/{}".format(base, repo_id, revision)
            return "{}/{}/tree/main".format(base, repo_id)
        if canonical == "civitai":
            base = getattr(adapter, "_api_base", "https://civitai.com/api")
            return "{}/v1/model-versions/{}".format(base.rstrip("/"), repo_id)
        return ""


# ---------------------------------------------------------------------------
# Public free-function entry point
# ---------------------------------------------------------------------------
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
) -> FetchResult:
    """Fetch a model from a registered source.

    Convenience wrapper around :meth:`ModelFetcher.fetch` that uses a
    process-level singleton :class:`ModelFetcher`.  Callers that
    want custom cache roots or custom allow-lists should construct
    their own :class:`ModelFetcher` and call :meth:`fetch` on it
    directly.

    Args:
        repo_id: Repository / version id.
        source: Source name (default ``"huggingface"``).
        revision: Source revision (default ``""`` == HEAD / latest).
        allow_license: Optional explicit allow-list.
        verify_cache: Whether to re-hash cached files on lookup.

    Returns:
        A :class:`FetchResult`.
    """
    return _get_default_fetcher().fetch(
        source=source,
        repo_id=repo_id,
        revision=revision,
        allow_license=allow_license,
        verify_cache=verify_cache,
    )
