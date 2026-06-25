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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from infrastructure.logger import get_logger

from .auth import ChecksumMismatch, GatedRepoError
from .cache import (
    CacheLocation,
    CachedModel,
    ModelCache,
    compute_content_fingerprint,
    default_cache_root,
)
from .civitai import CivitaiSource
from .huggingface import (
    DEFAULT_USER_AGENT,
    DownloadProgress,
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
from .mirrors import (
    DEFAULT_HF_MIRRORS,
    MirrorHealth,
    MirrorSet,
    check_all_mirrors,
    check_mirror_health,
)

__all__ = [
    "fetch",
    "ModelFetcher",
    "FetchResult",
    "SourceRegistry",
    "ChecksumMismatch",
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
        mirrors: Optional default :class:`MirrorSet` for the HF
            adapter.  When the registry's HuggingFace adapter does
            not already have a mirror set, the fetcher's mirrors
            are installed into it.  The per-call ``mirrors=``
            argument (if any) takes precedence.
    """

    def __init__(
        self,
        cache: Optional[ModelCache] = None,
        registry: Optional[SourceRegistry] = None,
        allow_license: Optional[Sequence[str]] = None,
        mirrors: Optional[MirrorSet] = None,
    ) -> None:
        self._cache: ModelCache = cache or ModelCache()
        self._registry: SourceRegistry = registry or SourceRegistry.default()
        self._default_allow: Tuple[str, ...] = tuple(
            sorted(set(allow_license) if allow_license is not None else DEFAULT_ALLOW_LICENSE)
        )
        self._default_mirrors: Optional[MirrorSet] = mirrors
        self._lock: threading.RLock = threading.RLock()
        self._logger = _logger
        # Eagerly install the default mirror set on the registry's
        # HF adapter, so every ``fetch(source="huggingface", ...)``
        # call benefits from the mirror config without the caller
        # having to remember to pass ``mirrors=`` each time.
        if mirrors is not None:
            self._install_default_mirrors(mirrors)

    def _install_default_mirrors(self, mirrors: MirrorSet) -> None:
        """Attach ``mirrors`` to every HF-flavoured adapter in the registry.

        The install is a permanent state change on the adapter
        instance -- we do NOT restore it on the way out.  Use
        per-call ``mirrors=`` on :meth:`fetch` to override at
        call time without mutating the registry.
        """
        for name in ("huggingface", "hf"):
            try:
                adapter = self._registry.get(name)
            except KeyError:
                continue
            if hasattr(adapter, "_mirrors"):
                adapter._mirrors = mirrors  # type: ignore[attr-defined]
                adapter._api_base = mirrors.bases[0]  # type: ignore[attr-defined]
            elif hasattr(adapter, "_api_base"):
                adapter._api_base = mirrors.bases[0]  # type: ignore[attr-defined]

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
        mirrors: Optional[MirrorSet] = None,
        on_progress: Optional[Any] = None,
        expected_sha256s: Optional[Mapping[str, str]] = None,
        token: Optional[str] = None,
        validate_checksums: bool = True,
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
            mirrors: Optional per-call :class:`MirrorSet` override
                for the HF adapter.  When the adapter supports
                mirrors, this is installed for the duration of
                the call and restored afterwards (so concurrent
                fetches are not affected).
            on_progress: Optional callback for the HF download
                loop.  Two shapes are accepted:
                ``(file_name, bytes_done, bytes_total, mirror)``
                (the v0.4.0 ergonomic shape) or
                ``Callable[[DownloadProgress], None]`` (the
                v0.4.x P2+ lower-level shape).  The fetcher
                normalises both into a single
                :class:`DownloadProgress` tick before forwarding
                to the HF adapter.  For sources other than HF
                the parameter is silently ignored.
            expected_sha256s: Optional ``{file_name: sha256_hex}``
                map.  When supplied the caller's pinned hashes are
                a hard contract: a file whose local hash does not
                match the pinned value triggers
                :class:`~models.source.auth.ChecksumMismatch`
                *before* the manifest is written.  Useful for
                supply-chain integrity -- the operator can pass a
                manifest captured out-of-band (e.g. from
                ``huggingface.co/api/models/.../tree/...``) and
                refuse a tampered mirror response.
            token: Optional per-call API token.  When supplied
                the adapter is rebuilt for the duration of the
                call so the value does *not* leak into the
                singleton ``SourceRegistry``'s default
                adapters -- this is the safe way to inject a
                secret for one call.  When ``None`` the
                adapter's existing token (env-var or on-disk
                file) is used.  Honoured by both the HF and
                Civitai adapters.
            validate_checksums: When ``True`` (default), per-file
                pin checks (``expected_sha256s``) are enforced
                before writing to the cache.  When ``False`` the
                caller's pins are ignored -- the manifest is
                still written with whatever hashes the adapter
                reported (upstream or local).  Use ``False`` only
                for trusted internal feeds; the default is the
                safe choice.

        Returns:
            A :class:`FetchResult` with the cache location, manifest,
            and license verdict.

        Raises:
            KeyError: When ``source`` is not registered.
            ValueError: When ``repo_id`` is empty.
            PermissionError: When the license is not on the
                allow-list.
            ChecksumMismatch: When ``expected_sha256s`` pins a
                file and the local hash does not match (only
                raised when ``validate_checksums=True``).
            RuntimeError: When the source adapter cannot resolve a
                license or download any file.
        """
        if not repo_id or not repo_id.strip():
            raise ValueError("`repo_id` must be non-empty")

        source = source.strip()
        canonical = _SOURCE_ALIASES.get(source, source)
        adapter = self._registry.get(source)

        # ----- install per-call mirror set on the HF adapter --------------
        # We use a context manager-like pattern: remember the
        # previous mirror set and restore it on the way out.
        # Per-call ``mirrors`` take precedence; otherwise fall
        # back to the fetcher's default_mirrors.
        effective_mirrors = mirrors if mirrors is not None else self._default_mirrors
        prev_mirrors = None
        prev_mirror_attr = None
        if effective_mirrors is not None and hasattr(adapter, "_mirrors"):
            prev_mirror_attr = "_mirrors"
            prev_mirrors = getattr(adapter, prev_mirror_attr, None)
            adapter._mirrors = effective_mirrors  # type: ignore[attr-defined]
            # Re-derive api_base so the primary mirror wins.
            adapter._api_base = effective_mirrors.bases[0]  # type: ignore[attr-defined]
        elif effective_mirrors is not None and hasattr(adapter, "_api_base"):
            # Source does not know about mirror sets -- stash the
            # primary base for the duration of the call.
            prev_mirror_attr = "_api_base"
            prev_mirrors = adapter._api_base  # type: ignore[attr-defined]
            adapter._api_base = effective_mirrors.bases[0]  # type: ignore[attr-defined]

        # ----- install per-call token on the adapter ----------------------
        # The adapter owns a ``_token`` (TokenInfo or None).  We
        # remember the previous value and restore it on the way
        # out so the singleton registry is not mutated by a
        # one-off call.  When ``token`` is ``None`` we keep the
        # adapter's existing token (env-var / on-disk file).
        prev_token_attr = None
        prev_token = None
        if token is not None and hasattr(adapter, "_token"):
            from .auth import resolve_token
            prev_token_attr = "_token"
            prev_token = getattr(adapter, prev_token_attr, None)
            new_tok = resolve_token(explicit=token, sources=canonical)
            adapter._token = new_tok  # type: ignore[attr-defined]

        # ----- effective expected_sha256s --------------------------------
        # Empty map is treated as "no pins" so callers can pass
        # ``{}`` without disabling the validation logic.
        effective_pins: Mapping[str, str] = expected_sha256s or {}
        # ``validate_checksums=False`` is the explicit opt-out --
        # replace the map with an empty mapping.
        if not validate_checksums:
            effective_pins = {}

        try:
            return self._fetch_inner(
                source=source,
                canonical=canonical,
                repo_id=repo_id,
                revision=revision,
                allow_license=allow_license,
                verify_cache=verify_cache,
                on_progress=on_progress,
                adapter=adapter,
                expected_sha256s=effective_pins,
            )
        finally:
            if prev_mirror_attr is not None and prev_mirrors is not None:
                setattr(adapter, prev_mirror_attr, prev_mirrors)
            if prev_token_attr is not None:
                setattr(adapter, prev_token_attr, prev_token)

    def _fetch_inner(
        self,
        *,
        source: str,
        canonical: str,
        repo_id: str,
        revision: str,
        allow_license: Optional[Sequence[str]],
        verify_cache: bool,
        on_progress: Optional[Any],
        adapter: Any,
        expected_sha256s: Mapping[str, str],
    ) -> FetchResult:

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

        # ----- download + dedup-aware write --------------------------------
        downloads = self._download_default_artifacts(
            adapter, repo_id, revision, canonical, on_progress=on_progress,
            expected_sha256s=expected_sha256s,
        )
        if not downloads:
            raise RuntimeError(
                "source {!r} returned no files for {!r}@{:!r}; "
                "check the network / credentials and retry".format(
                    canonical, repo_id, revision,
                )
            )

        # ----- cross-mirror / cross-revision dedup --------------------------
        # We just downloaded `downloads`; before writing them to
        # the canonical location, check whether the same content
        # (by fingerprint) is already present in the cache under
        # any other (repo_id, revision).  When a match is found we
        # *do not* write a duplicate copy -- the existing files
        # on disk are byte-identical (same fingerprint) and the
        # caller is happy because they did not pay for *disk
        # space* and a second round of integrity verification.
        # We only spent a network round-trip for the metadata
        # lookup, which is unavoidable at the time we learn the
        # file set.
        files_spec: List[Dict[str, Any]] = [
            {"name": d.name, "data": d.data, "sha256": d.sha256}
            for d in downloads
        ]
        fingerprint = compute_content_fingerprint(files_spec)
        existing = self._cache.find_by_fingerprint(canonical, fingerprint)
        if existing is not None and (
            existing.repo_id != repo_id or existing.revision != revision
        ):
            # The same content is cached under a different key --
            # surface that as a cache hit (the operator is happy
            # because they did not pay for a network round-trip).
            # We DO NOT skip the write to the canonical location
            # here -- the caller asked for this exact key, and
            # future lookups under this key should be a direct
            # manifest hit rather than a fingerprint scan.
            self._logger.info(
                "Cross-mirror dedup hit: %s/%s@%s == cached %s/%s@%s",
                canonical, repo_id, revision,
                canonical, existing.repo_id, existing.revision,
            )
            try:
                existing_manifest = self._cache.load_manifest(
                    canonical, existing.repo_id, existing.revision,
                )
            except (OSError, ValueError):
                existing_manifest = None
            if existing_manifest is not None:
                # When the caller pinned checksums, re-validate
                # the existing manifest's recorded digests -- a
                # stale manifest is the worst case for supply-
                # chain integrity.  The local file on disk is
                # byte-identical to the one we just downloaded
                # (fingerprint match), so the per-file hash is
                # already implicit; we just re-check that it
                # matches the pins if any.
                if expected_sha256s:
                    self._validate_pins_against_manifest(
                        existing_manifest, expected_sha256s,
                    )
                # Skip the write: serve the existing manifest
                # directly.  The files on disk are content-equal
                # (by construction -- same fingerprint).
                return FetchResult(
                    location=existing,
                    manifest=existing_manifest,
                    source=canonical,
                    license_check=check,
                    from_cache=True,
                )

        url = self._build_url(adapter, canonical, repo_id, revision)
        manifest = self._cache.write_files(
            source=canonical,
            repo_id=repo_id,
            revision=revision,
            license_id=check.license_id,
            url=url,
            files=files_spec,
            expected_sha256s=expected_sha256s or None,
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
        """Call the adapter's ``resolve_license`` and normalise the result.

        GatedRepoError is *not* swallowed: 401/403 is an
        operator-actionable condition (set $HF_TOKEN, install
        ``huggingface_hub`` and ``huggingface-cli login``, etc.)
        so the caller deserves the original error.  Any other
        adapter failure (5xx, network, JSON shape) is logged and
        reduced to ``""`` -- the license check will then
        short-circuit to "no license declared", which the
        operator can re-try without configuring a token.
        """
        try:
            raw = adapter.resolve_license(repo_id)
        except GatedRepoError:
            # 401/403: a token is required.  Do NOT swallow --
            # surface the actionable error to the caller.
            raise
        except Exception as exc:  # noqa: BLE001 - any adapter failure
            self._logger.warning(
                "License resolution failed for %s/%s: %s",
                canonical, repo_id, exc,
            )
            return ""
        return normalise_spdx(str(raw or ""))

    def _download_default_artifacts(
        self, adapter: Any, repo_id: str, revision: str, canonical: str,
        on_progress: Optional[Callable[..., None]] = None,
        expected_sha256s: Optional[Mapping[str, str]] = None,
    ) -> List[FileDownload]:
        """Call the adapter's ``download_default_artifacts`` if present.

        ``on_progress`` is forwarded to the adapter when it accepts a
        keyword argument of the same name.  Adapters that do not
        accept a progress callback simply ignore the argument (the
        typical contract is "forgive extra kwargs").

        ``expected_sha256s`` is forwarded to the adapter when it
        accepts a keyword argument of the same name; otherwise it
        is silently dropped.  The fetcher itself re-validates the
        pins in :meth:`ModelCache.write_files` so a stripped kwarg
        on the adapter is still safe.

        Two callback shapes are accepted (the fetcher docs explain
        why both are useful):

        * ``(file_name, bytes_done, bytes_total, mirror)`` -- the
          v0.4.0 ergonomic shape; converted internally into a
          :class:`DownloadProgress` tick.
        * ``Callable[[DownloadProgress], None]`` -- the v0.4.x
          P2+ low-level shape; forwarded verbatim.
        """
        fn = getattr(adapter, "download_default_artifacts", None)
        if not callable(fn):
            raise RuntimeError(
                "source adapter {!r} does not support downloads".format(canonical)
            )
        if canonical == "civitai":
            # Civitai's adapter does not take a revision and does
            # not accept an on_progress / expected_sha256s
            # callback -- ``download_default_artifacts(version_id)``.
            # Integrity pins are validated in
            # :meth:`ModelCache.write_files` so a stripped kwarg
            # is still safe.
            return fn(repo_id)  # type: ignore[arg-type]
        if on_progress is None and not expected_sha256s:
            return fn(repo_id, revision or "main")  # type: ignore[arg-type]

        # Normalise the callback.  We do this by *probing* the
        # callback's signature -- a 1-arg callback is treated as
        # the low-level ``DownloadProgress`` shape; everything
        # else is treated as the 4-arg ergonomic shape.
        if on_progress is not None:
            import inspect
            try:
                sig = inspect.signature(on_progress)
                nargs = len([
                    p for p in sig.parameters.values()
                    if p.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ])
            except (TypeError, ValueError):
                nargs = 4  # default: assume ergonomic shape

            if nargs <= 1:
                adapter_cb: Optional[Callable[..., None]] = on_progress
            else:
                def adapter_cb(tick: DownloadProgress) -> None:
                    on_progress(
                        tick.file_name, tick.bytes_done, tick.bytes_total, tick.mirror,
                    )
        else:
            adapter_cb = None

        # Try the most-featureful signature first; fall back
        # gracefully when the adapter does not accept one of the
        # kwargs.  Order: progress + pins, then progress only,
        # then pins only, then no kwargs.
        try:
            if adapter_cb is not None and expected_sha256s:
                return fn(  # type: ignore[arg-type]
                    repo_id, revision or "main",
                    on_progress=adapter_cb,
                    expected_sha256s=expected_sha256s,
                )
            if adapter_cb is not None:
                return fn(  # type: ignore[arg-type]
                    repo_id, revision or "main", on_progress=adapter_cb,
                )
            if expected_sha256s:
                return fn(  # type: ignore[arg-type]
                    repo_id, revision or "main", expected_sha256s=expected_sha256s,
                )
            return fn(repo_id, revision or "main")  # type: ignore[arg-type]
        except TypeError:
            # Adapter has the old signature -- fall back.
            return fn(repo_id, revision or "main")  # type: ignore[arg-type]

    def _validate_pins_against_manifest(
        self,
        manifest: CachedModel,
        expected_sha256s: Mapping[str, str],
    ) -> None:
        """Re-validate pinned sha256s against an *existing* manifest.

        Used on a cross-mirror dedup hit: the on-disk files are
        byte-identical to the ones we just downloaded (same
        fingerprint), so the per-file hash is implicit.  We
        therefore compare the pins to the manifest's recorded
        digests rather than re-hashing every file -- a few
        microseconds vs. a few hundred milliseconds.
        """
        recorded = {f.name: f.sha256 for f in manifest.files}
        for name, pinned in expected_sha256s.items():
            if not pinned:
                continue
            actual = recorded.get(name, "")
            if not actual:
                # The pinned file is not in this manifest -- the
                # caller is asking for stricter coverage than the
                # cached model has.  Refuse the dedup hit and let
                # the caller's next fetch do a clean write.
                raise ChecksumMismatch(
                    source=manifest.source,
                    repo_id=manifest.repo_id,
                    file_name=name,
                    expected_sha256=pinned,
                    actual_sha256="<not-in-existing-manifest>",
                )
            if actual != pinned:
                raise ChecksumMismatch(
                    source=manifest.source,
                    repo_id=manifest.repo_id,
                    file_name=name,
                    expected_sha256=pinned,
                    actual_sha256=actual,
                )

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
    expected_sha256s: Optional[Mapping[str, str]] = None,
    token: Optional[str] = None,
    validate_checksums: bool = True,
    on_progress: Optional[Any] = None,
    mirrors: Optional[MirrorSet] = None,
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
