"""The unified :class:`ModelFetcher` class (v0.6.x).

This module is the main entry point of the
:mod:`models.source.fetch` sub-package.  Most callers should not
import it directly -- the canonical entry point is the
:func:`models.source.fetch.fetch` free function, which is built on
top of a process-level :class:`ModelFetcher` singleton.

The :class:`ModelFetcher` owns a :class:`ModelCache`, a
:class:`SourceRegistry`, and an active allow-list.  A single
instance is safe to share across threads.

For the heavy lifting of :meth:`ModelFetcher._fetch_inner` the
implementation lives in
:mod:`models.source.fetch._inner` -- the method on this class is
a thin forwarder that keeps the public/private attribute names
stable for callers that introspect them via
``monkeypatch.setattr``.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from infrastructure.logger import get_logger

from ..cache import CachedModel, ModelCache
from ..mirrors import MirrorSet
from ._download_helpers import (
    download_default_artifacts as _download_default_artifacts_impl,
)
from ._download_helpers import (
    validate_pins_against_manifest as _validate_pins_impl,
)
from ._download_helpers import resolve_license_id as _resolve_license_id_impl
from ._registry import SOURCE_ALIASES, SourceRegistry

_logger = get_logger("models.source.fetch")


__all__ = ["ModelFetcher"]


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
        from ..license_check import DEFAULT_ALLOW_LICENSE

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
    ) -> "FetchResult":
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
        # Local imports to break the cycles
        # ``_fetcher -> _inner -> _fetcher`` and
        # ``_fetcher -> _result``.
        from ._inner import fetch_inner
        from ._result import FetchResult

        if not repo_id or not repo_id.strip():
            raise ValueError("`repo_id` must be non-empty")

        source = source.strip()
        canonical = SOURCE_ALIASES.get(source, source)
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
            from ..auth import resolve_token
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
            return fetch_inner(
                cache=self._cache,
                adapter=adapter,
                source=source,
                canonical=canonical,
                repo_id=repo_id,
                revision=revision,
                allow_license=allow_license,
                default_allow=self._default_allow,
                verify_cache=verify_cache,
                on_progress=on_progress,
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
    ) -> "FetchResult":
        """Thin forwarder around :func:`models.source.fetch._inner.fetch_inner`.

        Kept as a method (rather than inlined) so test code that
        ``monkeypatch.setattr(fetcher, "_fetch_inner", ...)``
        keeps working.
        """
        from ._inner import fetch_inner
        return fetch_inner(
            cache=self._cache,
            adapter=adapter,
            source=source,
            canonical=canonical,
            repo_id=repo_id,
            revision=revision,
            allow_license=allow_license,
            default_allow=self._default_allow,
            verify_cache=verify_cache,
            on_progress=on_progress,
            expected_sha256s=expected_sha256s,
        )

    # ------------------------------------------------------------------
    # Helper forwarders (preserved for introspection / monkeypatch)
    # ------------------------------------------------------------------
    def _resolve_license_id(
        self, adapter: Any, repo_id: str, canonical: str
    ) -> str:
        """Thin forwarder around
        :func:`models.source.fetch._download_helpers.resolve_license_id`.

        Kept as a method (rather than inlined) so tests / callers
        that introspect ``fetcher._resolve_license_id`` or
        ``monkeypatch.setattr(fetcher, "_resolve_license_id", ...)``
        keep working.
        """
        return _resolve_license_id_impl(adapter, repo_id, canonical)

    def _download_default_artifacts(
        self, adapter: Any, repo_id: str, revision: str, canonical: str,
        on_progress: Optional[Callable[..., None]] = None,
        expected_sha256s: Optional[Mapping[str, str]] = None,
    ):
        """Thin forwarder; see
        :func:`models.source.fetch._download_helpers.download_default_artifacts`.
        """
        return _download_default_artifacts_impl(
            adapter, repo_id, revision, canonical,
            on_progress=on_progress,
            expected_sha256s=expected_sha256s,
        )

    def _validate_pins_against_manifest(
        self,
        manifest: CachedModel,
        expected_sha256s: Mapping[str, str],
    ) -> None:
        """Thin forwarder; see
        :func:`models.source.fetch._download_helpers.validate_pins_against_manifest`.
        """
        return _validate_pins_impl(manifest, expected_sha256s)

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
