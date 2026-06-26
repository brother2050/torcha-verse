"""The :class:`HuggingFaceSource` core.

The :class:`HuggingFaceSource` is the v0.6.x adapter for the
HuggingFace Hub REST API.  The class itself is now focused on
the metadata API (resolve_license / list_files / download_files
/ download_default_artifacts); the byte-level download loop
that used to dominate the file is delegated to
:func:`download_one_with_fallback` in :mod:`._download`.

The :class:`HuggingFaceSource` is *testable without a network*:
the HTTP transport is provided by an injectable
:class:`HttpTransport` object (the default
:class:`UrllibTransport` uses the standard library).  Tests can
swap in a fake that records calls or returns canned responses
without monkey-patching.

This module depends on:

* :mod:`._constants` -- module-level constants.
* :mod:`._transport` / :mod:`._urllib_transport` -- the
  default HTTP transport.
* :mod:`._types` -- :class:`FileDownload` / :class:`DownloadProgress`.
* :mod:`._download` -- :func:`download_one_with_fallback`.
* :mod:`..auth` -- shared auth / checksum / gated-repo helpers.
"""

from __future__ import annotations

import threading
import time
import urllib.error
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from infrastructure.logger import get_logger

from ..auth import (
    ChecksumMismatch,
    GatedRepoError,
    TokenInfo,
    is_gated_http_error,
    resolve_token,
)
from ._constants import (
    CHUNK_SIZE,
    DEFAULT_API_BASE,
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    HF_API_URL,
    HF_RESOLVE_URL,
)
from ._download import download_default_artifacts, download_one_with_fallback
from ._transport import HttpTransport
from ._types import DownloadProgress, FileDownload
from ._urllib_transport import UrllibTransport

__all__ = ["HuggingFaceSource"]


_logger = get_logger("models.source.huggingface.source")


class HuggingFaceSource:
    """Adapter for the HuggingFace Hub API.

    The class owns a single :class:`HttpTransport` and a
    thread-safe cache of repo-metadata lookups (HF's metadata
    is heavy enough that you don't want to re-fetch it on
    every file).

    Args:
        api_base: Base URL for the HF API.  Defaults to
            :data:`DEFAULT_API_BASE`.  Ignored when ``mirrors``
            is provided -- the first mirror becomes the primary
            base.
        transport: Optional :class:`HttpTransport` (mainly for
            testing).  When ``None`` a :class:`UrllibTransport`
            is used.
        token: Optional HuggingFace API token for gated /
            private repos.  Passed as ``Authorization: Bearer
            <token>``.  When ``None`` the constructor falls
            back to :func:`models.source.auth.resolve_token`.
        mirrors: Optional ordered list of mirror base URLs.
            The first one is used as the primary, the rest are
            fallbacks tried in order when a download fails.
    """

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        transport: Optional[HttpTransport] = None,
        token: Optional[str] = None,
        mirrors: Optional[Any] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        # The ``mirrors`` argument is duck-typed: when truthy and
        # it exposes a ``.bases`` attribute (the
        # :class:`models.source.mirrors.MirrorSet` contract) the
        # adapter stores it verbatim; otherwise we wrap the
        # ``api_base`` in a single-entry tuple.  The duck-typing
        # avoids a hard import on :mod:`models.source.mirrors`
        # from the sub-package.
        if mirrors is not None and hasattr(mirrors, "bases"):
            self._mirrors = mirrors
            # Keep ``api_base`` pointing at the primary mirror for
            # backwards compatibility (e.g. repr, _build_url
            # callers).
            self._api_base: str = mirrors.bases[0]
        else:
            self._mirrors = None
            self._api_base = str(api_base).rstrip("/")

        self._transport: HttpTransport = transport or UrllibTransport(
            timeout=timeout,
        )
        # ``resolve_token`` may also accept an explicit token.
        # The result is normalised to a :class:`TokenInfo` so the
        # same downstream code handles ``None``, ``str`` and a
        # pre-resolved :class:`TokenInfo`.
        #
        # The ``_token`` attribute is the *late-mutated* slot
        # used by :func:`models.source.fetch.fetch` to install
        # a per-call token -- the rest of the class reads
        # through :meth:`_auth_headers` (which inspects the
        # latest value of ``_token`` every time) so the swap
        # takes effect on the very next call.
        self._token_info: TokenInfo = resolve_token(token)
        self._token: TokenInfo = self._token_info

        # Thread-safe repo-metadata cache.
        self._meta_cache: Dict[str, Any] = {}
        self._meta_lock: threading.Lock = threading.Lock()

        # Thread-safe mirror-failure memory: a base URL that
        # just failed is *suppressed* for the remainder of the
        # process so we do not pay the network round-trip
        # twice.
        self._dead_mirrors: Dict[str, float] = {}
        self._dead_lock: threading.Lock = threading.Lock()
        self._dead_ttl_s: float = 60.0  # 1 minute TTL on "dead" memory

    @property
    def mirrors(self) -> Tuple[str, ...]:
        """Return the ordered mirror base URLs the adapter will try."""
        if self._mirrors is not None:
            return self._mirrors.bases
        return (self._api_base,)

    def _is_mirror_dead(self, base: str) -> bool:
        """Return ``True`` if ``base`` recently failed and is still cooling off."""
        with self._dead_lock:
            ts = self._dead_mirrors.get(base)
            if ts is None:
                return False
            if (time.time() - ts) > self._dead_ttl_s:
                # Expired: drop the entry and try again next time.
                self._dead_mirrors.pop(base, None)
                return False
            return True

    def _mark_mirror_dead(self, base: str) -> None:
        """Record ``base`` as a recently-failed mirror."""
        with self._dead_lock:
            self._dead_mirrors[base] = time.time()

    def _for_each_live_mirror(self, method: str, path: str):
        """Yield every live mirror in order, skipping the dead ones.

        This is implemented as a generator because the caller
        needs to start a new HTTP request as soon as a mirror
        fails, without buffering the whole list.
        """
        bases = (
            self._mirrors.bases
            if self._mirrors is not None
            else (self._api_base,)
        )
        for base in bases:
            if self._is_mirror_dead(base):
                continue
            yield base

    def _auth_headers(self) -> Dict[str, str]:
        """Return the auth headers (Bearer when a token is configured)."""
        # Read through ``self._token`` (not the cached
        # ``self._token_info``) so that any late mutation -- e.g.
        # :func:`models.source.fetch.fetch` swapping in a
        # per-call token by overwriting ``adapter._token`` --
        # takes effect on the very next call.
        token = self._token
        if token is not None and getattr(token, "value", ""):
            return {"Authorization": "Bearer {}".format(token.value)}
        return {}

    def _api_url(self, repo_id: str, path: str, base: Optional[str] = None) -> str:
        """Return the API URL for ``repo_id`` + ``path`` on ``base``.

        The path is concatenated after ``"/api/models/"`` so the
        caller always supplies the ``/models/{id}`` prefix
        explicitly (e.g. ``path="/models/bert-base/tree/main"``).
        This keeps the URL template in :mod:`._constants`
        agnostic of the ``models/`` keyword.
        """
        if path and not path.startswith("/"):
            path = "/" + path
        return "{}/api/models/{}{}".format(
            (base or self._api_base).rstrip("/"),
            repo_id,
            path,
        )

    def _resolve_url(
        self,
        repo_id: str,
        filename: str,
        revision: str,
        base: Optional[str] = None,
    ) -> str:
        """Return the CDN URL for ``repo_id``/``filename``@``revision`` on ``base``."""
        return HF_RESOLVE_URL.format(
            base=(base or self._api_base).rstrip("/"),
            repo_id=repo_id,
            revision=revision,
            filename=filename,
        )

    # ------------------------------------------------------------------
    # Metadata API
    # ------------------------------------------------------------------
    def resolve_license(
        self,
        repo_id: str,
        *,
        use_cache: bool = True,
    ) -> str:
        """Return the SPDX license id declared by the repo.

        The HF metadata endpoint returns either a ``cardData`` block
        with a ``license`` field, or a top-level ``license`` string.
        This method unifies the two shapes and returns the SPDX id
        verbatim (the caller is expected to run it through
        :func:`models.source.license_check.normalise_spdx` /
        :func:`check_license`).

        Returns:
            A license id string.  Empty string when the source does
            not declare a license (the caller is expected to reject
            this with the appropriate "no license declared" error).
        """
        if use_cache:
            with self._meta_lock:
                if repo_id in self._meta_cache:
                    return str(self._meta_cache[repo_id].get("license", ""))

        # Try every live mirror in order; record "useful" failures
        # so we do not retry the same broken mirror on the next
        # file in the same fetch.
        data: Any = None
        for base in self._for_each_live_mirror(
            "GET", "/api/models/{}".format(repo_id)
        ):
            url = self._api_url(repo_id, "", base=base)  # no extra path => /api/models/{id}
            try:
                data = self._transport.get_json(
                    url, headers=self._auth_headers()
                )
                break  # success -- fall through to parsing below
            except Exception as exc:  # noqa: BLE001 - any mirror failure
                if is_gated_http_error(exc):
                    raise GatedRepoError(
                        source="huggingface",
                        repo_id=repo_id,
                        status_code=int(getattr(exc, "code", 0)),
                        hint=(
                            "Set $HF_TOKEN or pass "
                            "HuggingFaceSource(token=...) and retry."
                        ),
                    ) from exc
                _logger.warning(
                    "HF metadata fetch failed for %s on %s: %s",
                    repo_id, base, exc,
                )
                code = (
                    int(getattr(exc, "code", 0))
                    if isinstance(exc, urllib.error.HTTPError)
                    else 0
                )
                if 500 <= code < 600:
                    self._mark_mirror_dead(base)
                elif not isinstance(exc, urllib.error.HTTPError):
                    # Network-class failure -- assume the mirror is down.
                    self._mark_mirror_dead(base)
                continue
        if data is None:
            # Every mirror failed; return empty so the license
            # check short-circuits to "no license declared".
            return ""

        if not isinstance(data, dict):
            return ""

        # Unify the two shapes (cardData / top-level).
        license_id = ""
        card = data.get("cardData")
        if isinstance(card, dict):
            raw = card.get("license")
            if isinstance(raw, str):
                license_id = raw
            elif isinstance(raw, list) and raw:
                first = raw[0]
                if isinstance(first, dict):
                    license_id = str(first.get("id", ""))
                else:
                    license_id = str(first)
        if not license_id:
            top = data.get("license")
            if isinstance(top, str):
                license_id = top
            elif isinstance(top, list) and top:
                first = top[0]
                if isinstance(first, dict):
                    license_id = str(first.get("id", ""))
                else:
                    license_id = str(first)
        if not license_id:
            for k in ("license_name", "licenseLink", "license_link"):
                v = data.get(k)
                if isinstance(v, str) and v:
                    license_id = v
                    break

        # Cache the parsed metadata so subsequent lookups
        # (e.g. ``list_files``) avoid a second round-trip.
        with self._meta_lock:
            self._meta_cache.setdefault(repo_id, data)
            self._meta_cache[repo_id]["license"] = license_id
        return license_id

    def list_files(
        self,
        repo_id: str,
        revision: str = "main",
    ) -> List[str]:
        """Enumerate the files attached to a revision.

        Tries every live mirror in order; returns the first
        non-empty result.  Sub-directory entries
        (``"type": "directory"``) and in-progress upload markers
        (filenames starting with ``"~"``) are filtered out.

        Args:
            repo_id: HF repository id.
            revision: Source revision (default ``"main"``).

        Returns:
            A list of filenames (e.g. ``["config.json",
            "model.safetensors", ...]``).  Empty when every
            mirror failed.
        """
        for base in self._for_each_live_mirror(
            "GET", "/api/models/{}/tree/{}".format(repo_id, revision or "main")
        ):
            url = self._api_url(repo_id, "/tree/{}".format(revision), base=base)
            try:
                data = self._transport.get_json(
                    url, headers=self._auth_headers()
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "Failed to list files for %s on %s: %s",
                    repo_id, base, exc,
                )
                code = (
                    int(getattr(exc, "code", 0))
                    if isinstance(exc, urllib.error.HTTPError)
                    else 0
                )
                if 500 <= code < 600:
                    self._mark_mirror_dead(base)
                elif not isinstance(exc, urllib.error.HTTPError):
                    self._mark_mirror_dead(base)
                continue
            if not isinstance(data, list):
                continue
            files: List[str] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                # ``type`` is "file" / "directory" / "lfs" --
                # only ``"file"`` (and the absence of a ``type``
                # field, which happens on some legacy mirrors)
                # is a real file.
                t = item.get("type")
                if t == "directory":
                    continue
                path = item.get("path", "")
                if not isinstance(path, str) or not path:
                    continue
                # ``~incomplete-foo`` is HF's marker for an
                # in-progress upload -- never download these.
                if path.startswith("~"):
                    continue
                files.append(path)
            if files:
                return files
            # An empty list is ambiguous (the repo might be
            # genuinely empty or the mirror might be out of
            # date).  Try the next mirror before giving up.
            if data == []:
                continue
        return []

    # ------------------------------------------------------------------
    # Download API (delegates to _download.download_one_with_fallback)
    # ------------------------------------------------------------------
    def _download_one_with_fallback(
        self,
        repo_id: str,
        revision: str,
        name: str,
        *,
        expected_sha256: str = "",
        on_progress: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> Optional[FileDownload]:
        """Public method that delegates to the free function."""
        return download_one_with_fallback(
            self, repo_id, revision, name,
            expected_sha256=expected_sha256,
            on_progress=on_progress,
        )

    def download_files(
        self,
        repo_id: str,
        revision: str,
        names: Sequence[str],
        *,
        expected_sha256s: Optional[Mapping[str, str]] = None,
        on_progress: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> List[FileDownload]:
        """Download a list of files and return ``FileDownload`` entries.

        Names starting with ``"~"`` (HF's in-progress upload
        markers) are silently skipped to avoid picking up a
        half-uploaded blob.

        Args:
            repo_id: HF repository id.
            revision: Source revision.
            names: Filenames to download.
            expected_sha256s: Optional per-file integrity map.
            on_progress: Optional progress callback.

        Returns:
            A list of :class:`FileDownload` entries, in the
            same order as ``names``.  Files that failed every
            mirror are omitted from the result.
        """
        results: List[FileDownload] = []
        for name in names:
            # In-progress upload markers -- never download.
            if name.startswith("~"):
                continue
            expected = (expected_sha256s or {}).get(name, "") or ""
            res = self._download_one_with_fallback(
                repo_id, revision, name,
                expected_sha256=expected,
                on_progress=on_progress,
            )
            if res is not None:
                results.append(res)
        return results

    def download_default_artifacts(
        self,
        repo_id: str,
        revision: str = "main",
        *,
        expected_sha256s: Optional[Mapping[str, str]] = None,
        on_progress: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> List[FileDownload]:
        """Download the four canonical HF artifacts (config, tokenizer, weights).

        See :func:`download_default_artifacts` in
        :mod:`._download` for the full description.
        """
        return download_default_artifacts(
            self, repo_id, revision,
            expected_sha256s=expected_sha256s,
            on_progress=on_progress,
        )

    def __repr__(self) -> str:
        return "HuggingFaceSource(api_base={!r})".format(self._api_base)
