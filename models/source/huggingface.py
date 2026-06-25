"""HuggingFace Hub source adapter for the TorchaVerse model fetcher (v0.4.0).

This module implements the HuggingFace side of the v0.4.0 model
fetcher.  It exposes a thin ``HuggingFaceSource`` class that wraps
the upstream ``https://huggingface.co`` API:

* ``resolve_license(repo_id)`` -- reads the repo metadata and
  returns the SPDX license id (or ``""`` when the source does not
  declare one).
* ``list_files(repo_id, revision)`` -- enumerates the files
  associated with a revision.
* ``download_files(repo_id, revision, names)`` -- downloads a list
  of files and returns ``[{name, data, sha256}, ...]``.

The adapter is designed to be *testable without a network*.  The
HTTP transport is provided by an injectable :class:`HttpTransport`
object -- the default implementation is :class:`UrllibTransport`
(``urllib.request`` from the standard library), but tests can swap
in a fake that records calls or returns canned responses.  This is
why the module works in any environment that has ``torch`` (and
therefore any environment in which the rest of TorchaVerse runs),
without forcing a hard dependency on the optional ``huggingface_hub``
package.

When ``huggingface_hub`` *is* installed the class
:class:`HuggingFaceHubTransport` is exposed so a future caller can
opt in.  The default transport remains pure stdlib for v0.4.0
minimum-viable.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``models.source`` (this module) -- HuggingFace adapter.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from infrastructure.logger import get_logger

__all__ = [
    "HuggingFaceSource",
    "HttpTransport",
    "UrllibTransport",
    "FileDownload",
    "DownloadProgress",
    # Constants exported for the sibling Civitai adapter so it can
    # construct a default transport with the same user-agent /
    # timeout.
    "DEFAULT_USER_AGENT",
    "DEFAULT_TIMEOUT",
    "DEFAULT_API_BASE",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Default HuggingFace API base URL.
DEFAULT_API_BASE: str = "https://huggingface.co"

#: Default user-agent sent with every request.
DEFAULT_USER_AGENT: str = "torcha-verse/0.4.0 (+https://github.com/torcha-verse/torcha-verse)"

#: Default request timeout (seconds).
DEFAULT_TIMEOUT: float = 30.0

#: Read buffer size when downloading a file body.
_CHUNK_SIZE = 1 << 16  # 64 KiB

#: Module-level logger.
_logger = get_logger("models.source.huggingface")


# ---------------------------------------------------------------------------
# Transport abstraction
# ---------------------------------------------------------------------------
@dataclass
class FileDownload:
    """A single downloaded file.

    Attributes:
        name: Filename (e.g. ``"config.json"``).
        data: File contents as bytes.
        sha256: Optional hex-encoded SHA-256 of ``data``.  When the
            source provides one (HF ``x-linked-etag`` / ``x-repo-commit``
            / ``ETag``), it is preferred; otherwise the adapter hashes
            locally.
    """

    name: str
    data: bytes
    sha256: str = ""


@dataclass
class DownloadProgress:
    """A single progress tick for a long-running download.

    Attributes:
        file_name: The file being downloaded.
        bytes_done: Bytes received so far for ``file_name``.
        bytes_total: Total expected bytes (``-1`` when the server
            did not advertise ``Content-Length``).
        mirror: The mirror base URL the bytes came from.  Useful
            for showing the user which mirror actually worked.
        started_at: Unix timestamp at the start of the download.
        finished: ``True`` on the final tick (i.e. when ``bytes_done``
            equals ``bytes_total``, or when the download errored
            before completion and ``error`` is set).
        error: Empty string on success, otherwise a short error
            description.
    """

    file_name: str
    bytes_done: int = 0
    bytes_total: int = -1
    mirror: str = ""
    started_at: float = 0.0
    finished: bool = False
    error: str = ""

    @property
    def percent(self) -> float:
        """``bytes_done / bytes_total`` as a fraction in ``[0, 1]``."""
        if self.bytes_total <= 0:
            return 0.0
        return min(1.0, max(0.0, self.bytes_done / self.bytes_total))

    def as_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "mirror": self.mirror,
            "started_at": self.started_at,
            "finished": self.finished,
            "error": self.error,
            "percent": round(self.percent, 4),
        }


class HttpTransport:
    """Pluggable HTTP transport interface.

    The fetcher talks to HuggingFace through this interface so that
    tests can swap in a fake transport without monkey-patching
    ``urllib``.  The default implementation
    :class:`UrllibTransport` uses the standard library.
    """

    def get_json(
        self, url: str, *, headers: Optional[Dict[str, str]] = None
    ) -> Any:
        """Issue a GET, decode the response as JSON, return the value."""
        raise NotImplementedError

    def get_bytes(
        self, url: str, *, headers: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, Dict[str, str]]:
        """Issue a GET, return ``(body, response_headers)``."""
        raise NotImplementedError


class UrllibTransport(HttpTransport):
    """Default :class:`HttpTransport` backed by ``urllib.request``.

    No third-party dependencies; works in any Python 3.9+
    environment.
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._user_agent = str(user_agent)
        self._timeout = float(timeout)

    def _request(
        self, url: str, headers: Optional[Dict[str, str]] = None
    ) -> Tuple[Any, Dict[str, str]]:
        hdrs = {"User-Agent": self._user_agent}
        if headers:
            hdrs.update(headers)
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            raw = resp.read()
            # urllib's headers are case-insensitive; casefold for safety.
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
        return raw, resp_headers

    def get_json(
        self, url: str, *, headers: Optional[Dict[str, str]] = None
    ) -> Any:
        hdrs = {"Accept": "application/json"}
        if headers:
            hdrs.update(headers)
        raw, _ = self._request(url, headers=hdrs)
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))

    def get_bytes(
        self, url: str, *, headers: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, Dict[str, str]]:
        raw, resp_headers = self._request(url, headers=headers)
        return raw, resp_headers


# ---------------------------------------------------------------------------
# HuggingFaceSource
# ---------------------------------------------------------------------------
class HuggingFaceSource:
    """Adapter for the HuggingFace Hub API.

    The class owns a single :class:`HttpTransport` and a thread-safe
    cache of repo-metadata lookups (HF's metadata is heavy enough
    that you don't want to re-fetch it on every file).

    Args:
        api_base: Base URL for the HF API.  Defaults to
            ``"https://huggingface.co"``.  Ignored when ``mirrors``
            is provided -- the first mirror becomes the primary
            base.
        transport: Optional :class:`HttpTransport` (mainly for
            testing).  When ``None`` a :class:`UrllibTransport` is
            used.
        token: Optional HuggingFace API token for gated / private
            repos.  Passed as ``Authorization: Bearer <token>``.
        mirrors: Optional :class:`~models.source.mirrors.MirrorSet`.
            When provided the adapter will try every mirror in
            order and silently fall back on network-level errors
            (see :func:`models.source.mirrors.is_useful_mirror_error`).
            When ``None`` the adapter uses the single ``api_base``
            URL (legacy behaviour).
    """

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        transport: Optional[HttpTransport] = None,
        token: Optional[str] = None,
        mirrors: Optional["MirrorSet"] = None,  # noqa: F821 - forward ref
    ) -> None:
        self._mirrors: "MirrorSet"  # type: ignore[valid-type]
        if mirrors is not None:
            self._mirrors = mirrors
            # Keep ``api_base`` pointing at the primary mirror for
            # backwards compatibility (e.g. repr, _build_url callers).
            self._api_base: str = mirrors.bases[0]
        else:
            self._api_base = api_base.rstrip("/")
        self._transport: HttpTransport = transport or UrllibTransport()
        self._token: Optional[str] = token
        self._meta_cache: Dict[str, Dict[str, Any]] = {}
        self._meta_lock: threading.Lock = threading.Lock()
        self._logger = _logger
        # Thread-safe mirror-failure memory: a base URL that just
        # failed is *suppressed* for the remainder of the process
        # so we do not pay the network round-trip twice.
        self._dead_mirrors: Dict[str, float] = {}
        self._dead_lock: threading.Lock = threading.Lock()
        self._dead_ttl_s: float = 60.0  # 1 minute TTL on "dead" memory

    @property
    def mirrors(self) -> Tuple[str, ...]:
        """Return the ordered mirror base URLs the adapter will try."""
        if hasattr(self, "_mirrors"):
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
        with self._dead_lock:
            self._dead_mirrors[base] = time.time()

    def _for_each_live_mirror(
        self,
        method: str,
        path: str,
        *,
        auth: bool = True,
    ) -> List[str]:
        """Return the live mirror base URLs (most-preferred first)
        for the given ``method + path``.

        "Live" means: not in the recent-failure set.  Used by the
        download loop to iterate the mirrors without paying for
        network calls against known-broken ones.
        """
        out: List[str] = []
        for base in self.mirrors:
            if self._is_mirror_dead(base):
                continue
            out.append(base)
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _auth_headers(self) -> Dict[str, str]:
        if self._token:
            return {"Authorization": "Bearer {}".format(self._token)}
        return {}

    def _api_url(self, repo_id: str, path: str, base: Optional[str] = None) -> str:
        return "{}/api/models/{}".format(base or self._api_base, repo_id)
        # NOTE: ``path`` is reserved for future endpoint expansion
        # (e.g. tree listing) without changing the public signature.
        _ = path

    def _resolve_url(
        self, repo_id: str, name: str, revision: str, base: Optional[str] = None
    ) -> str:
        """Build the public download URL for a file."""
        b = (base or self._api_base).rstrip("/")
        if revision:
            return "{}/{}/resolve/{}/{}".format(
                b, repo_id, revision, name,
            )
        return "{}/{}/resolve/HEAD/{}".format(
            b, repo_id, name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def resolve_license(
        self, repo_id: str, *, use_cache: bool = True
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
        for base in self._for_each_live_mirror("GET", "/api/models/{}".format(repo_id)):
            url = self._api_url(repo_id, "", base=base)
            try:
                data = self._transport.get_json(
                    url, headers=self._auth_headers()
                )
                break  # success -- fall through to parsing below
            except Exception as exc:  # noqa: BLE001 - any mirror failure
                self._logger.warning(
                    "HF metadata fetch failed for %s on %s: %s",
                    repo_id, base, exc,
                )
                # 5xx / network -> mark this mirror dead; 4xx -> caller problem.
                code = int(getattr(exc, "code", 0)) if isinstance(exc, urllib.error.HTTPError) else 0
                if 500 <= code < 600:
                    self._mark_mirror_dead(base)
                else:
                    # Treat all non-HTTPError as network-class -- the
                    # mirror is *probably* down for us right now.
                    if not isinstance(exc, urllib.error.HTTPError):
                        self._mark_mirror_dead(base)
                continue
        if data is None:
            # Every mirror failed; return empty so the license
            # check short-circuits to "no license declared".
            return ""

        if not isinstance(data, dict):
            return ""
        # Unify the two shapes.
        license_id = ""
        card = data.get("cardData")
        if isinstance(card, dict):
            raw = card.get("license")
            if isinstance(raw, str):
                license_id = raw
            elif isinstance(raw, list) and raw:
                # Some cards list multiple licenses; pick the first.
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
            # Some repos stash a "license_name" / "license_link"
            # instead of an SPDX id; we still surface them so the
            # whitelist can decide.
            for k in ("license_name", "licenseLink", "license_link"):
                v = data.get(k)
                if isinstance(v, str) and v:
                    license_id = v
                    break

        with self._meta_lock:
            self._meta_cache[repo_id] = dict(data)
            self._meta_cache[repo_id]["license"] = license_id
        return license_id

    def list_files(
        self,
        repo_id: str,
        revision: str = "main",
    ) -> List[str]:
        """Return the list of file names attached to a revision.

        Uses the ``/api/models/{repo_id}/tree/{revision}`` endpoint
        and returns the ``path`` of every entry whose ``type`` is
        ``"file"``.  When the request fails the method falls back
        to the next mirror in the configured :class:`MirrorSet`,
        and only when *every* mirror has failed returns an empty
        list (so the caller can fall back to a configured default
        file list, if any).
        """
        # Try every live mirror in order; the first successful
        # response wins.
        for base in self._for_each_live_mirror(
            "GET", "/api/models/{}/tree/{}".format(repo_id, revision or "main"),
        ):
            url = "{}/api/models/{}/tree/{}".format(
                base, repo_id, revision or "main",
            )
            try:
                data = self._transport.get_json(
                    url, headers=self._auth_headers()
                )
            except Exception as exc:  # noqa: BLE001 - mirror is best-effort
                self._logger.warning(
                    "HF tree listing failed for %s@%s on %s: %s",
                    repo_id, revision, base, exc,
                )
                # 5xx / network -> mark this mirror dead; 4xx -> caller-broken.
                code = int(getattr(exc, "code", 0)) if isinstance(exc, urllib.error.HTTPError) else 0
                if 500 <= code < 600:
                    self._mark_mirror_dead(base)
                elif not isinstance(exc, urllib.error.HTTPError):
                    self._mark_mirror_dead(base)
                continue
            if not isinstance(data, list):
                return []
            names: List[str] = []
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                if entry.get("type") != "file":
                    continue
                path = entry.get("path")
                if isinstance(path, str) and path:
                    names.append(path)
            return names
        return []

    def download_files(
        self,
        repo_id: str,
        revision: str,
        names: Sequence[str],
        *,
        on_progress: Optional[Callable[["DownloadProgress"], None]] = None,
    ) -> List[FileDownload]:
        """Download a list of files from a HF repo.

        Args:
            repo_id: HuggingFace repository id.
            revision: Source revision (``"main"``, a tag, a commit
                hash, ...).
            names: Sequence of file names to download.  Names that
                start with ``"~"`` are skipped (HF uses this prefix
                for non-public checkpoint fragments).
            on_progress: Optional callback invoked once per file
                with a :class:`DownloadProgress` describing the
                download.  Called at the *boundary* of each file
                (start tick + end tick) -- transport-level byte
                streaming is not exposed by :class:`HttpTransport`.

        Returns:
            A list of :class:`FileDownload` entries, in the same
            order as ``names``.  Names that 404 are simply omitted
            from the result -- the caller can decide whether to
            treat the missing file as fatal.
        """
        results: List[FileDownload] = []
        for name in names:
            if not name or name.startswith("~"):
                continue
            # Try every live mirror in order; fall back on the
            # next mirror when the current one fails with a
            # "useful" error (network / 5xx).
            file_download = self._download_one_with_fallback(
                repo_id, revision, name, on_progress=on_progress,
            )
            if file_download is not None:
                results.append(file_download)
        return results

    def _download_one_with_fallback(
        self,
        repo_id: str,
        revision: str,
        name: str,
        *,
        on_progress: Optional[Callable[["DownloadProgress"], None]] = None,
    ) -> Optional[FileDownload]:
        """Try every mirror for a single file; return the first success."""
        last_error = ""
        for base in self._for_each_live_mirror("GET", "{}/{}".format(repo_id, name)):
            url = self._resolve_url(repo_id, name, revision, base=base)
            t0 = time.time()
            if on_progress is not None:
                try:
                    on_progress(
                        DownloadProgress(
                            file_name=name,
                            bytes_done=0,
                            bytes_total=-1,
                            mirror=base,
                            started_at=t0,
                            finished=False,
                        )
                    )
                except Exception:  # noqa: BLE001 - progress must never break the download
                    pass
            try:
                body, resp_headers = self._transport.get_bytes(
                    url, headers=self._auth_headers()
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                last_error = "{}: {}".format(type(exc).__name__, exc)
                self._logger.warning(
                    "HF download failed for %s@%s/%s on %s: %s",
                    repo_id, revision, name, base, exc,
                )
                # 5xx / network -> mark this mirror dead; 4xx -> caller problem.
                if isinstance(exc, urllib.error.HTTPError):
                    code = int(getattr(exc, "code", 0))
                    if 500 <= code < 600:
                        self._mark_mirror_dead(base)
                else:
                    self._mark_mirror_dead(base)
                # Emit a finished-with-error tick so the UI can stop the spinner.
                if on_progress is not None:
                    try:
                        on_progress(
                            DownloadProgress(
                                file_name=name,
                                bytes_done=0,
                                bytes_total=-1,
                                mirror=base,
                                started_at=t0,
                                finished=True,
                                error=last_error,
                            )
                        )
                    except Exception:  # noqa: BLE001
                        pass
                continue
            # HF exposes the blob sha256 via x-linked-etag (the
            # Git LFS pointer) and the git blob sha via the
            # x-repo-commit + ETag combination.  We do not enforce
            # either -- we just record them when present.
            upstream_sha = (
                resp_headers.get("x-linked-etag", "")
                or resp_headers.get("etag", "")
            ).strip().strip('"')
            local_sha = hashlib.sha256(body).hexdigest()
            if on_progress is not None:
                try:
                    on_progress(
                        DownloadProgress(
                            file_name=name,
                            bytes_done=len(body),
                            bytes_total=len(body),
                            mirror=base,
                            started_at=t0,
                            finished=True,
                            error="",
                        )
                    )
                except Exception:  # noqa: BLE001
                    pass
            return FileDownload(
                name=name,
                data=body,
                sha256=upstream_sha or local_sha,
            )
        # Every mirror failed.
        self._logger.warning(
            "HF download exhausted all mirrors for %s@%s/%s (last error: %s)",
            repo_id, revision, name, last_error,
        )
        return None

    def download_default_artifacts(
        self,
        repo_id: str,
        revision: str = "main",
        *,
        on_progress: Optional[Callable[["DownloadProgress"], None]] = None,
    ) -> List[FileDownload]:
        """Download the four canonical HF artifacts (config, tokenizer, weights).

        Convenience helper that fetches:

        * ``config.json`` -- the model config.
        * ``tokenizer.json`` / ``tokenizer_config.json`` -- whichever
          the repo has.
        * ``model.safetensors`` (preferred) or ``pytorch_model.bin``
          (fallback) -- the weights.

        Args:
            repo_id: HF repository id.
            revision: Source revision (``"main"``, tag, commit hash,
                ...).
            on_progress: Optional progress callback forwarded to
                :meth:`download_files`.

        Returns:
            A list of :class:`FileDownload` entries (in the order
            they were resolved).  May be empty if the repo is
            missing every expected artifact -- the caller is
            expected to surface a useful error.
        """
        names: List[str] = []
        available = self.list_files(repo_id, revision or "main")
        has = set(available)
        for candidate in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "model.safetensors",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        ):
            if candidate in has:
                names.append(candidate)
        if not names and available:
            # Last resort: pull every file under 4 MiB (configs and
            # tokenizers are usually small) and skip big weight
            # blobs -- those will be requested explicitly.
            for n in available[:8]:
                if n.endswith((".json", ".txt", ".md", ".model")):
                    names.append(n)
        return self.download_files(
            repo_id, revision or "main", names, on_progress=on_progress,
        )

    def __repr__(self) -> str:
        return "HuggingFaceSource(api_base={!r})".format(self._api_base)
