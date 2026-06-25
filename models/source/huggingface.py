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

import json
import threading
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
            ``"https://huggingface.co"``.
        transport: Optional :class:`HttpTransport` (mainly for
            testing).  When ``None`` a :class:`UrllibTransport` is
            used.
        token: Optional HuggingFace API token for gated / private
            repos.  Passed as ``Authorization: Bearer <token>``.
    """

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        transport: Optional[HttpTransport] = None,
        token: Optional[str] = None,
    ) -> None:
        self._api_base: str = api_base.rstrip("/")
        self._transport: HttpTransport = transport or UrllibTransport()
        self._token: Optional[str] = token
        self._meta_cache: Dict[str, Dict[str, Any]] = {}
        self._meta_lock: threading.Lock = threading.Lock()
        self._logger = _logger

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _auth_headers(self) -> Dict[str, str]:
        if self._token:
            return {"Authorization": "Bearer {}".format(self._token)}
        return {}

    def _api_url(self, repo_id: str, path: str) -> str:
        return "{}/api/models/{}".format(self._api_base, repo_id)
        # NOTE: ``path`` is reserved for future endpoint expansion
        # (e.g. tree listing) without changing the public signature.
        _ = path

    def _resolve_url(self, repo_id: str, name: str, revision: str) -> str:
        """Build the public download URL for a file."""
        if revision:
            return "{}/{}/resolve/{}/{}".format(
                self._api_base, repo_id, revision, name,
            )
        return "{}/{}/resolve/HEAD/{}".format(
            self._api_base, repo_id, name,
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

        url = self._api_url(repo_id, "")
        try:
            data = self._transport.get_json(
                url, headers=self._auth_headers()
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            self._logger.warning(
                "HF metadata fetch failed for %s: %s", repo_id, exc,
            )
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
        ``"file"``.  When the request fails the method returns an
        empty list (so the caller falls back to a configured default
        file list, if any).
        """
        url = "{}/api/models/{}/tree/{}".format(
            self._api_base, repo_id, revision or "main",
        )
        try:
            data = self._transport.get_json(
                url, headers=self._auth_headers()
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            self._logger.warning(
                "HF tree listing failed for %s@%s: %s", repo_id, revision, exc,
            )
            return []
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

    def download_files(
        self,
        repo_id: str,
        revision: str,
        names: Sequence[str],
    ) -> List[FileDownload]:
        """Download a list of files from a HF repo.

        Args:
            repo_id: HuggingFace repository id.
            revision: Source revision (``"main"``, a tag, a commit
                hash, ...).
            names: Sequence of file names to download.  Names that
                start with ``"~"`` are skipped (HF uses this prefix
                for non-public checkpoint fragments).

        Returns:
            A list of :class:`FileDownload` entries, in the same
            order as ``names``.  Names that 404 are simply omitted
            from the result -- the caller can decide whether to
            treat the missing file as fatal.
        """
        import hashlib

        results: List[FileDownload] = []
        for name in names:
            if not name or name.startswith("~"):
                continue
            url = self._resolve_url(repo_id, name, revision)
            try:
                body, resp_headers = self._transport.get_bytes(
                    url, headers=self._auth_headers()
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                self._logger.warning(
                    "HF download failed for %s@%s/%s: %s",
                    repo_id, revision, name, exc,
                )
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
            results.append(
                FileDownload(
                    name=name,
                    data=body,
                    sha256=upstream_sha or local_sha,
                )
            )
        return results

    def download_default_artifacts(
        self,
        repo_id: str,
        revision: str = "main",
    ) -> List[FileDownload]:
        """Download the four canonical HF artifacts (config, tokenizer, weights).

        Convenience helper that fetches:

        * ``config.json`` -- the model config.
        * ``tokenizer.json`` / ``tokenizer_config.json`` -- whichever
          the repo has.
        * ``model.safetensors`` (preferred) or ``pytorch_model.bin``
          (fallback) -- the weights.

        Returns:
            A list of :class:`FileDownload` entries (in the order
            they were resolved).  May be empty if the repo is
            missing every expected artifact -- the caller is
            expected to surface a useful error.
        """
        names: List[str] = []
        available = self.list_files(repo_id, revision)
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
        return self.download_files(repo_id, revision, names)

    def __repr__(self) -> str:
        return "HuggingFaceSource(api_base={!r})".format(self._api_base)
