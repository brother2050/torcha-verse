"""The :func:`download_one_with_fallback` helper.

This module contains the single largest method of the original
:class:`HuggingFaceSource` -- ``_download_one_with_fallback``
(155 lines in the v0.5.x single-file implementation) --
factored into a module-level function that takes the
``HuggingFaceSource`` instance as its first argument.

Why a module-level function and not a method?

* It is the *one* method that does enough network / mirror
  bookkeeping that it benefits from being unit-testable in
  isolation, which is awkward from a class method.
* The :class:`HuggingFaceSource` class is now focused on the
  metadata API (``resolve_license`` / ``list_files`` /
  ``download_files``), while the actual byte-level download
  with progress reporting lives here.

The function is exported as a public helper so the
:meth:`HuggingFaceSource._download_one_with_fallback` method
can simply delegate to it.
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
from typing import Callable, List, Mapping, Optional, TYPE_CHECKING

from infrastructure.logger import get_logger

from ..auth import (
    ChecksumMismatch,
    GatedRepoError,
    extract_expected_sha256_from_headers,
    is_gated_http_error,
)
from ._types import DownloadProgress, FileDownload

if TYPE_CHECKING:
    from ._source import HuggingFaceSource

__all__ = [
    "download_one_with_fallback",
    "DEFAULT_ARTIFACT_CANDIDATES",
    "download_default_artifacts",
]


_logger = get_logger("models.source.huggingface.download")


#: Default set of candidate file names for
#: :func:`download_default_artifacts`.  Ordered by preference;
#: the first ones that exist in the repo are downloaded.
DEFAULT_ARTIFACT_CANDIDATES: tuple = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def download_one_with_fallback(
    source: "HuggingFaceSource",
    repo_id: str,
    revision: str,
    name: str,
    *,
    expected_sha256: str = "",
    on_progress: Optional[Callable[[DownloadProgress], None]] = None,
) -> Optional[FileDownload]:
    """Try every mirror for a single file; return the first success.

    Args:
        source: The :class:`HuggingFaceSource` instance providing
            the transport, the mirror list, and the auth
            headers.
        repo_id, revision, name: The HF resource locator.
        expected_sha256: Optional 64-char hex SHA-256 the
            caller *requires* this file to match.  When
            non-empty and the local hash does not match, the
            function raises :class:`~models.source.auth.ChecksumMismatch`.
        on_progress: Optional progress callback.  Receives a
            :class:`DownloadProgress` event at start, finish,
            and on errors so the caller can drive a UI.

    Returns:
        A :class:`FileDownload` on success, or ``None`` when
        every mirror failed.
    """
    last_error = ""
    for base in source._for_each_live_mirror(
        "GET", "{}/{}".format(repo_id, name)
    ):
        url = source._resolve_url(repo_id, name, revision, base=base)
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
            except Exception as exc:  # noqa: BLE001
                _logger.debug("progress update (started) suppressed: %s", exc)
        try:
            body, resp_headers = source._transport.get_bytes(
                url, headers=source._auth_headers()
            )
        except Exception as exc:  # noqa: BLE001
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
            last_error = "{}: {}".format(type(exc).__name__, exc)
            _logger.warning(
                "HF download failed for %s@%s/%s on %s: %s",
                repo_id, revision, name, base, exc,
            )
            code = (
                int(getattr(exc, "code", 0))
                if isinstance(exc, urllib.error.HTTPError)
                else 0
            )
            if 500 <= code < 600:
                source._mark_mirror_dead(base)
            elif not isinstance(exc, urllib.error.HTTPError):
                source._mark_mirror_dead(base)
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
                except Exception as exc:  # noqa: BLE001
                    _logger.debug("progress update (retry) suppressed: %s", exc)
            continue
        # HF exposes the *LFS pointer* oid via ``x-linked-etag``
        # and the *git blob* sha via ``x-repo-commit`` + ``ETag``.
        # **Neither is guaranteed to equal the content sha256**:
        # LFS-tracked files (.safetensors, .bin, .gguf, ...) are
        # served as a 100-ish-byte pointer file whose own sha is
        # useless; the real sha is ``sha256(body)`` of the resolved
        # bytes.  We use the header value only as a *debug hint* --
        # if it disagrees with the locally-computed sha we log a
        # warning so the operator can spot a misconfigured mirror,
        # but we never let it override the authoritative local
        # digest.  The previous behaviour of using
        # ``upstream_sha or local_sha`` made the cache fingerprint
        # non-deterministic across mirrors, because two mirrors
        # resolving the same LFS file could legitimately return
        # different ``x-linked-etag`` values.
        upstream_hint = extract_expected_sha256_from_headers(
            resp_headers, file_name=name,
        )
        local_sha = hashlib.sha256(body).hexdigest()
        if upstream_hint and upstream_hint != local_sha:
            _logger.debug(
                "HF header sha hint for %s/%s from %s "
                "(hint=%s) differs from content sha %s; "
                "using local sha as authoritative (LFS pointer "
                "oid is not a content digest).",
                repo_id, name, base, upstream_hint, local_sha,
            )
        if expected_sha256 and local_sha != expected_sha256:
            _logger.error(
                "Checksum mismatch on %s/%s from %s: expected=%s actual=%s",
                repo_id, name, base, expected_sha256, local_sha,
            )
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
                            error="checksum mismatch",
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    _logger.debug("progress update (mismatch) suppressed: %s", exc)
            raise ChecksumMismatch(
                source="huggingface",
                repo_id=repo_id,
                file_name=name,
                expected_sha256=expected_sha256,
                actual_sha256=local_sha,
            )
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
            except Exception as exc:  # noqa: BLE001
                _logger.debug("progress update (finished) suppressed: %s", exc)
        return FileDownload(
            name=name,
            data=body,
            sha256=local_sha,
        )
    _logger.warning(
        "HF download exhausted all mirrors for %s@%s/%s (last error: %s)",
        repo_id, revision, name, last_error,
    )
    return None


def download_default_artifacts(
    source: "HuggingFaceSource",
    repo_id: str,
    revision: str = "main",
    *,
    expected_sha256s: Optional[Mapping[str, str]] = None,
    on_progress: Optional[Callable[[DownloadProgress], None]] = None,
) -> List[FileDownload]:
    """Download the four canonical HF artifacts (config, tokenizer, weights).

    Args:
        source: The :class:`HuggingFaceSource` to drive.
        repo_id: HF repository id.
        revision: Source revision (``"main"``, tag, commit hash,
            ...).
        expected_sha256s: Optional per-file integrity map.
        on_progress: Optional progress callback.

    Returns:
        A list of :class:`FileDownload` entries (in the order
        they were resolved).
    """
    names: List[str] = []
    available = source.list_files(repo_id, revision or "main")
    has = set(available)
    for candidate in DEFAULT_ARTIFACT_CANDIDATES:
        if candidate in has:
            names.append(candidate)
    if not names and available:
        for n in available[:8]:
            if n.endswith((".json", ".txt", ".md", ".model")):
                names.append(n)
    return source.download_files(
        repo_id, revision or "main", names,
        expected_sha256s=expected_sha256s,
        on_progress=on_progress,
    )
