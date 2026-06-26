"""Data types and progress-reporting for the v0.6.x HuggingFace sub-package.

Two related dataclasses live here:

* :class:`FileDownload` -- the *result* of a single file
  download (filename + bytes + optional sha256).
* :class:`DownloadProgress` -- a *progress callback* payload
  that adapters emit to the caller during a long download.  The
  :attr:`DownloadProgress.percent` property and
  :meth:`DownloadProgress.as_dict` method are deliberately tiny
  so the adapter can construct it cheaply on every progress
  tick (a 1 MiB file may emit thousands of ticks).

These classes are pure data, no side effects, and importable
without any third-party dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

__all__ = ["FileDownload", "DownloadProgress"]


@dataclass
class FileDownload:
    """A single downloaded file.

    Attributes:
        name: Filename (e.g. ``"config.json"``).
        data: File contents as bytes.
        sha256: Hex-encoded SHA-256 digest **of ``data``** --
            always computed locally via ``hashlib.sha256(data)``
            by the adapter.  This is the *content* sha, NOT a
            header-extracted value:

            * HTTP headers like HF's ``x-linked-etag`` or
              ``ETag`` are **not** content digests for
              LFS-tracked files -- ``x-linked-etag`` is the
              LFS *pointer* git blob oid, and a few CDNs
              return the storage backend's blob oid.  Trusting
              them produced a non-deterministic cache
              fingerprint that broke cross-mirror dedup.
            * The cache manifest's ``CachedFile.sha256`` is
              keyed off this field, so the value must be the
              *content* sha to make ``ModelCache.verify()``
              succeed and to make the
              ``compute_content_fingerprint`` stable across
              mirrors.
            Defaults to ``""`` for backward compatibility with
            the v0.4.x on-disk cache format (older manifests
            may have empty sha, in which case ``verify()``
            will recompute the digest from the file bytes).
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

    def as_dict(self) -> Dict[str, object]:
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
