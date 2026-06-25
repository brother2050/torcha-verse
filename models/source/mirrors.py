"""Mirror catalog and helper for the TorchaVerse model fetcher (v0.4.x P2+).

HuggingFace Hub (``https://huggingface.co``) is sometimes slow,
inaccessible, or otherwise restricted in certain network
environments.  Operators regularly publish third-party *mirrors* that
expose the same repository / file layout with a different base URL;
for example ``https://hf-mirror.com`` and several
CDN-backed academic mirrors.

This module is the single source of truth for the known mirror
list, the :class:`MirrorSet` configuration object that the
:class:`~models.source.huggingface.HuggingFaceSource` consumes, and
a small :func:`check_mirror_health` helper that probes a base URL
and reports whether the mirror is reachable.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``models.source`` (this module) -- mirror catalog.

Threading: every public type is safe to share across threads.
"""
from __future__ import annotations

import os
import threading
import urllib.error
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from infrastructure.logger import get_logger

from .huggingface import DEFAULT_USER_AGENT, HttpTransport, UrllibTransport

__all__ = [
    "DEFAULT_HF_MIRRORS",
    "MirrorSet",
    "MirrorHealth",
    "check_mirror_health",
    "is_useful_mirror_error",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Known HuggingFace mirror base URLs, ordered from the operator's
#: preferred default to the long-tail public mirrors.  Each entry
#: preserves the trailing-path semantics of the upstream Hub, i.e.
#: ``{base}/{repo_id}/resolve/{revision}/{name}`` is a valid download
#: URL.
DEFAULT_HF_MIRRORS: Tuple[str, ...] = (
    # 1. Upstream (always tried first -- it is the canonical
    #    authority and the only one that knows about *brand-new*
    #    private repos).  Listed here for completeness; callers can
    #    omit it if they only want third-party mirrors.
    "https://huggingface.co",
    # 2. Community mirror widely used in mainland China.
    "https://hf-mirror.com",
)

#: Default request timeout for health probes (seconds).
DEFAULT_HEALTH_TIMEOUT: float = 5.0

#: A small JSON endpoint we use to probe mirror reachability.  The
#: endpoint is intentionally tiny (returns the upstream metadata for
#: a well-known public repo) so the probe is cheap.
_HEALTH_PROBE_REPO: str = "bert-base-uncased"
_HEALTH_PROBE_PATH: str = "/api/models/{}".format(_HEALTH_PROBE_REPO)

#: Module-level logger.
_logger = get_logger("models.source.mirrors")


# ---------------------------------------------------------------------------
# MirrorSet
# ---------------------------------------------------------------------------
@dataclass
class MirrorSet:
    """An ordered, deduplicated list of mirror base URLs.

    Attributes:
        bases: Tuple of base URLs, in the order the
            :class:`~models.source.huggingface.HuggingFaceSource`
            should try them.  The first entry is the *primary*
            (typically the upstream Hub).  Subsequent entries are
            fallback mirrors.
        health_check: When ``True`` (default) the
            :class:`~models.source.huggingface.HuggingFaceSource`
            will probe each mirror with a tiny ``GET`` before
            falling through; the probe is skipped when the request
            obviously cannot be useful (e.g. the URL is malformed).
        probe_timeout: Timeout in seconds for the health probe.
    """

    bases: Tuple[str, ...] = DEFAULT_HF_MIRRORS
    health_check: bool = True
    probe_timeout: float = DEFAULT_HEALTH_TIMEOUT

    def __post_init__(self) -> None:
        # Deduplicate while preserving order, and normalise trailing
        # slashes (some users paste ``.../`` accidentally).
        seen: List[str] = []
        for raw in self.bases:
            if not raw or not isinstance(raw, str):
                continue
            b = raw.strip().rstrip("/")
            if b and b not in seen:
                seen.append(b)
        if not seen:
            raise ValueError("MirrorSet requires at least one base URL")
        object.__setattr__(self, "bases", tuple(seen))

    @classmethod
    def default(cls) -> "MirrorSet":
        """Return the catalog default (:data:`DEFAULT_HF_MIRRORS`)."""
        return cls()

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
    ) -> "MirrorSet":
        """Build a :class:`MirrorSet` from ``$TORCHA_VERSE_HF_MIRRORS``.

        The environment variable holds a comma-separated list of
        base URLs.  When unset, the catalog default is used.  The
        upstream Hub (``https://huggingface.co``) is always
        prepended so private / brand-new repos still resolve.
        """
        if env is None:
            env = os.environ
        raw = env.get("TORCHA_VERSE_HF_MIRRORS", "").strip()
        if not raw:
            return cls()
        # Prepend the upstream to keep it authoritative, then append
        # the user's list (preserving user order within their own
        # list).
        upstream = "https://huggingface.co"
        bases: List[str] = [upstream]
        for chunk in raw.split(","):
            b = chunk.strip().rstrip("/")
            if b and b not in bases:
                bases.append(b)
        return cls(bases=tuple(bases))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@dataclass
class MirrorHealth:
    """Outcome of a :func:`check_mirror_health` probe.

    Attributes:
        base: The mirror base URL that was probed.
        reachable: ``True`` when the probe returned a 2xx or 3xx
            status (the exact status does not matter -- HF returns
            ``200`` for the metadata endpoint of a public repo).
        status_code: The HTTP status code returned by the probe
            (``0`` when the request never reached the server).
        elapsed_s: Wall-clock time spent probing.
        error: A short, human-readable error description.  Empty
            string on success.
    """

    base: str
    reachable: bool
    status_code: int = 0
    elapsed_s: float = 0.0
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "base": self.base,
            "reachable": self.reachable,
            "status_code": self.status_code,
            "elapsed_s": round(self.elapsed_s, 4),
            "error": self.error,
        }

    def __repr__(self) -> str:
        return (
            "MirrorHealth(base={!r}, reachable={}, status={}, "
            "elapsed_s={:.3f}, error={!r})"
        ).format(
            self.base,
            self.reachable,
            self.status_code,
            self.elapsed_s,
            self.error,
        )


def is_useful_mirror_error(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` justifies trying the next mirror.

    Network-level errors (DNS, connection refused, timeout) and
    5xx HTTP errors are "useful" -- the mirror is broken for us.
    Authentication / 4xx errors are NOT useful -- they signal that
    the caller did something wrong (e.g. the repo does not exist
    on any mirror) and trying the next mirror would just waste a
    network round-trip.

    Note: :class:`urllib.error.HTTPError` is a subclass of
    :class:`urllib.error.URLError`, so we have to inspect
    :class:`HTTPError` *before* the catch-all ``URLError`` branch
    otherwise every HTTP error -- including 4xx -- would be
    classified as "network trouble" and trigger a mirror
    fallback.
    """
    if isinstance(exc, urllib.error.HTTPError):
        # 5xx -> mirror-side trouble.  4xx -> caller-side trouble.
        return 500 <= int(getattr(exc, "code", 0)) < 600
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, OSError)):
        return True
    return False


def check_mirror_health(
    base: str,
    *,
    transport: Optional[HttpTransport] = None,
    timeout: float = DEFAULT_HEALTH_TIMEOUT,
) -> MirrorHealth:
    """Probe a mirror with a tiny ``GET`` to a well-known repo.

    The probe is *advisory*: a transient failure is recorded but
    the caller is free to ignore it.  The function never raises;
    every error is captured into :attr:`MirrorHealth.error` so the
    caller can build a report.
    """
    import time

    base = base.rstrip("/")
    url = base + _HEALTH_PROBE_PATH
    tr = transport or UrllibTransport(timeout=timeout)
    t0 = time.time()
    try:
        _ = tr.get_json(url, headers={"User-Agent": DEFAULT_USER_AGENT})
        return MirrorHealth(
            base=base,
            reachable=True,
            status_code=200,
            elapsed_s=time.time() - t0,
        )
    except urllib.error.HTTPError as exc:
        # 404 on a public repo == mirror is reachable but the
        # well-known repo is missing (rare).  Treat 2xx/3xx/404
        # as "reachable" -- the mirror answered us.
        code = int(getattr(exc, "code", 0))
        reachable = 200 <= code < 500
        return MirrorHealth(
            base=base,
            reachable=reachable,
            status_code=code,
            elapsed_s=time.time() - t0,
            error="" if reachable else "HTTP {}".format(code),
        )
    except Exception as exc:  # noqa: BLE001 - probe is best-effort
        return MirrorHealth(
            base=base,
            reachable=False,
            status_code=0,
            elapsed_s=time.time() - t0,
            error="{}: {}".format(type(exc).__name__, exc),
        )


def check_all_mirrors(
    bases: Sequence[str],
    *,
    transport: Optional[HttpTransport] = None,
    timeout: float = DEFAULT_HEALTH_TIMEOUT,
) -> List[MirrorHealth]:
    """Run :func:`check_mirror_health` against every base in ``bases``.

    Returns a list of :class:`MirrorHealth` in the same order as
    ``bases``.  Useful for an operator who wants to see "which
    mirror works for me right now" before triggering a fetch.
    """
    out: List[MirrorHealth] = []
    for b in bases:
        out.append(
            check_mirror_health(b, transport=transport, timeout=timeout)
        )
    return out
