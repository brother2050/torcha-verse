"""Model download via the v0.4.x P2+ mirror + dedup + progress path.

This example demonstrates the full ``ModelFetcher`` API surface
relevant to production model fetches:

1. **Mirror list construction** -- build a :class:`MirrorSet` from
   the default catalog or from ``$TORCHA_VERSE_HF_MIRRORS``.
2. **Mirror health check** -- probe each mirror with a tiny
   :func:`check_mirror_health` call before kicking off a real
   fetch, so the operator can see "which mirror works for me
   right now".
3. **Per-file progress callback** -- a :class:`DownloadProgress`
   callback prints a live progress line for every file the HF
   adapter downloads.
4. **Cross-mirror dedup** -- fetch the *same* file set twice under
   different (repo_id, revision) keys; the second call should
   short-circuit on the content fingerprint and return
   ``from_cache=True`` without re-downloading.

The example is **zero-network** by default: it uses an
in-memory :class:`FakeTransport` (defined inline) instead of
``UrllibTransport``.  Operators who want to talk to real mirrors
can pass ``--real`` to swap in the default urllib transport.

Run with::

    python examples/model_download.py
    python examples/model_download.py --real     # talk to real mirrors (network)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.source import (
    DownloadProgress,
    HttpTransport,
    HuggingFaceSource,
    ModelCache,
    ModelFetcher,
    MirrorHealth,
    MirrorSet,
    SourceRegistry,
    check_all_mirrors,
)


# ---------------------------------------------------------------------------
# Fake transport (zero-network)
# ---------------------------------------------------------------------------
class FakeTransport(HttpTransport):
    """An in-memory :class:`HttpTransport` for the demo.

    Serves a fixed "tiny model" payload from a configurable base
    URL, and *fails* on every other base URL -- so we can prove
    that the :class:`ModelFetcher` falls back across mirrors
    without ever touching the network.

    The "etag" we return is a real SHA-256 of the payload so the
    cache integrity check passes.
    """

    def __init__(self, working_base: str, bytes_per_file: int = 256) -> None:
        self._working_base = working_base.rstrip("/")
        self._bytes_per_file = bytes_per_file
        self._hits: Dict[str, int] = {}
        self._lock = threading.Lock()

    def _count(self, url: str) -> None:
        with self._lock:
            self._hits[url] = self._hits.get(url, 0) + 1

    @property
    def hit_count(self) -> int:
        with self._lock:
            return sum(self._hits.values())

    def get_json(self, url: str, headers: Optional[Dict[str, str]] = None) -> Any:
        self._count(url)
        if not url.startswith(self._working_base):
            raise ConnectionError("mirror not reachable: {}".format(url))
        if "/tree/" in url:
            return [
                {"type": "file", "path": "config.json"},
                {"type": "file", "path": "model.safetensors"},
            ]
        return {
            "cardData": {"license": "apache-2.0"},
            "license": "apache-2.0",
        }

    def get_bytes(
        self, url: str, headers: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, Dict[str, str]]:
        self._count(url)
        if not url.startswith(self._working_base):
            raise ConnectionError("mirror not reachable: {}".format(url))
        # Pretend every file is 256 bytes; the SHA will be
        # different because we use the file name as a salt.
        name = url.rsplit("/", 1)[-1]
        payload = ("{}|{}".format(name, "x" * self._bytes_per_file)).encode("utf-8")
        etag = hashlib.sha256(payload).hexdigest()
        return payload, {"x-linked-etag": etag}


# ---------------------------------------------------------------------------
# Progress + health printing
# ---------------------------------------------------------------------------
def _print_progress(tick: DownloadProgress) -> None:
    pct = tick.percent * 100
    if tick.error and not tick.finished:
        return  # mid-failure ticks are noisy -- skip
    if tick.finished and tick.error:
        sys.stdout.write(
            "\n  ✗ {} ({}): {}\n".format(tick.file_name, tick.mirror, tick.error)
        )
        sys.stdout.flush()
        return
    sys.stdout.write(
        "\r  ▏{} {:6.2f}% {:>10}/{:>10} bytes (via {})".format(
            tick.file_name, pct, tick.bytes_done, max(tick.bytes_total, 0), tick.mirror,
        )
    )
    sys.stdout.flush()


def _print_health(label: str, health: List[MirrorHealth]) -> None:
    print("\n  mirror health ({}):".format(label))
    for h in health:
        mark = "✓" if h.reachable else "✗"
        print(
            "    {} {:40s} reachable={} status={} elapsed_s={:.3f} error={!r}".format(
                mark, h.base, h.reachable, h.status_code, h.elapsed_s, h.error,
            )
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real", action="store_true",
        help="use UrllibTransport (talks to real mirrors; needs network)",
    )
    parser.add_argument(
        "--cache", default=None,
        help="cache root (default: $TORCHA_VERSE_CACHE or ~/.cache/torcha-verse-demo)",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("TorchaVerse — v0.4.x P2+ Model Download (mirror + dedup + progress)")
    print("=" * 60)

    # 1. Build a mirror set.
    mirrors = MirrorSet.from_env()
    print("\n[1] mirror set: {}".format(list(mirrors.bases)))

    # 2. Pick transport.  In demo mode the "working" mirror is
    #    hf-mirror.com (so we can prove the upstream fails and
    #    the fallback wins).
    if args.real:
        transport = None  # use UrllibTransport
    else:
        working = (
            "https://hf-mirror.com"
            if "https://hf-mirror.com" in mirrors.bases
            else mirrors.bases[0]
        )
        transport = FakeTransport(working_base=working)

    # 3. Health check.
    health = check_all_mirrors(mirrors.bases, transport=transport)
    _print_health("initial probe", health)

    # 4. Build the fetcher.
    cache_root = args.cache or os.environ.get(
        "TORCHA_VERSE_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "torcha-verse-demo"),
    )
    cache = ModelCache(root=cache_root)
    print("\n[2] cache root: {}".format(cache.root))

    # The default registry instantiates HF + Civitai with the
    # real UrllibTransport.  In demo mode we want zero network,
    # so we *replace* the HF adapter with one that uses our
    # FakeTransport.
    registry = SourceRegistry.default()
    if transport is not None:
        registry.register(
            "huggingface",
            HuggingFaceSource(mirrors=mirrors, transport=transport),
        )
    fetcher = ModelFetcher(
        cache=cache, registry=registry, mirrors=mirrors,
    )

    # 5. First fetch -- expect from_cache=False.
    repo = "demo/tiny-model"
    rev = "v1"
    print("\n[3] first fetch ({}@{}) -- expect from_cache=False".format(repo, rev))
    t0 = time.time()
    res1 = fetcher.fetch(
        "huggingface", repo, revision=rev,
        on_progress=_print_progress,
    )
    print("\n  -> from_cache={} files={} elapsed={:.2f}s".format(
        res1.from_cache, len(res1.manifest.files), time.time() - t0,
    ))

    # 6. Second fetch -- same key, expect from_cache=True.
    print("\n[4] second fetch ({}@{}) -- expect from_cache=True (same key)".format(repo, rev))
    t0 = time.time()
    res2 = fetcher.fetch("huggingface", repo, revision=rev)
    print("  -> from_cache={} elapsed={:.2f}s".format(res2.from_cache, time.time() - t0))

    # 7. Third fetch -- *different* revision, same content: cross-mirror
    #    dedup should kick in (content_fingerprint matches).
    print("\n[5] third fetch ({}@{}) -- expect from_cache=True (content dedup)".format(repo, "v1.1"))
    t0 = time.time()
    res3 = fetcher.fetch("huggingface", repo, revision="v1.1")
    print("  -> from_cache={} cached_as={}@{} elapsed={:.2f}s".format(
        res3.from_cache, res3.manifest.repo_id, res3.manifest.revision,
        time.time() - t0,
    ))

    # 8. Health check after traffic -- the "broken" mirror should
    #    have been suppressed (dead-mirror memory) for the
    #    remainder of the process.
    print("\n[6] mirror health (after traffic):")
    health2 = check_all_mirrors(mirrors.bases, transport=transport)
    _print_health("post-traffic", health2)

    if isinstance(transport, FakeTransport):
        print("\n[stats] fake transport served {} HTTP calls".format(transport.hit_count))

    print("\nDemo complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
