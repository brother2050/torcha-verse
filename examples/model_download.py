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
5. **Token resolution + expected_sha256s** -- show how the
   fetcher picks up ``$HF_TOKEN`` automatically and how the
   caller can pin a per-file ``sha256`` to refuse a tampered
   mirror response.  Both ``GatedRepoError`` (401/403) and
   :class:`ChecksumMismatch` are demonstrated.

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
    ChecksumMismatch,
    DownloadProgress,
    GatedRepoError,
    HttpTransport,
    HuggingFaceSource,
    ModelCache,
    ModelFetcher,
    MirrorHealth,
    MirrorSet,
    SourceRegistry,
    TokenInfo,
    auth_headers,
    check_all_mirrors,
    resolve_token,
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

    A *gated* base URL is supported (constructor arg) so the
    token-resolution / :class:`GatedRepoError` demo can show
    401/403 handling without a real network call.
    """

    def __init__(
        self,
        working_base: str,
        bytes_per_file: int = 256,
        gated_base: Optional[str] = None,
    ) -> None:
        self._working_base = working_base.rstrip("/")
        self._gated_base = gated_base.rstrip("/") if gated_base else None
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
        if self._gated_base and url.startswith(self._gated_base):
            import urllib.error
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
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
        if self._gated_base and url.startswith(self._gated_base):
            import urllib.error
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
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

    # ------------------------------------------------------------------
    # 9. Token resolution demo -- show the lookup chain.
    # ------------------------------------------------------------------
    print("\n[7] token resolution (env + on-disk file fallback)")
    ti_env = resolve_token(
        env={"HF_TOKEN": "demo_hf_env"}, sources="huggingface",
    )
    print("  env var:        present={} source={} value_redacted={}".format(
        ti_env.is_present, ti_env.source, ti_env.as_dict()["value_redacted"],
    ))
    print("  explicit None:  present={} source={}".format(
        ti_env.is_present, ti_env.source,
    ))
    print("  empty explicit: present={} source={}".format(
        resolve_token(explicit="", env={}).is_present,
        resolve_token(explicit="", env={}).source,
    ))
    print("  no token:       present={} source={}".format(
        resolve_token(env={}, sources="huggingface").is_present,
        resolve_token(env={}, sources="huggingface").source,
    ))
    # Show the auth_headers helper builds the right shape.
    print("  auth_headers(present) = {}".format(
        auth_headers(ti_env),
    ))
    print("  auth_headers(None)    = {}".format(auth_headers(None)))

    # ------------------------------------------------------------------
    # 10. expected_sha256s + ChecksumMismatch demo.  We pin the
    #     *correct* sha for one file and a *wrong* sha for another,
    #     and the second fetch should raise ChecksumMismatch before
    #     any bytes hit the cache.
    # ------------------------------------------------------------------
    print("\n[8] expected_sha256s demo")
    # Compute the actual hashes of the "tiny" payload the
    # FakeTransport serves.
    config_payload = ("config.json|{}".format(
        "x" * transport._bytes_per_file,
    )).encode("utf-8")
    weights_payload = ("model.safetensors|{}".format(
        "x" * transport._bytes_per_file,
    )).encode("utf-8")
    config_sha = hashlib.sha256(config_payload).hexdigest()
    weights_sha = hashlib.sha256(weights_payload).hexdigest()
    # Use a fresh fetcher pointed at a *fresh* cache so we
    # exercise the full download path.
    cache2 = ModelCache(root=cache_root + "-integrity")
    fetcher2 = ModelFetcher(cache=cache2, registry=registry, mirrors=mirrors)

    # 10a. Correct pins -- write succeeds.
    print("  [a] correct pins -- expect from_cache=False")
    ok = fetcher2.fetch(
        "huggingface", "demo/tiny-model", revision="v2",
        expected_sha256s={
            "config.json": config_sha,
            "model.safetensors": weights_sha,
        },
    )
    print("      from_cache={} files={}".format(
        ok.from_cache, len(ok.manifest.files),
    ))

    # 10b. Wrong pin -- ChecksumMismatch.
    print("  [b] wrong pin (config.json) -- expect ChecksumMismatch")
    cache3 = ModelCache(root=cache_root + "-integrity-bad")
    fetcher3 = ModelFetcher(cache=cache3, registry=registry, mirrors=mirrors)
    try:
        fetcher3.fetch(
            "huggingface", "demo/tiny-model", revision="v3",
            expected_sha256s={
                "config.json": "f" * 64,  # intentionally wrong
                "model.safetensors": weights_sha,
            },
        )
        print("      ERROR: no exception raised!")
    except ChecksumMismatch as exc:
        print("      caught: {} file={} expected={}...".format(
            type(exc).__name__, exc.file_name,
            exc.expected_sha256[:8],
        ))
        # Crucially: no manifest was written.
        assert not cache3.has("huggingface", "demo/tiny-model", "v3")
        print("      manifest NOT written (cache stays clean)")

    # 10c. validate_checksums=False -- the wrong pin is silently
    #     accepted.  Use only for trusted internal feeds.
    print("  [c] validate_checksums=False -- expect the wrong pin to pass")
    cache4 = ModelCache(root=cache_root + "-integrity-skip")
    fetcher4 = ModelFetcher(cache=cache4, registry=registry, mirrors=mirrors)
    skipped = fetcher4.fetch(
        "huggingface", "demo/tiny-model", revision="v4",
        expected_sha256s={"config.json": "f" * 64},
        validate_checksums=False,
    )
    print("      from_cache={} files={}".format(
        skipped.from_cache, len(skipped.manifest.files),
    ))

    # ------------------------------------------------------------------
    # 11. GatedRepoError demo -- point the fetch at a "gated" base
    #     and confirm the 401 surfaces as a GatedRepoError rather
    #     than a generic failure.
    # ------------------------------------------------------------------
    print("\n[9] GatedRepoError demo (401 surfaces with helpful hint)")
    gated_mirrors = MirrorSet(
        bases=("https://gated.example.com",) + tuple(mirrors.bases),
    )
    gated_transport = FakeTransport(
        working_base=mirrors.bases[0], gated_base="https://gated.example.com",
    )
    gated_cache = ModelCache(root=cache_root + "-gated")
    gated_reg = SourceRegistry()
    gated_reg.register(
        "huggingface",
        HuggingFaceSource(mirrors=gated_mirrors, transport=gated_transport),
    )
    gated_fetcher = ModelFetcher(
        cache=gated_cache, registry=gated_reg, mirrors=gated_mirrors,
    )
    try:
        gated_fetcher.fetch(
            "huggingface", "gated/secret-model", revision="main",
            token="demo-secret-token",  # the fake still 401s -- the demo
                                         # is about the *error* path
        )
        print("      ERROR: no exception raised!")
    except GatedRepoError as exc:
        print("      caught: GatedRepoError source={} status={}".format(
            exc.source, exc.status_code,
        ))
        print("      hint:   {}".format(exc.hint))

    print("\nDemo complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
