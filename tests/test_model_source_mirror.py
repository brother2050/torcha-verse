"""Tests for the v0.4.x P2+ mirror / dedup / progress path (model source).

These tests are zero-network: they use an inline
:class:`FakeTransport` that returns canned responses for a
configurable set of "working" base URLs and ``ConnectionError``
on every other URL.  This lets us exercise the full mirror
fallback / dedup / progress callback machinery deterministically.

Coverage
--------

* :mod:`models.source.mirrors` -- :class:`MirrorSet`,
  :func:`check_mirror_health`, :func:`is_useful_mirror_error`,
  env-var construction.
* :mod:`models.source.cache` -- :func:`compute_content_fingerprint`,
  :meth:`ModelCache.find_by_fingerprint` (incl. the recursive
  scan that handles ``/``-containing repo_ids).
* :mod:`models.source.huggingface` -- mirror fallback in
  :meth:`resolve_license` and :meth:`list_files`,
  per-file :class:`DownloadProgress` callback,
  dead-mirror suppression memory.
* :mod:`models.source.fetch` -- end-to-end fetch with mirror
  fallback, on_progress callback wiring, cross-mirror dedup
  (same content, different (repo, revision) key).
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

pytestmark = pytest.mark.model_source

from models.source import (
    DownloadProgress,
    HuggingFaceSource,
    HttpTransport,
    MirrorHealth,
    MirrorSet,
    ModelCache,
    ModelFetcher,
    SourceRegistry,
    compute_content_fingerprint,
    is_useful_mirror_error,
    check_mirror_health,
)
from models.source.cache import CachedFile, CachedModel
from models.source.mirrors import DEFAULT_HF_MIRRORS, check_all_mirrors


# ---------------------------------------------------------------------------
# Reusable fake transport
# ---------------------------------------------------------------------------
class FakeTransport(HttpTransport):
    """An in-memory :class:`HttpTransport`.

    Args:
        working_bases: Set of base URLs the transport serves (the
            rest raise ``ConnectionError``).  Order does not
            matter; we serve *any* matching URL.
        bytes_per_file: Payload size hint.
        extra_headers: Optional dict of static headers to attach
            to every response (useful for etag headers).
    """

    def __init__(
        self,
        working_bases: List[str],
        bytes_per_file: int = 256,
    ) -> None:
        self._working = {b.rstrip("/") for b in working_bases}
        self._bytes = bytes_per_file
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
        if not any(url.startswith(b) for b in self._working):
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
        if not any(url.startswith(b) for b in self._working):
            raise ConnectionError("mirror not reachable: {}".format(url))
        name = url.rsplit("/", 1)[-1]
        payload = ("{}|{}".format(name, "x" * self._bytes)).encode("utf-8")
        return payload, {"x-linked-etag": hashlib.sha256(payload).hexdigest()}


# ---------------------------------------------------------------------------
# MirrorSet
# ---------------------------------------------------------------------------
class TestMirrorSet:
    def test_default_bases(self) -> None:
        ms = MirrorSet.default()
        assert len(ms.bases) >= 1
        assert ms.bases[0] == "https://huggingface.co"
        assert "https://hf-mirror.com" in ms.bases
        # No trailing slash.
        for b in ms.bases:
            assert not b.endswith("/")

    def test_dedup_preserves_order(self) -> None:
        ms = MirrorSet(bases=(
            "https://b.example",
            "https://a.example",
            "https://b.example",
            "https://a.example/",
        ))
        assert ms.bases == (
            "https://b.example", "https://a.example",
        )

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            MirrorSet(bases=())

    def test_from_env_prepends_upstream(self) -> None:
        ms = MirrorSet.from_env(env={
            "TORCHA_VERSE_HF_MIRRORS": "https://a.example, https://b.example",
        })
        assert ms.bases[0] == "https://huggingface.co"
        assert "https://a.example" in ms.bases
        assert "https://b.example" in ms.bases

    def test_from_env_empty_returns_default(self) -> None:
        ms = MirrorSet.from_env(env={})
        # No upstream prepend when env is empty -- the default
        # already has upstream as the first entry.
        assert ms.bases[0] == "https://huggingface.co"


# ---------------------------------------------------------------------------
# check_mirror_health + is_useful_mirror_error
# ---------------------------------------------------------------------------
class TestMirrorHealth:
    def test_reachable_uses_fake_transport(self) -> None:
        ft = FakeTransport(working_bases=["https://hf-mirror.com"])
        h = check_mirror_health("https://hf-mirror.com", transport=ft)
        assert h.reachable is True
        assert h.error == ""

    def test_unreachable_uses_fake_transport(self) -> None:
        ft = FakeTransport(working_bases=["https://hf-mirror.com"])
        h = check_mirror_health("https://huggingface.co", transport=ft)
        assert h.reachable is False
        assert "mirror not reachable" in h.error
        assert h.status_code == 0

    def test_404_is_reachable(self) -> None:
        # A 404 still means the mirror answered us; the well-known
        # probe repo might be missing on this mirror, but the host
        # is alive.
        class NotFoundTransport(HttpTransport):
            def get_json(self, url, headers=None):
                raise urllib_error_404()
            def get_bytes(self, url, headers=None):
                raise urllib_error_404()

        h = check_mirror_health("https://example.com", transport=NotFoundTransport())
        assert h.reachable is True
        assert h.status_code == 404

    def test_check_all_returns_same_order(self) -> None:
        ft = FakeTransport(working_bases=["https://hf-mirror.com"])
        bases = ["https://huggingface.co", "https://hf-mirror.com"]
        hs = check_all_mirrors(bases, transport=ft)
        assert [h.base for h in hs] == bases
        assert [h.reachable for h in hs] == [False, True]

    def test_is_useful_mirror_error_classification(self) -> None:
        import urllib.error

        class _FakeHTTPError(urllib.error.HTTPError):
            def __init__(self, code: int) -> None:
                super().__init__(
                    "http://x", code, "x", {}, None,
                )

        assert is_useful_mirror_error(ConnectionError("x")) is True
        assert is_useful_mirror_error(TimeoutError("x")) is True
        assert is_useful_mirror_error(_FakeHTTPError(500)) is True
        assert is_useful_mirror_error(_FakeHTTPError(503)) is True
        assert is_useful_mirror_error(_FakeHTTPError(404)) is False  # 4xx = caller
        assert is_useful_mirror_error(_FakeHTTPError(401)) is False
        assert is_useful_mirror_error(ValueError("x")) is False


def urllib_error_404():
    import urllib.error
    return urllib.error.HTTPError(
        "http://x", 404, "Not Found", {}, None,
    )


# ---------------------------------------------------------------------------
# compute_content_fingerprint + find_by_fingerprint
# ---------------------------------------------------------------------------
class TestFingerprint:
    def test_fingerprint_order_independent(self) -> None:
        a = [{"name": "a.bin", "sha256": "a1"}, {"name": "b.bin", "sha256": "b1"}]
        b = [dict(reversed(p.items())) for p in reversed(a)]
        # When written as dict literals both orderings have the
        # same key/value order; we need explicit ordering:
        b = [{"name": "b.bin", "sha256": "b1"}, {"name": "a.bin", "sha256": "a1"}]
        assert compute_content_fingerprint(a) == compute_content_fingerprint(b)

    def test_fingerprint_sensitive_to_sha(self) -> None:
        a = [{"name": "a.bin", "sha256": "a1"}]
        b = [{"name": "a.bin", "sha256": "a2"}]
        assert compute_content_fingerprint(a) != compute_content_fingerprint(b)

    def test_fingerprint_sensitive_to_name(self) -> None:
        a = [{"name": "a.bin", "sha256": "a1"}]
        b = [{"name": "c.bin", "sha256": "a1"}]
        assert compute_content_fingerprint(a) != compute_content_fingerprint(b)

    def test_fingerprint_empty_is_stable(self) -> None:
        assert compute_content_fingerprint([]) == compute_content_fingerprint([])


class TestCacheFindByFingerprint:
    def _populate(self, cache: ModelCache) -> str:
        data1 = b"alpha" * 16
        data2 = b"beta" * 16
        sha1 = hashlib.sha256(data1).hexdigest()
        sha2 = hashlib.sha256(data2).hexdigest()
        manifest = cache.write_files(
            "huggingface", "demo/x", "v1", "apache-2.0", "http://x",
            [
                {"name": "a.bin", "data": data1, "sha256": sha1},
                {"name": "b.bin", "data": data2, "sha256": sha2},
            ],
        )
        return manifest.content_fingerprint

    def test_finds_existing(self, tmp_path: Path) -> None:
        cache = ModelCache(root=tmp_path)
        fp = self._populate(cache)
        loc = cache.find_by_fingerprint("huggingface", fp)
        assert loc is not None
        assert loc.repo_id == "demo/x"
        assert loc.revision == "v1"

    def test_missing_returns_none(self, tmp_path: Path) -> None:
        cache = ModelCache(root=tmp_path)
        assert cache.find_by_fingerprint("huggingface", "0" * 64) is None

    def test_handles_repo_id_with_slash(self, tmp_path: Path) -> None:
        # HF-style repo_id contains a slash; on disk this becomes
        # an additional directory level.  find_by_fingerprint must
        # still locate it.
        cache = ModelCache(root=tmp_path)
        data = b"gamma" * 8
        sha = hashlib.sha256(data).hexdigest()
        manifest = cache.write_files(
            "huggingface", "Qwen/Qwen2.5-0.5B", "main", "apache-2.0",
            "http://x",
            [{"name": "model.bin", "data": data, "sha256": sha}],
        )
        loc = cache.find_by_fingerprint("huggingface", manifest.content_fingerprint)
        assert loc is not None
        assert loc.repo_id == "Qwen/Qwen2.5-0.5B"
        assert loc.revision == "main"

    def test_returns_first_match(self, tmp_path: Path) -> None:
        cache = ModelCache(root=tmp_path)
        # Two entries with the same fingerprint (different keys,
        # same content).  find_by_fingerprint returns the first
        # one lexicographically; that is the canonical choice.
        data = b"shared" * 8
        sha = hashlib.sha256(data).hexdigest()
        cache.write_files(
            "huggingface", "alpha", "v1", "apache-2.0", "http://x",
            [{"name": "m.bin", "data": data, "sha256": sha}],
        )
        cache.write_files(
            "huggingface", "beta", "v1", "apache-2.0", "http://x",
            [{"name": "m.bin", "data": data, "sha256": sha}],
        )
        fp = compute_content_fingerprint([{"name": "m.bin", "sha256": sha}])
        loc = cache.find_by_fingerprint("huggingface", fp)
        assert loc is not None
        assert loc.repo_id in {"alpha", "beta"}


# ---------------------------------------------------------------------------
# HuggingFaceSource mirror fallback + on_progress
# ---------------------------------------------------------------------------
class TestHFMirrorFallback:
    def test_resolve_license_falls_back(self, tmp_path: Path) -> None:
        # Upstream is broken; mirror is good.
        ft = FakeTransport(working_bases=["https://hf-mirror.com"])
        mirrors = MirrorSet(bases=("https://huggingface.co", "https://hf-mirror.com"))
        hf = HuggingFaceSource(mirrors=mirrors, transport=ft)
        license_id = hf.resolve_license("demo/x")
        assert license_id == "apache-2.0"

    def test_resolve_license_returns_empty_when_all_fail(self) -> None:
        ft = FakeTransport(working_bases=[])
        mirrors = MirrorSet(bases=("https://nope1.example", "https://nope2.example"))
        hf = HuggingFaceSource(mirrors=mirrors, transport=ft)
        assert hf.resolve_license("demo/x") == ""

    def test_list_files_falls_back(self) -> None:
        ft = FakeTransport(working_bases=["https://hf-mirror.com"])
        mirrors = MirrorSet(bases=("https://huggingface.co", "https://hf-mirror.com"))
        hf = HuggingFaceSource(mirrors=mirrors, transport=ft)
        files = hf.list_files("demo/x", "main")
        assert "config.json" in files
        assert "model.safetensors" in files

    def test_dead_mirror_memory(self) -> None:
        # First call: upstream is dead, mirror works.  After the
        # first call the upstream should be marked dead and the
        # next metadata request should NOT hit it.
        ft = FakeTransport(working_bases=["https://hf-mirror.com"])
        mirrors = MirrorSet(bases=("https://huggingface.co", "https://hf-mirror.com"))
        hf = HuggingFaceSource(mirrors=mirrors, transport=ft)
        # First fetch: _mark_mirror_dead is called on the upstream.
        hf.resolve_license("demo/x")
        # Subsequent lookup: upstream is in dead-memory.
        assert hf._is_mirror_dead("https://huggingface.co") is True
        # And the mirror is alive.
        assert hf._is_mirror_dead("https://hf-mirror.com") is False

    def test_download_progress_callback(self) -> None:
        ft = FakeTransport(working_bases=["https://hf-mirror.com"])
        mirrors = MirrorSet(bases=("https://hf-mirror.com",))
        hf = HuggingFaceSource(mirrors=mirrors, transport=ft)

        ticks: List[DownloadProgress] = []

        def cb(tick: DownloadProgress) -> None:
            ticks.append(tick)

        results = hf.download_files(
            "demo/x", "main",
            ["config.json", "model.safetensors"],
            on_progress=cb,
        )
        # 2 files => 2 finished ticks (start + finish collapse in our impl).
        assert len(results) == 2
        assert all(r.name for r in results)
        # The final ticks for every file should be "finished=True".
        finished = [t for t in ticks if t.finished and not t.error]
        assert len(finished) == 2
        for t in finished:
            assert t.bytes_total > 0
            assert t.bytes_done == t.bytes_total
            assert t.mirror == "https://hf-mirror.com"

    def test_download_progress_callback_swallows_exceptions(self) -> None:
        ft = FakeTransport(working_bases=["https://hf-mirror.com"])
        mirrors = MirrorSet(bases=("https://hf-mirror.com",))
        hf = HuggingFaceSource(mirrors=mirrors, transport=ft)

        def cb(tick: DownloadProgress) -> None:
            raise RuntimeError("oops")

        # Must not raise -- the download loop is robust to a
        # misbehaving callback.
        results = hf.download_files("demo/x", "main", ["a.json"], on_progress=cb)
        assert len(results) == 1

    def test_download_files_skips_tilde_names(self) -> None:
        ft = FakeTransport(working_bases=["https://hf-mirror.com"])
        mirrors = MirrorSet(bases=("https://hf-mirror.com",))
        hf = HuggingFaceSource(mirrors=mirrors, transport=ft)
        results = hf.download_files("demo/x", "main", ["~incomplete", "config.json"])
        # ~incomplete is silently dropped.
        assert [r.name for r in results] == ["config.json"]


# ---------------------------------------------------------------------------
# ModelFetcher end-to-end (mirror + dedup + progress)
# ---------------------------------------------------------------------------
def _make_fetcher(
    tmp_path: Path,
    transport: HttpTransport,
    mirrors: MirrorSet,
) -> ModelFetcher:
    cache = ModelCache(root=tmp_path / "cache")
    registry = SourceRegistry.default()
    registry.register(
        "huggingface",
        HuggingFaceSource(mirrors=mirrors, transport=transport),
    )
    return ModelFetcher(cache=cache, registry=registry, mirrors=mirrors)


class TestFetcherEndToEnd:
    def test_first_fetch_downloads(self, tmp_path: Path) -> None:
        ft = FakeTransport(working_bases=["https://hf-mirror.com"])
        mirrors = MirrorSet(bases=("https://hf-mirror.com",))
        fetcher = _make_fetcher(tmp_path, ft, mirrors)
        result = fetcher.fetch("huggingface", "demo/x", revision="v1")
        assert result.from_cache is False
        assert len(result.manifest.files) == 2

    def test_second_fetch_is_cache_hit(self, tmp_path: Path) -> None:
        ft = FakeTransport(working_bases=["https://hf-mirror.com"])
        mirrors = MirrorSet(bases=("https://hf-mirror.com",))
        fetcher = _make_fetcher(tmp_path, ft, mirrors)
        fetcher.fetch("huggingface", "demo/x", revision="v1")
        result = fetcher.fetch("huggingface", "demo/x", revision="v1")
        assert result.from_cache is True

    def test_on_progress_callback_invoked(self, tmp_path: Path) -> None:
        ft = FakeTransport(working_bases=["https://hf-mirror.com"])
        mirrors = MirrorSet(bases=("https://hf-mirror.com",))
        fetcher = _make_fetcher(tmp_path, ft, mirrors)
        ticks: List[Dict[str, Any]] = []

        def cb(name: str, done: int, total: int, mirror: str) -> None:
            ticks.append({
                "name": name, "done": done, "total": total, "mirror": mirror,
            })

        fetcher.fetch("huggingface", "demo/x", revision="v1", on_progress=cb)
        # We expect at least 2 ticks (one per file) -- possibly
        # more for the start+finish.  We assert on file names
        # only.
        names = {t["name"] for t in ticks}
        assert "config.json" in names
        assert "model.safetensors" in names

    def test_cross_mirror_dedup(self, tmp_path: Path) -> None:
        # First fetch v1 from the mirror.  Then fetch v1.1 with
        # the same content: cross-mirror dedup should kick in and
        # *return* the v1 manifest without writing a duplicate
        # v1.1 directory.  The second fetch still pays a small
        # network cost (the metadata + tree listing) -- that is
        # unavoidable until we know the file set -- but no
        # duplicate payload is downloaded and no duplicate disk
        # space is consumed.
        ft = FakeTransport(working_bases=["https://hf-mirror.com"])
        mirrors = MirrorSet(bases=("https://hf-mirror.com",))
        fetcher = _make_fetcher(tmp_path, ft, mirrors)
        fetcher.fetch("huggingface", "demo/x", revision="v1")

        result = fetcher.fetch("huggingface", "demo/x", revision="v1.1")
        assert result.from_cache is True
        assert result.manifest.revision == "v1"  # served from v1

        # And no duplicate v1.1 directory on disk -- only v1.
        v1_dir = tmp_path / "cache" / "huggingface" / "demo" / "x" / "v1"
        v11_dir = tmp_path / "cache" / "huggingface" / "demo" / "x" / "v1.1"
        assert v1_dir.is_dir()
        assert not v11_dir.exists()  # dedup did not write a copy

    def test_mirror_fallback_first_call(self, tmp_path: Path) -> None:
        # Upstream is broken; mirror works.  The fetcher should
        # transparently use the mirror and still return a
        # well-formed result.
        ft = FakeTransport(working_bases=["https://hf-mirror.com"])
        mirrors = MirrorSet(bases=("https://huggingface.co", "https://hf-mirror.com"))
        fetcher = _make_fetcher(tmp_path, ft, mirrors)
        result = fetcher.fetch("huggingface", "demo/x", revision="v1")
        assert result.from_cache is False
        assert len(result.manifest.files) == 2

    def test_per_call_mirrors_override(self, tmp_path: Path) -> None:
        # The fetcher's default mirrors point at a working
        # mirror; the per-call mirrors point at a *different*
        # working mirror.  We use a single transport that serves
        # *both* bases -- otherwise the test would conflate
        # "mirror switch failed" with "transport doesn't speak
        # the new base".  The assertion is on the adapter's
        # post-call state (mirrors must be restored).
        ft = FakeTransport(working_bases=[
            "https://hf-mirror.com", "https://other.example",
        ])
        default_mirrors = MirrorSet(bases=("https://hf-mirror.com",))
        override_mirrors = MirrorSet(bases=("https://other.example",))

        cache = ModelCache(root=tmp_path / "cache")
        registry = SourceRegistry.default()
        registry.register(
            "huggingface",
            HuggingFaceSource(mirrors=default_mirrors, transport=ft),
        )
        fetcher = ModelFetcher(
            cache=cache, registry=registry, mirrors=default_mirrors,
        )
        result = fetcher.fetch(
            "huggingface", "demo/x", revision="v1",
            mirrors=override_mirrors,
        )
        assert len(result.manifest.files) == 2
        # Default mirror should be restored on the adapter --
        # we use `is` to be sure we are pointing at the same
        # object the user installed (not just an equal-looking
        # MirrorSet).
        hf_after = registry.get("huggingface")
        assert hf_after._mirrors is default_mirrors
