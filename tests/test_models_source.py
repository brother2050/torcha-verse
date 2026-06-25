"""Tests for the TorchaVerse model source framework (v0.4.0).

These tests are deliberately self-contained: they only need
``torch`` and the standard library, and they exercise every
component of :mod:`models.source` through a fake
:class:`HttpTransport` so no real network call is ever made.

Coverage
--------

* :mod:`license_check` -- SPDX normalisation, allow-list, NC and
  no-derivatives short-circuits, ``extend_default_allow_license``.
* :mod:`cache` -- atomic write, sha256 verification, manifest
  round-trip, ``clear`` idempotency, ``default_cache_root`` env
  override.
* :mod:`huggingface` -- JSON / bytes transport, license
  resolution, file listing, default-artifact download.
* :mod:`civitai` -- license resolution, file listing, default
  artifact download.
* :mod:`fetch` -- license rejection, cache hit / miss, cache
  verification failure path, source aliases, custom
  :class:`ModelCache` and :class:`SourceRegistry`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from models.source import (
    DEFAULT_ALLOW_LICENSE,
    CacheLocation,
    CachedFile,
    CachedModel,
    CivitaiSource,
    FetchResult,
    FileDownload,
    HttpTransport,
    HuggingFaceSource,
    LicenseCheckResult,
    ModelCache,
    ModelFetcher,
    SourceRegistry,
    UrllibTransport,
    check_license,
    default_cache_root,
    extend_default_allow_license,
    fetch,
    get_default_allow_license,
    is_known_non_commercial,
    normalise_spdx,
)


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.model_source


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_cache(tmp_path: Path) -> ModelCache:
    """A :class:`ModelCache` rooted at a temporary directory."""
    cache_root = tmp_path / "cache"
    return ModelCache(root=cache_root)


@pytest.fixture
def monkeypatch_cache_root(monkeypatch, tmp_path):
    """Point ``$TORCHA_VERSE_CACHE`` at ``tmp_path`` for the duration."""
    monkeypatch.setenv("TORCHA_VERSE_CACHE", str(tmp_path / "envcache"))


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------
class FakeTransport(HttpTransport):
    """A configurable in-memory :class:`HttpTransport`.

    Routes are registered as ``(method, url_substring)`` -> response.
    The default fallback (when no route matches) raises so that
    tests fail loudly if they forgot to wire a fixture.
    """

    def __init__(self) -> None:
        self.routes: List[Dict[str, Any]] = []
        self.calls: List[Tuple[str, str, Dict[str, str]]] = []

    def route(
        self,
        url_substring: str,
        *,
        json_body: Any = None,
        bytes_body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        entry: Dict[str, Any] = {"url": url_substring}
        if json_body is not None:
            entry["json"] = json_body
        if bytes_body is not None:
            entry["bytes"] = bytes_body
        if headers:
            entry["headers"] = headers
        self.routes.append(entry)

    def _match(self, url: str) -> Optional[Dict[str, Any]]:
        """Find the most specific (longest-matching) route for ``url``."""
        best: Optional[Dict[str, Any]] = None
        best_len = -1
        for entry in self.routes:
            if entry["url"] in url and len(entry["url"]) > best_len:
                best = entry
                best_len = len(entry["url"])
        return best

    def get_json(
        self, url: str, *, headers: Optional[Dict[str, str]] = None
    ) -> Any:
        self.calls.append(("json", url, dict(headers or {})))
        entry = self._match(url)
        if entry is None or "json" not in entry:
            raise RuntimeError("no fake JSON route for {}".format(url))
        return entry["json"]

    def get_bytes(
        self, url: str, *, headers: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, Dict[str, str]]:
        self.calls.append(("bytes", url, dict(headers or {})))
        entry = self._match(url)
        if entry is None or "bytes" not in entry:
            raise RuntimeError("no fake bytes route for {}".format(url))
        return entry["bytes"], dict(entry.get("headers", {}))


@pytest.fixture
def fake_hf_transport() -> FakeTransport:
    """A :class:`FakeTransport` pre-loaded with a small HF repo."""
    t = FakeTransport()
    t.route(
        "/api/models/Qwen/Qwen2.5-0.5B-Instruct",
        json_body={"cardData": {"license": "apache-2.0"}},
    )
    t.route(
        "/api/models/Qwen/Qwen2.5-0.5B-Instruct/tree/main",
        json_body=[
            {"type": "file", "path": "config.json"},
            {"type": "file", "path": "model.safetensors"},
            {"type": "file", "path": "tokenizer.json"},
            {"type": "directory", "path": "onnx"},
        ],
    )
    t.route(
        "/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json",
        bytes_body=b'{"hidden_size": 64}',
    )
    t.route(
        "/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json",
        bytes_body=b'{"tokenizer": "fake"}',
    )
    t.route(
        "/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors",
        bytes_body=b"\x00\x01\x02fake-weights",
    )
    return t


# ---------------------------------------------------------------------------
# license_check.py
# ---------------------------------------------------------------------------
class TestNormaliseSpdx:
    def test_basic(self) -> None:
        assert normalise_spdx("Apache-2.0") == "apache-2.0"
        assert normalise_spdx("  MIT  ") == "mit"
        assert normalise_spdx("BSD 3-Clause") == "bsd-3-clause"

    def test_empty(self) -> None:
        assert normalise_spdx("") == ""
        assert normalise_spdx("   ") == ""


class TestIsKnownNonCommercial:
    def test_nc_token(self) -> None:
        assert is_known_non_commercial("cc-by-nc-4.0")
        assert is_known_non_commercial("Research-Only")
        assert not is_known_non_commercial("apache-2.0")
        assert not is_known_non_commercial("")


class TestCheckLicense:
    def test_default_allow(self) -> None:
        for lic in DEFAULT_ALLOW_LICENSE:
            r = check_license(lic)
            assert r.accepted, r.reason

    def test_empty_rejected(self) -> None:
        r = check_license("")
        assert not r.accepted
        assert "no license" in r.reason.lower()

    def test_nc_rejected(self) -> None:
        r = check_license("cc-by-nc-4.0")
        assert not r.accepted
        assert "non-commercial" in r.reason.lower()

    def test_nc_opt_in(self) -> None:
        r = check_license("cc-by-nc-4.0", allow_license=["cc-by-nc-4.0"])
        assert r.accepted

    def test_known_ok_but_not_in_allow(self) -> None:
        r = check_license("gpl-3.0")
        assert not r.accepted
        assert "gpl-3.0" in r.reason

    def test_known_ok_opt_in(self) -> None:
        r = check_license("gpl-3.0", allow_license=["gpl-3.0"])
        assert r.accepted

    def test_unknown_rejected(self) -> None:
        r = check_license("definitely-not-a-spdx-id")
        assert not r.accepted

    def test_unknown_opt_in(self) -> None:
        r = check_license("definitely-not-a-spdx-id",
                          allow_license=["definitely-not-a-spdx-id"])
        assert r.accepted

    def test_no_derivatives_rejected(self) -> None:
        r = check_license("cc-by-nd-4.0")
        assert not r.accepted

    def test_normalisation_works(self) -> None:
        r = check_license("  Apache 2.0  ")
        assert r.accepted
        assert r.license_id == "apache-2.0"


class TestExtendDefaultAllowLicense:
    def setup_method(self) -> None:
        # Snapshot so the test doesn't leak into other tests.
        self._snapshot = get_default_allow_license()

    def teardown_method(self) -> None:
        # Reset the module-level allow-list to the snapshot.
        import models.source.license_check as lc
        lc._effective_default = self._snapshot

    def test_extend_idempotent(self) -> None:
        before = get_default_allow_license()
        extend_default_allow_license(["gpl-3.0"])
        assert "gpl-3.0" in get_default_allow_license()
        extend_default_allow_license(["gpl-3.0"])  # idempotent
        after = get_default_allow_license()
        assert before.union({"gpl-3.0"}) == after


# ---------------------------------------------------------------------------
# cache.py
# ---------------------------------------------------------------------------
class TestDefaultCacheRoot:
    def test_env_override(self, monkeypatch_cache_root, tmp_path) -> None:
        # ``default_cache_root`` reads $TORCHA_VERSE_CACHE if set.
        expected = (tmp_path / "envcache").resolve()
        assert default_cache_root() == expected


class TestModelCache:
    def test_location_for(self, tmp_cache) -> None:
        loc = tmp_cache.location_for("huggingface", "Qwen/Qwen2.5-0.5B", "main")
        assert isinstance(loc, CacheLocation)
        assert loc.source == "huggingface"
        assert loc.repo_id == "Qwen/Qwen2.5-0.5B"
        assert loc.revision == "main"
        assert str(tmp_cache.root) in str(loc.path())

    def test_location_for_empty_repo_id_raises(self, tmp_cache) -> None:
        with pytest.raises(ValueError):
            tmp_cache.location_for("huggingface", "")
        with pytest.raises(ValueError):
            tmp_cache.location_for("", "x")

    def test_has_and_load_manifest(self, tmp_cache) -> None:
        assert not tmp_cache.has("huggingface", "x", "main")
        tmp_cache.write_files(
            source="huggingface", repo_id="x", revision="main",
            license_id="apache-2.0", url="https://example.com",
            files=[{"name": "a.txt", "data": b"hello"}],
        )
        assert tmp_cache.has("huggingface", "x", "main")
        manifest = tmp_cache.load_manifest("huggingface", "x", "main")
        assert isinstance(manifest, CachedModel)
        assert manifest.license_id == "apache-2.0"
        assert manifest.files[0].name == "a.txt"
        assert manifest.files[0].size == 5

    def test_write_requires_files(self, tmp_cache) -> None:
        with pytest.raises(ValueError):
            tmp_cache.write_files(
                source="huggingface", repo_id="x", revision="main",
                license_id="apache-2.0", url="",
                files=[],
            )

    def test_write_rejects_invalid_spec(self, tmp_cache) -> None:
        with pytest.raises(ValueError):
            tmp_cache.write_files(
                source="huggingface", repo_id="x", revision="main",
                license_id="apache-2.0", url="",
                files=[{"name": ""}],
            )
        with pytest.raises(TypeError):
            tmp_cache.write_files(
                source="huggingface", repo_id="x", revision="main",
                license_id="apache-2.0", url="",
                files=[{"name": "a.txt", "data": 12345}],
            )

    def test_sha256_computed_when_missing(self, tmp_cache) -> None:
        manifest = tmp_cache.write_files(
            source="huggingface", repo_id="x", revision="main",
            license_id="apache-2.0", url="",
            files=[{"name": "a.txt", "data": b"hello"}],
        )
        # SHA-256 of b"hello"
        assert manifest.files[0].sha256 == (
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )

    def test_sha256_mismatch_writing_raises(self, tmp_cache) -> None:
        with pytest.raises(ValueError):
            tmp_cache.write_files(
                source="huggingface", repo_id="x", revision="main",
                license_id="apache-2.0", url="",
                files=[{
                    "name": "a.txt",
                    "data": b"hello",
                    "sha256": "0" * 64,  # wrong
                }],
            )

    def test_verify_passes_after_write(self, tmp_cache) -> None:
        tmp_cache.write_files(
            source="huggingface", repo_id="x", revision="main",
            license_id="apache-2.0", url="",
            files=[{"name": "a.txt", "data": b"hello"}],
        )
        assert tmp_cache.verify("huggingface", "x", "main")

    def test_verify_detects_tampering(self, tmp_cache) -> None:
        loc = tmp_cache.location_for("huggingface", "x", "main")
        tmp_cache.write_files(
            source="huggingface", repo_id="x", revision="main",
            license_id="apache-2.0", url="",
            files=[{"name": "a.txt", "data": b"hello"}],
        )
        # Tamper with the file.
        (loc.path() / "a.txt").write_bytes(b"corrupted")
        assert not tmp_cache.verify("huggingface", "x", "main")

    def test_verify_returns_false_when_missing(self, tmp_cache) -> None:
        assert not tmp_cache.verify("huggingface", "nope", "main")

    def test_clear_idempotent(self, tmp_cache) -> None:
        tmp_cache.write_files(
            source="huggingface", repo_id="x", revision="main",
            license_id="apache-2.0", url="",
            files=[{"name": "a.txt", "data": b"hello"}],
        )
        assert tmp_cache.clear("huggingface", "x", "main")
        assert not tmp_cache.has("huggingface", "x", "main")
        # Second call returns False.
        assert not tmp_cache.clear("huggingface", "x", "main")

    def test_manifest_round_trip(self, tmp_cache) -> None:
        tmp_cache.write_files(
            source="huggingface", repo_id="x", revision="v1",
            license_id="mit", url="https://example.com",
            files=[
                {"name": "a.txt", "data": b"alpha"},
                {"name": "b.txt", "data": b"beta"},
            ],
        )
        m = tmp_cache.load_manifest("huggingface", "x", "v1")
        d = m.as_dict()
        rebuilt = CachedModel.from_dict(d)
        assert rebuilt.license_id == m.license_id
        assert rebuilt.repo_id == m.repo_id
        assert [f.name for f in rebuilt.files] == ["a.txt", "b.txt"]
        # JSON round-trip preserves everything.
        s = m.to_json()
        d2 = json.loads(s)
        assert d2 == d


# ---------------------------------------------------------------------------
# huggingface.py
# ---------------------------------------------------------------------------
class TestUrllibTransportSmoke:
    def test_urllib_is_default(self) -> None:
        src = HuggingFaceSource()
        assert isinstance(src._transport, UrllibTransport)


class TestHuggingFaceSource:
    def test_resolve_license_card_data(self, fake_hf_transport) -> None:
        src = HuggingFaceSource(transport=fake_hf_transport)
        assert src.resolve_license("Qwen/Qwen2.5-0.5B-Instruct") == "apache-2.0"

    def test_resolve_license_uses_cache(self, fake_hf_transport) -> None:
        src = HuggingFaceSource(transport=fake_hf_transport)
        src.resolve_license("Qwen/Qwen2.5-0.5B-Instruct")
        # Second call must not re-issue the HTTP request.
        calls_before = len(fake_hf_transport.calls)
        assert src.resolve_license("Qwen/Qwen2.5-0.5B-Instruct") == "apache-2.0"
        assert len(fake_hf_transport.calls) == calls_before

    def test_list_files_filters_directories(self, fake_hf_transport) -> None:
        src = HuggingFaceSource(transport=fake_hf_transport)
        files = src.list_files("Qwen/Qwen2.5-0.5B-Instruct", "main")
        assert "config.json" in files
        assert "model.safetensors" in files
        assert "onnx" not in files  # directories are filtered

    def test_download_default_artifacts(self, fake_hf_transport) -> None:
        src = HuggingFaceSource(transport=fake_hf_transport)
        downloads = src.download_default_artifacts(
            "Qwen/Qwen2.5-0.5B-Instruct", "main"
        )
        names = [d.name for d in downloads]
        assert "config.json" in names
        assert "model.safetensors" in names
        # All downloads have non-empty bodies and a sha256.
        for d in downloads:
            assert isinstance(d, FileDownload)
            assert len(d.data) > 0
            assert len(d.sha256) > 0

    def test_bearer_token_header(self, fake_hf_transport) -> None:
        src = HuggingFaceSource(transport=fake_hf_transport, token="abc123")
        src.resolve_license("Qwen/Qwen2.5-0.5B-Instruct")
        # Find the metadata call.
        for kind, url, hdrs in fake_hf_transport.calls:
            if kind == "json" and "/api/models/" in url:
                assert hdrs.get("Authorization") == "Bearer abc123"
                break
        else:
            pytest.fail("no JSON call was issued")


# ---------------------------------------------------------------------------
# civitai.py
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_civitai_transport() -> FakeTransport:
    t = FakeTransport()
    t.route(
        "/api/v1/model-versions/12345",
        json_body={
            "id": 12345,
            "license": "CC-BY-4.0",
            "files": [
                {"name": "model.safetensors",
                 "downloadUrl": "https://cdn.test/model.safetensors"},
                {"name": "config.json",
                 "downloadUrl": "https://cdn.test/config.json"},
            ],
        },
    )
    t.route(
        "https://cdn.test/model.safetensors",
        bytes_body=b"fake-civitai-weights",
    )
    t.route(
        "https://cdn.test/config.json",
        bytes_body=b'{"unet": [64, 64]}',
    )
    return t


class TestCivitaiSource:
    def test_resolve_license(self, fake_civitai_transport) -> None:
        src = CivitaiSource(transport=fake_civitai_transport)
        assert src.resolve_license("12345") == "CC-BY-4.0"

    def test_list_files(self, fake_civitai_transport) -> None:
        src = CivitaiSource(transport=fake_civitai_transport)
        assert src.list_files("12345") == ["model.safetensors", "config.json"]

    def test_download_default_artifacts(self, fake_civitai_transport) -> None:
        src = CivitaiSource(transport=fake_civitai_transport)
        downloads = src.download_default_artifacts("12345")
        names = [d.name for d in downloads]
        assert "model.safetensors" in names
        assert "config.json" in names


# ---------------------------------------------------------------------------
# fetch.py
# ---------------------------------------------------------------------------
class TestSourceRegistry:
    def test_default_has_hf_and_civitai(self) -> None:
        reg = SourceRegistry.default()
        assert "huggingface" in reg.available()
        assert "civitai" in reg.available()
        assert isinstance(reg.get("hf"), HuggingFaceSource)
        assert isinstance(reg.get("cv"), CivitaiSource)

    def test_alias_resolution(self) -> None:
        reg = SourceRegistry()
        reg.register("huggingface", "fake-hf")
        reg.register("civitai", "fake-cv")
        assert reg.get("huggingface") == "fake-hf"
        assert reg.get("hf") == "fake-hf"
        assert reg.get("civitai") == "fake-cv"
        assert reg.get("cv") == "fake-cv"

    def test_unknown_raises(self) -> None:
        reg = SourceRegistry()
        with pytest.raises(KeyError):
            reg.get("unknown")

    def test_register_empty_name_raises(self) -> None:
        reg = SourceRegistry()
        with pytest.raises(ValueError):
            reg.register("", "x")

    def test_register_overwrites(self) -> None:
        reg = SourceRegistry()
        reg.register("huggingface", "v1")
        reg.register("huggingface", "v2")
        assert reg.get("huggingface") == "v2"


class TestModelFetcher:
    def _fetcher(
        self, tmp_cache, fake_hf_transport,
    ) -> ModelFetcher:
        reg = SourceRegistry()
        reg.register("huggingface", HuggingFaceSource(transport=fake_hf_transport))
        reg.register("civitai", CivitaiSource(transport=FakeTransport()))
        return ModelFetcher(cache=tmp_cache, registry=reg)

    def test_fetch_miss_then_hit(
        self, tmp_cache, fake_hf_transport,
    ) -> None:
        fetcher = self._fetcher(tmp_cache, fake_hf_transport)
        result1 = fetcher.fetch(
            source="huggingface",
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision="main",
        )
        assert isinstance(result1, FetchResult)
        assert result1.accepted
        assert not result1.from_cache
        assert result1.license_check.license_id == "apache-2.0"
        # Cache files exist.
        loc = result1.location
        assert (loc.path() / "config.json").is_file()

        result2 = fetcher.fetch(
            source="huggingface",
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision="main",
        )
        assert result2.from_cache
        assert result2.manifest.license_id == "apache-2.0"

    def test_fetch_uses_alias(
        self, tmp_cache, fake_hf_transport,
    ) -> None:
        fetcher = self._fetcher(tmp_cache, fake_hf_transport)
        result = fetcher.fetch(
            source="hf",  # alias
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision="main",
        )
        assert result.source == "huggingface"

    def test_fetch_rejects_unknown_license(
        self, tmp_cache,
    ) -> None:
        transport = FakeTransport()
        transport.route(
            "/api/models/Secrets/Bad",
            json_body={"cardData": {"license": "non-commercial"}},
        )
        reg = SourceRegistry()
        reg.register("huggingface", HuggingFaceSource(transport=transport))
        reg.register("civitai", CivitaiSource(transport=FakeTransport()))
        fetcher = ModelFetcher(cache=tmp_cache, registry=reg)
        with pytest.raises(PermissionError):
            fetcher.fetch(
                source="huggingface", repo_id="Secrets/Bad", revision="main",
            )

    def test_fetch_rejects_no_license(
        self, tmp_cache,
    ) -> None:
        transport = FakeTransport()
        transport.route("/api/models/No/License", json_body={})
        reg = SourceRegistry()
        reg.register("huggingface", HuggingFaceSource(transport=transport))
        reg.register("civitai", CivitaiSource(transport=FakeTransport()))
        fetcher = ModelFetcher(cache=tmp_cache, registry=reg)
        with pytest.raises(PermissionError):
            fetcher.fetch(source="huggingface", repo_id="No/License")

    def test_fetch_explicit_allow_list(
        self, tmp_cache, fake_hf_transport,
    ) -> None:
        # The HF model is apache-2.0 -- allow only ``mit`` and the
        # call must be rejected.
        fetcher = self._fetcher(tmp_cache, fake_hf_transport)
        with pytest.raises(PermissionError):
            fetcher.fetch(
                source="huggingface",
                repo_id="Qwen/Qwen2.5-0.5B-Instruct",
                revision="main",
                allow_license=["mit"],
            )

    def test_fetch_detects_cache_tampering(
        self, tmp_cache, fake_hf_transport,
    ) -> None:
        fetcher = self._fetcher(tmp_cache, fake_hf_transport)
        result = fetcher.fetch(
            source="huggingface",
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision="main",
        )
        # Tamper with a cached file.
        (result.location.path() / "config.json").write_bytes(b"corrupted")
        # Next fetch must re-download (cache miss + verify failure).
        result2 = fetcher.fetch(
            source="huggingface",
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision="main",
        )
        assert not result2.from_cache
        # The fresh download restores the file.
        assert (result2.location.path() / "config.json").read_bytes() == (
            b'{"hidden_size": 64}'
        )

    def test_fetch_empty_repo_id_raises(self, tmp_cache) -> None:
        fetcher = ModelFetcher(cache=tmp_cache)
        with pytest.raises(ValueError):
            fetcher.fetch(source="huggingface", repo_id="")

    def test_fetch_unknown_source_raises(self, tmp_cache) -> None:
        fetcher = ModelFetcher(cache=tmp_cache)
        with pytest.raises(KeyError):
            fetcher.fetch(source="unknown", repo_id="x")

    def test_fetch_result_as_dict(
        self, tmp_cache, fake_hf_transport,
    ) -> None:
        fetcher = self._fetcher(tmp_cache, fake_hf_transport)
        result = fetcher.fetch(
            source="huggingface",
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision="main",
        )
        d = result.as_dict()
        assert d["source"] == "huggingface"
        assert d["from_cache"] is False
        assert d["license_check"]["accepted"] is True


class TestModuleLevelFetch:
    def test_module_level_fetch_resolves(
        self, tmp_path, monkeypatch, fake_hf_transport,
    ) -> None:
        """``from models.source import fetch`` works end-to-end with a
        custom registry/cache bound to the singleton."""
        # Reach for the *module* (not the re-exported function) so we
        # can swap the singleton without name-shadowing issues.
        import importlib
        fetch_mod = importlib.import_module("models.source.fetch")
        monkeypatch.setattr(fetch_mod, "_default_fetcher", None)
        cache = ModelCache(root=tmp_path / "cache")
        reg = SourceRegistry()
        reg.register(
            "huggingface",
            HuggingFaceSource(transport=fake_hf_transport),
        )
        reg.register("civitai", CivitaiSource(transport=FakeTransport()))
        custom = ModelFetcher(cache=cache, registry=reg)
        monkeypatch.setattr(fetch_mod, "_default_fetcher", custom)
        result = fetch(
            "Qwen/Qwen2.5-0.5B-Instruct",
            source="huggingface",
            revision="main",
        )
        assert result.accepted


# ---------------------------------------------------------------------------
# Top-level re-exports via ``models`` package
# ---------------------------------------------------------------------------
class TestModelsTopLevelReExports:
    def test_fetch_exposed_at_models_root(self) -> None:
        from models import fetch as top_level_fetch
        from models.source import fetch as sub_level_fetch
        assert top_level_fetch is sub_level_fetch

    def test_models_source_module_importable(self) -> None:
        # Just importing the module is a useful smoke test.
        import models.source  # noqa: F401
        assert hasattr(models.source, "__version__")
        assert models.source.__version__ == "0.4.0"
