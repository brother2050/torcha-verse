"""Tests for the v0.4.x P2++ supply-chain integrity layer.

This file covers the *new* capabilities added on top of the
v0.4.0 model-fetcher minimum-viable:

* Token resolution: explicit / env-var / on-disk file fallback
  chain.  Token value is never serialised by ``TokenInfo.as_dict()``.
* Header-based SHA-256 extraction (HF LFS pointer, ETag, weak
  ETag with ``W/`` prefix, ``x-checksum-sha256``).
* :class:`GatedRepoError` raised on 401/403 (HF + Civitai).
* :class:`ChecksumMismatch` raised on caller-pinned hash
  mismatch -- in :meth:`ModelCache.write_files`,
  :meth:`ModelFetcher.fetch`, and the cross-mirror dedup hit
  path.
* Per-call ``token=`` / ``expected_sha256s=`` /
  ``validate_checksums=`` API on :meth:`ModelFetcher.fetch` and
  :func:`fetch`.

The tests are pure-stdlib: no real network access.  HF and Civitai
are driven through the same :class:`FakeTransport` the v0.4.0
tests use, with a small extension to optionally raise
``urllib.error.HTTPError`` so the gated-repo code path is
covered end-to-end.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from models.source import (
    ChecksumMismatch,
    CivitaiSource,
    FetchResult,
    FileDownload,
    GatedRepoError,
    HttpTransport,
    HuggingFaceSource,
    ModelCache,
    ModelFetcher,
    SourceRegistry,
    TokenInfo,
    auth_headers,
    extract_expected_sha256_from_headers,
    fetch,
    is_gated_http_error,
    resolve_token,
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
    return ModelCache(root=tmp_path / "cache")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _http_error(code: int, url: str = "https://example.com") -> urllib.error.HTTPError:
    """Build an ``urllib.error.HTTPError`` with the given code."""
    return urllib.error.HTTPError(
        url, code, "HTTP Error", {}, None,
    )


# ---------------------------------------------------------------------------
# Fake transport (with HTTPError support for gated-repo tests)
# ---------------------------------------------------------------------------
class FakeTransport(HttpTransport):
    """Configurable in-memory :class:`HttpTransport`."""

    def __init__(self) -> None:
        self.routes: List[Dict[str, Any]] = []
        self.calls: List[Tuple[str, str, Dict[str, str]]] = []

    def route(
        self,
        url_substring: str,
        *,
        json_body: Any = None,
        bytes_body: Optional[bytes] = None,
        bytes_headers: Optional[Dict[str, str]] = None,
        raise_http_error: Optional[int] = None,
    ) -> None:
        entry: Dict[str, Any] = {"url": url_substring}
        if json_body is not None:
            entry["json"] = json_body
        if bytes_body is not None:
            entry["bytes"] = bytes_body
        if bytes_headers is not None:
            entry["headers"] = bytes_headers
        if raise_http_error is not None:
            entry["raise_http_error"] = int(raise_http_error)
        self.routes.append(entry)

    def _match(self, url: str) -> Optional[Dict[str, Any]]:
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
        if entry is None:
            raise RuntimeError("no fake JSON route for {}".format(url))
        if "raise_http_error" in entry:
            raise _http_error(entry["raise_http_error"], url)
        if "json" not in entry:
            raise RuntimeError("no fake JSON body for {}".format(url))
        return entry["json"]

    def get_bytes(
        self, url: str, *, headers: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, Dict[str, str]]:
        self.calls.append(("bytes", url, dict(headers or {})))
        entry = self._match(url)
        if entry is None:
            raise RuntimeError("no fake bytes route for {}".format(url))
        if "raise_http_error" in entry:
            raise _http_error(entry["raise_http_error"], url)
        if "bytes" not in entry:
            raise RuntimeError("no fake bytes body for {}".format(url))
        return entry["bytes"], dict(entry.get("headers", {}))


@pytest.fixture
def hf_repo_with_sha_headers() -> FakeTransport:
    """A small HF repo whose download endpoint advertises an
    ``x-linked-etag`` (LFS pointer) SHA.  This is the canonical
    "we know the upstream hash" shape.
    """
    config_bytes = b'{"hidden_size": 64}'
    weights_bytes = b"\x00\x01\x02fake-weights"
    tok = FakeTransport()
    tok.route(
        "/api/models/example/repo",
        json_body={"cardData": {"license": "apache-2.0"}},
    )
    tok.route(
        "/api/models/example/repo/tree/main",
        json_body=[
            {"type": "file", "path": "config.json"},
            {"type": "file", "path": "model.safetensors"},
        ],
    )
    tok.route(
        "/example/repo/resolve/main/config.json",
        bytes_body=config_bytes,
        bytes_headers={"x-linked-etag": '"{}"'.format(_sha256(config_bytes))},
    )
    tok.route(
        "/example/repo/resolve/main/model.safetensors",
        bytes_body=weights_bytes,
        bytes_headers={
            "x-linked-etag": '"{}"'.format(_sha256(weights_bytes)),
            "x-repo-commit": "deadbeef",
        },
    )
    return tok


# ===========================================================================
# TokenInfo / resolve_token
# ===========================================================================
class TestResolveToken:
    def test_explicit_wins(self) -> None:
        ti = resolve_token(explicit="abc", env={"HF_TOKEN": "ignored"})
        assert ti.is_present
        assert ti.value == "abc"
        assert ti.source == "explicit"

    def test_explicit_empty_is_empty(self) -> None:
        ti = resolve_token(explicit="", env={"HF_TOKEN": "ignored"})
        assert not ti.is_present
        assert ti.source == "empty-explicit"

    def test_env_var_hf(self) -> None:
        ti = resolve_token(
            env={"HF_TOKEN": "  hf_xyz  "}, sources="huggingface",
        )
        assert ti.is_present
        assert ti.value == "hf_xyz"  # whitespace stripped
        assert ti.source == "env"
        assert ti.env_var == "HF_TOKEN"

    def test_env_var_hf_legacy(self) -> None:
        ti = resolve_token(
            env={"HUGGING_FACE_HUB_TOKEN": "hf_legacy"},
            sources="huggingface",
        )
        assert ti.is_present
        assert ti.value == "hf_legacy"
        assert ti.env_var == "HUGGING_FACE_HUB_TOKEN"

    def test_env_var_civitai(self) -> None:
        ti = resolve_token(env={"CIVITAI_TOKEN": "cv_abc"}, sources="civitai")
        assert ti.is_present
        assert ti.value == "cv_abc"
        assert ti.env_var == "CIVITAI_TOKEN"

    def test_env_var_generic(self) -> None:
        ti = resolve_token(
            env={"TORCHA_VERSE_TOKEN": "tv_abc"}, sources="generic",
        )
        assert ti.is_present
        assert ti.value == "tv_abc"
        assert ti.env_var == "TORCHA_VERSE_TOKEN"

    def test_empty_env_falls_through(self) -> None:
        ti = resolve_token(
            env={"HF_TOKEN": "   "}, sources="huggingface",
        )
        # Empty / whitespace-only env vars do not satisfy the lookup.
        assert not ti.is_present
        assert ti.source == "none"

    def test_no_token_returns_none(self) -> None:
        ti = resolve_token(env={}, sources="huggingface")
        assert not ti.is_present
        assert ti.source == "none"

    def test_on_disk_token_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Simulate ``~/.cache/huggingface/token`` by overriding
        # the home directory via the home_dir= kwarg.
        tok_file = tmp_path / "token"
        tok_file.write_text("hf_on_disk\n", encoding="utf-8")
        home = tmp_path / "home"
        home.mkdir()
        cache = home / ".cache" / "huggingface"
        cache.mkdir(parents=True)
        (cache / "token").write_text("hf_on_disk\n", encoding="utf-8")
        ti = resolve_token(
            env={}, sources="huggingface", home_dir=str(home),
        )
        assert ti.is_present
        assert ti.value == "hf_on_disk"
        assert ti.source == "file"
        assert ti.file_path is not None and ti.file_path.endswith("token")

    def test_on_disk_empty_file_falls_through(
        self, tmp_path: Path,
    ) -> None:
        home = tmp_path / "home"
        cache = home / ".cache" / "huggingface"
        cache.mkdir(parents=True)
        (cache / "token").write_text("   \n", encoding="utf-8")
        ti = resolve_token(
            env={}, sources="huggingface", home_dir=str(home),
        )
        assert not ti.is_present
        assert ti.source == "none"

    def test_unknown_source_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_token(sources="bogus")  # type: ignore[arg-type]


class TestTokenInfoAsDict:
    def test_redacts_value(self) -> None:
        ti = TokenInfo(value="secret", source="explicit")
        d = ti.as_dict()
        assert d["present"] is True
        assert d["value_redacted"] == "***"
        assert "secret" not in json.dumps(d)

    def test_empty(self) -> None:
        ti = TokenInfo(value="", source="none")
        d = ti.as_dict()
        assert d["present"] is False
        assert d["value_redacted"] == ""


class TestAuthHeaders:
    def test_present(self) -> None:
        h = auth_headers(TokenInfo(value="hf_abc", source="explicit"))
        assert h == {"Authorization": "Bearer hf_abc"}

    def test_absent(self) -> None:
        assert auth_headers(None) == {}
        assert auth_headers(TokenInfo(value="", source="none")) == {}


# ===========================================================================
# extract_expected_sha256_from_headers
# ===========================================================================
class TestExtractSha:
    def test_x_linked_etag(self) -> None:
        h = {"x-linked-etag": '"abc123"'}
        assert extract_expected_sha256_from_headers(h) == "abc123"

    def test_x_linked_etag_no_quotes(self) -> None:
        h = {"x-linked-etag": "abc123"}
        assert extract_expected_sha256_from_headers(h) == "abc123"

    def test_etag_with_weak_prefix(self) -> None:
        h = {"etag": 'W/"deadbeef"'}
        assert extract_expected_sha256_from_headers(h) == "deadbeef"

    def test_etag_strong(self) -> None:
        h = {"etag": '"cafebabe"'}
        assert extract_expected_sha256_from_headers(h) == "cafebabe"

    def test_x_checksum_sha256(self) -> None:
        h = {"x-checksum-sha256": "ffeeddcc"}
        assert extract_expected_sha256_from_headers(h) == "ffeeddcc"

    def test_priority_x_linked_first(self) -> None:
        # ``x-linked-etag`` is the most reliable for LFS; prefer
        # it over plain ETag.
        h = {
            "x-linked-etag": '"from_lfs"',
            "etag": '"from_blob"',
        }
        assert extract_expected_sha256_from_headers(h) == "from_lfs"

    def test_empty(self) -> None:
        assert extract_expected_sha256_from_headers({}) == ""
        assert extract_expected_sha256_from_headers(
            {"x-linked-etag": ""},  # explicit empty
        ) == ""


# ===========================================================================
# is_gated_http_error
# ===========================================================================
class TestIsGatedHttpError:
    def test_401(self) -> None:
        assert is_gated_http_error(_http_error(401)) is True

    def test_403(self) -> None:
        assert is_gated_http_error(_http_error(403)) is True

    def test_404_is_not_gated(self) -> None:
        assert is_gated_http_error(_http_error(404)) is False

    def test_500_is_not_gated(self) -> None:
        assert is_gated_http_error(_http_error(500)) is False

    def test_non_http_error(self) -> None:
        assert is_gated_http_error(RuntimeError("nope")) is False
        assert is_gated_http_error(ConnectionError("nope")) is False


# ===========================================================================
# ChecksumMismatch / GatedRepoError
# ===========================================================================
class TestExceptions:
    def test_checksum_mismatch_str(self) -> None:
        e = ChecksumMismatch(
            source="huggingface", repo_id="r/f", file_name="a.bin",
            expected_sha256="a" * 64, actual_sha256="b" * 64,
        )
        msg = str(e)
        assert "huggingface" in msg
        assert "a.bin" in msg
        assert "a" * 64 in msg
        assert "b" * 64 in msg
        d = e.as_dict()
        assert d["file_name"] == "a.bin"

    def test_gated_repo_error_str(self) -> None:
        e = GatedRepoError(
            source="huggingface", repo_id="r/f", status_code=401,
            hint="Set $HF_TOKEN.",
        )
        msg = str(e)
        assert "401" in msg
        assert "huggingface" in msg
        assert "r/f" in msg
        assert "Set $HF_TOKEN." in msg


# ===========================================================================
# ModelCache.write_files + expected_sha256s
# ===========================================================================
class TestModelCacheIntegrity:
    def test_write_files_passes_with_correct_pins(
        self, tmp_cache: ModelCache,
    ) -> None:
        a = b"alpha"
        b = b"bravo"
        pins = {"a.bin": _sha256(a), "b.bin": _sha256(b)}
        manifest = tmp_cache.write_files(
            source="huggingface", repo_id="r/f", revision="main",
            license_id="apache-2.0", url="https://example.com",
            files=[
                {"name": "a.bin", "data": a},
                {"name": "b.bin", "data": b},
            ],
            expected_sha256s=pins,
        )
        assert len(manifest.files) == 2
        assert tmp_cache.verify("huggingface", "r/f", "main")

    def test_write_files_raises_on_mismatch(
        self, tmp_cache: ModelCache,
    ) -> None:
        a = b"alpha"
        # Pin a wrong hash.
        with pytest.raises(ChecksumMismatch) as excinfo:
            tmp_cache.write_files(
                source="huggingface", repo_id="r/f", revision="main",
                license_id="apache-2.0", url="https://example.com",
                files=[{"name": "a.bin", "data": a}],
                expected_sha256s={"a.bin": "0" * 64},
            )
        assert excinfo.value.file_name == "a.bin"
        # No manifest is written on a pre-flight mismatch --
        # the cache directory stays empty so the next fetch
        # starts clean.
        assert not tmp_cache.has("huggingface", "r/f", "main")

    def test_write_files_ignores_empty_pins(
        self, tmp_cache: ModelCache,
    ) -> None:
        a = b"alpha"
        # Empty-string pin is treated as "no pin".
        manifest = tmp_cache.write_files(
            source="huggingface", repo_id="r/f", revision="main",
            license_id="apache-2.0", url="https://example.com",
            files=[{"name": "a.bin", "data": a}],
            expected_sha256s={"a.bin": "", "missing.bin": _sha256(b"x")},
        )
        assert len(manifest.files) == 1

    def test_write_files_without_pins_unchanged(
        self, tmp_cache: ModelCache,
    ) -> None:
        # No pins at all: legacy behaviour is preserved.
        a = b"alpha"
        manifest = tmp_cache.write_files(
            source="huggingface", repo_id="r/f", revision="main",
            license_id="apache-2.0", url="https://example.com",
            files=[{"name": "a.bin", "data": a}],
        )
        assert manifest.files[0].sha256 == _sha256(a)


# ===========================================================================
# HF adapter: x-linked-etag → expected_sha256, 401/403 → GatedRepoError
# ===========================================================================
class TestHuggingFaceIntegrity:
    def test_download_records_upstream_sha(
        self, hf_repo_with_sha_headers: FakeTransport,
    ) -> None:
        src = HuggingFaceSource(transport=hf_repo_with_sha_headers)
        # Use the public download path via a tiny fetcher.
        cache = ModelCache(
            root=hf_repo_with_sha_headers.routes[0]["url"]
            if False else "/tmp/tv-int-test",
        )
        # ``ModelCache.write_files`` is what consumes the
        # ``FileDownload.sha256`` -- check directly.
        downloads = src.download_files(
            "example/repo", "main",
            names=["config.json", "model.safetensors"],
        )
        assert len(downloads) == 2
        # Both files advertise x-linked-etag; the adapter must
        # propagate that hash.
        for d in downloads:
            assert d.sha256, "expected non-empty upstream hash"
            assert len(d.sha256) == 64  # sha256 hex

    def test_401_raises_gated_repo_error(self) -> None:
        t = FakeTransport()
        t.route(
            "/api/models/gated/repo",
            raise_http_error=401,
        )
        src = HuggingFaceSource(transport=t)
        with pytest.raises(GatedRepoError) as excinfo:
            src.resolve_license("gated/repo")
        assert excinfo.value.source == "huggingface"
        assert excinfo.value.status_code == 401
        assert "HF_TOKEN" in excinfo.value.hint

    def test_404_is_not_gated(self) -> None:
        t = FakeTransport()
        t.route("/api/models/missing/repo", raise_http_error=404)
        src = HuggingFaceSource(transport=t)
        # 404 returns "" (publicly-visible "no license" sentinel)
        # -- it must NOT raise GatedRepoError.
        assert src.resolve_license("missing/repo") == ""


# ===========================================================================
# Civitai adapter: header-based SHA + 401/403 → GatedRepoError
# ===========================================================================
class TestCivitaiIntegrity:
    def _build_transport(self) -> FakeTransport:
        t = FakeTransport()
        t.route(
            "/api/v1/model-versions/42",
            json_body={
                "license": "CC-BY-4.0",
                "files": [
                    {
                        "name": "model.safetensors",
                        "downloadUrl": "https://cdn.civitai.com/42/model.safetensors",
                        "hashes": {"SHA256": _sha256(b"weights")},
                    },
                ],
            },
        )
        t.route(
            "/42/model.safetensors",
            bytes_body=b"weights",
            bytes_headers={"etag": '"{}"'.format(_sha256(b"weights"))},
        )
        return t

    def test_download_uses_metadata_sha(self) -> None:
        src = CivitaiSource(transport=self._build_transport())
        downloads = src.download_files(
            "42", ["model.safetensors"],
        )
        assert len(downloads) == 1
        # Metadata ``hashes.SHA256`` is preferred over the ETag.
        assert downloads[0].sha256 == _sha256(b"weights")

    def test_401_on_metadata_raises(self) -> None:
        t = FakeTransport()
        t.route("/api/v1/model-versions/42", raise_http_error=401)
        src = CivitaiSource(transport=t)
        with pytest.raises(GatedRepoError) as excinfo:
            src.resolve_license("42")
        assert excinfo.value.source == "civitai"
        assert "CIVITAI_TOKEN" in excinfo.value.hint

    def test_401_on_download_raises(self) -> None:
        t = FakeTransport()
        t.route(
            "/api/v1/model-versions/42",
            json_body={
                "license": "CC-BY-4.0",
                "files": [
                    {
                        "name": "model.safetensors",
                        "downloadUrl": "https://cdn.civitai.com/42/model.safetensors",
                    },
                ],
            },
        )
        t.route(
            "/42/model.safetensors", raise_http_error=403,
        )
        src = CivitaiSource(transport=t)
        with pytest.raises(GatedRepoError) as excinfo:
            src.download_files("42", ["model.safetensors"])
        assert excinfo.value.status_code == 403

    def test_pinned_mismatch_raises(self) -> None:
        t = self._build_transport()
        src = CivitaiSource(transport=t)
        with pytest.raises(ChecksumMismatch):
            src.download_files(
                "42", ["model.safetensors"],
                expected_sha256s={"model.safetensors": "f" * 64},
            )

    def test_pinned_match_passes(self) -> None:
        t = self._build_transport()
        src = CivitaiSource(transport=t)
        downloads = src.download_files(
            "42", ["model.safetensors"],
            expected_sha256s={"model.safetensors": _sha256(b"weights")},
        )
        assert len(downloads) == 1

    def test_list_files_401(self) -> None:
        t = FakeTransport()
        t.route("/api/v1/model-versions/42", raise_http_error=401)
        src = CivitaiSource(transport=t)
        with pytest.raises(GatedRepoError):
            src.list_files("42")


# ===========================================================================
# ModelFetcher: per-call token / expected_sha256s / validate_checksums
# ===========================================================================
def _build_hf_fetcher(tmp_path: Path) -> Tuple[ModelFetcher, FakeTransport]:
    """A :class:`ModelFetcher` whose HF adapter is a fake.

    Returns ``(fetcher, transport)`` so tests can mutate the
    transport in-place (e.g. raise 401).
    """
    t = FakeTransport()
    t.route(
        "/api/models/example/repo",
        json_body={"cardData": {"license": "apache-2.0"}},
    )
    t.route(
        "/api/models/example/repo/tree/main",
        json_body=[
            {"type": "file", "path": "config.json"},
            {"type": "file", "path": "model.safetensors"},
        ],
    )
    config_bytes = b'{"hidden_size": 64}'
    weights_bytes = b"\x00\x01\x02fake-weights"
    t.route(
        "/example/repo/resolve/main/config.json",
        bytes_body=config_bytes,
        bytes_headers={"x-linked-etag": '"{}"'.format(_sha256(config_bytes))},
    )
    t.route(
        "/example/repo/resolve/main/model.safetensors",
        bytes_body=weights_bytes,
        bytes_headers={"x-linked-etag": '"{}"'.format(_sha256(weights_bytes))},
    )
    cache = ModelCache(root=tmp_path / "cache")
    reg = SourceRegistry()
    reg.register("huggingface", HuggingFaceSource(transport=t))
    return ModelFetcher(cache=cache, registry=reg), t


class TestFetcherToken:
    def test_token_kwarg_sets_header(
        self, tmp_path: Path,
    ) -> None:
        fetcher, transport = _build_hf_fetcher(tmp_path)
        # Token "secret" must be present in the Authorization
        # header of every transport call after the first.
        result = fetcher.fetch(
            source="huggingface",
            repo_id="example/repo",
            revision="main",
            token="secret-hf-token",
        )
        assert result.from_cache is False
        # Find the bytes call -- it should carry the auth header.
        bytes_calls = [
            c for c in transport.calls if c[0] == "bytes"
        ]
        assert bytes_calls
        for _, _, headers in bytes_calls:
            assert headers.get("Authorization") == "Bearer secret-hf-token"

    def test_token_does_not_leak_after_call(
        self, tmp_path: Path,
    ) -> None:
        fetcher, transport = _build_hf_fetcher(tmp_path)
        fetcher.fetch(
            source="huggingface", repo_id="example/repo",
            revision="main", token="one-off",
        )
        n_after_first = len(transport.calls)
        # Clear the cache so the second call is forced to hit
        # the transport again.  This proves the registry's
        # adapter is restored -- the second batch of transport
        # calls must not carry ``one-off``.
        fetcher.cache.clear("huggingface", "example/repo", "main")
        fetcher.fetch(
            source="huggingface", repo_id="example/repo",
            revision="main",
        )
        # Every call after the first batch must lack
        # ``Authorization``.
        later = transport.calls[n_after_first:]
        assert later, "expected transport calls after cache clear"
        for _, _, headers in later:
            assert "Authorization" not in headers


class TestFetcherExpectedSha:
    def test_matching_pins_succeed(
        self, tmp_path: Path,
    ) -> None:
        fetcher, _ = _build_hf_fetcher(tmp_path)
        # The fixture advertises x-linked-etag; we re-use the
        # locally-computed hash to satisfy the pin contract.
        config_sha = _sha256(b'{"hidden_size": 64}')
        weights_sha = _sha256(b"\x00\x01\x02fake-weights")
        result = fetcher.fetch(
            source="huggingface", repo_id="example/repo",
            revision="main",
            expected_sha256s={
                "config.json": config_sha,
                "model.safetensors": weights_sha,
            },
        )
        assert result.from_cache is False
        assert not fetcher.cache.verify(
            "huggingface", "example/repo", "main",
        ) is False  # sanity

    def test_mismatched_pins_raise(
        self, tmp_path: Path,
    ) -> None:
        fetcher, _ = _build_hf_fetcher(tmp_path)
        with pytest.raises(ChecksumMismatch):
            fetcher.fetch(
                source="huggingface", repo_id="example/repo",
                revision="main",
                expected_sha256s={
                    "config.json": "f" * 64,  # wrong
                    "model.safetensors": _sha256(
                        b"\x00\x01\x02fake-weights",
                    ),
                },
            )
        # The failed write must not leave a manifest behind.
        assert not fetcher.cache.has(
            "huggingface", "example/repo", "main",
        )

    def test_validate_checksums_false_skips_pins(
        self, tmp_path: Path,
    ) -> None:
        fetcher, _ = _build_hf_fetcher(tmp_path)
        # Even with a deliberately wrong pin, validate_checksums=False
        # is the explicit opt-out -- the write goes through.
        result = fetcher.fetch(
            source="huggingface", repo_id="example/repo",
            revision="main",
            expected_sha256s={"config.json": "f" * 64},
            validate_checksums=False,
        )
        assert result.from_cache is False

    def test_validate_checksums_does_not_leak_pins(
        self, tmp_path: Path,
    ) -> None:
        fetcher, _ = _build_hf_fetcher(tmp_path)
        # First call: pin mismatch + validate off.
        fetcher.fetch(
            source="huggingface", repo_id="example/repo",
            revision="main",
            expected_sha256s={"config.json": "f" * 64},
            validate_checksums=False,
        )
        # Second call: no pins -- must succeed via cache hit.
        result = fetcher.fetch(
            source="huggingface", repo_id="example/repo",
            revision="main",
        )
        assert result.from_cache is True


class TestFetcherGated:
    def test_401_propagates_from_adapter(
        self, tmp_path: Path,
    ) -> None:
        fetcher, transport = _build_hf_fetcher(tmp_path)
        # Replace the metadata route with a 401.
        transport.routes.clear()
        transport.route("/api/models/example/repo", raise_http_error=401)
        with pytest.raises(GatedRepoError) as excinfo:
            fetcher.fetch(
                source="huggingface", repo_id="example/repo",
                revision="main",
            )
        assert excinfo.value.source == "huggingface"


# ===========================================================================
# Top-level fetch() wrapper passes through the new kwargs
# ===========================================================================
class TestFetchWrapper:
    def test_top_level_passes_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Point the default cache at tmp_path so the top-level
        # singleton fetcher does not pollute the home dir.
        monkeypatch.setenv("TORCHA_VERSE_CACHE", str(tmp_path / "global"))
        # Build a fetcher explicitly with a fake transport and
        # install it as the module-level singleton.  We use
        # ``sys.modules`` directly because :mod:`models.source`'s
        # ``__init__`` re-binds ``fetch`` to the free function,
        # so the bare attribute is the function, not the module.
        import sys
        fetch_module = sys.modules["models.source.fetch"]
        fetcher, _ = _build_hf_fetcher(tmp_path)
        monkeypatch.setattr(fetch_module, "_default_fetcher", fetcher)
        # Call the public top-level ``fetch`` (the re-exported
        # function) -- it must route through the patched
        # singleton.
        from models.source import fetch as top_level_fetch
        result = top_level_fetch(
            "example/repo", source="huggingface",
            revision="main",
            expected_sha256s={
                "config.json": _sha256(b'{"hidden_size": 64}'),
                "model.safetensors": _sha256(
                    b"\x00\x01\x02fake-weights",
                ),
            },
        )
        assert result.from_cache is False
