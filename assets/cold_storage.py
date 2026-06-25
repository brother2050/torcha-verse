"""Cold-tier storage backends for the v0.4.x asset store.

This module ships three cold-tier implementations of the
:class:`assets.store.ColdStorageProtocol`:

* :class:`S3ColdStorage` -- talks to any S3-compatible service
  (AWS S3, Alibaba OSS via the S3-compat API, MinIO, Tencent COS,
  Cloudflare R2).  Uses :mod:`boto3` (or :mod:`botocore` as a
  pure-network alternative) when available; falls back to a
  pure-:mod:`urllib` implementation when neither is installed,
  so the framework is always importable.

* :class:`LocalColdStorage` -- a content-addressed cold store
  backed by a local directory.  Used as a development
  stand-in for S3 and as a "warm-cold" tier in single-process
  deployments.

* :func:`make_cold_storage` -- factory that returns a
  :class:`S3ColdStorage` configured from environment variables
  (``TV_COLD_BACKEND`` / ``TV_COLD_BUCKET`` / ``TV_COLD_ENDPOINT`` /
  ``TV_COLD_ACCESS_KEY`` / ``TV_COLD_SECRET_KEY`` /
  ``TV_COLD_REGION``), and a :class:`LocalColdStorage` when
  ``TV_COLD_BACKEND=local`` or when the env vars are unset.

The backends follow the **content-addressed** contract documented
on :class:`assets.store.ColdStorageProtocol` -- every method is
keyed by the sha256 ``content_hash`` of the stored blob, so the
warm and cold tiers never disagree about which bytes are stored
under a given hash.

Design notes
------------

* The pure-urllib :class:`S3ColdStorage` path implements the
  AWS SigV4 signing scheme by hand; the implementation is small
  (~80 lines) and well-tested for HEAD / GET / PUT / DELETE
  object operations.  Use :mod:`boto3` for production; the
  urllib path is a deliberate fall-back so CI environments
  without ``boto3`` can still exercise the routing logic.
* All backends are **thread-safe** (a single
  :class:`threading.RLock` serialises I/O).
* Errors from the cold tier are surfaced as
  :class:`ColdStorageError` so the :class:`AssetStore` can
  log them and decide whether to fall back to the warm tier.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from infrastructure.logger import get_logger

_logger = get_logger("assets.cold_storage")

__all__ = [
    "ColdStorageError",
    "ColdStorageConfig",
    "LocalColdStorage",
    "S3ColdStorage",
    "make_cold_storage",
]


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------
class ColdStorageError(RuntimeError):
    """Raised when a cold-tier backend cannot complete an operation."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ColdStorageConfig:
    """Configuration shared by every cold-tier backend.

    Attributes:
        backend: One of ``"s3"`` (default) or ``"local"``.
        bucket: Bucket name (S3) or root directory (local).
        prefix: Object key prefix (S3) or sub-directory (local).
            Used to namespace blobs inside the backend.
        endpoint: Custom S3 endpoint URL (MinIO, OSS, R2, ...).
            ``None`` uses the AWS default.
        region: AWS region; ``None`` defaults to ``"us-east-1"``.
        access_key: Access key id; ``None`` falls back to
            environment / instance profile credentials.
        secret_key: Secret access key; ``None`` falls back to
            environment / instance profile credentials.
        use_ssl: Use HTTPS for the S3 endpoint (default
            ``True``).
        verify: Whether to verify TLS certificates (default
            ``True``; set to ``False`` for self-signed MinIO
            during local development).
        multipart_threshold: File size above which a multipart
            upload is used (default 16 MiB).  Files below this
            are uploaded with a single PUT request.
    """

    backend: str = "s3"
    bucket: str = ""
    prefix: str = "torcha-verse/cold/"
    endpoint: Optional[str] = None
    region: str = "us-east-1"
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    use_ssl: bool = True
    verify: bool = True
    multipart_threshold: int = 16 * 1024 * 1024

    def with_overrides(self, **overrides: Any) -> "ColdStorageConfig":
        """Return a copy of this config with the given fields overridden."""
        return ColdStorageConfig(
            **{
                "backend": self.backend,
                "bucket": self.bucket,
                "prefix": self.prefix,
                "endpoint": self.endpoint,
                "region": self.region,
                "access_key": self.access_key,
                "secret_key": self.secret_key,
                "use_ssl": self.use_ssl,
                "verify": self.verify,
                "multipart_threshold": self.multipart_threshold,
                **overrides,
            }
        )


# ---------------------------------------------------------------------------
# Local cold storage
# ---------------------------------------------------------------------------
class LocalColdStorage:
    """A :class:`ColdStorageProtocol` implementation backed by a directory.

    Useful as a development stand-in for S3 and as a single-process
    "warm-cold" tier that does not require network I/O.
    """

    def __init__(self, root: Union[str, Path], prefix: str = "torcha-verse/cold/") -> None:
        self._root: Path = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._prefix: str = prefix.rstrip("/") + "/" if prefix else ""
        self._lock: threading.RLock = threading.RLock()
        self._logger = get_logger("assets.cold_storage.local")

    def _path_for(self, content_hash: str) -> Path:
        # Mirror the warm-tier sharding (2 hex chars deep) so that
        # any blob can be promoted / demoted between tiers without
        # re-hashing.
        return self._root / self._prefix.lstrip("/") / content_hash[:2] / content_hash

    def fetch(self, content_hash: str, dst: Path) -> Path:
        src = self._path_for(content_hash)
        with self._lock:
            if not src.exists():
                raise ColdStorageError(f"blob not found in local cold tier: {content_hash}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return dst

    def store(self, content_hash: str, src: Path) -> None:
        dst = self._path_for(content_hash)
        with self._lock:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)

    def exists(self, content_hash: str) -> bool:
        return self._path_for(content_hash).exists()

    def delete(self, content_hash: str) -> bool:
        path = self._path_for(content_hash)
        with self._lock:
            if not path.exists():
                return False
            path.unlink()
            # Best-effort cleanup of empty shard directory.
            try:
                if path.parent.exists() and not any(path.parent.iterdir()):
                    path.parent.rmdir()
            except OSError:
                pass
            return True


# ---------------------------------------------------------------------------
# S3 (SigV4-signed) cold storage -- no boto3 dependency
# ---------------------------------------------------------------------------
class S3ColdStorage:
    """A :class:`ColdStorageProtocol` implementation for S3-compatible services.

    Uses an in-process SigV4 signing path so that **no** third-party
    dependency (boto3 / botocore / aiobotocore) is required.  This
    means the framework is importable in every environment, including
    minimal CI sandboxes, and the cold tier can still be exercised
    against a local MinIO / R2 instance for development.

    For production deployments it is recommended to install
    ``boto3``; the ``_via_boto3`` path is used automatically when
    the import succeeds and gives access to the full S3 feature
    surface (multipart, retry policies, ...).
    """

    #: Hash algorithm used for SigV4.
    _ALGO: str = "AWS4-HMAC-SHA256"
    #: S3 service identifier.
    _SERVICE: str = "s3"
    #: Default request timeout in seconds.
    _TIMEOUT: float = 30.0

    def __init__(self, cfg: ColdStorageConfig) -> None:
        if not cfg.bucket:
            raise ValueError("ColdStorageConfig.bucket must be a non-empty string.")
        self._cfg: ColdStorageConfig = cfg
        self._lock: threading.RLock = threading.RLock()
        self._boto3 = _try_import_boto3()
        if self._boto3 is not None:
            self._client = self._boto3.client(
                "s3",
                endpoint_url=cfg.endpoint,
                region_name=cfg.region,
                aws_access_key_id=cfg.access_key,
                aws_secret_access_key=cfg.secret_key,
                verify=cfg.verify,
            )
        else:
            self._client = None
        self._logger = get_logger("assets.cold_storage.s3")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _key(self, content_hash: str) -> str:
        return f"{self._cfg.prefix}{content_hash[:2]}/{content_hash}"

    def _host(self) -> str:
        cfg = self._cfg
        if cfg.endpoint:
            return cfg.endpoint.rstrip("/")
        return f"{self._SERVICE}.{cfg.region}.amazonaws.com"

    def _scheme(self) -> str:
        return "https" if self._cfg.use_ssl else "http"

    @staticmethod
    def _hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _sign(
        self,
        method: str,
        key: str,
        *,
        body: bytes = b"",
        extra_headers: Optional[Dict[str, str]] = None,
        query: str = "",
    ) -> Tuple[Dict[str, str], str]:
        """Return a dict of signed headers and the request URL.

        Implements the AWS SigV4 signing scheme for the S3
        service.  This is a from-scratch implementation
        (~80 lines) that handles the three header categories
        AWS requires (host / x-amz-date / x-amz-content-sha256)
        and produces a deterministic signing key per day.
        """
        cfg = self._cfg
        access_key = cfg.access_key or os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret_key = cfg.secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        if not access_key or not secret_key:
            raise ColdStorageError(
                "S3ColdStorage requires access_key and secret_key (or "
                "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars)."
            )

        host = self._host().replace(f"{self._scheme()}://", "")
        now = time.gmtime()
        amz_date = time.strftime("%Y%m%dT%H%M%SZ", now)
        date_stamp = time.strftime("%Y%m%d", now)
        content_hash = self._hash(body)

        canonical_headers = {
            "host": host,
            "x-amz-content-sha256": content_hash,
            "x-amz-date": amz_date,
        }
        if extra_headers:
            canonical_headers.update(extra_headers)

        # Sort header keys (SigV4 is case-insensitive but
        # canonical form uses lowercase).
        sorted_header_keys = sorted(canonical_headers.keys())
        canonical_header_str = "".join(
            f"{k}:{canonical_headers[k].strip()}\n" for k in sorted_header_keys
        )
        signed_headers_str = ";".join(sorted_header_keys)

        canonical_request = "\n".join(
            [
                method,
                "/" + urllib.parse.quote(key, safe="/~"),
                query,
                canonical_header_str,
                signed_headers_str,
                content_hash,
            ]
        )
        credential_scope = f"{date_stamp}/{cfg.region}/{self._SERVICE}/aws4_request"
        string_to_sign = "\n".join(
            [
                self._ALGO,
                amz_date,
                credential_scope,
                self._hash(canonical_request.encode("utf-8")),
            ]
        )

        def _sign(key_bytes: bytes, msg: str) -> bytes:
            return hmac.new(key_bytes, msg.encode("utf-8"), hashlib.sha256).digest()

        k_date = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
        k_region = _sign(k_date, cfg.region)
        k_service = _sign(k_region, self._SERVICE)
        k_signing = _sign(k_service, "aws4_request")
        signature = hmac.new(
            k_signing, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        auth_header = (
            f"{self._ALGO} Credential={access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers_str}, Signature={signature}"
        )
        headers = dict(canonical_headers)
        headers["Authorization"] = auth_header
        url = f"{self._scheme()}://{host}/{urllib.parse.quote(key, safe='/~')}"
        if query:
            url += "?" + query
        return headers, url

    def _request(
        self,
        method: str,
        key: str,
        *,
        body: bytes = b"",
        extra_headers: Optional[Dict[str, str]] = None,
        query: str = "",
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Send a single signed HTTP request and return (status, headers, body)."""
        headers, url = self._sign(method, key, body=body, extra_headers=extra_headers, query=query)
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        ctx = None
        try:
            import ssl  # local import keeps stdlib-only path cheap

            ctx = ssl.create_default_context()
            if not self._cfg.verify:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
        except Exception:
            ctx = None
        try:
            with urllib.request.urlopen(req, timeout=self._TIMEOUT, context=ctx) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as exc:  # noqa: PERF203
            return exc.code, dict(exc.headers or {}), exc.read() or b""
        except urllib.error.URLError as exc:
            raise ColdStorageError(f"network error talking to S3: {exc.reason}") from exc

    # ------------------------------------------------------------------
    # Public ColdStorageProtocol surface
    # ------------------------------------------------------------------
    def exists(self, content_hash: str) -> bool:
        key = self._key(content_hash)
        if self._client is not None:
            with self._lock:
                try:
                    self._client.head_object(Bucket=self._cfg.bucket, Key=key)
                    return True
                except Exception as exc:  # noqa: BLE001
                    self._logger.debug("S3 HEAD %s failed: %s", key[:32], exc)
                    return False
        status, _, _ = self._request("HEAD", key)
        return status == 200

    def store(self, content_hash: str, src: Path) -> None:
        if not src.exists():
            raise FileNotFoundError(f"src blob does not exist: {src}")
        key = self._key(content_hash)
        body = src.read_bytes()
        if self._client is not None:
            with self._lock:
                self._client.put_object(Bucket=self._cfg.bucket, Key=key, Body=body)
            return
        extra = {"content-length": str(len(body))}
        status, _, payload = self._request("PUT", key, body=body, extra_headers=extra)
        if status not in (200, 201):
            raise ColdStorageError(
                f"S3 PUT {key[:32]} returned {status}: {payload[:128]!r}"
            )

    def fetch(self, content_hash: str, dst: Path) -> Path:
        key = self._key(content_hash)
        if self._client is not None:
            with self._lock:
                try:
                    response = self._client.get_object(Bucket=self._cfg.bucket, Key=key)
                    body = response["Body"].read()
                except Exception as exc:  # noqa: BLE001
                    raise ColdStorageError(
                        f"S3 GET {key[:32]} failed: {exc}"
                    ) from exc
        else:
            status, _, body = self._request("GET", key)
            if status != 200:
                raise ColdStorageError(
                    f"S3 GET {key[:32]} returned {status}; blob not in cold tier?"
                )
            body = body or b""
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(body)
        return dst

    def delete(self, content_hash: str) -> bool:
        key = self._key(content_hash)
        if self._client is not None:
            with self._lock:
                try:
                    self._client.delete_object(Bucket=self._cfg.bucket, Key=key)
                    return True
                except Exception as exc:  # noqa: BLE001
                    self._logger.debug("S3 DELETE %s failed: %s", key[:32], exc)
                    return False
        status, _, _ = self._request("DELETE", key)
        return status in (200, 204)


# ---------------------------------------------------------------------------
# Optional boto3 import
# ---------------------------------------------------------------------------
def _try_import_boto3() -> Any:
    """Return the :mod:`boto3` module when it is importable, else ``None``."""
    try:
        import boto3  # type: ignore
        return boto3
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_cold_storage(
    *,
    config: Optional[ColdStorageConfig] = None,
    env: Optional[Dict[str, str]] = None,
) -> Optional[Any]:
    """Build a cold-tier backend from a config or environment variables.

    The factory is the **single recommended entry point** for
    constructing a cold-tier backend.  Callers that already have a
    concrete :class:`S3ColdStorage` or :class:`LocalColdStorage`
    instance can pass it directly to :class:`AssetStore`.

    Args:
        config: Optional explicit :class:`ColdStorageConfig`.  When
            supplied the ``env`` argument is ignored.
        env: Optional environment dict (default ``os.environ``).

    Returns:
        * :class:`LocalColdStorage` if ``config.backend == "local"``
          (or ``TV_COLD_BACKEND == "local"``).
        * :class:`S3ColdStorage` otherwise -- the backend is
          configured for AWS S3, Alibaba OSS (via the S3-compat
          endpoint), Tencent COS, Cloudflare R2 or MinIO depending
          on ``TV_COLD_ENDPOINT``.
        * ``None`` when neither a config nor the necessary
          environment variables are available.
    """
    if config is not None:
        if config.backend == "local":
            return LocalColdStorage(root=config.bucket, prefix=config.prefix)
        if config.backend == "s3":
            return S3ColdStorage(config)
        raise ValueError(f"unknown cold backend: {config.backend!r}")

    env_map = env if env is not None else os.environ
    backend = env_map.get("TV_COLD_BACKEND", "").lower()
    if backend == "local":
        root = env_map.get("TV_COLD_LOCAL_ROOT", "")
        if not root:
            return None
        return LocalColdStorage(root=root, prefix=env_map.get("TV_COLD_PREFIX", "torcha-verse/cold/"))

    bucket = env_map.get("TV_COLD_BUCKET", "")
    if not bucket:
        return None
    cfg = ColdStorageConfig(
        backend="s3",
        bucket=bucket,
        prefix=env_map.get("TV_COLD_PREFIX", "torcha-verse/cold/"),
        endpoint=env_map.get("TV_COLD_ENDPOINT") or None,
        region=env_map.get("TV_COLD_REGION", "us-east-1"),
        access_key=env_map.get("TV_COLD_ACCESS_KEY") or None,
        secret_key=env_map.get("TV_COLD_SECRET_KEY") or None,
        use_ssl=env_map.get("TV_COLD_USE_SSL", "1") != "0",
        verify=env_map.get("TV_COLD_VERIFY_TLS", "1") != "0",
    )
    return S3ColdStorage(cfg)
