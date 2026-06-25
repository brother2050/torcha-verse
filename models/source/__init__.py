"""TorchaVerse model source adapters (v0.4.0).

This subpackage implements the v0.4.0 "model source auto-fetch +
license audit" milestone: a single line

.. code-block:: python

    from models.source import fetch
    result = fetch("Qwen/Qwen2.5-0.5B-Instruct")

verifies the model's license against the default allow-list, pulls
the weights to ``~/.cache/torcha-verse/``, and returns a
:class:`FetchResult` describing the cache location and the
whitelist verdict.

Modules
-------

* :mod:`license_check` -- SPDX-id normalisation and the allow-list
  check (:data:`DEFAULT_ALLOW_LICENSE`).
* :mod:`cache` -- the on-disk :class:`ModelCache` with atomic
  writes and sha256 integrity verification.
* :mod:`huggingface` -- the HuggingFace Hub adapter, with an
  injectable :class:`HttpTransport` so tests can run without the
  network.
* :mod:`civitai` -- the Civitai adapter (same transport interface).
* :mod:`fetch` -- the unified :class:`ModelFetcher` and the
  :func:`fetch` convenience function.

Threading: every public type is safe to share across threads.  The
``fetch()`` free function uses a process-level singleton
:class:`ModelFetcher` so multiple calls in the same evaluation /
serving process share the same cache and metadata state.
"""

from __future__ import annotations

from .auth import (
    ChecksumMismatch,
    GatedRepoError,
    TokenInfo,
    auth_headers,
    extract_expected_sha256_from_headers,
    is_gated_http_error,
    resolve_token,
)
from .cache import (
    CacheLocation,
    CachedFile,
    CachedModel,
    ModelCache,
    compute_content_fingerprint,
    default_cache_root,
)
from .civitai import CivitaiSource
from .fetch import FetchResult, ModelFetcher, SourceRegistry, fetch
from .huggingface import (
    DEFAULT_API_BASE,
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    DownloadProgress,
    FileDownload,
    HttpTransport,
    HuggingFaceSource,
    OllamaTransport,
    OpenAICompatTransport,
    UrllibTransport,
)
from .license_check import (
    DEFAULT_ALLOW_LICENSE,
    LicenseCheckResult,
    check_license,
    extend_default_allow_license,
    get_default_allow_license,
    is_known_non_commercial,
    normalise_spdx,
)
from .mirrors import (
    DEFAULT_HF_MIRRORS,
    MirrorHealth,
    MirrorSet,
    check_all_mirrors,
    check_mirror_health,
    is_useful_mirror_error,
)

__all__ = [
    # license_check
    "DEFAULT_ALLOW_LICENSE",
    "LicenseCheckResult",
    "check_license",
    "normalise_spdx",
    "is_known_non_commercial",
    "get_default_allow_license",
    "extend_default_allow_license",
    # auth
    "TokenInfo",
    "resolve_token",
    "auth_headers",
    "GatedRepoError",
    "ChecksumMismatch",
    "extract_expected_sha256_from_headers",
    "is_gated_http_error",
    # cache
    "CacheLocation",
    "CachedFile",
    "CachedModel",
    "ModelCache",
    "default_cache_root",
    "compute_content_fingerprint",
    # huggingface
    "FileDownload",
    "HttpTransport",
    "HuggingFaceSource",
    "OllamaTransport",
    "OpenAICompatTransport",
    "UrllibTransport",
    "DownloadProgress",
    "DEFAULT_API_BASE",
    "DEFAULT_TIMEOUT",
    "DEFAULT_USER_AGENT",
    # civitai
    "CivitaiSource",
    # mirrors
    "DEFAULT_HF_MIRRORS",
    "MirrorSet",
    "MirrorHealth",
    "check_mirror_health",
    "check_all_mirrors",
    "is_useful_mirror_error",
    # fetch
    "fetch",
    "ModelFetcher",
    "FetchResult",
    "SourceRegistry",
]


__version__ = "0.4.0"
