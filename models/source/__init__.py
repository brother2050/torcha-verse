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

from .cache import (
    CacheLocation,
    CachedFile,
    CachedModel,
    ModelCache,
    default_cache_root,
)
from .civitai import CivitaiSource
from .fetch import FetchResult, ModelFetcher, SourceRegistry, fetch
from .huggingface import (
    FileDownload,
    HttpTransport,
    HuggingFaceSource,
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

__all__ = [
    # license_check
    "DEFAULT_ALLOW_LICENSE",
    "LicenseCheckResult",
    "check_license",
    "normalise_spdx",
    "is_known_non_commercial",
    "get_default_allow_license",
    "extend_default_allow_license",
    # cache
    "CacheLocation",
    "CachedFile",
    "CachedModel",
    "ModelCache",
    "default_cache_root",
    # huggingface
    "FileDownload",
    "HttpTransport",
    "HuggingFaceSource",
    "UrllibTransport",
    # civitai
    "CivitaiSource",
    # fetch
    "fetch",
    "ModelFetcher",
    "FetchResult",
    "SourceRegistry",
]


__version__ = "0.4.0"
