"""Module-level constants shared by the HuggingFace sub-package.

A small, dependency-free module so the rest of the sub-package
can import shared constants without a circular dependency on
:mod:`.__init__` (which depends on every sub-module).
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_API_BASE",
    "DEFAULT_USER_AGENT",
    "DEFAULT_TIMEOUT",
    "CHUNK_SIZE",
    "HF_RESOLVE_URL",
    "HF_API_URL",
]


#: Default HuggingFace API base URL.
DEFAULT_API_BASE: str = "https://huggingface.co"

#: Default user-agent sent with every request.
DEFAULT_USER_AGENT: str = (
    "torcha-verse/0.6.0 (+https://github.com/torcha-verse/torcha-verse)"
)

#: Default request timeout (seconds).
DEFAULT_TIMEOUT: float = 30.0

#: Read buffer size when downloading a file body.
CHUNK_SIZE: int = 1 << 16  # 64 KiB

#: URL template for HuggingFace's CDN (resolve endpoint).
#: ``{base}`` is replaced by the chosen base; ``{repo_id}`` /
#: ``{revision}`` / ``{filename}`` are passed through verbatim.
HF_RESOLVE_URL: str = "{base}/{repo_id}/resolve/{revision}/{filename}"

#: URL template for HuggingFace's REST API (api endpoint).
#: ``{base}`` / ``{repo_id}`` / ``{path}`` are passed through.
#: Note the ``models/`` prefix is included in the ``{path}``
#: argument (e.g. ``"/models/{id}"``) so a single
#: ``HF_API_URL`` template can serve both the
#: ``/api/models/{id}`` and ``/api/models/{id}/tree/{rev}``
#: endpoints without divergence.
HF_API_URL: str = "{base}/api{path}"
