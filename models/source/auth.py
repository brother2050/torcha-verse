"""Authentication helpers for the TorchaVerse model fetcher (v0.4.x P2++).

Several model hosts (HuggingFace, Civitai, OpenAI-compatible
endpoints) require an *API token* to access gated or private
content.  The token is used as ``Authorization: Bearer <token>`` on
every request.

Resolution order
----------------

The :func:`resolve_token` helper centralises the lookup so the HF
and Civitai adapters do not have to re-implement it.  The order
is:

1. Explicit value passed in (typically via
   ``HuggingFaceSource(token=...)`` or
   :meth:`ModelFetcher.fetch(token=...)``).
2. ``$TORCHA_VERSE_TOKEN`` -- a TorchaVerse-wide default.
3. ``$HF_TOKEN`` / ``$HUGGING_FACE_HUB_TOKEN`` -- HuggingFace's own
   env vars, kept in sync with the standard tooling
   (``huggingface-cli login`` writes ``$HF_TOKEN`` to the
   user's shell profile).
4. ``$CIVITAI_TOKEN`` -- Civitai's own env var.
5. ``~/.cache/huggingface/token`` -- the on-disk cache written by
   ``huggingface_hub``'s ``login()``.  We strip whitespace and
   refuse an empty file; a malicious or stray ``token`` file in
   the home directory must never be treated as a credential.
6. ``~/.cache/civitai/token`` -- same idea, less standardised.

The result is *always* either an explicit string the caller trusts
or ``None`` (no token available, public-only access).

Threading: every public type is safe to share across threads; the
file-system reads are best-effort and protected by a per-path
lock.
"""
from __future__ import annotations

import os
import threading
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from infrastructure.logger import get_logger

__all__ = [
    "TokenInfo",
    "resolve_token",
    "GatedRepoError",
    "ChecksumMismatch",
    "extract_expected_sha256_from_headers",
    "auth_headers",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: ``$HF_TOKEN`` / ``$HUGGING_FACE_HUB_TOKEN`` is the official
#: env var name -- we accept both.  The HF tooling writes
#: ``HF_TOKEN`` since 0.10.
_HF_TOKEN_ENV_VARS: tuple = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

#: ``$CIVITAI_TOKEN`` is the de-facto standard.
_CIVITAI_TOKEN_ENV_VARS: tuple = ("CIVITAI_TOKEN",)

#: Generic TorchaVerse-wide default.
_GENERIC_TOKEN_ENV_VAR: str = "TORCHA_VERSE_TOKEN"

#: On-disk locations we look at *after* the env vars.  The HF
#: location is what ``huggingface_hub.login()`` writes.
_HF_TOKEN_FILE: str = "~/.cache/huggingface/token"
_CIVITAI_TOKEN_FILE: str = "~/.cache/civitai/token"

#: Module-level logger.
_logger = get_logger("models.source.auth")


# ---------------------------------------------------------------------------
# TokenInfo
# ---------------------------------------------------------------------------
@dataclass
class TokenInfo:
    """Where a token came from -- the *provenance* matters for tests.

    Attributes:
        value: The token string.  Empty string when no token is
            available; in that case the ``source`` is one of
            ``"none"`` / ``"empty-explicit"`` / ``"empty-env"``.
        source: Provenance tag.  One of:
            * ``"explicit"`` -- passed in by the caller.
            * ``"env"`` -- read from one of the env vars.
            * ``"file"`` -- read from a well-known token file.
            * ``"none"`` -- no token available.
        env_var: When ``source == "env"``, the env-var name that
            held the token.  ``None`` otherwise.
        file_path: When ``source == "file"``, the path that held
            the token.  ``None`` otherwise.
    """

    value: str
    source: str
    env_var: Optional[str] = None
    file_path: Optional[str] = None

    @property
    def is_present(self) -> bool:
        return bool(self.value)

    def as_dict(self) -> dict:
        return {
            "present": self.is_present,
            "source": self.source,
            "env_var": self.env_var,
            "file_path": self.file_path,
            # We do NOT serialise the token value -- never log it.
            "value_redacted": "***" if self.is_present else "",
        }


# ---------------------------------------------------------------------------
# File-system lookup
# ---------------------------------------------------------------------------
_file_locks: dict = {}
_file_locks_guard: threading.Lock = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    """Return a per-path lock so we do not race two readers on the same file."""
    with _file_locks_guard:
        lk = _file_locks.get(path)
        if lk is None:
            lk = threading.Lock()
            _file_locks[path] = lk
        return lk


def _read_token_file(path: str) -> Optional[str]:
    """Return the first non-empty line of ``path`` or ``None``.

    Empty files, missing files, and unreadable files all return
    ``None``.  We do not raise -- the caller treats "no token
    available" as a perfectly valid outcome.
    """
    expanded = os.path.expanduser(path)
    p = Path(expanded)
    with _lock_for(expanded):
        try:
            if not p.is_file():
                return None
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped:
                        return stripped
            return None
        except OSError as exc:
            _logger.debug("token file %s unreadable: %s", expanded, exc)
            return None


# ---------------------------------------------------------------------------
# resolve_token
# ---------------------------------------------------------------------------
def resolve_token(
    explicit: Optional[str] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    sources: str = "huggingface",
    home_dir: Optional[str] = None,
) -> TokenInfo:
    """Resolve a model-host API token.

    Args:
        explicit: Token value passed in by the caller.  Wins over
            every other source.
        env: Optional env-var mapping for testing.  When ``None``
            :data:`os.environ` is used.
        sources: One of ``"huggingface"`` / ``"civitai"`` /
            ``"generic"``.  Selects which env-var / file paths to
            consult; ``"generic"`` looks at ``$TORCHA_VERSE_TOKEN``
            only.
        home_dir: Override for the home directory (used to locate
            the on-disk token files).  Defaults to
            :func:`os.path.expanduser` with no prefix.

    Returns:
        A :class:`TokenInfo` describing the resolution.  When no
        token is available the returned object has
        ``is_present == False`` and ``source == "none"``.
    """
    if explicit is not None:
        if not explicit:
            return TokenInfo(value="", source="empty-explicit")
        return TokenInfo(value=explicit, source="explicit")

    env_map = env if env is not None else os.environ

    if sources == "huggingface":
        env_vars = _HF_TOKEN_ENV_VARS
        file_paths = (_HF_TOKEN_FILE,)
    elif sources == "civitai":
        env_vars = _CIVITAI_TOKEN_ENV_VARS
        file_paths = (_CIVITAI_TOKEN_FILE,)
    elif sources == "generic":
        env_vars = (_GENERIC_TOKEN_ENV_VAR,)
        file_paths = ()
    else:
        raise ValueError(
            "Unknown token source {!r}; expected 'huggingface' / 'civitai' / 'generic'".format(
                sources
            )
        )

    for var in env_vars:
        raw = env_map.get(var)
        if raw is None:
            continue
        stripped = raw.strip()
        if not stripped:
            continue
        return TokenInfo(value=stripped, source="env", env_var=var)

    for fpath in file_paths:
        candidate = fpath
        if home_dir is not None and candidate.startswith("~/"):
            candidate = os.path.join(home_dir, candidate[2:])
        tok = _read_token_file(candidate)
        if tok:
            return TokenInfo(value=tok, source="file", file_path=os.path.expanduser(candidate))

    return TokenInfo(value="", source="none")


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------
def auth_headers(token: Optional[TokenInfo]) -> dict:
    """Build the canonical ``Authorization`` header dict (or empty).

    When ``token`` is ``None`` or :attr:`TokenInfo.is_present` is
    ``False``, returns an empty dict -- callers can always
    splat the result into their request headers.
    """
    if token is None or not token.is_present:
        return {}
    return {"Authorization": "Bearer {}".format(token.value)}


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------
class GatedRepoError(RuntimeError):
    """Raised when a model host refuses access with 401/403.

    The model exists but requires a token the caller has not
    supplied.  The error message tells the operator which env
    var to set, *without* leaking the (possibly empty) token
    value.
    """

    def __init__(
        self,
        source: str,
        repo_id: str,
        status_code: int,
        hint: str = "",
    ) -> None:
        msg = (
            "Access to {source} repo {repo!r} denied (HTTP {code}). "
            "The repo is gated -- a valid token is required. "
            "{hint}"
        ).format(source=source, repo=repo_id, code=status_code, hint=hint)
        super().__init__(msg)
        self.source = source
        self.repo_id = repo_id
        self.status_code = status_code
        self.hint = hint


class ChecksumMismatch(RuntimeError):
    """Raised when a downloaded file's hash does not match the expected.

    Carries the per-file mismatch detail so the operator can
    decide whether to retry the download, abort, or accept a
    different hash from the upstream (rare; usually a sign of
    a corrupted cache or a malicious mirror).
    """

    def __init__(
        self,
        source: str,
        repo_id: str,
        file_name: str,
        expected_sha256: str,
        actual_sha256: str,
    ) -> None:
        msg = (
            "Checksum mismatch for {source} {repo}/{file}: "
            "expected sha256={expected}, got {actual}"
        ).format(
            source=source,
            repo=repo_id,
            file=file_name,
            expected=expected_sha256 or "<none>",
            actual=actual_sha256,
        )
        super().__init__(msg)
        self.source = source
        self.repo_id = repo_id
        self.file_name = file_name
        self.expected_sha256 = expected_sha256
        self.actual_sha256 = actual_sha256

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "repo_id": self.repo_id,
            "file_name": self.file_name,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
        }


# ---------------------------------------------------------------------------
# Upstream-sha extraction
# ---------------------------------------------------------------------------
def extract_expected_sha256_from_headers(
    headers: Mapping[str, str],
    file_name: str = "",
) -> str:
    """Pull a SHA256 *hint* from a download response headers.

    **This function returns a debug hint, NOT an authoritative
    content digest.**  Callers MUST recompute the content sha
    via ``hashlib.sha256(body).hexdigest()`` and use that for
    the cache manifest.  The hint is useful for:

    * operator-facing logs ("mirror X is serving a different
      sha than mirror Y -- check your mirror config");
    * quick sanity checks in tests (mocked HTTP responses can
      stamp an expected sha without writing the full body to
      disk);
    * documentation of the HTTP header layout.

    We look at:

    * ``x-linked-etag`` -- HF LFS pointer git blob oid.  This
      is **NOT** a content sha for LFS-tracked files
      (``.safetensors`` / ``.bin`` / ``.gguf``), because the
      server first returns the 100-byte pointer file and the
      CDN resolves it to a *different* LFS object whose sha
      is only computable after the resolved body is fetched.
      We accept the value as a hint, but the adapter will
      override it with ``sha256(body)``.
    * ``x-repo-commit`` -- the git commit SHA.  This is the
      *blob* hash, not the content hash.  Used as a secondary
      signal only.
    * ``etag`` -- the standard HTTP ETag header, which HF
      populates with the blob SHA for non-LFS files
      (``config.json`` etc.).  We strip the surrounding
      double quotes and the weak prefix.
    * ``x-checksum-sha256`` -- occasionally set by CDN frontends.

    The result is the first non-empty value found, lower-cased
    and stripped.  Empty string when nothing useful is in the
    response.  An empty result is the *expected* outcome in
    many real-world mirrors; do not treat it as an error.
    """
    if not headers:
        return ""
    candidates = (
        "x-linked-etag",
        "x-checksum-sha256",
        "x-sha256",
        "etag",
    )
    for key in candidates:
        raw = headers.get(key)
        if not raw:
            continue
        # ETag is "W/\"hash\"" or "\"hash\"" -- strip the
        # weak prefix and the surrounding quotes.
        cleaned = raw.strip()
        if cleaned.startswith("W/"):
            cleaned = cleaned[2:]
        cleaned = cleaned.strip().strip('"').lower()
        if not cleaned:
            continue
        return cleaned
    return ""


def is_gated_http_error(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` looks like a 401/403 from a gated repo."""
    if isinstance(exc, urllib.error.HTTPError):
        return int(getattr(exc, "code", 0)) in (401, 403)
    return False
