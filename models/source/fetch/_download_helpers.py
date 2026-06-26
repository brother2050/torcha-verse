"""Download / progress / pin-validation helpers for ModelFetcher.

These were inline methods on :class:`ModelFetcher` in v0.4.x.  In
v0.6.x they live in their own module so the :class:`ModelFetcher`
class body stays focused on the main fetch flow.  They are imported
into :class:`ModelFetcher` via a private alias.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, List, Mapping, Optional, Sequence

from infrastructure.logger import get_logger

from ..auth import ChecksumMismatch, GatedRepoError
from ..cache import CachedModel
from ..huggingface import DownloadProgress, FileDownload
from ..license_check import normalise_spdx

_logger = get_logger("models.source.fetch")


def resolve_license_id(
    adapter: Any, repo_id: str, canonical: str
) -> str:
    """Call the adapter's ``resolve_license`` and normalise the result.

    GatedRepoError is *not* swallowed: 401/403 is an
    operator-actionable condition (set $HF_TOKEN, install
    ``huggingface_hub`` and ``huggingface-cli login``, etc.)
    so the caller deserves the original error.  Any other
    adapter failure (5xx, network, JSON shape) is logged and
    reduced to ``""`` -- the license check will then
    short-circuit to "no license declared", which the
    operator can re-try without configuring a token.
    """
    try:
        raw = adapter.resolve_license(repo_id)
    except GatedRepoError:
        # 401/403: a token is required.  Do NOT swallow --
        # surface the actionable error to the caller.
        raise
    except Exception as exc:  # noqa: BLE001 - any adapter failure
        _logger.warning(
            "License resolution failed for %s/%s: %s",
            canonical, repo_id, exc,
        )
        return ""
    return normalise_spdx(str(raw or ""))


def download_default_artifacts(
    adapter: Any,
    repo_id: str,
    revision: str,
    canonical: str,
    on_progress: Optional[Callable[..., None]] = None,
    expected_sha256s: Optional[Mapping[str, str]] = None,
) -> List[FileDownload]:
    """Call the adapter's ``download_default_artifacts`` if present.

    ``on_progress`` is forwarded to the adapter when it accepts a
    keyword argument of the same name.  Adapters that do not
    accept a progress callback simply ignore the argument (the
    typical contract is "forgive extra kwargs").

    ``expected_sha256s`` is forwarded to the adapter when it
    accepts a keyword argument of the same name; otherwise it
    is silently dropped.  The fetcher itself re-validates the
    pins in :meth:`ModelCache.write_files` so a stripped kwarg
    on the adapter is still safe.

    Two callback shapes are accepted (the fetcher docs explain
    why both are useful):

    * ``(file_name, bytes_done, bytes_total, mirror)`` -- the
      v0.4.0 ergonomic shape; converted internally into a
      :class:`DownloadProgress` tick.
    * ``Callable[[DownloadProgress], None]`` -- the v0.4.x
      P2+ low-level shape; forwarded verbatim.
    """
    fn = getattr(adapter, "download_default_artifacts", None)
    if not callable(fn):
        raise RuntimeError(
            "source adapter {!r} does not support downloads".format(canonical)
        )
    if canonical == "civitai":
        # Civitai's adapter does not take a revision and does
        # not accept an on_progress / expected_sha256s
        # callback -- ``download_default_artifacts(version_id)``.
        # Integrity pins are validated in
        # :meth:`ModelCache.write_files` so a stripped kwarg
        # is still safe.
        return fn(repo_id)  # type: ignore[arg-type]
    if on_progress is None and not expected_sha256s:
        return fn(repo_id, revision or "main")  # type: ignore[arg-type]

    # Normalise the callback.  We do this by *probing* the
    # callback's signature -- a 1-arg callback is treated as
    # the low-level ``DownloadProgress`` shape; everything
    # else is treated as the 4-arg ergonomic shape.
    if on_progress is not None:
        try:
            sig = inspect.signature(on_progress)
            nargs = len([
                p for p in sig.parameters.values()
                if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ])
        except (TypeError, ValueError):
            nargs = 4  # default: assume ergonomic shape

        if nargs <= 1:
            adapter_cb: Optional[Callable[..., None]] = on_progress
        else:
            def adapter_cb(tick: DownloadProgress) -> None:
                on_progress(
                    tick.file_name, tick.bytes_done, tick.bytes_total, tick.mirror,
                )
    else:
        adapter_cb = None

    # Try the most-featureful signature first; fall back
    # gracefully when the adapter does not accept one of the
    # kwargs.  Order: progress + pins, then progress only,
    # then pins only, then no kwargs.
    try:
        if adapter_cb is not None and expected_sha256s:
            return fn(  # type: ignore[arg-type]
                repo_id, revision or "main",
                on_progress=adapter_cb,
                expected_sha256s=expected_sha256s,
            )
        if adapter_cb is not None:
            return fn(  # type: ignore[arg-type]
                repo_id, revision or "main", on_progress=adapter_cb,
            )
        if expected_sha256s:
            return fn(  # type: ignore[arg-type]
                repo_id, revision or "main", expected_sha256s=expected_sha256s,
            )
        return fn(repo_id, revision or "main")  # type: ignore[arg-type]
    except TypeError:
        # Adapter has the old signature -- fall back.
        return fn(repo_id, revision or "main")  # type: ignore[arg-type]


def validate_pins_against_manifest(
    manifest: CachedModel,
    expected_sha256s: Mapping[str, str],
) -> None:
    """Re-validate pinned sha256s against an *existing* manifest.

    Used on a cross-mirror dedup hit: the on-disk files are
    byte-identical to the ones we just downloaded (same
    fingerprint), so the per-file hash is implicit.  We
    therefore compare the pins to the manifest's recorded
    digests rather than re-hashing every file -- a few
    microseconds vs. a few hundred milliseconds.
    """
    recorded = {f.name: f.sha256 for f in manifest.files}
    for name, pinned in expected_sha256s.items():
        if not pinned:
            continue
        actual = recorded.get(name, "")
        if not actual:
            # The pinned file is not in this manifest -- the
            # caller is asking for stricter coverage than the
            # cached model has.  Refuse the dedup hit and let
            # the caller's next fetch do a clean write.
            raise ChecksumMismatch(
                source=manifest.source,
                repo_id=manifest.repo_id,
                file_name=name,
                expected_sha256=pinned,
                actual_sha256="<not-in-existing-manifest>",
            )
        if actual != pinned:
            raise ChecksumMismatch(
                source=manifest.source,
                repo_id=manifest.repo_id,
                file_name=name,
                expected_sha256=pinned,
                actual_sha256=actual,
            )
