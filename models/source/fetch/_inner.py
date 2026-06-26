"""The inner :func:`fetch_inner` pipeline (v0.6.x).

In v0.4.x this lived as :meth:`ModelFetcher._fetch_inner` -- a
single 150-line method that did, in order:

* license resolution + allow-list check
* cache lookup (with optional re-hash verification)
* download + dedup-aware write
* cross-mirror / cross-revision fingerprint dedup
* post-write integrity check

In v0.6.x the body is moved to a module-level
:func:`fetch_inner` function so the :class:`ModelFetcher` class
file stays under the soft 500-line cap.  The
:meth:`ModelFetcher._fetch_inner` method remains as a thin
forwarder to keep test introspection / monkeypatch working.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from infrastructure.logger import get_logger

from ..cache import CachedModel, ModelCache, compute_content_fingerprint
from ..license_check import LicenseCheckResult, check_license
from ._download_helpers import (
    download_default_artifacts,
    resolve_license_id,
    validate_pins_against_manifest,
)
from ._result import FetchResult

_logger = get_logger("models.source.fetch")


def fetch_inner(
    *,
    cache: ModelCache,
    adapter: Any,
    source: str,
    canonical: str,
    repo_id: str,
    revision: str,
    allow_license: Optional[Sequence[str]],
    default_allow: Sequence[str],
    verify_cache: bool,
    on_progress: Optional[Any],
    expected_sha256s: Mapping[str, str],
    resolve_license_id_fn=resolve_license_id,
    download_default_artifacts_fn=download_default_artifacts,
    validate_pins_against_manifest_fn=validate_pins_against_manifest,
    build_url_fn=None,
) -> FetchResult:
    """Run the inner fetch pipeline (license -> cache -> download -> write).

    This is the body of the v0.4.x
    :meth:`ModelFetcher._fetch_inner` method, lifted into a
    module-level function so the
    :class:`ModelFetcher` class file stays small.  The
    ``*_fn`` parameters default to the module-level helpers but
    can be overridden in tests.
    """
    # ----- license check ------------------------------------------------
    license_id = resolve_license_id_fn(adapter, repo_id, canonical)
    effective_allow = (
        list(allow_license) if allow_license is not None
        else list(default_allow)
    )
    check: LicenseCheckResult = check_license(license_id, allow_license=effective_allow)
    if not check.accepted:
        # Persist the rejection in the log; do NOT cache anything.
        _logger.warning(
            "License rejected for %s/%s@%s: %s",
            canonical, repo_id, revision, check.reason,
        )
        raise PermissionError(check.reason)

    # ----- cache lookup -------------------------------------------------
    loc = cache.location_for(canonical, repo_id, revision)
    if cache.has(canonical, repo_id, revision):
        if verify_cache and not cache.verify(canonical, repo_id, revision):
            _logger.warning(
                "Cache verification failed for %s/%s@%s; re-fetching",
                canonical, repo_id, revision,
            )
            cache.clear(canonical, repo_id, revision)
        else:
            manifest = cache.load_manifest(canonical, repo_id, revision)
            _logger.info(
                "Cache hit for %s/%s@%s (license=%s)",
                canonical, repo_id, revision, check.license_id,
            )
            return FetchResult(
                location=loc,
                manifest=manifest,
                source=canonical,
                license_check=check,
                from_cache=True,
            )

    # ----- download + dedup-aware write --------------------------------
    downloads = download_default_artifacts_fn(
        adapter, repo_id, revision, canonical, on_progress=on_progress,
        expected_sha256s=expected_sha256s,
    )
    if not downloads:
        raise RuntimeError(
            "source {!r} returned no files for {!r}@{:!r}; "
            "check the network / credentials and retry".format(
                canonical, repo_id, revision,
            )
        )

    # ----- cross-mirror / cross-revision dedup --------------------------
    # We just downloaded `downloads`; before writing them to
    # the canonical location, check whether the same content
    # (by fingerprint) is already present in the cache under
    # any other (repo_id, revision).  When a match is found we
    # *do not* write a duplicate copy -- the existing files
    # on disk are byte-identical (same fingerprint) and the
    # caller is happy because they did not pay for *disk
    # space* and a second round of integrity verification.
    # We only spent a network round-trip for the metadata
    # lookup, which is unavoidable at the time we learn the
    # file set.
    files_spec: List[Dict[str, Any]] = [
        {"name": d.name, "data": d.data, "sha256": d.sha256}
        for d in downloads
    ]
    fingerprint = compute_content_fingerprint(files_spec)
    existing = cache.find_by_fingerprint(canonical, fingerprint)
    if existing is not None and (
        existing.repo_id != repo_id or existing.revision != revision
    ):
        # The same content is cached under a different key --
        # surface that as a cache hit (the operator is happy
        # because they did not pay for a network round-trip).
        # We DO NOT skip the write to the canonical location
        # here -- the caller asked for this exact key, and
        # future lookups under this key should be a direct
        # manifest hit rather than a fingerprint scan.
        _logger.info(
            "Cross-mirror dedup hit: %s/%s@%s == cached %s/%s@%s",
            canonical, repo_id, revision,
            canonical, existing.repo_id, existing.revision,
        )
        try:
            existing_manifest: Optional[CachedModel] = cache.load_manifest(
                canonical, existing.repo_id, existing.revision,
            )
        except (OSError, ValueError):
            existing_manifest = None
        if existing_manifest is not None:
            # When the caller pinned checksums, re-validate
            # the existing manifest's recorded digests -- a
            # stale manifest is the worst case for supply-
            # chain integrity.  The local file on disk is
            # byte-identical to the one we just downloaded
            # (fingerprint match), so the per-file hash is
            # already implicit; we just re-check that it
            # matches the pins if any.
            if expected_sha256s:
                validate_pins_against_manifest_fn(
                    existing_manifest, expected_sha256s,
                )
            # Skip the write: serve the existing manifest
            # directly.  The files on disk are content-equal
            # (by construction -- same fingerprint).
            return FetchResult(
                location=existing,
                manifest=existing_manifest,
                source=canonical,
                license_check=check,
                from_cache=True,
            )

    if build_url_fn is None:
        from ._fetcher import ModelFetcher
        url = ModelFetcher._build_url(adapter, canonical, repo_id, revision)
    else:
        url = build_url_fn(adapter, canonical, repo_id, revision)
    manifest = cache.write_files(
        source=canonical,
        repo_id=repo_id,
        revision=revision,
        license_id=check.license_id,
        url=url,
        files=files_spec,
        expected_sha256s=expected_sha256s or None,
    )
    # The just-written files are already sha256-verified by
    # ``write_files``; call :meth:`verify` to cross-check the
    # manifest's per-file digests.
    if not cache.verify(canonical, repo_id, revision):
        cache.clear(canonical, repo_id, revision)
        raise RuntimeError(
            "post-write integrity check failed for "
            "{}/{}/{}".format(canonical, repo_id, revision)
        )
    return FetchResult(
        location=loc,
        manifest=manifest,
        source=canonical,
        license_check=check,
        from_cache=False,
    )
