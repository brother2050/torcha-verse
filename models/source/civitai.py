"""Civitai source adapter for the TorchaVerse model fetcher (v0.4.0).

Civitai is a community model-sharing platform focused on image /
video generation checkpoints (Stable Diffusion, etc.).  Its public
REST API exposes model metadata, version lists, and download URLs;
this module wraps the version-fetch and download steps behind a
:class:`CivitaiSource` adapter with the same shape as
:class:`models.source.huggingface.HuggingFaceSource`.

The adapter speaks the same :class:`HttpTransport` interface as the
HuggingFace adapter, so a test fixture (or a future production
override) can swap both at once.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``models.source`` (this module) -- Civitai adapter.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from infrastructure.logger import get_logger

from .auth import (
    ChecksumMismatch,
    GatedRepoError,
    TokenInfo,
    extract_expected_sha256_from_headers,
    is_gated_http_error,
    resolve_token,
)

from .huggingface import (
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    FileDownload,
    HttpTransport,
    UrllibTransport,
)

__all__ = ["CivitaiSource"]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Default Civitai API base URL.
_DEFAULT_API_BASE = "https://civitai.com/api"

#: Module-level logger.
_logger = get_logger("models.source.civitai")


# ---------------------------------------------------------------------------
# CivitaiSource
# ---------------------------------------------------------------------------
class CivitaiSource:
    """Adapter for the Civitai model-version API.

    Civitai identifies models by numeric id and *versions* by a
    separate version id; for the v0.4.0 minimum-viable milestone we
    treat the version id as the "revision" field of the cache key
    (the same slot the HuggingFace adapter uses for ``main`` /
    commit hash).  The public :meth:`fetch` on the unified
    :mod:`models.source.fetch` module passes through whatever the
    caller supplies.

    Args:
        api_base: Base URL for the Civitai API.  Defaults to
            ``"https://civitai.com/api"``.
        transport: Optional :class:`HttpTransport` (mainly for
            testing).  Defaults to :class:`UrllibTransport`.
        token: Optional Civitai API token.  Forwarded as
            ``Authorization: Bearer <token>`` (Civitai accepts both
            this and the legacy ``x-civitai-key`` header; we use
            the standard one).
    """

    def __init__(
        self,
        api_base: str = _DEFAULT_API_BASE,
        transport: Optional[HttpTransport] = None,
        token: Optional[str] = None,
    ) -> None:
        self._api_base: str = api_base.rstrip("/")
        self._transport: HttpTransport = transport or UrllibTransport()
        # Resolve the token through the centralised helper so
        # env vars / on-disk files work out of the box.
        self._token: TokenInfo = resolve_token(explicit=token, sources="civitai")
        self._logger = _logger

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _auth_headers(self) -> Dict[str, str]:
        if self._token and self._token.is_present:
            return {"Authorization": "Bearer {}".format(self._token.value)}
        return {}

    def _version_url(self, version_id: str) -> str:
        return "{}/v1/model-versions/{}".format(self._api_base, version_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def resolve_license(self, version_id: str) -> str:
        """Return the SPDX license id declared by a model version.

        Civitai returns the license as a free-form string (most
        often ``"CC-BY-4.0"``, ``"Apache 2.0"``, ``"MIT"`` or
        ``"AllowCommercialUse.Image"`` style flags).  We normalise
        the recognised ones to SPDX-ids and pass the rest through
        verbatim -- the caller is expected to run the value through
        :func:`models.source.license_check.check_license` so an
        unknown id is rejected by default.

        Raises:
            GatedRepoError: When the version metadata endpoint
                returns 401/403 (token required).
        """
        try:
            data = self._transport.get_json(
                self._version_url(version_id), headers=self._auth_headers(),
            )
        except Exception as exc:  # noqa: BLE001 - any failure
            if is_gated_http_error(exc):
                raise GatedRepoError(
                    source="civitai",
                    repo_id=version_id,
                    status_code=int(getattr(exc, "code", 0)),
                    hint="Set $CIVITAI_TOKEN or pass CivitaiSource(token=...) and retry.",
                ) from exc
            self._logger.warning(
                "Civitai metadata fetch failed for %s: %s", version_id, exc,
            )
            return ""
        if not isinstance(data, dict):
            return ""
        # Civitai sometimes puts the license in two places.
        for key in ("license", "trainedWords", "commercialUse"):
            v = data.get(key)
            if key == "license" and isinstance(v, str):
                return v
            if key == "commercialUse" and isinstance(v, str):
                # Map the "commercialUse" flag to a license hint
                # so the whitelist can still decide.
                return {
                    "Image": "cc-by-4.0",
                    "Rent": "cc-by-4.0",
                    "Sell": "cc-by-4.0",
                    "None": "cc-by-nc-4.0",
                    "NoneCC": "cc-by-nc-4.0",
                }.get(v, v)
        # Fall back to a hint embedded in the model object.
        model = data.get("model")
        if isinstance(model, dict):
            v = model.get("license")
            if isinstance(v, str):
                return v
        return ""

    def list_files(
        self, version_id: str
    ) -> List[str]:
        """List the file names attached to a Civitai version.

        Returns:
            A list of file names (e.g. ``["model.safetensors",
            "config.json"]``).  Empty on error.

        Raises:
            GatedRepoError: When the version metadata endpoint
                returns 401/403 (token required).
        """
        try:
            data = self._transport.get_json(
                self._version_url(version_id), headers=self._auth_headers(),
            )
        except Exception as exc:  # noqa: BLE001
            if is_gated_http_error(exc):
                raise GatedRepoError(
                    source="civitai",
                    repo_id=version_id,
                    status_code=int(getattr(exc, "code", 0)),
                    hint="Set $CIVITAI_TOKEN or pass CivitaiSource(token=...) and retry.",
                ) from exc
            self._logger.warning(
                "Civitai file listing failed for %s: %s", version_id, exc,
            )
            return []
        if not isinstance(data, dict):
            return []
        files = data.get("files")
        if not isinstance(files, list):
            return []
        names: List[str] = []
        for entry in files:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if isinstance(name, str) and name:
                names.append(name)
        return names

    def download_files(
        self,
        version_id: str,
        names: Sequence[str],
        *,
        expected_sha256s: Optional[Mapping[str, str]] = None,
    ) -> List[FileDownload]:
        """Download a list of files from a Civitai version.

        Resolves the per-file download URL from the version
        metadata, then issues one ``GET`` per name.  Missing files
        (404) are silently skipped -- the caller decides whether the
        absence is fatal.

        Args:
            version_id: Civitai model-version id.
            names: Sequence of file names to download.  Names that
                start with ``"~"`` are skipped (Civitai uses this
                prefix for non-public checkpoint fragments, mirroring
                the HF convention).
            expected_sha256s: Optional ``{file_name: sha256_hex}``
                map.  When a file's local hash does not match the
                pinned value we raise
                :class:`~models.source.auth.ChecksumMismatch`.

        Raises:
            GatedRepoError: When the version metadata endpoint
                returns 401/403 (token required).
            ChecksumMismatch: When ``expected_sha256s`` pins a file
                and the local hash does not match.
        """
        try:
            data = self._transport.get_json(
                self._version_url(version_id), headers=self._auth_headers(),
            )
        except Exception as exc:  # noqa: BLE001 - any failure
            if is_gated_http_error(exc):
                raise GatedRepoError(
                    source="civitai",
                    repo_id=version_id,
                    status_code=int(getattr(exc, "code", 0)),
                    hint="Set $CIVITAI_TOKEN or pass CivitaiSource(token=...) and retry.",
                ) from exc
            self._logger.warning(
                "Civitai metadata fetch (for download) failed for %s: %s",
                version_id, exc,
            )
            return []
        if not isinstance(data, dict):
            return []

        # Build a name -> downloadUrl map.
        url_by_name: Dict[str, str] = {}
        for entry in data.get("files", []) or []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            url = entry.get("downloadUrl") or entry.get("url")
            if isinstance(name, str) and isinstance(url, str):
                url_by_name[name] = url

        results: List[FileDownload] = []
        for name in names:
            if not name or name.startswith("~"):
                continue
            url = url_by_name.get(name)
            if not url:
                self._logger.warning(
                    "Civitai file %s not found in version %s", name, version_id,
                )
                continue
            try:
                body, resp_headers = self._transport.get_bytes(
                    url, headers=self._auth_headers(),
                )
            except Exception as exc:  # noqa: BLE001
                if is_gated_http_error(exc):
                    raise GatedRepoError(
                        source="civitai",
                        repo_id=version_id,
                        status_code=int(getattr(exc, "code", 0)),
                        hint="Set $CIVITAI_TOKEN or pass CivitaiSource(token=...) and retry.",
                    ) from exc
                self._logger.warning(
                    "Civitai download failed for %s/%s: %s",
                    version_id, name, exc,
                )
                continue
            local_sha = hashlib.sha256(body).hexdigest()
            # Civitai exposes TWO different sha256 values that we
            # have to be careful to distinguish:
            #
            # 1. ``x-checksum-sha256`` / similar HTTP header -- this
            #    is *usually* the content sha, but mirrors do not
            #    always set it, and a few CDNs set it to the blob
            #    storage oid (an LFS-style pitfall).  Treat it as a
            #    debug hint only.
            #
            # 2. The version metadata API ``data["files"][i]
            #    ["hashes"]["SHA256"]`` -- this is the *content*
            #    sha reported by Civitai's own content-addressable
            #    storage, and is the only header-shaped value we
            #    trust as an authoritative content digest.
            #
            # We compute the *content* sha locally (``local_sha``)
            # and use it for the manifest.  The metadata API sha
            # is only used as a *cross-check* -- if the operator
            # pinned one via ``expected_sha256s``, we trust that
            # pin; otherwise we use local_sha unconditionally.
            header_hint = extract_expected_sha256_from_headers(
                resp_headers, file_name=name,
            )
            metadata_sha = ""
            for entry in data.get("files", []) or []:
                if not isinstance(entry, dict):
                    continue
                if entry.get("name") == name:
                    hashes = entry.get("hashes")
                    if isinstance(hashes, dict):
                        sha = hashes.get("SHA256")
                        if isinstance(sha, str) and sha:
                            metadata_sha = sha
                    break
            if header_hint and header_hint != local_sha and (
                not metadata_sha or header_hint != metadata_sha
            ):
                _logger.debug(
                    "Civitai header sha hint for %s/%s "
                    "(hint=%s) differs from content sha %s; "
                    "using local sha as authoritative.",
                    version_id, name, header_hint, local_sha,
                )
            pinned = ""
            if expected_sha256s is not None:
                pinned = str(expected_sha256s.get(name, "") or "")
            if pinned and local_sha != pinned:
                raise ChecksumMismatch(
                    source="civitai",
                    repo_id=version_id,
                    file_name=name,
                    expected_sha256=pinned,
                    actual_sha256=local_sha,
                )
            # Sanity-check the cross-check when we have a trusted
            # metadata sha and no explicit pin -- if local and
            # metadata disagree, the mirror is serving stale /
            # corrupted bytes and we must not cache them.
            if (
                metadata_sha
                and not pinned
                and local_sha != metadata_sha
            ):
                raise ChecksumMismatch(
                    source="civitai",
                    repo_id=version_id,
                    file_name=name,
                    expected_sha256=metadata_sha,
                    actual_sha256=local_sha,
                )
            results.append(
                FileDownload(
                    name=name,
                    data=body,
                    sha256=local_sha,
                )
            )
        return results

    def download_default_artifacts(
        self, version_id: str
    ) -> List[FileDownload]:
        """Download the canonical Civitai artifacts (config + weights)."""
        names: List[str] = []
        available = self.list_files(version_id)
        has = set(available)
        for candidate in (
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "model.ckpt",
            "vae.pt",
            "text_encoder.pt",
        ):
            if candidate in has:
                names.append(candidate)
        if not names and available:
            names = available[:4]
        return self.download_files(version_id, names)

    def __repr__(self) -> str:
        return "CivitaiSource(api_base={!r})".format(self._api_base)
