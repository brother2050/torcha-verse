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

import urllib.error
from typing import Any, Dict, List, Optional, Sequence, Tuple

from infrastructure.logger import get_logger

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
        self._token: Optional[str] = token
        self._logger = _logger

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _auth_headers(self) -> Dict[str, str]:
        if self._token:
            return {"Authorization": "Bearer {}".format(self._token)}
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
        """
        try:
            data = self._transport.get_json(
                self._version_url(version_id), headers=self._auth_headers(),
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
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
        """
        try:
            data = self._transport.get_json(
                self._version_url(version_id), headers=self._auth_headers(),
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
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
    ) -> List[FileDownload]:
        """Download a list of files from a Civitai version.

        Resolves the per-file download URL from the version
        metadata, then issues one ``GET`` per name.  Missing files
        (404) are silently skipped -- the caller decides whether the
        absence is fatal.
        """
        import hashlib

        try:
            data = self._transport.get_json(
                self._version_url(version_id), headers=self._auth_headers(),
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
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
            if not name:
                continue
            url = url_by_name.get(name)
            if not url:
                self._logger.warning(
                    "Civitai file %s not found in version %s", name, version_id,
                )
                continue
            try:
                body, _ = self._transport.get_bytes(
                    url, headers=self._auth_headers(),
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                self._logger.warning(
                    "Civitai download failed for %s/%s: %s",
                    version_id, name, exc,
                )
                continue
            local_sha = hashlib.sha256(body).hexdigest()
            results.append(
                FileDownload(name=name, data=body, sha256=local_sha)
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
