"""Result container for the unified model fetcher (v0.6.x).

This module is an internal sub-module of :mod:`models.source.fetch`;
importing it directly is supported but the canonical entry point
remains :func:`models.source.fetch.fetch`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..cache import CacheLocation, CachedModel
from ..license_check import LicenseCheckResult

__all__ = ["FetchResult"]


@dataclass
class FetchResult:
    """Outcome of a :func:`fetch` call.

    Attributes:
        location: The :class:`CacheLocation` of the fetched model.
        manifest: The :class:`CachedModel` manifest that was loaded
            from the cache (either pre-existing or freshly written).
        source: The canonical source id used in the cache.
        license_check: The :class:`LicenseCheckResult` from the
            license whitelist verification.
        from_cache: ``True`` when the model was already cached and
            no network call was needed; ``False`` when a fresh
            download happened.
    """

    location: CacheLocation
    manifest: CachedModel
    source: str
    license_check: LicenseCheckResult
    from_cache: bool

    @property
    def accepted(self) -> bool:
        """``True`` when the license was accepted by the whitelist."""
        return self.license_check.accepted

    def as_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "location": self.location.as_dict(),
            "manifest": self.manifest.as_dict(),
            "source": self.source,
            "license_check": {
                "accepted": self.license_check.accepted,
                "reason": self.license_check.reason,
                "license_id": self.license_check.license_id,
            },
            "from_cache": self.from_cache,
        }

    def __repr__(self) -> str:
        return (
            "FetchResult(source={!r}, repo_id={!r}, license={!r}, "
            "from_cache={})".format(
                self.source,
                self.manifest.repo_id,
                self.license_check.license_id,
                self.from_cache,
            )
        )
