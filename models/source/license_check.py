"""SPDX-style license whitelist for the TorchaVerse model fetcher (v0.4.0).

When a model is fetched from an external source (HuggingFace Hub, Civitai,
etc.) the fetcher must verify that the model's license is on a curated
allow-list before downloading the weights.  This module centralises the
allow-list, the SPDX-id normaliser, and the verification function used
by :mod:`models.source.fetch` and the per-source adapters.

Why a whitelist?
----------------
Some open-weight model releases carry restrictions (``non-commercial``,
``research-only``, ``no-derivatives``) that make them unsuitable for
inclusion in a generated-AI framework.  The default allow-list covers
the four most permissive SPDX ids that are commonly used for
open-weight models.  Callers may extend or replace the list per-fetch
through the ``allow_license=[...]`` argument.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``models.source`` (this module) -- license policy.
"""

from __future__ import annotations

import threading
from typing import FrozenSet, Iterable, Optional, Sequence, Tuple

from infrastructure.logger import get_logger

__all__ = [
    "DEFAULT_ALLOW_LICENSE",
    "LicenseCheckResult",
    "check_license",
    "normalise_spdx",
    "is_known_non_commercial",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: The default SPDX-ids accepted by :func:`fetch` when the caller does
#: not pass an explicit ``allow_license=`` argument.  These are the
#: four most permissive open-source licenses commonly used for
#: open-weight model releases.
DEFAULT_ALLOW_LICENSE: FrozenSet[str] = frozenset({
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "cc-by-4.0",
})

#: Lower-case substrings that, when present in a license string, mark
#: it as non-commercial.  Used by :func:`is_known_non_commercial` to
#: short-circuit license checks when the license is obviously
#: restricted.
_NON_COMMERCIAL_TOKENS: Tuple[str, ...] = (
    "noncommercial",
    "non-commercial",
    "nc-",
    "research-only",
    "research_use_only",
    "no-commercial",
)

#: Lower-case substrings that mark a license as "no derivatives",
#: which the default allow-list rejects.
_NO_DERIVATIVES_TOKENS: Tuple[str, ...] = (
    "no-derivatives",
    "nd-",
    "noderivatives",
)

#: SPDX-ids that are *known* to be acceptable but live outside the
#: permissive default set (e.g. copyleft licenses).  These are
#: *always* allowed -- the default deny-list does not contain them --
#: but a user that has explicitly requested a stricter allow-list
#: (e.g. only MIT) can still reject them.
_KNOWN_OK_SPDX: FrozenSet[str] = frozenset({
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "unlicense",
    "mpl-2.0",
    "lgpl-3.0",
    "gpl-3.0",
    "epl-2.0",
})

#: Module-level logger.
_logger = get_logger("models.source.license_check")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalise_spdx(license_id: str) -> str:
    """Normalise a license string to a lowercase SPDX-style id.

    Performs three normalisations:

    1. Strip leading / trailing whitespace.
    2. Lower-case.
    3. Replace spaces with hyphens (SPDX-ids never contain spaces).

    The function is intentionally permissive -- it does not try to
    resolve free-form license names like "Apache License 2.0" to
    their SPDX-ids.  Callers that need that resolution should do it
    upstream; this function only canonicalises a known SPDX-id.

    Args:
        license_id: Any string.  Empty strings are returned as-is.

    Returns:
        A normalised license id (lowercase, hyphens, no whitespace).
    """
    if not license_id:
        return ""
    return license_id.strip().lower().replace(" ", "-")


def is_known_non_commercial(license_id: str) -> bool:
    """Return ``True`` if ``license_id`` contains a known NC token.

    Useful for early rejection of obviously-restricted licenses
    (e.g. ``cc-by-nc-4.0``, ``research-only``) without having to
    resolve the SPDX id.
    """
    if not license_id:
        return False
    haystack = license_id.lower()
    return any(token in haystack for token in _NON_COMMERCIAL_TOKENS)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
class LicenseCheckResult:
    """Outcome of a license check.

    Attributes:
        accepted: ``True`` if the license is allowed.
        reason: Human-readable explanation.  Always populated so
            callers can log / surface it.
        license_id: The normalised license id that was checked.
    """

    __slots__ = ("accepted", "reason", "license_id")

    def __init__(
        self, accepted: bool, reason: str, license_id: str,
    ) -> None:
        self.accepted = bool(accepted)
        self.reason = str(reason)
        self.license_id = str(license_id)

    def __repr__(self) -> str:
        return "LicenseCheckResult(accepted={}, license_id={!r}, reason={!r})".format(
            self.accepted, self.license_id, self.reason,
        )

    def __bool__(self) -> bool:
        return self.accepted


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------
def check_license(
    license_id: str,
    allow_license: Optional[Sequence[str]] = None,
) -> LicenseCheckResult:
    """Verify a license id against an allow-list.

    The check has three short-circuits:

    1. Empty / missing license -> denied ("no license declared").
    2. License string contains a known non-commercial token -> denied
       ("non-commercial restriction detected").
    3. License string contains a "no-derivatives" token -> denied
       ("derivative restriction detected"), unless the caller
       explicitly allow-lists it (e.g. ``cc-by-nd-4.0``).

    Otherwise the normalised id is matched against ``allow_license``;
    when the caller does not pass an allow-list the module-level
    :data:`DEFAULT_ALLOW_LICENSE` is used.

    Args:
        license_id: A license id or free-form string.  May be empty.
        allow_license: Optional explicit allow-list.  When ``None``
            :data:`DEFAULT_ALLOW_LICENSE` is used.

    Returns:
        A :class:`LicenseCheckResult` with ``accepted`` and ``reason``.
    """
    if not license_id or not license_id.strip():
        return LicenseCheckResult(
            accepted=False,
            reason="no license declared by the source",
            license_id="",
        )

    normalised = normalise_spdx(license_id)

    # Compute the effective allow-list once -- it is reused below.
    explicit_allow: FrozenSet[str] = frozenset(
        normalise_spdx(s) for s in (allow_license or DEFAULT_ALLOW_LICENSE)
        if s
    )

    # If the caller *explicitly* allow-listed this id (e.g.
    # ``allow_license=["cc-by-nc-4.0"]``), honour that -- they have
    # opted in to the non-commercial / no-derivatives restriction.
    if normalised in explicit_allow:
        return LicenseCheckResult(
            accepted=True,
            reason=(
                "license {!r} is in the caller's explicit allow-list"
            ).format(normalised),
            license_id=normalised,
        )

    # Short-circuit: known non-commercial restriction.
    if is_known_non_commercial(normalised):
        return LicenseCheckResult(
            accepted=False,
            reason=(
                "license {!r} contains a non-commercial restriction "
                "(e.g. 'non-commercial' / 'research-only')"
            ).format(normalised),
            license_id=normalised,
        )

    # No-derivatives check: the default allow-list rejects ND
    # licenses; a caller that has not explicitly listed the ND id
    # is denied.
    if any(token in normalised for token in _NO_DERIVATIVES_TOKENS):
        return LicenseCheckResult(
            accepted=False,
            reason=(
                "license {!r} carries a no-derivatives restriction "
                "and is not in the caller's allow-list"
            ).format(normalised),
            license_id=normalised,
        )

    # The id is not in the allow-list.  If it is a known-OK SPDX
    # (e.g. a copyleft license) we surface a helpful hint, but we
    # still deny because the default allow-list is the source of
    # truth.
    if normalised in _KNOWN_OK_SPDX:
        return LicenseCheckResult(
            accepted=False,
            reason=(
                "license {!r} is a recognised permissive / copyleft "
                "SPDX id but is not in the caller's allow-list; "
                "pass allow_license=[...] to opt in"
            ).format(normalised),
            license_id=normalised,
        )

    return LicenseCheckResult(
        accepted=False,
        reason=(
            "license {!r} is unknown to the whitelist; pass "
            "allow_license=[...] to explicitly opt in"
        ).format(normalised),
        license_id=normalised,
    )


# ---------------------------------------------------------------------------
# Process-level cached allow-list
# ---------------------------------------------------------------------------
_default_lock = threading.Lock()
_effective_default: FrozenSet[str] = DEFAULT_ALLOW_LICENSE


def get_default_allow_license() -> FrozenSet[str]:
    """Return the effective default allow-list (frozen)."""
    return _effective_default


def extend_default_allow_license(extra: Iterable[str]) -> None:
    """Add licenses to the module-level default allow-list (idempotent).

    Useful for one-off opt-ins (e.g. a script that wants to enable
    ``gpl-3.0`` globally for a single run).  The change is process-
    local and thread-safe.
    """
    global _effective_default
    with _default_lock:
        _effective_default = _effective_default.union(
            normalise_spdx(s) for s in extra if s
        )
