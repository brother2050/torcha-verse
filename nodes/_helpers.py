"""Shared helper functions for node modules.

This module centralises the small coercion / extraction helpers that were
previously duplicated across :mod:`nodes.image`, :mod:`nodes.video`,
:mod:`nodes.audio` and :mod:`nodes.consistency`.  Keeping them in one place
avoids drift and makes future changes a single-edit affair.

The coercion helpers deliberately **exclude** :class:`bool` (which is a
subclass of :class:`int` in Python) and only accept genuinely numeric
values, matching the behaviour previously inlined in each node module.
"""
from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "_MEGAPIXEL_PIXELS",
    "coerce_dim",
    "coerce_int",
    "coerce_float",
    "ref_id",
]

#: Number of pixels in one megapixel (used to normalise spatial estimates).
_MEGAPIXEL_PIXELS: float = 1_000_000.0


def coerce_dim(value: Any) -> Optional[int]:
    """Return ``value`` as an ``int`` when it is an integer-like number.

    ``bool`` is explicitly excluded (it is a subclass of :class:`int` in
    Python but is not a valid dimension).  Non-integer floats (e.g.
    ``5.7``) and strings are rejected.
    """
    if isinstance(value, bool):  # bool is a subclass of int -- exclude it.
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def coerce_int(value: Any) -> Optional[int]:
    """Return ``value`` as an ``int`` when it is an integer-like number.

    Alias of :func:`coerce_dim`; kept as a separate name for readability at
    call sites that are not specifically about image / video dimensions.
    """
    return coerce_dim(value)


def coerce_float(value: Any) -> Optional[float]:
    """Return ``value`` as a ``float`` when it is a real number.

    ``bool`` is explicitly excluded.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def ref_id(ref: Any) -> Optional[str]:
    """Return the asset id of ``ref`` for ``AssetRef``, ``str`` or id-bearing objects.

    Resolution order:

    1. ``None`` -> ``None``.
    2. :class:`str` -> the string itself (a raw asset id).
    3. Objects with an ``asset_id`` attribute (e.g. :class:`~assets.base.AssetRef`).
    4. Objects with an ``id`` attribute.
    5. Anything else -> ``None``.
    """
    if ref is None:
        return None
    if isinstance(ref, str):
        return ref
    if hasattr(ref, "asset_id"):
        return ref.asset_id
    if hasattr(ref, "id"):
        return ref.id
    return None
