"""Helpers shared by the hardcoding rules (v0.6.x).

* :func:`_truncate` -- shortens a string for display.
* :func:`_looks_like_path` -- heuristic path detection (mirrors
  the v0.4.x behaviour exactly so the path detection is
  identical).
* :data:`STRING_MIN_LENGTH` / :data:`LIST_MAX_ELEMENTS` /
  :data:`_EXEMPT_NUMBERS` / :data:`_STRUCTURAL_MIN` /
  :data:`_STRUCTURAL_MAX` / :data:`_STRUCTURAL_PACKAGES` -- the
  v0.4.x constants that drive the built-in rules.
"""

from __future__ import annotations

import re
from typing import Any, FrozenSet, Tuple

__all__ = [
    "STRING_MIN_LENGTH",
    "LIST_MAX_ELEMENTS",
    "EXEMPT_NUMBERS",
    "STRUCTURAL_MIN",
    "STRUCTURAL_MAX",
    "STRUCTURAL_PACKAGES",
    "truncate",
    "looks_like_path",
    "_looks_like_path",
    "_truncate",
]


#: Minimum string length that triggers :class:`StringLiteralRule`.
STRING_MIN_LENGTH: int = 10

#: Maximum number of list elements allowed before :class:`ListLiteralRule`
#: triggers.
LIST_MAX_ELEMENTS: int = 3

#: Numeric literals exempt from :class:`NumericLiteralRule`.
EXEMPT_NUMBERS: FrozenSet[Any] = frozenset({0, 1, -1, 0.0, 1.0, -1.0})

#: Numeric range for the ``is_structural_init`` heuristic.  Values
#: outside this range are *not* considered structural.
STRUCTURAL_MIN: int = 2
STRUCTURAL_MAX: int = 10000

#: Top-level package prefixes the structural-init heuristic considers
#: to be "model definitions".
STRUCTURAL_PACKAGES: Tuple[str, ...] = ("models/",)

#: Module-level functions whose call result is "reading from
#: runtime config" (mirrors :data:`RUNTIME_FUNCS` in the scanner).
RUNTIME_FUNCS: FrozenSet[str] = frozenset({
    "getenv", "environ", "expanduser", "expandvars", "getattr",
    "argv", "get",
})


def truncate(text: str, limit: int = 120) -> str:
    """Truncate ``text`` to ``limit`` characters, appending ``...`` when over."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


#: Underscore-prefixed alias for tests / external callers that
#: import the v0.4.x private name ``_truncate``.
_truncate = truncate


#: Regex used by :func:`looks_like_path` (mirrors the v0.4.x scanner).
_PATH_PATTERN: "re.Pattern[str]" = re.compile(
    r"(?:^[/~.])"
    r"|(?:[A-Za-z]:[\\/])"
    r"|(?:[\w.\-]+/[\w.\-]+/[\w.\-]*)"
    r"|(?:[\w.\-]+/[\w.\-]+\.\w{1,8})"
    r"|(?:[\w.\-]+\\[\w.\-]+\.\w{1,8})"
)


def looks_like_path(value: str) -> bool:
    """Return ``True`` if ``value`` resembles a filesystem path.

    Mirrors the v0.4.x behaviour exactly (single-character
    separators are excluded; only strong path indicators count).
    """
    if len(value) < 2:
        return False
    if "/" not in value and "\\" not in value:
        return False
    return _PATH_PATTERN.search(value) is not None


#: Underscore-prefixed alias for tests that import the v0.4.x
#: private name ``_looks_like_path``.
_looks_like_path = looks_like_path
