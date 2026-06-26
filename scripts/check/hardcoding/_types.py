"""Data classes for the hardcoding scanner (v0.6.x).

The :class:`Violation` and :class:`Exemption` dataclasses are kept
in their own module so the AST visitor / scan / CLI modules can
each stay well under the soft 500-line cap.  The classes are
re-exported from :mod:`scripts.check.hardcoding.__init__` so
callers can keep using ``from scripts.check_hardcoding import
Violation``.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Optional, Set

from ._constants import SEVERITY_CRITICAL, SEVERITY_INFO, WILDCARD_TYPES

__all__ = ["Violation", "Exemption"]


@dataclass
class Violation:
    """A single hardcoding violation found by the scanner.

    Attributes:
        file: Path of the offending file, relative to the scan root.
        line: 1-based line number where the violation occurs.
        col: 0-based column offset of the offending node.
        type: Violation type identifier (one of the ``*_LITERAL``
            constants).
        content: A short textual representation of the offending value.
        severity: ``critical`` / ``warn`` / ``info`` -- the v0.4.x D1
            extension; see :doc:`/docs/hardcoding_convention`.
    """

    file: str
    line: int
    col: int
    type: str
    content: str
    severity: str = SEVERITY_CRITICAL

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "file": self.file,
            "line": self.line,
            "column": self.col,
            "type": self.type,
            "content": self.content,
            "severity": self.severity,
        }


@dataclass
class Exemption:
    """A single whitelist exemption.

    A violation is exempt when *all* specified fields match.  Omitted
    (``None``) fields match anything.

    Attributes:
        file: glob pattern matched against the violation's relative
            path (e.g. ``"core/*.py"``).
        type: Violation type to exempt, or ``"*"``/``"all"`` for any.
        line: Exact line number to exempt, or ``None`` for any line.
        content_contains: Substring that must appear in the violation
            content, or ``None`` to ignore content.
        severity: When the exemption matches, downgrade the violation's
            severity to this level.  ``None`` keeps the original.
        protocol_format: When ``True``, mark the violation as
            ``severity=info`` (protocol-bound literal, see D1
            convention section 1.3).
        rules: Optional set of *rule names* this exemption applies
            to.  ``None`` (the default) matches every rule.  Set this
            to opt out of a *specific* rule without affecting the
            other rules' behaviour on the same file/line -- e.g.
            opt out of :class:`NumericLiteralRule` for a model
            dimension that lives outside the
            ``[_STRUCTURAL_MIN, _STRUCTURAL_MAX]`` heuristic range.
        reason: Optional human-readable rationale, persisted in
            exports.
    """

    file: str
    type: str = "*"
    line: Optional[int] = None
    content_contains: Optional[str] = None
    severity: Optional[str] = None
    protocol_format: bool = False
    rules: Optional[Set[str]] = None
    reason: Optional[str] = None

    def matches(self, violation: "Violation") -> bool:
        """Return ``True`` if this exemption covers ``violation``.

        When ``protocol_format`` is set, the matching violation's
        ``severity`` is *downgraded* to ``info`` but the violation is
        still returned (for audit).  When ``severity`` is set on the
        exemption, the matching violation's severity is set to that
        level.  When ``rules`` is set, the exemption only matches
        violations whose ``type`` is in the set (this is the
        *per-rule opt-out* path).
        """
        if not fnmatch.fnmatch(violation.file, self.file):
            return False
        if self.type not in WILDCARD_TYPES and violation.type != self.type:
            return False
        if self.line is not None and violation.line != self.line:
            return False
        if self.content_contains is not None and self.content_contains not in violation.content:
            return False
        if self.rules is not None and violation.type not in self.rules:
            return False
        return True

    def apply(self, violation: "Violation") -> bool:
        """Try to apply this exemption to ``violation`` (in-place).

        Returns ``True`` if the exemption matched (whether or not it
        actually changed the violation -- ``protocol_format: true``
        exemptions still let the violation through, just with
        ``severity=info``).  Returns ``False`` if the exemption does
        not match.
        """
        if not self.matches(violation):
            return False
        if self.protocol_format:
            violation.severity = SEVERITY_INFO
        if self.severity is not None:
            violation.severity = self.severity
        return True

    def is_terminal(self) -> bool:
        """``True`` when the exemption fully *removes* the violation.

        A non-terminal exemption only downgrades severity.  Terminal
        exemptions are those that match but do not specify
        ``protocol_format`` and do not specify ``severity``.
        """
        return not (self.protocol_format or self.severity is not None)
