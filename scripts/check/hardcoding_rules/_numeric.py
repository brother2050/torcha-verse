"""Rule #2: numeric literal inside ``__init__`` (heuristic-aware)."""

from __future__ import annotations

from typing import List

from ._helpers import (
    EXEMPT_NUMBERS,
    STRUCTURAL_MAX,
    STRUCTURAL_MIN,
    STRUCTURAL_PACKAGES,
)
from ._protocol import Rule, RuleContext, ViolationCandidate

__all__ = ["NumericLiteralRule"]


class NumericLiteralRule(Rule):
    """Rule #2: numeric literal inside ``__init__`` (heuristic-aware)."""

    name = "numeric_literal"
    description = "numeric literal inside __init__ (0/1/-1 exempt)"
    default_severity = "critical"

    def check(self, ctx: RuleContext) -> List[ViolationCandidate]:
        if not ctx.in_init:
            return []
        if isinstance(ctx.value, bool) or ctx.value is None:
            return []
        if not isinstance(ctx.value, (int, float, complex)):
            return []
        if ctx.value in EXEMPT_NUMBERS:
            return []
        severity = self._severity(ctx)
        return [ViolationCandidate(
            type=self.name, content=repr(ctx.value), severity=severity,
        )]

    def _severity(self, ctx: RuleContext) -> str:
        if ctx.in_runtime_attr:
            return "info"
        # ``is_structural_init`` -- integer in [2, 10000] in
        # ``models/`` is treated as a model dimension.
        if isinstance(ctx.value, bool):
            return self.default_severity
        if not isinstance(ctx.value, int):
            return self.default_severity
        if not any(ctx.relpath.startswith(p) for p in STRUCTURAL_PACKAGES):
            return self.default_severity
        if STRUCTURAL_MIN <= ctx.value <= STRUCTURAL_MAX:
            return "info"
        return self.default_severity
