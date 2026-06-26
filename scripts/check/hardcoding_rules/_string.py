"""Rule #1: long string literal inside a function body (v0.6.x)."""

from __future__ import annotations

from typing import List

from ._helpers import STRING_MIN_LENGTH, truncate
from ._protocol import Rule, RuleContext, ViolationCandidate

__all__ = ["StringLiteralRule"]


class StringLiteralRule(Rule):
    """Rule #1: long string literal inside a function body."""

    name = "string_literal"
    description = "long string literal inside a function body"
    default_severity = "critical"

    def check(self, ctx: RuleContext) -> List[ViolationCandidate]:
        if not isinstance(ctx.value, str):
            return []
        if ctx.in_docstring or ctx.in_excluded_str or ctx.in_all:
            return []
        if not (ctx.in_function and len(ctx.value) > STRING_MIN_LENGTH):
            return []
        severity = "info" if (ctx.in_log_message_format or ctx.in_log_call or ctx.in_runtime_attr) else "critical"
        return [ViolationCandidate(
            type=self.name, content=truncate(repr(ctx.value)), severity=severity,
        )]
