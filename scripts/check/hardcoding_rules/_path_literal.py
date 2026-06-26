"""Rule #3: path-like string literal (checked everywhere)."""

from __future__ import annotations

from typing import List

from ._helpers import looks_like_path, truncate
from ._protocol import Rule, RuleContext, ViolationCandidate

__all__ = ["PathLiteralRule"]


class PathLiteralRule(Rule):
    """Rule #3: path-like string literal (checked everywhere)."""

    name = "path_literal"
    description = "string literal that looks like a filesystem path"
    default_severity = "critical"

    def check(self, ctx: RuleContext) -> List[ViolationCandidate]:
        if not isinstance(ctx.value, str):
            return []
        if ctx.in_docstring or ctx.in_excluded_str or ctx.in_all:
            return []
        if not looks_like_path(ctx.value):
            return []
        severity = "info" if (ctx.in_log_message_format or ctx.in_log_call or ctx.in_runtime_attr) else "critical"
        return [ViolationCandidate(
            type=self.name, content=truncate(repr(ctx.value)), severity=severity,
        )]
