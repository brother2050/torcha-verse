"""Rule #4: list literal with more than 3 elements inside a function."""

from __future__ import annotations

import ast
from typing import List

from ._helpers import LIST_MAX_ELEMENTS
from ._protocol import Rule, RuleContext, ViolationCandidate

__all__ = ["ListLiteralRule"]


class ListLiteralRule(Rule):
    """Rule #4: list literal with more than 3 elements inside a function."""

    name = "list_literal"
    description = "list literal with more than 3 elements inside a function"
    default_severity = "critical"

    def applies_to(self, node: ast.AST) -> bool:
        return isinstance(node, ast.List)

    def check(self, ctx: RuleContext) -> List[ViolationCandidate]:
        # The visitor dispatches to us with the list value (a
        # list, not a scalar).  We mirror the v0.4.x semantics:
        # * 4+ elements,
        # * inside a function body,
        # * default ``critical`` severity.
        if not ctx.in_function:
            return []
        try:
            n = len(ctx.value)
        except TypeError:
            return []
        if n <= LIST_MAX_ELEMENTS:
            return []
        return [ViolationCandidate(
            type=self.name,
            content="[{} elements]".format(n),
            severity=self.default_severity,
        )]
