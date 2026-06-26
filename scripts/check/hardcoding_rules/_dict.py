"""Rule #7: large dict literal inside a function (informational)."""

from __future__ import annotations

import ast
from typing import List

from ._protocol import Rule, RuleContext, ViolationCandidate

__all__ = ["DictLiteralRule"]


class DictLiteralRule(Rule):
    """Rule #7: large dict literal inside a function (informational).

    Fires on an :class:`ast.Dict` whose number of keys is at least
    :data:`DICT_MIN_KEYS` and which lives inside a function body.
    Docstrings are exempt.

    The default severity is ``info`` because the project's
    sub-systems (``ConfigCenter`` defaults, ``ModuleBus`` mappings,
    per-node ``schema``) often contain static dict literals that
    are de-facto protocol definitions.
    """

    name = "dict_literal"
    description = "dict literal with many keys inside a function (informational)"
    default_severity = "info"

    #: A dict with fewer than this many keys is treated as a
    #: "regular kwargs" / "small mapping" and ignored.
    DICT_MIN_KEYS: int = 5

    def applies_to(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Dict)

    def check(self, ctx: RuleContext) -> List[ViolationCandidate]:
        if ctx.in_docstring or ctx.in_excluded_str or ctx.in_all:
            return []
        if not ctx.in_function:
            return []
        d: ast.Dict = ctx.node  # type: ignore[assignment]
        n = len(d.keys)
        if n < self.DICT_MIN_KEYS:
            return []
        return [ViolationCandidate(
            type=self.name,
            content="{{{} keys}}".format(n),
            severity=self.default_severity,
        )]
