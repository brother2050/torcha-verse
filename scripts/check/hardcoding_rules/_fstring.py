"""Rule #5: f-string template literal (informational by default)."""

from __future__ import annotations

import ast
from typing import Any, List

from ._helpers import truncate
from ._protocol import Rule, RuleContext, ViolationCandidate

__all__ = ["FStringTemplateRule"]


class FStringTemplateRule(Rule):
    """Rule #5: f-string template literal (informational by default).

    Fires on an :class:`ast.JoinedStr` whose template parts contain
    at least one :class:`ast.Constant` that is non-empty AND whose
    raw (concatenated) template is longer than
    :data:`FSTRING_MIN_LENGTH`.  Docstrings and ``__all__`` are
    exempt.

    The default severity is ``info`` because f-string templates
    in TorchaVerse are almost always protocol/format identifiers
    (e.g. ``logger.info("loading {path}")``).  Sites that need
    runtime-configurable templates should move the format string
    to ConfigCenter and downgrade / exempt the rule.
    """

    name = "fstring_template"
    description = "f-string template literal (informational)"
    default_severity = "info"

    #: F-strings shorter than this are noise (they're often inline
    #: debug / repr); we ignore them.
    FSTRING_MIN_LENGTH: int = 20

    def applies_to(self, node: ast.AST) -> bool:
        return isinstance(node, ast.JoinedStr)

    def check(self, ctx: RuleContext) -> List[ViolationCandidate]:
        if ctx.in_docstring or ctx.in_excluded_str or ctx.in_all:
            return []
        # ``ctx.value`` is the list of JoinedStr parts.  We accept
        # either an ``ast.JoinedStr`` itself (the visitor passes
        # the whole node) or a list of parts.
        parts: list
        if isinstance(ctx.node, ast.JoinedStr):
            parts = list(ctx.node.values)
        else:
            parts = ctx.value
        template = "".join(
            v.value for v in parts if isinstance(v, ast.Constant) and isinstance(v.value, str)
        )
        if not template:
            return []
        if len(template) < self.FSTRING_MIN_LENGTH:
            return []
        severity = "info" if (ctx.in_log_message_format or ctx.in_log_call or ctx.in_runtime_attr) else self.default_severity
        return [ViolationCandidate(
            type=self.name,
            content=truncate("f'" + template + "'"),
            severity=severity,
        )]
