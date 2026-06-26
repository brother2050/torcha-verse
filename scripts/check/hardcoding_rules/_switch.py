"""Rule #8: a boolean literal used as a behavioural switch (warn)."""

from __future__ import annotations

import ast
from typing import List

from ._protocol import Rule, RuleContext, ViolationCandidate

__all__ = ["HardcodedSwitchRule"]


class HardcodedSwitchRule(Rule):
    """Rule #8: a boolean literal used as a behavioural switch.

    Fires on a bare ``True`` / ``False`` literal that sits *inside a
    function body* (not in ``__init__``) and is the *value* of an
    assignment whose target is not a private name (i.e. not
    ``_foo``).  Docstrings and ``__all__`` are exempt.

    The default severity is ``"warn"`` -- boolean switches in
    function bodies are almost always operator-controlled
    ("show this progress bar", "fail fast on error", "use the
    fast path").  Hardcoding them removes the operator's ability
    to toggle behaviour at runtime, so the v0.5.x D1 stage flags
    the site and suggests moving the flag to ConfigCenter.
    """

    name = "hardcoded_switch"
    description = "boolean literal used as a behavioural switch (warn)"
    default_severity = "warn"

    def applies_to(self, node: ast.AST) -> bool:
        # We only inspect ast.Constant nodes; the visitor does
        # the parent-stack dance to figure out whether we are in
        # an assignment context.
        return isinstance(node, ast.Constant) and isinstance(node.value, bool)

    def check(self, ctx: RuleContext) -> List[ViolationCandidate]:
        if ctx.in_docstring or ctx.in_excluded_str or ctx.in_all:
            return []
        # Bool literals in __init__ are *expected* (they're
        # default values for instance attributes).  We only fire
        # for body-level switches.
        if ctx.in_init:
            return []
        # The visitor tags every Constant with ``in_function``,
        # so a body-level bool is always in_function.  Use the
        # explicit flag to be defensive.
        if not ctx.in_function:
            return []
        # Bool in the log-call / runtime-attr positions are
        # already whitelisted by their own exemptions -- demote
        # to info to avoid noise.
        severity = "info" if (ctx.in_log_message_format or ctx.in_log_call or ctx.in_runtime_attr) else self.default_severity
        return [ViolationCandidate(
            type=self.name,
            content=repr(ctx.value),
            severity=severity,
        )]
