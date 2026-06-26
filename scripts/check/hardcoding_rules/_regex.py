"""Rule #6: regex pattern string passed to ``re.*`` (informational)."""

from __future__ import annotations

import ast
from typing import FrozenSet, List, Optional

from ._helpers import truncate
from ._protocol import Rule, RuleContext, ViolationCandidate

__all__ = ["RegexPatternRule"]


class RegexPatternRule(Rule):
    """Rule #6: regex pattern string passed to ``re.*`` (informational).

    Fires on the first positional argument of a call whose function
    is the ``re`` module (``re.compile`` / ``re.match`` / ``re.search`` /
    ``re.sub`` / ``re.findall`` / ``re.split`` / ``re.fullmatch``).

    The default severity is ``info`` because regex patterns in
    TorchaVerse are almost always protocol/format identifiers
    (e.g. ``_RE_DETERMINISTIC_FLAG = re.compile(r'^-{0,2}d+')``).
    """

    name = "regex_pattern"
    description = "regex pattern string passed to re.* (informational)"
    default_severity = "info"

    #: The ``re`` module attributes we recognise.
    _RE_METHODS: FrozenSet[str] = frozenset({
        "compile", "match", "search", "sub", "findall",
        "split", "fullmatch", "subn",
    })

    def applies_to(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Call)

    def check(self, ctx: RuleContext) -> List[ViolationCandidate]:
        call: ast.Call = ctx.node  # type: ignore[assignment]
        func = call.func
        if not isinstance(func, ast.Attribute):
            return []
        if func.attr not in self._RE_METHODS:
            return []
        if not isinstance(func.value, ast.Name) or func.value.id != "re":
            return []
        # The first positional argument (or ``pattern=`` kwarg) is
        # the regex pattern.
        pattern_node: Optional[ast.AST] = None
        if call.args:
            pattern_node = call.args[0]
        else:
            for kw in call.keywords:
                if kw.arg == "pattern":
                    pattern_node = kw.value
                    break
        if pattern_node is None or not isinstance(pattern_node, ast.Constant):
            return []
        if not isinstance(pattern_node.value, str):
            return []
        if not pattern_node.value:
            return []
        return [ViolationCandidate(
            type=self.name,
            content=truncate("re.{}({!r})".format(func.attr, pattern_node.value)),
            severity=self.default_severity,
        )]
