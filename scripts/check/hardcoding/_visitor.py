"""AST visitor for the hardcoding scanner (v0.6.x).

The :class:`HardcodingVisitor` is a thin dispatcher: it builds a
:class:`~scripts.check.hardcoding_rules.RuleContext` for each AST
node and asks every rule in :data:`~scripts.check.hardcoding_rules.DEFAULT_RULES`
to evaluate it.  The rule-based design (D1 stage three) lets
callers add new rules, opt out per-rule, or run a single rule via
the ``--only-rule`` CLI flag -- without changing the visitor
itself.

Placeholder notes
-----------------
* The ``if node.value is None or isinstance(node.value, bool): pass``
  on line 41 of this module is the v0.4.x line 569 placeholder
  (\"#47 -- booleans / None are never numeric violations\").  We
  keep the explicit ``pass`` so the line number stays stable for
  tooling that grep-pins it.
"""

from __future__ import annotations

import ast
from typing import Any, List, Optional, Set, Tuple

from scripts.check.hardcoding_rules import (
    DEFAULT_RULES,
    Rule,
    RuleContext,
    ViolationCandidate,
    get_rule,
    list_rule_names,
)

from ._ast_helpers import (
    collect_docstring_ids,
    is_log_message_format,
    is_runtime_attr,
    is_structural_init,
)
from ._constants import CONTENT_LIMIT, SEVERITY_CRITICAL, SEVERITY_INFO
from ._types import Violation

__all__ = ["HardcodingVisitor", "build_context"]


class HardcodingVisitor(ast.NodeVisitor):
    """Walks an AST tree collecting hardcoding violations.

    See module docstring for the dispatch contract.
    """

    def __init__(
        self,
        relpath: str,
        docstring_ids: Set[int],
        excluded_str_ids: Set[int],
        rules: Optional[List[Rule]] = None,
    ) -> None:
        self.relpath: str = relpath
        self._docstring_ids: Set[int] = docstring_ids
        self._excluded_str_ids: Set[int] = excluded_str_ids
        self._rules: List[Rule] = list(rules) if rules is not None else list(DEFAULT_RULES)
        self.violations: List[Violation] = []
        # Stack of (name, kind) for each function-like scope entered.
        self._func_stack: List[Tuple[str, str]] = []

    @property
    def in_function(self) -> bool:
        """``True`` when inside any function/lambda body."""
        return bool(self._func_stack)

    @property
    def in_init(self) -> bool:
        """``True`` when inside an ``__init__`` method (any depth)."""
        return any(
            name == "__init__"
            for name, kind in self._func_stack
            if kind == "func"
        )

    # -- scope management ------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_stack.append((node.name, "func"))
        self.generic_visit(node)
        self._func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._func_stack.append(("<lambda>", "lambda"))
        self.generic_visit(node)
        self._func_stack.pop()

    # -- literal dispatch ------------------------------------------------
    def build_context(self, node: ast.AST, value: Any) -> RuleContext:
        """Build a :class:`RuleContext` for ``node``.

        Sub-helpers require ``.parent`` to be set on the AST node;
        the scan pipeline attaches that in
        :func:`scripts.check.hardcoding._ast_helpers.attach_parents`.
        """
        in_docstring = id(node) in self._docstring_ids
        in_all = id(node) in self._excluded_str_ids and not in_docstring
        in_log = bool(is_log_message_format(node)) if isinstance(node, ast.Constant) else False
        in_runtime = bool(is_runtime_attr(node)) if isinstance(node, ast.Constant) else False
        return RuleContext(
            relpath=self.relpath,
            node=node,
            value=value,
            in_function=self.in_function,
            in_init=self.in_init,
            in_docstring=in_docstring,
            in_excluded_str=in_all,
            in_log_message_format=in_log,
            in_runtime_attr=in_runtime,
        )

    def _record(self, ctx: RuleContext, candidate: ViolationCandidate) -> None:
        """Convert a :class:`ViolationCandidate` into a :class:`Violation`.

        Applies the structural-init heuristic (downgrade to ``info``
        when applicable) and appends the result to
        :attr:`self.violations`.
        """
        if candidate.severity == SEVERITY_CRITICAL and is_structural_init(
            self.relpath, ctx.value
        ):
            severity = SEVERITY_INFO
        else:
            severity = candidate.severity
        self.violations.append(
            Violation(
                file=self.relpath,
                line=ctx.node.lineno,
                col=getattr(ctx.node, "col_offset", 0),
                type=candidate.type,
                content=candidate.content[:CONTENT_LIMIT],
                severity=severity,
            )
        )

    def visit_Constant(self, node: ast.Constant) -> None:
        """Apply every rule to a single ``Constant`` AST node."""
        if node.value is None or isinstance(node.value, bool):
            # booleans / None are never numeric violations
            pass  # placeholder #47 (was v0.4.x line 569)
        ctx = self.build_context(node, node.value)
        for rule in self._rules:
            if not rule.applies_to(node):
                continue
            for candidate in rule.check(ctx):
                self._record(ctx, candidate)

    def visit_List(self, node: ast.List) -> None:
        """Apply every rule to a single ``List`` AST node."""
        ctx = self.build_context(node, node.elts)
        for rule in self._rules:
            if not rule.applies_to(node):
                continue
            for candidate in rule.check(ctx):
                self._record(ctx, candidate)
        # Don't recurse into individual list elements (they were
        # already inspected as ``Constant`` nodes when the
        # visitor walks the tree normally).  We *do* still want
        # the visitor to walk into anything *inside* the list
        # (e.g. a function call inside a list element), so we
        # fall through to ``generic_visit``.
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        """Apply every rule to a single ``Dict`` AST node."""
        ctx = self.build_context(node, list(node.keys))
        for rule in self._rules:
            if not rule.applies_to(node):
                continue
            for candidate in rule.check(ctx):
                self._record(ctx, candidate)
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        """Apply every rule to a single ``JoinedStr`` (f-string) AST node."""
        ctx = self.build_context(node, list(node.values))
        for rule in self._rules:
            if not rule.applies_to(node):
                continue
            for candidate in rule.check(ctx):
                self._record(ctx, candidate)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Apply every rule to a single ``Call`` AST node.

        The :class:`RegexPatternRule` uses this to detect
        ``re.compile(...)`` / ``re.match(...)`` / etc.
        """
        ctx = self.build_context(node, None)
        for rule in self._rules:
            if not rule.applies_to(node):
                continue
            for candidate in rule.check(ctx):
                self._record(ctx, candidate)
        # Recurse into the Call's children so that
        # ``visit_Constant`` fires on the arguments (the
        # ``StringLiteralRule`` lives there).
        self.generic_visit(node)
