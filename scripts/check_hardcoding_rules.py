"""Pluggable hardcoding rules (D1 stage three, v0.4.x).

The v0.4.x D1 scanner historically had 4 hardcoded rules baked into
:mod:`scripts.check_hardcoding` (string / numeric / path / list
literals).  D1 stage three splits them into independent, *pluggable*
:class:`Rule` classes so that:

* A new rule (e.g. "f-string" or "regex pattern") can be added
  without editing the visitor.
* An :class:`~scripts.check_hardcoding.Exemption` can opt out of a
  *specific* rule (per-rule opt-out).  This is the bit that lets
  ``is_structural_init`` be replaced by "this numeric literal is
  documented as a model dimension" instead of "any integer in
  [2, 10000] is structural".

A rule is a stateless object: its :meth:`Rule.check` method takes
a :class:`RuleContext` (the file path, the AST node, the parent
stack, exemption status) and returns a list of
:class:`ViolationCandidate` objects.  The visitor walks the AST
once, dispatches to every rule in :data:`DEFAULT_RULES` and
collects the candidates.

The default rules preserve the v0.4.x semantics exactly -- the
46 existing ``test_hardcoding_severity.py`` tests are an
intentional regression baseline.
"""
from __future__ import annotations

import abc
import ast
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Set, Type

__all__ = [
    "RuleContext",
    "ViolationCandidate",
    "Rule",
    "StringLiteralRule",
    "NumericLiteralRule",
    "PathLiteralRule",
    "ListLiteralRule",
    "FStringTemplateRule",
    "RegexPatternRule",
    "DictLiteralRule",
    "DEFAULT_RULES",
    "get_rule",
    "list_rule_names",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RuleContext:
    """Per-node context handed to a :class:`Rule.check` method.

    Attributes:
        relpath: File path (POSIX, relative to the scan root).
        node: The ``ast.Constant`` (or list) node being inspected.
        value: The literal value (``str``/``int``/``float``/``list``).
        in_function: Whether the node is inside a function body.
        in_init: Whether the node is inside ``__init__``.
        in_docstring: Whether the node is a docstring (already
            exempted by the visitor).
        in_all: Whether the node is a string in ``__all__``.
        in_log_call: Whether the node is the format string of a
            logger ``.info()/.warning()/...`` call.
        in_runtime_attr: Whether the node is the argument of an
            ``os.environ[...]`` / ``Path(...)`` / ``sys.argv[...]``
            expression.
    """

    relpath: str
    node: ast.AST
    value: Any
    in_function: bool
    in_init: bool
    in_docstring: bool = False
    in_all: bool = False
    in_log_call: bool = False
    in_runtime_attr: bool = False


@dataclass
class ViolationCandidate:
    """A rule-emitted violation, before the visitor wraps it as a
    :class:`~scripts.check_hardcoding.Violation`.

    Attributes:
        type: Rule name (e.g. ``"string_literal"``) -- becomes the
            :attr:`Violation.type` field.
        content: Short textual representation of the offending value.
        severity: ``critical`` / ``warn`` / ``info``.
    """

    type: str
    content: str
    severity: str = "critical"


# ---------------------------------------------------------------------------
# Helpers shared between rules
# ---------------------------------------------------------------------------
def _truncate(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Rule base class
# ---------------------------------------------------------------------------
class Rule(abc.ABC):
    """Base class for a pluggable hardcoding rule.

    Subclasses must set :attr:`name` and :attr:`description`, and
    implement :meth:`applies_to` and :meth:`check`.

    The default :meth:`applies_to` accepts any ``ast.Constant`` so
    a rule that does *not* depend on the AST node type can simply
    override :meth:`check` only.
    """

    #: Short, stable identifier -- the ``type`` field on emitted
    #: violations and the value of the YAML ``type:`` key in the
    #: whitelist.
    name: str = ""

    #: Human-readable description for ``--list-rules``.
    description: str = ""

    #: Default severity when the rule fires.  Subclasses can return
    #: a different severity dynamically via :meth:`check`.
    default_severity: str = "critical"

    @abc.abstractmethod
    def check(self, ctx: RuleContext) -> List[ViolationCandidate]:
        """Inspect ``ctx`` and return zero or more candidates."""
        raise NotImplementedError

    def applies_to(self, node: ast.AST) -> bool:
        """Return ``True`` if this rule inspects ``node``.

        Default: any ``ast.Constant``.  Override for rules that
        look at composite nodes (e.g. list literals).
        """
        return isinstance(node, ast.Constant)


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------
#: Minimum string length that triggers :class:`StringLiteralRule`.
STRING_MIN_LENGTH: int = 10

#: Maximum number of list elements allowed before :class:`ListLiteralRule`
#: triggers.
LIST_MAX_ELEMENTS: int = 3

#: Numeric range for the ``is_structural_init`` heuristic.  Values
#: outside this range are *not* considered structural.
_STRUCTURAL_MIN: int = 2
_STRUCTURAL_MAX: int = 10000

#: Top-level package prefixes the structural-init heuristic considers
#: to be "model definitions".
_STRUCTURAL_PACKAGES: tuple[str, ...] = ("models/",)

#: Numeric literals exempt from :class:`NumericLiteralRule`.
_EXEMPT_NUMBERS: frozenset[Any] = frozenset({0, 1, -1, 0.0, 1.0, -1.0})


def _looks_like_path(value: str) -> bool:
    """Return ``True`` if ``value`` resembles a filesystem path.

    Mirrors the v0.4.x behaviour exactly (single-character
    separators are excluded; only strong path indicators count).
    """
    if len(value) < 2:
        return False
    if "/" not in value and "\\" not in value:
        return False
    # The original scanner uses a single regex; we keep that
    # here verbatim so the path detection is identical.
    import re
    pattern = re.compile(
        r"(?:^[/~.])"
        r"|(?:[A-Za-z]:[\\/])"
        r"|(?:[\w.\-]+/[\w.\-]+/[\w.\-]*)"
        r"|(?:[\w.\-]+/[\w.\-]+\.\w{1,8})"
        r"|(?:[\w.\-]+\\[\w.\-]+\.\w{1,8})"
    )
    return bool(pattern.search(value))


class StringLiteralRule(Rule):
    """Rule #1: long string literal inside a function body."""

    name = "string_literal"
    description = "long string literal inside a function body"
    default_severity = "critical"

    def check(self, ctx: RuleContext) -> List[ViolationCandidate]:
        if not isinstance(ctx.value, str):
            return []
        if ctx.in_docstring or ctx.in_all:
            return []
        if not (ctx.in_function and len(ctx.value) > STRING_MIN_LENGTH):
            return []
        severity = "info" if (ctx.in_log_call or ctx.in_runtime_attr) else "critical"
        return [ViolationCandidate(
            type=self.name, content=_truncate(repr(ctx.value)), severity=severity,
        )]


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
        if ctx.value in _EXEMPT_NUMBERS:
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
        if not any(ctx.relpath.startswith(p) for p in _STRUCTURAL_PACKAGES):
            return self.default_severity
        if _STRUCTURAL_MIN <= ctx.value <= _STRUCTURAL_MAX:
            return "info"
        return self.default_severity


class PathLiteralRule(Rule):
    """Rule #3: path-like string literal (checked everywhere)."""

    name = "path_literal"
    description = "string literal that looks like a filesystem path"
    default_severity = "critical"

    def check(self, ctx: RuleContext) -> List[ViolationCandidate]:
        if not isinstance(ctx.value, str):
            return []
        if ctx.in_docstring or ctx.in_all:
            return []
        if not _looks_like_path(ctx.value):
            return []
        severity = "info" if (ctx.in_log_call or ctx.in_runtime_attr) else "critical"
        return [ViolationCandidate(
            type=self.name, content=_truncate(repr(ctx.value)), severity=severity,
        )]


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
        if ctx.in_docstring or ctx.in_all:
            return []
        # ``ctx.value`` is the list of JoinedStr parts.  We accept
        # either an ``ast.JoinedStr`` itself (the visitor passes
        # the whole node) or a list of parts.
        parts: list[Any]
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
        severity = "info" if (ctx.in_log_call or ctx.in_runtime_attr) else self.default_severity
        return [ViolationCandidate(
            type=self.name,
            content=_truncate("f'" + template + "'"),
            severity=severity,
        )]


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
    _RE_METHODS: frozenset[str] = frozenset({
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
            content=_truncate("re.{}({!r})".format(func.attr, pattern_node.value)),
            severity=self.default_severity,
        )]


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
        if ctx.in_docstring or ctx.in_all:
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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
DEFAULT_RULES: tuple[Rule, ...] = (
    StringLiteralRule(),
    NumericLiteralRule(),
    PathLiteralRule(),
    ListLiteralRule(),
    FStringTemplateRule(),
    RegexPatternRule(),
    DictLiteralRule(),
)


def get_rule(name: str) -> Optional[Rule]:
    """Return the default rule with ``name`` or ``None``."""
    for rule in DEFAULT_RULES:
        if rule.name == name:
            return rule
    return None


def list_rule_names() -> List[str]:
    """Return the names of all default rules in dispatch order."""
    return [rule.name for rule in DEFAULT_RULES]
