"""Pluggable hardcoding rules (D1 stage three, v0.4.x) -- shared
protocol (v0.6.x).

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

This module hosts the base :class:`Rule` ABC and the
:class:`RuleContext` / :class:`ViolationCandidate` data classes.
The concrete rules live in their own sub-modules so this file
stays well under the soft 500-line cap.

``RuleContext`` exposes *two* aliases for the ``__all__`` and the
log-call flag (``in_all`` / ``in_excluded_str`` and
``in_log_call`` / ``in_log_message_format``).  The v0.4.x tests
spell the flags as ``in_all`` / ``in_log_call``; the v0.6.x
visitor uses the more verbose names.  Both spellings are
supported via the ``__init__`` keyword arguments and property
aliases so that test fixtures and runtime code can use either.
"""

from __future__ import annotations

import abc
import ast
from dataclasses import dataclass, field
from typing import Any, List, Optional

__all__ = ["RuleContext", "ViolationCandidate", "Rule"]


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
        in_excluded_str: Verbose alias for :attr:`in_all`.
        in_all: Whether the node is a string in ``__all__`` /
            import call / etc.  ``__all__``-style flag.
        in_log_message_format: Verbose alias for :attr:`in_log_call`.
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
    in_excluded_str: bool = False
    in_all: bool = False
    in_log_message_format: bool = False
    in_log_call: bool = False
    in_runtime_attr: bool = False

    def __post_init__(self) -> None:
        # Backwards-compat: ``in_all`` is the v0.4.x spelling;
        # ``in_excluded_str`` is the v0.6.x visitor's spelling.
        # If only one is set, mirror it to the other.
        if self.in_excluded_str and not self.in_all:
            self.in_all = self.in_excluded_str
        elif self.in_all and not self.in_excluded_str:
            self.in_excluded_str = self.in_all
        if self.in_log_message_format and not self.in_log_call:
            self.in_log_call = self.in_log_message_format
        elif self.in_log_call and not self.in_log_message_format:
            self.in_log_message_format = self.in_log_call


@dataclass
class ViolationCandidate:
    """A rule-emitted violation, before the visitor wraps it as a
    :class:`~scripts.check.hardcoding.Violation`.

    Attributes:
        type: Rule name (e.g. ``"string_literal"``) -- becomes the
            :attr:`Violation.type` field.
        content: Short textual representation of the offending value.
        severity: ``critical`` / ``warn`` / ``info``.
    """

    type: str
    content: str
    severity: str = "critical"


class Rule(abc.ABC):
    """Base class for a pluggable hardcoding rule.

    Subclasses must set :attr:`name` and :attr:`description`, and
    implement :meth:`check`.  The default :meth:`applies_to`
    accepts any ``ast.Constant`` so a rule that does *not* depend
    on the AST node type can simply override :meth:`check` only.
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
