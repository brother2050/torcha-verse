"""Pluggable hardcoding rules (D1 stage three, v0.4.x / v0.6.x).

The v0.4.x D1 scanner historically had 4 hardcoded rules baked into
:mod:`scripts.check_hardcoding` (string / numeric / path / list
literals).  D1 stage three splits them into independent, *pluggable*
:class:`Rule` classes so that:

* A new rule (e.g. "f-string" or "regex pattern") can be added
  without editing the visitor.
* An :class:`~scripts.check.hardcoding.Exemption` can opt out of a
  *specific* rule (per-rule opt-out).  This is the bit that lets
  ``is_structural_init`` be replaced by "this numeric literal is
  documented as a model dimension" instead of "any integer in
  [2, 10000] is structural".

Sub-modules
-----------

* :mod:`._protocol` -- the :class:`Rule` ABC plus the
  :class:`RuleContext` / :class:`ViolationCandidate` data
  classes.
* :mod:`._helpers` -- :func:`truncate` / :func:`looks_like_path`
  and the v0.4.x constants that drive the built-in rules.
* :mod:`._string` -- :class:`StringLiteralRule` (rule #1).
* :mod:`._numeric` -- :class:`NumericLiteralRule` (rule #2).
* :mod:`._path_literal` -- :class:`PathLiteralRule` (rule #3).
* :mod:`._list` -- :class:`ListLiteralRule` (rule #4).
* :mod:`._fstring` -- :class:`FStringTemplateRule` (rule #5).
* :mod:`._regex` -- :class:`RegexPatternRule` (rule #6).
* :mod:`._dict` -- :class:`DictLiteralRule` (rule #7).
* :mod:`._switch` -- :class:`HardcodedSwitchRule` (rule #8).
* :mod:`._apikey` -- :class:`ApiKeyPatternRule` (rule #9).

Public surface (preserved from v0.4.x):

* :class:`Rule` -- the base class for new rules.
* :class:`RuleContext` / :class:`ViolationCandidate` -- the
  data classes passed into :meth:`Rule.check`.
* :data:`DEFAULT_RULES` -- the tuple of built-in rule instances
  in dispatch order.
* :func:`get_rule` / :func:`list_rule_names` -- registry
  helpers used by ``--only-rule`` and ``--list-rules``.
"""

from __future__ import annotations

# Re-export the public API at the sub-package level so that
# ``from scripts.check.hardcoding_rules import ...`` (the
# v0.4.x import path) and ``from scripts.check.hardcoding_rules.X
# import Rule`` both work.
from ._apikey import ApiKeyPatternRule
from ._dict import DictLiteralRule
from ._fstring import FStringTemplateRule
from ._helpers import (
    EXEMPT_NUMBERS,
    LIST_MAX_ELEMENTS,
    STRING_MIN_LENGTH,
    STRUCTURAL_MAX,
    STRUCTURAL_MIN,
    STRUCTURAL_PACKAGES,
    _looks_like_path,
    _truncate,
    looks_like_path,
    truncate,
)
from ._list import ListLiteralRule
from ._numeric import NumericLiteralRule
from ._path_literal import PathLiteralRule
from ._protocol import Rule, RuleContext, ViolationCandidate
from ._regex import RegexPatternRule
from ._string import StringLiteralRule
from ._switch import HardcodedSwitchRule

__all__ = [
    # Base classes
    "Rule",
    "RuleContext",
    "ViolationCandidate",
    # Built-in rules (rules #1 .. #9)
    "StringLiteralRule",
    "NumericLiteralRule",
    "PathLiteralRule",
    "ListLiteralRule",
    "FStringTemplateRule",
    "RegexPatternRule",
    "DictLiteralRule",
    "HardcodedSwitchRule",
    "ApiKeyPatternRule",
    # Helpers
    "looks_like_path",
    "_looks_like_path",
    "truncate",
    "_truncate",
    "STRING_MIN_LENGTH",
    "LIST_MAX_ELEMENTS",
    "EXEMPT_NUMBERS",
    "STRUCTURAL_MIN",
    "STRUCTURAL_MAX",
    "STRUCTURAL_PACKAGES",
]


# ---------------------------------------------------------------------------
# Default rule registry (built lazily so import order is irrelevant).
# ---------------------------------------------------------------------------
DEFAULT_RULES: tuple = (
    StringLiteralRule(),
    NumericLiteralRule(),
    PathLiteralRule(),
    ListLiteralRule(),
    FStringTemplateRule(),
    RegexPatternRule(),
    DictLiteralRule(),
    HardcodedSwitchRule(),
    ApiKeyPatternRule(),
)


def get_rule(name: str):
    """Return the default rule with ``name`` or ``None``."""
    for rule in DEFAULT_RULES:
        if rule.name == name:
            return rule
    return None


def list_rule_names():
    """Return the names of all default rules in dispatch order."""
    return [rule.name for rule in DEFAULT_RULES]
