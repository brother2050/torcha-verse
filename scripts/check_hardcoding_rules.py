"""Backwards-compatible shim for the v0.6.x pluggable hardcoding rules.

The actual implementation lives in
:mod:`scripts.check.hardcoding_rules`.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.check.hardcoding_rules import *  # noqa: F401,F403
from scripts.check.hardcoding_rules import (  # noqa: F401
    DEFAULT_RULES,
    EXEMPT_NUMBERS,
    LIST_MAX_ELEMENTS,
    STRING_MIN_LENGTH,
    STRUCTURAL_MAX,
    STRUCTURAL_MIN,
    STRUCTURAL_PACKAGES,
    ApiKeyPatternRule,
    DictLiteralRule,
    FStringTemplateRule,
    HardcodedSwitchRule,
    ListLiteralRule,
    NumericLiteralRule,
    PathLiteralRule,
    RegexPatternRule,
    Rule,
    RuleContext,
    StringLiteralRule,
    ViolationCandidate,
    _looks_like_path,
    _truncate,
    get_rule,
    list_rule_names,
    looks_like_path,
    truncate,
)

__all__ = [  # noqa: F405
    "Rule",
    "RuleContext",
    "ViolationCandidate",
    "StringLiteralRule",
    "NumericLiteralRule",
    "PathLiteralRule",
    "ListLiteralRule",
    "FStringTemplateRule",
    "RegexPatternRule",
    "DictLiteralRule",
    "HardcodedSwitchRule",
    "ApiKeyPatternRule",
    "DEFAULT_RULES",
    "get_rule",
    "list_rule_names",
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

if __name__ == "__main__":
    sys.stderr.write(
        "check_hardcoding_rules.py is a library module, not a CLI; "
        "use check_hardcoding.py for the scanner.\n"
    )
    sys.exit(0)
