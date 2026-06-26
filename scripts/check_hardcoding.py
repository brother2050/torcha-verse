"""Backwards-compatible shim for the v0.6.x hardcoding scanner.

This module is preserved so that existing tooling (CI scripts, tests
that import ``from scripts.check_hardcoding import ...``) continues
to work.  The actual implementation lives in
:mod:`scripts.check.hardcoding`.
"""
from __future__ import annotations

import os
import sys

# Make the project root importable when the script is run as a
# subprocess from any cwd (the v0.4.x convention is that
# ``python scripts/check_hardcoding.py --path <x>`` works without
# setting ``PYTHONPATH``).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.check.hardcoding import *  # noqa: F401,F403
from scripts.check.hardcoding import (  # noqa: F401
    Exemption,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_ORDER,
    SEVERITY_WARN,
    Violation,
    export_critical,
    filter_by_severity,
    format_json,
    format_text,
    is_log_message_format,
    is_runtime_attr,
    is_structural_init,
    list_rule_names,
    load_whitelist,
    main,
    scan_directory,
    scan_file,
)

# Re-export the rule framework so the v0.4.x ``from
# scripts.check_hardcoding import Rule, ...`` import path keeps
# working for tests that previously pulled rule-related symbols
# from this module.
try:
    from scripts.check.hardcoding_rules import (  # noqa: F401
        DEFAULT_RULES,
        Rule,
        RuleContext,
        ViolationCandidate,
        get_rule,
    )
except ImportError:  # pragma: no cover
    pass  # placeholder #91 (scripts/check_hardcoding.py:57) -- shim "import anything from sub-package failed"

__all__ = [  # noqa: F405
    "Violation",
    "Exemption",
    "scan_file",
    "scan_directory",
    "load_whitelist",
    "filter_by_severity",
    "format_text",
    "format_json",
    "export_critical",
    "is_log_message_format",
    "is_runtime_attr",
    "is_structural_init",
    "list_rule_names",
    "main",
    "SEVERITY_CRITICAL",
    "SEVERITY_WARN",
    "SEVERITY_INFO",
    "SEVERITY_ORDER",
    "DEFAULT_RULES",
    "Rule",
    "RuleContext",
    "ViolationCandidate",
    "get_rule",
]


if __name__ == "__main__":
    sys.exit(main())
