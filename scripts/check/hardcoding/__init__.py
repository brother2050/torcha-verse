"""Hardcoding scanner for the TorchaVerse framework (v0.6.x).

Scans Python source files (excluding the ``config/`` directory) for
patterns that indicate *hard-coded* values which should instead come
from configuration files or the :class:`~core.module_bus.ModuleBus`
registry.  Analysis is performed with the standard library
:mod:`ast` module -- no third-party parsing libraries are required.

This sub-package was extracted from the original
``scripts/check_hardcoding.py`` (1300 lines).  Sub-modules:

* :mod:`._constants` -- module-level configuration constants
  (rule identifiers, severity levels, log/runtime-attr sets).
* :mod:`._types` -- the :class:`Violation` and :class:`Exemption`
  dataclasses.
* :mod:`._ast_helpers` -- AST utilities (parent links, docstring
  collection, path / log / runtime-attr heuristics,
  :func:`is_structural_init`).
* :mod:`._visitor` -- the :class:`HardcodingVisitor` (a thin
  rule dispatcher).
* :mod:`._scan` -- :func:`scan_file` / :func:`scan_directory`.
* :mod:`._whitelist` -- YAML whitelist loader +
  :func:`filter_by_severity`.
* :mod:`._formatters` -- text / JSON formatters and the
  ``--export`` YAML writer.
* :mod:`._cli` -- :func:`main` argparse entry point.

Detected patterns
-----------------
1. ``string_literal``
        String literals longer than 10 characters that appear inside a
        function body.  Docstrings, ``import``-related call arguments,
        ``__all__`` entries and arguments to *logging* calls are
        excluded.

2. ``numeric_literal``
        Numeric literals that appear inside an ``__init__`` constructor.
        The values ``0``, ``1``, ``-1`` (and ``True``/``False``/``None``)
        are excluded as they are commonly used for initialisation.

3. ``path_literal``
        String literals that look like filesystem paths.

4. ``list_literal``
        List literals with more than three elements that appear inside a
        function body.

Severity classification (D1, v0.4.x)
-----------------------------------
Every violation is tagged with a ``severity`` of one of:

* ``critical`` -- runtime config that should come from ConfigCenter /
  defaults.  CI ``--severity critical`` will fail on these.
* ``warn`` -- borderline cases (currently unused; reserved for future
  rules).
* ``info`` -- model structural hyperparams, protocol/format identifiers
  and other legitimate constants.  Reported but not CI-failing.

The mapping is driven by the v0.4.x D1 convention document:
``docs/hardcoding_convention.md``.

Usage
-----
::

    python scripts/check_hardcoding.py --path . --format text
    python scripts/check_hardcoding.py --whitelist config/hardcoded_whitelist.yaml
    python scripts/check_hardcoding.py --severity critical --export config/hardcoding_critical.yaml
    python scripts/check_hardcoding.py --severity info

Exit codes
----------
* ``0`` -- no violations at the requested severity (or above).
* ``1`` -- violations found.
* ``2`` -- usage / configuration error.

The scanner always emits a report (even when violations are present) so
it can be wired into CI without masking the underlying issues.
"""

from __future__ import annotations

# Re-export the public API at the sub-package level so that
# ``from scripts.check.hardcoding import ...`` and the
# thin shim in ``scripts/check_hardcoding.py`` both work.
from ._ast_helpers import (
    is_log_message_format,
    is_runtime_attr,
    is_structural_init,
    looks_like_path,
)
from ._cli import main
from ._constants import (
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_ORDER,
    SEVERITY_WARN,
)
from ._formatters import export_critical, format_json, format_text
from ._scan import scan_directory, scan_file
from ._types import Exemption, Violation
from ._whitelist import filter_by_severity, load_whitelist

# Re-export the rule-name helper so ``--list-rules`` works from the
# shim (and the same name is exposed here for tests).
from scripts.check.hardcoding_rules import (
    DEFAULT_RULES,
    Rule,
    RuleContext,
    ViolationCandidate,
    get_rule,
    list_rule_names,
)

__all__ = [
    # Public data classes
    "Violation",
    "Exemption",
    # Public API
    "scan_file",
    "scan_directory",
    "load_whitelist",
    "filter_by_severity",
    "format_text",
    "format_json",
    "export_critical",
    "main",
    # Helpers re-exported for tests
    "is_log_message_format",
    "is_runtime_attr",
    "is_structural_init",
    "looks_like_path",
    # Severity constants (re-exported for tests)
    "SEVERITY_CRITICAL",
    "SEVERITY_WARN",
    "SEVERITY_INFO",
    "SEVERITY_ORDER",
    "list_rule_names",
    # Rule framework re-exports
    "DEFAULT_RULES",
    "Rule",
    "RuleContext",
    "ViolationCandidate",
    "get_rule",
]
