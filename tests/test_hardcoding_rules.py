"""Tests for the D1 stage three rules-based scanner.

Covers the v0.4.x D1 stage three work-stream: split the scanner's
4 hardcoded rules into pluggable :class:`~scripts.check_hardcoding_rules.Rule`
classes, add per-rule opt-out via :attr:`Exemption.rules`, add the
``--only-rule`` / ``--list-rules`` / ``--ci`` CLI flags, and wire
``scripts/check_ci_gates.py`` to be the unified CI entry point.

Coverage map:

* :class:`RuleContext` and :class:`ViolationCandidate` data classes
  are importable and have the documented fields.
* :class:`Rule` is abstract; the 4 built-in subclasses implement it.
* :data:`DEFAULT_RULES` contains exactly the 4 named rules in the
  documented order.
* :func:`get_rule` / :func:`list_rule_names` resolve the registry.
* :class:`StringLiteralRule` fires only on long-in-function strings
  and respects ``in_docstring`` / ``in_all`` / ``in_log_call`` /
  ``in_runtime_attr`` exemptions.
* :class:`NumericLiteralRule` fires only in ``__init__`` and exempts
  ``0``/``1``/``-1``/booleans/``None``.  The ``models/`` structural
  range downgrades to ``info``.
* :class:`PathLiteralRule` fires when the string matches the path
  regex; respects the same exemptions as :class:`StringLiteralRule`.
* :class:`ListLiteralRule` fires only inside a function and only
  when there are more than 3 elements; ignores shorter lists.
* :class:`Exemption` ``rules`` field performs per-rule opt-out.
* :class:`Exemption.is_terminal` is ``True`` exactly when no
  ``protocol_format`` and no ``severity`` override is set.
* ``scan_directory(only_rule=...)`` returns only violations emitted
  by that rule.
* The ``--ci`` flag, when wired with the project ``pyproject.toml``,
  returns exit code 0 on a clean tree and exit code 1 on a dirty
  tree.
* The ``--list-rules`` flag returns the 4 expected names.
* :mod:`scripts.check_ci_gates` exposes a registry with at least
  the ``hardcoding`` and ``placeholders`` gates and can be
  imported without errors.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List

import pytest

from scripts.check_hardcoding import (
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    Exemption,
    Violation,
    list_rule_names,
    scan_directory,
    scan_file,
)
from scripts.check_hardcoding_rules import (
    DEFAULT_RULES,
    LIST_MAX_ELEMENTS,
    Rule,
    RuleContext,
    StringLiteralRule,
    NumericLiteralRule,
    PathLiteralRule,
    ListLiteralRule,
    FStringTemplateRule,
    RegexPatternRule,
    DictLiteralRule,
    STRING_MIN_LENGTH,
    ViolationCandidate,
    _looks_like_path,
    get_rule,
)
from scripts.ci_config import (
    DEFAULT_CI_SETTINGS,
    load_hardcoding_ci_settings,
)
import scripts.check_ci_gates as ci_gates


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.hardcoding_rules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ctx(
    *,
    relpath: str = "x.py",
    value=None,
    in_function: bool = False,
    in_init: bool = False,
    in_docstring: bool = False,
    in_all: bool = False,
    in_log_call: bool = False,
    in_runtime_attr: bool = False,
    node=None,
) -> RuleContext:
    """Build a RuleContext with sensible defaults for tests."""
    if node is None:
        node = _fake_constant(value)
    return RuleContext(
        relpath=relpath, node=node, value=value,
        in_function=in_function, in_init=in_init,
        in_docstring=in_docstring, in_all=in_all,
        in_log_call=in_log_call, in_runtime_attr=in_runtime_attr,
    )


def _fake_constant(value):
    """Build a minimal ast.Constant node carrying ``value``."""
    import ast
    return ast.Constant(value=value)


def _violation(**kwargs) -> Violation:
    return Violation(
        file=kwargs.get("file", "x.py"),
        line=kwargs.get("line", 1),
        col=kwargs.get("col", 0),
        type=kwargs.get("type", "string_literal"),
        content=kwargs.get("content", "'foo'"),
        severity=kwargs.get("severity", SEVERITY_CRITICAL),
    )


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Registry / Rule base class
# ---------------------------------------------------------------------------
class TestRuleRegistry:
    """The default rules are registered in the documented order.

    As of the 3-new-Rule extension the registry has 7 rules: the
    4 original ones (string / numeric / path / list) plus the 3
    informational extensions (fstring_template / regex_pattern /
    dict_literal).
    """

    def test_default_rules_count(self):
        assert len(DEFAULT_RULES) == 7

    def test_default_rules_names(self):
        names = [r.name for r in DEFAULT_RULES]
        assert names == [
            "string_literal", "numeric_literal", "path_literal", "list_literal",
            "fstring_template", "regex_pattern", "dict_literal",
        ]

    def test_default_rules_have_descriptions(self):
        for rule in DEFAULT_RULES:
            assert rule.description, f"{rule.name} has no description"

    def test_get_rule_returns_correct_instance(self):
        assert isinstance(get_rule("string_literal"), StringLiteralRule)
        assert isinstance(get_rule("numeric_literal"), NumericLiteralRule)
        assert isinstance(get_rule("path_literal"), PathLiteralRule)
        assert isinstance(get_rule("list_literal"), ListLiteralRule)

    def test_get_rule_unknown_returns_none(self):
        assert get_rule("nonexistent_rule") is None

    def test_list_rule_names_matches_default(self):
        assert list_rule_names() == [r.name for r in DEFAULT_RULES]


class TestRuleBase:
    """The Rule base class enforces the abstract method contract."""

    def test_rule_is_abstract(self):
        with pytest.raises(TypeError):
            # Cannot instantiate a Rule without overriding check().
            Rule()

    def test_rule_default_applies_to_constant(self):
        rule = StringLiteralRule()
        import ast
        assert rule.applies_to(ast.Constant(value="x")) is True
        # List node not handled by the default applies_to().
        assert rule.applies_to(ast.List(elts=[])) is False

    def test_list_rule_overrides_applies_to(self):
        rule = ListLiteralRule()
        import ast
        assert rule.applies_to(ast.List(elts=[])) is True
        assert rule.applies_to(ast.Constant(value=1)) is False


# ---------------------------------------------------------------------------
# StringLiteralRule
# ---------------------------------------------------------------------------
class TestStringLiteralRule:
    """Rule #1: long string literal inside a function body."""

    def test_short_string_does_not_fire(self):
        rule = StringLiteralRule()
        ctx = _ctx(value="short", in_function=True)
        assert rule.check(ctx) == []

    def test_long_string_in_function_fires(self):
        rule = StringLiteralRule()
        ctx = _ctx(value="a" * (STRING_MIN_LENGTH + 1), in_function=True)
        cands = rule.check(ctx)
        assert len(cands) == 1
        assert cands[0].type == "string_literal"
        assert cands[0].severity == "critical"

    def test_long_string_outside_function_does_not_fire(self):
        rule = StringLiteralRule()
        ctx = _ctx(value="a" * (STRING_MIN_LENGTH + 1), in_function=False)
        assert rule.check(ctx) == []

    def test_docstring_is_exempt(self):
        rule = StringLiteralRule()
        ctx = _ctx(value="a" * (STRING_MIN_LENGTH + 1), in_function=True, in_docstring=True)
        assert rule.check(ctx) == []

    def test_all_entry_is_exempt(self):
        rule = StringLiteralRule()
        ctx = _ctx(value="a" * (STRING_MIN_LENGTH + 1), in_function=True, in_all=True)
        assert rule.check(ctx) == []

    def test_log_call_downgrades_to_info(self):
        rule = StringLiteralRule()
        ctx = _ctx(value="a" * (STRING_MIN_LENGTH + 1), in_function=True, in_log_call=True)
        cands = rule.check(ctx)
        assert cands[0].severity == "info"

    def test_runtime_attr_downgrades_to_info(self):
        rule = StringLiteralRule()
        ctx = _ctx(value="a" * (STRING_MIN_LENGTH + 1), in_function=True, in_runtime_attr=True)
        cands = rule.check(ctx)
        assert cands[0].severity == "info"

    def test_non_string_value_does_not_fire(self):
        rule = StringLiteralRule()
        ctx = _ctx(value=42, in_function=True)
        assert rule.check(ctx) == []


# ---------------------------------------------------------------------------
# NumericLiteralRule
# ---------------------------------------------------------------------------
class TestNumericLiteralRule:
    """Rule #2: numeric literal inside __init__ (heuristic-aware)."""

    def test_in_init_critical(self):
        rule = NumericLiteralRule()
        ctx = _ctx(value=500, in_init=True, relpath="core/utils.py")
        cands = rule.check(ctx)
        assert len(cands) == 1
        assert cands[0].severity == "critical"

    def test_outside_init_does_not_fire(self):
        rule = NumericLiteralRule()
        ctx = _ctx(value=500, in_init=False)
        assert rule.check(ctx) == []

    @pytest.mark.parametrize("exempt", [0, 1, -1, 0.0, 1.0, -1.0])
    def test_exempt_numbers_do_not_fire(self, exempt):
        rule = NumericLiteralRule()
        ctx = _ctx(value=exempt, in_init=True)
        assert rule.check(ctx) == []

    def test_booleans_do_not_fire(self):
        rule = NumericLiteralRule()
        for v in (True, False):
            assert rule.check(_ctx(value=v, in_init=True)) == []

    def test_none_does_not_fire(self):
        rule = NumericLiteralRule()
        assert rule.check(_ctx(value=None, in_init=True)) == []

    def test_structural_range_in_models_downgrades_to_info(self):
        rule = NumericLiteralRule()
        ctx = _ctx(value=512, in_init=True, relpath="models/text/transformer.py")
        cands = rule.check(ctx)
        assert cands[0].severity == "info"

    def test_structural_range_outside_models_remains_critical(self):
        rule = NumericLiteralRule()
        ctx = _ctx(value=512, in_init=True, relpath="core/utils.py")
        cands = rule.check(ctx)
        assert cands[0].severity == "critical"

    def test_outside_structural_range_remains_critical_in_models(self):
        rule = NumericLiteralRule()
        ctx = _ctx(value=1_000_000, in_init=True, relpath="models/text/transformer.py")
        cands = rule.check(ctx)
        assert cands[0].severity == "critical"

    def test_runtime_attr_downgrades_to_info(self):
        rule = NumericLiteralRule()
        ctx = _ctx(value=500, in_init=True, relpath="core/utils.py", in_runtime_attr=True)
        cands = rule.check(ctx)
        assert cands[0].severity == "info"


# ---------------------------------------------------------------------------
# PathLiteralRule
# ---------------------------------------------------------------------------
class TestPathLiteralRule:
    """Rule #3: path-like string literal."""

    def test_absolute_unix_path_fires(self):
        rule = PathLiteralRule()
        ctx = _ctx(value="/var/log/app.log", in_function=True)
        cands = rule.check(ctx)
        assert len(cands) == 1
        assert cands[0].type == "path_literal"

    def test_relative_path_fires(self):
        rule = PathLiteralRule()
        ctx = _ctx(value="config/settings.yaml", in_function=True)
        cands = rule.check(ctx)
        assert len(cands) == 1

    def test_windows_drive_letter_fires(self):
        rule = PathLiteralRule()
        ctx = _ctx(value="C:\\Users\\me\\file.txt", in_function=True)
        cands = rule.check(ctx)
        assert len(cands) == 1

    def test_plain_word_does_not_fire(self):
        rule = PathLiteralRule()
        ctx = _ctx(value="hello", in_function=True)
        assert rule.check(ctx) == []

    def test_docstring_is_exempt(self):
        rule = PathLiteralRule()
        ctx = _ctx(value="/var/log/x.log", in_function=True, in_docstring=True)
        assert rule.check(ctx) == []

    def test_log_call_downgrades_to_info(self):
        rule = PathLiteralRule()
        ctx = _ctx(value="/var/log/x.log", in_function=True, in_log_call=True)
        cands = rule.check(ctx)
        assert cands[0].severity == "info"

    def test_non_string_value_does_not_fire(self):
        rule = PathLiteralRule()
        ctx = _ctx(value=42)
        assert rule.check(ctx) == []


# ---------------------------------------------------------------------------
# ListLiteralRule
# ---------------------------------------------------------------------------
class TestListLiteralRule:
    """Rule #4: list literal with >3 elements inside a function body."""

    def test_short_list_does_not_fire(self):
        rule = ListLiteralRule()
        ctx = _ctx(value=[1, 2, 3], in_function=True)
        assert rule.check(ctx) == []

    def test_max_length_list_does_not_fire(self):
        """A list of exactly LIST_MAX_ELEMENTS (3) does not trigger."""
        rule = ListLiteralRule()
        ctx = _ctx(value=[1] * LIST_MAX_ELEMENTS, in_function=True)
        assert rule.check(ctx) == []

    def test_long_list_in_function_fires(self):
        rule = ListLiteralRule()
        ctx = _ctx(value=[1, 2, 3, 4], in_function=True)
        cands = rule.check(ctx)
        assert len(cands) == 1
        assert cands[0].type == "list_literal"

    def test_long_list_outside_function_does_not_fire(self):
        rule = ListLiteralRule()
        ctx = _ctx(value=[1, 2, 3, 4, 5], in_function=False)
        assert rule.check(ctx) == []


# ---------------------------------------------------------------------------
# FStringTemplateRule
# ---------------------------------------------------------------------------
class TestFStringTemplateRule:
    """Rule #5: long f-string template literal (default ``info``)."""

    def test_default_severity_is_info(self):
        rule = FStringTemplateRule()
        assert rule.default_severity == "info"

    def test_long_fstring_fires(self):
        rule = FStringTemplateRule()
        import ast
        node = ast.JoinedStr(values=[
            ast.Constant(value="hello, "),
            ast.FormattedValue(
                value=ast.Name(id="name"),
                conversion=-1, format_spec=None,
            ),
            ast.Constant(value="! Welcome to the TorchaVerse playground."),
        ])
        ctx = _ctx(node=node, value=list(node.values), in_function=True)
        cands = rule.check(ctx)
        assert len(cands) == 1
        assert cands[0].severity == "info"
        assert cands[0].type == "fstring_template"

    def test_short_fstring_does_not_fire(self):
        rule = FStringTemplateRule()
        import ast
        node = ast.JoinedStr(values=[
            ast.Constant(value="hi "),
            ast.FormattedValue(value=ast.Name(id="x"), conversion=-1, format_spec=None),
        ])
        ctx = _ctx(node=node, value=list(node.values))
        assert rule.check(ctx) == []

    def test_all_string_fstring_does_not_fire(self):
        """A f-string with no FormattedValue is a no-op template."""
        rule = FStringTemplateRule()
        import ast
        node = ast.JoinedStr(values=[ast.Constant(value="just a constant")])
        ctx = _ctx(node=node, value=list(node.values))
        assert rule.check(ctx) == []

    def test_docstring_is_exempt(self):
        rule = FStringTemplateRule()
        import ast
        node = ast.JoinedStr(values=[
            ast.Constant(value="prefix "),
            ast.FormattedValue(value=ast.Name(id="x"), conversion=-1, format_spec=None),
            ast.Constant(value=" suffix long enough to trigger"),
        ])
        ctx = _ctx(node=node, value=list(node.values), in_docstring=True)
        assert rule.check(ctx) == []

    def test_log_call_downgrades_to_info(self):
        rule = FStringTemplateRule()
        import ast
        node = ast.JoinedStr(values=[
            ast.Constant(value="a "),
            ast.FormattedValue(value=ast.Name(id="x"), conversion=-1, format_spec=None),
            ast.Constant(value=" b long enough to trigger indeed"),
        ])
        ctx = _ctx(node=node, value=list(node.values), in_log_call=True)
        cands = rule.check(ctx)
        assert cands[0].severity == "info"

    def test_applies_to_only_joinedstr(self):
        rule = FStringTemplateRule()
        import ast
        assert rule.applies_to(ast.JoinedStr(values=[])) is True
        assert rule.applies_to(ast.Constant(value="x")) is False
        assert rule.applies_to(ast.List(elts=[])) is False


# ---------------------------------------------------------------------------
# RegexPatternRule
# ---------------------------------------------------------------------------
class TestRegexPatternRule:
    """Rule #6: regex pattern string in ``re.*`` calls (default ``info``)."""

    def test_default_severity_is_info(self):
        rule = RegexPatternRule()
        assert rule.default_severity == "info"

    def _make_call(self, attr_name: str, pattern: str) -> "ast.Call":
        import ast
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="re"),
                attr=attr_name,
            ),
            args=[ast.Constant(value=pattern)],
            keywords=[],
        )

    def test_re_compile_fires(self):
        rule = RegexPatternRule()
        import ast
        node = self._make_call("compile", r"^foo\d+$")
        ctx = _ctx(node=node, value=None, in_function=True)
        cands = rule.check(ctx)
        assert len(cands) == 1
        assert cands[0].severity == "info"
        assert cands[0].type == "regex_pattern"
        assert "compile" in cands[0].content

    def test_re_search_fires(self):
        rule = RegexPatternRule()
        import ast
        node = self._make_call("search", r"\bword\b")
        ctx = _ctx(node=node, value=None)
        cands = rule.check(ctx)
        assert cands[0].type == "regex_pattern"

    def test_re_sub_fires(self):
        rule = RegexPatternRule()
        import ast
        # re.sub(pattern, repl, string) -- rule inspects the
        # first positional argument (pattern) only.
        node = ast.Call(
            func=ast.Attribute(value=ast.Name(id="re"), attr="sub"),
            args=[ast.Constant(value=r"x+"), ast.Constant(value="y"),
                  ast.Constant(value="abc")],
            keywords=[],
        )
        ctx = _ctx(node=node, value=None)
        cands = rule.check(ctx)
        assert cands[0].type == "regex_pattern"
        assert cands[0].content.startswith("re.sub(")
        assert r"x+" in cands[0].content

    def test_non_re_call_does_not_fire(self):
        """A call to ``os.path.join`` must NOT fire this rule."""
        rule = RegexPatternRule()
        import ast
        node = ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id="os"),
                    attr="path",
                ),
                attr="join",
            ),
            args=[ast.Constant(value="a"), ast.Constant(value="b")],
            keywords=[],
        )
        ctx = _ctx(node=node, value=None)
        assert rule.check(ctx) == []

    def test_non_string_first_arg_does_not_fire(self):
        rule = RegexPatternRule()
        import ast
        node = ast.Call(
            func=ast.Attribute(value=ast.Name(id="re"), attr="match"),
            args=[ast.Constant(value=42)],  # not a string!
            keywords=[],
        )
        ctx = _ctx(node=node, value=None)
        assert rule.check(ctx) == []

    def test_empty_pattern_does_not_fire(self):
        rule = RegexPatternRule()
        import ast
        node = self._make_call("match", "")
        ctx = _ctx(node=node, value=None)
        assert rule.check(ctx) == []

    def test_keyword_pattern_arg_fires(self):
        """``re.match(pattern=r'x+', string='foo')`` must fire."""
        rule = RegexPatternRule()
        import ast
        node = ast.Call(
            func=ast.Attribute(value=ast.Name(id="re"), attr="match"),
            args=[],
            keywords=[
                ast.keyword(
                    arg="pattern",
                    value=ast.Constant(value=r"x+"),
                ),
            ],
        )
        ctx = _ctx(node=node, value=None)
        cands = rule.check(ctx)
        assert cands[0].type == "regex_pattern"

    def test_applies_to_only_call(self):
        rule = RegexPatternRule()
        import ast
        assert rule.applies_to(ast.Call(func=None, args=[], keywords=[])) is True
        assert rule.applies_to(ast.Constant(value="x")) is False


# ---------------------------------------------------------------------------
# DictLiteralRule
# ---------------------------------------------------------------------------
class TestDictLiteralRule:
    """Rule #7: large dict literal inside a function (default ``info``)."""

    def test_default_severity_is_info(self):
        rule = DictLiteralRule()
        assert rule.default_severity == "info"

    def _make_dict(self, n_keys: int) -> "ast.Dict":
        import ast
        keys = [ast.Constant(value="k{}".format(i)) for i in range(n_keys)]
        values = [ast.Constant(value=i) for i in range(n_keys)]
        return ast.Dict(keys=keys, values=values)

    def test_large_dict_in_function_fires(self):
        rule = DictLiteralRule()
        import ast
        node = self._make_dict(6)
        ctx = _ctx(node=node, value=None, in_function=True)
        cands = rule.check(ctx)
        assert len(cands) == 1
        assert cands[0].severity == "info"
        assert cands[0].type == "dict_literal"
        assert "6 keys" in cands[0].content

    def test_small_dict_does_not_fire(self):
        rule = DictLiteralRule()
        import ast
        node = self._make_dict(3)
        ctx = _ctx(node=node, value=None, in_function=True)
        assert rule.check(ctx) == []

    def test_dict_outside_function_does_not_fire(self):
        rule = DictLiteralRule()
        import ast
        node = self._make_dict(10)
        ctx = _ctx(node=node, value=None, in_function=False)
        assert rule.check(ctx) == []

    def test_dict_in_docstring_does_not_fire(self):
        rule = DictLiteralRule()
        import ast
        node = self._make_dict(10)
        ctx = _ctx(node=node, value=None, in_function=True, in_docstring=True)
        assert rule.check(ctx) == []

    def test_applies_to_only_dict(self):
        rule = DictLiteralRule()
        import ast
        assert rule.applies_to(ast.Dict(keys=[], values=[])) is True
        assert rule.applies_to(ast.List(elts=[])) is False
        assert rule.applies_to(ast.Constant(value="x")) is False


# ---------------------------------------------------------------------------
# Exemption.rules -- per-rule opt-out
# ---------------------------------------------------------------------------
class TestExemptionRules:
    """Exemption.rules gives per-rule opt-out."""

    def test_rules_none_matches_every_type(self):
        v = _violation(type="string_literal")
        ex = Exemption(file="x.py", rules=None)
        assert ex.matches(v) is True

    def test_rules_set_filters_to_named_types(self):
        v_string = _violation(type="string_literal")
        v_numeric = _violation(type="numeric_literal")
        ex = Exemption(file="x.py", rules={"string_literal"})
        assert ex.matches(v_string) is True
        assert ex.matches(v_numeric) is False

    def test_rules_multiple_allowed(self):
        v_string = _violation(type="string_literal")
        v_path = _violation(type="path_literal")
        v_numeric = _violation(type="numeric_literal")
        ex = Exemption(file="x.py", rules={"string_literal", "path_literal"})
        assert ex.matches(v_string) is True
        assert ex.matches(v_path) is True
        assert ex.matches(v_numeric) is False

    def test_per_rule_opt_out_preserves_other_violations(self):
        """An exemption with rules={string_literal} does NOT swallow
        a numeric_literal violation on the same line."""
        v_string = _violation(type="string_literal", file="x.py", line=10)
        v_numeric = _violation(type="numeric_literal", file="x.py", line=10)
        ex = Exemption(file="x.py", line=10, rules={"string_literal"})
        assert ex.matches(v_string) is True
        assert ex.matches(v_numeric) is False


# ---------------------------------------------------------------------------
# Exemption.is_terminal / apply
# ---------------------------------------------------------------------------
class TestExemptionTerminality:
    """Terminal vs non-terminal exemptions behave as documented."""

    def test_terminal_no_protocol_no_severity(self):
        ex = Exemption(file="x.py")
        assert ex.is_terminal() is True

    def test_protocol_format_makes_non_terminal(self):
        ex = Exemption(file="x.py", protocol_format=True)
        assert ex.is_terminal() is False

    def test_explicit_severity_makes_non_terminal(self):
        ex = Exemption(file="x.py", severity="info")
        assert ex.is_terminal() is False

    def test_apply_terminal_does_not_change_severity(self):
        v = _violation()
        ex = Exemption(file="x.py")
        ex.apply(v)
        assert v.severity == "critical"  # unchanged

    def test_apply_protocol_format_downgrades_to_info(self):
        v = _violation()
        ex = Exemption(file="x.py", protocol_format=True)
        ex.apply(v)
        assert v.severity == "info"

    def test_apply_explicit_severity_overrides(self):
        v = _violation()
        ex = Exemption(file="x.py", severity="warn")
        ex.apply(v)
        assert v.severity == "warn"


# ---------------------------------------------------------------------------
# scan_directory / only_rule / CLI flags
# ---------------------------------------------------------------------------
class TestScannerCLI:
    """The new --only-rule and --list-rules CLI flags."""

    def test_list_rule_names_via_module(self):
        names = list_rule_names()
        for expected in (
            "string_literal", "numeric_literal", "path_literal",
            "list_literal", "fstring_template", "regex_pattern",
            "dict_literal",
        ):
            assert expected in names
        assert len(names) == 7

    def test_scan_directory_only_rule_string(self, tmp_path):
        """When only_rule='string_literal', numeric and path and list
        violations are not reported."""
        # Create a file with all 4 rule types of violations.
        code = textwrap.dedent("""
            def f():
                s = "this is a long string literal"   # string
                n = 500                                # numeric
                lst = [1, 2, 3, 4, 5]                 # list
                # path is rare inside function bodies; use os.path.join
                import os
                p = "/var/log/app.log"                # path
                return s, n, lst, p
        """)
        # NOTE: the `n = 500` is NOT in __init__, so numeric won't fire.
        # Make a class with __init__.
        code = textwrap.dedent("""
            class C:
                def __init__(self):
                    self.s = "this is a long string literal"   # string
                    self.n = 500                                # numeric
                    self.lst = [1, 2, 3, 4, 5]                 # list
                    self.p = "/var/log/app.log"                # path
        """)
        _write(tmp_path, "x.py", code)
        all_v = scan_directory(tmp_path)
        only_string = scan_directory(tmp_path, only_rule="string_literal")
        # only_string should be a strict subset of all_v
        all_types = {v.type for v in all_v}
        string_types = {v.type for v in only_string}
        assert "string_literal" in string_types
        assert "numeric_literal" not in string_types
        assert "list_literal" not in string_types
        assert "path_literal" not in string_types
        assert string_types.issubset(all_types)

    def test_scan_directory_only_rule_unknown_raises(self, tmp_path):
        _write(tmp_path, "x.py", "def f(): pass\n")
        with pytest.raises(ValueError):
            scan_directory(tmp_path, only_rule="nonexistent_rule")

    def test_scan_file_returns_violations(self, tmp_path):
        code = textwrap.dedent("""
            class C:
                def __init__(self):
                    self.x = "a long string inside init"
                    self.n = 500
        """)
        f = _write(tmp_path, "x.py", code)
        violations = scan_file(f, root=tmp_path)
        assert isinstance(violations, list)
        assert all(isinstance(v, Violation) for v in violations)
        # 1 string + 1 numeric at minimum
        assert len(violations) >= 2


# ---------------------------------------------------------------------------
# scripts.ci_config -- mini-TOML parser
# ---------------------------------------------------------------------------
class TestCIConfig:
    """scripts.ci_config exposes the [tool.torcha-verse.hardcoding] API."""

    def test_default_settings(self):
        assert DEFAULT_CI_SETTINGS["path"] == "."
        assert DEFAULT_CI_SETTINGS["whitelist"] == "config/hardcoded_whitelist.yaml"
        assert DEFAULT_CI_SETTINGS["ci_fail_on"] == "critical"
        assert DEFAULT_CI_SETTINGS["enabled"] is True

    def test_load_returns_dict(self, tmp_path):
        pyproject = _write(tmp_path, "pyproject.toml", textwrap.dedent("""
            [tool.torcha-verse.hardcoding]
            path = "./src"
            whitelist = "config/whitelist.yaml"
            ci_fail_on = "warn"
            enabled = false
        """))
        result = load_hardcoding_ci_settings(pyproject)
        assert result["path"] == "./src"
        assert result["whitelist"] == "config/whitelist.yaml"
        assert result["ci_fail_on"] == "warn"
        assert result["enabled"] is False

    def test_load_merges_defaults(self, tmp_path):
        """When a key is missing, the default fills in."""
        pyproject = _write(tmp_path, "pyproject.toml", textwrap.dedent("""
            [tool.torcha-verse.hardcoding]
            path = "./src"
        """))
        result = load_hardcoding_ci_settings(pyproject)
        # path overridden
        assert result["path"] == "./src"
        # other keys come from defaults
        assert result["whitelist"] == "config/hardcoded_whitelist.yaml"
        assert result["ci_fail_on"] == "critical"
        assert result["enabled"] is True

    def test_load_missing_section_returns_defaults(self, tmp_path):
        """If the [tool.torcha-verse.hardcoding] section is absent,
        the function returns the defaults unchanged."""
        pyproject = _write(tmp_path, "pyproject.toml", textwrap.dedent("""
            [tool.other]
            key = "value"
        """))
        result = load_hardcoding_ci_settings(pyproject)
        assert result == dict(DEFAULT_CI_SETTINGS)

    def test_load_invalid_enabled_exits(self, tmp_path):
        """An invalid boolean for ``enabled`` triggers SystemExit."""
        pyproject = _write(tmp_path, "pyproject.toml", textwrap.dedent("""
            [tool.torcha-verse.hardcoding]
            enabled = "not-a-bool"
        """))
        with pytest.raises(SystemExit) as exc:
            load_hardcoding_ci_settings(pyproject)
        # SystemExit carries an error message (not a numeric code).
        assert "enabled" in str(exc.value).lower()

    def test_load_invalid_ci_fail_on_exits(self, tmp_path):
        """An invalid severity for ``ci_fail_on`` triggers SystemExit."""
        pyproject = _write(tmp_path, "pyproject.toml", textwrap.dedent("""
            [tool.torcha-verse.hardcoding]
            ci_fail_on = "fatal"
        """))
        with pytest.raises(SystemExit) as exc:
            load_hardcoding_ci_settings(pyproject)
        assert "ci_fail_on" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# scripts.check_ci_gates -- unified CI entry
# ---------------------------------------------------------------------------
class TestCIGates:
    """scripts.check_ci_gates exposes a registry with the right gates."""

    def test_registry_has_hardcoding_and_placeholders(self):
        assert "hardcoding" in ci_gates.GATE_REGISTRY
        assert "placeholders" in ci_gates.GATE_REGISTRY

    def test_registry_runners_are_callable(self):
        for name, spec in ci_gates.GATE_REGISTRY.items():
            assert callable(spec["runner"]), f"{name} runner not callable"

    def test_registry_default_enabled_true(self):
        """The hardcoding and placeholders gates are enabled by
        default; the degrade_logging gate is intentionally off
        until the 38 known silent-degrade sites are fixed."""
        for name, spec in ci_gates.GATE_REGISTRY.items():
            if name == "degrade_logging":
                # D3 stage three -- default off, opt-in via
                # [tool.torcha-verse.ci-gates.degrade_logging].
                assert spec["default_enabled"] is False
            else:
                assert spec["default_enabled"] is True, f"{name} not enabled by default"

    def test_main_list_returns_0(self, capsys):
        """`--list` mode is informational and exits 0."""
        rc = ci_gates.main(["--list"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "hardcoding" in captured.out
        assert "placeholders" in captured.out

    def test_main_returns_int(self):
        """The runner's main() is callable and returns an int exit code."""
        # We don't actually run it (would re-execute scanners); just
        # verify the signature.
        import inspect
        sig = inspect.signature(ci_gates.main)
        assert "argv" in sig.parameters
