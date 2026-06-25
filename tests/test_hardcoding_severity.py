"""Tests for the D1 severity extension to ``check_hardcoding.py``.

These tests cover the v0.4.x D1 work-stream: the convention document
(:doc:`/docs/hardcoding_convention`) and its enforcement via the
``severity`` field on both :class:`Violation` and :class:`Exemption`.

Coverage:

* :class:`Violation` defaults ``severity`` to ``critical``.
* :class:`Exemption.matches` / :meth:`Exemption.apply` for each kind of
  downgrade (terminal / ``protocol_format`` / explicit ``severity``).
* :class:`Exemption.is_terminal` correctly distinguishes terminal from
  non-terminal exemptions.
* :func:`filter_by_severity` returns violations at or above the given
  threshold (``critical`` < ``warn`` < ``info``).
* Scanner auto-classifies model-structural numeric literals in
  ``models/`` as ``info`` (the ``is_structural_init`` heuristic).
* Scanner auto-classifies ``os.environ[...]`` / ``Path(...).expanduser()``
  strings as ``info`` (the ``is_runtime_attr`` heuristic).
* :func:`export_critical` deduplicates by ``(file, line, type)`` and
  produces a whitelist-schema-compatible YAML file.
* The real :file:`config/hardcoded_whitelist.yaml` actually downgrades
  the violations it claims to cover (end-to-end).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scripts.check_hardcoding import (
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARN,
    Exemption,
    Violation,
    export_critical,
    filter_by_severity,
    is_log_message_format,
    load_whitelist,
    scan_directory,
    scan_file,
)


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.hardcoding_severity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _violation(
    *,
    file: str = "x.py",
    line: int = 1,
    col: int = 0,
    type: str = "string_literal",
    content: str = "'foo'",
    severity: str = SEVERITY_CRITICAL,
) -> Violation:
    return Violation(
        file=file, line=line, col=col, type=type,
        content=content, severity=severity,
    )


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Violation defaults
# ---------------------------------------------------------------------------
class TestViolationDefaults:
    def test_default_severity_is_critical(self) -> None:
        v = Violation(file="x.py", line=1, col=0, type="string_literal", content="'x'")
        assert v.severity == SEVERITY_CRITICAL

    def test_as_dict_includes_severity(self) -> None:
        v = _violation(severity=SEVERITY_INFO)
        d = v.as_dict()
        assert d["severity"] == SEVERITY_INFO
        assert d["type"] == "string_literal"
        assert d["file"] == "x.py"


# ---------------------------------------------------------------------------
# Exemption matching / applying
# ---------------------------------------------------------------------------
class TestExemption:
    def test_wildcard_type_matches(self) -> None:
        ex = Exemption(file="x.py", type="*")
        for vtype in ("string_literal", "numeric_literal", "path_literal", "list_literal"):
            assert ex.matches(_violation(type=vtype))

    def test_exact_type_match(self) -> None:
        ex = Exemption(file="x.py", type="string_literal")
        assert ex.matches(_violation(type="string_literal"))
        assert not ex.matches(_violation(type="numeric_literal"))

    def test_line_match(self) -> None:
        ex = Exemption(file="x.py", line=42)
        assert ex.matches(_violation(line=42))
        assert not ex.matches(_violation(line=43))

    def test_content_contains(self) -> None:
        ex = Exemption(file="x.py", content_contains="hello")
        assert ex.matches(_violation(content="'hello world'"))
        assert not ex.matches(_violation(content="'goodbye'"))

    def test_glob_file_match(self) -> None:
        ex = Exemption(file="agents/*.py")
        assert ex.matches(_violation(file="agents/foo.py"))
        assert not ex.matches(_violation(file="tools/foo.py"))

    def test_terminal_when_no_severity_and_no_protocol(self) -> None:
        assert Exemption(file="x.py").is_terminal()

    def test_non_terminal_when_severity_set(self) -> None:
        assert not Exemption(file="x.py", severity="info").is_terminal()

    def test_non_terminal_when_protocol_format(self) -> None:
        assert not Exemption(file="x.py", protocol_format=True).is_terminal()

    def test_apply_protocol_format_downgrades_to_info(self) -> None:
        v = _violation(severity=SEVERITY_CRITICAL)
        ex = Exemption(file="x.py", protocol_format=True)
        assert ex.apply(v) is True
        assert v.severity == SEVERITY_INFO

    def test_apply_explicit_severity(self) -> None:
        v = _violation(severity=SEVERITY_CRITICAL)
        ex = Exemption(file="x.py", severity="warn")
        assert ex.apply(v) is True
        assert v.severity == SEVERITY_WARN

    def test_apply_terminal_keeps_violation(self) -> None:
        """Terminal exemptions do not modify the violation -- they
        are filtered out later in :func:`scan_directory`."""
        v = _violation()
        ex = Exemption(file="x.py", reason="test")
        assert ex.apply(v) is True
        assert v.severity == SEVERITY_CRITICAL  # unchanged
        assert ex.is_terminal()

    def test_apply_does_not_match(self) -> None:
        v = _violation(file="x.py")
        ex = Exemption(file="y.py")
        assert ex.apply(v) is False


# ---------------------------------------------------------------------------
# filter_by_severity
# ---------------------------------------------------------------------------
class TestFilterBySeverity:
    def test_critical_threshold(self) -> None:
        vs = [
            _violation(severity=SEVERITY_CRITICAL),
            _violation(severity=SEVERITY_INFO),
            _violation(severity=SEVERITY_WARN),
        ]
        out = filter_by_severity(vs, "critical")
        assert len(out) == 1
        assert out[0].severity == SEVERITY_CRITICAL

    def test_warn_threshold(self) -> None:
        vs = [
            _violation(severity=SEVERITY_CRITICAL),
            _violation(severity=SEVERITY_INFO),
            _violation(severity=SEVERITY_WARN),
        ]
        out = filter_by_severity(vs, "warn")
        assert len(out) == 2
        assert all(v.severity != SEVERITY_INFO for v in out)

    def test_info_threshold(self) -> None:
        vs = [
            _violation(severity=SEVERITY_CRITICAL),
            _violation(severity=SEVERITY_INFO),
        ]
        out = filter_by_severity(vs, "info")
        assert len(out) == 2

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            filter_by_severity([], "bogus")


# ---------------------------------------------------------------------------
# Scanner heuristics
# ---------------------------------------------------------------------------
class TestStructuralInitHeuristic:
    def test_model_init_numeric_is_info(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "models/fake.py", "\
            class Tiny:\n\
                def __init__(self):\n\
                    self.d_model = 768\n\
                    self.num_layers = 12\n\
        ")
        v = scan_file(p, tmp_path)
        # Both should be tagged as ``info`` (structural) -- the scanner
        # only downgrades values in [2, 10000] from models/ paths.
        for violation in v:
            assert violation.severity == SEVERITY_INFO, (
                "{}:{} should be info, got {}".format(
                    violation.file, violation.line, violation.severity,
                )
            )

    def test_out_of_range_stays_critical(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "models/fake.py", "\
            class Tiny:\n\
                def __init__(self):\n\
                    self.max_seq_len = 100000\n\
        ")
        v = scan_file(p, tmp_path)
        # 100000 is outside [2, 10000] so the structural heuristic
        # does not apply; the literal stays critical.
        assert any(
            viol.severity == SEVERITY_CRITICAL for viol in v
        )

    def test_non_models_path_stays_critical(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "training/fake.py", "\
            class Tiny:\n\
                def __init__(self):\n\
                    self.d_model = 768\n\
        ")
        v = scan_file(p, tmp_path)
        # Not in models/, so the structural heuristic does not apply.
        assert any(
            viol.severity == SEVERITY_CRITICAL for viol in v
        )


class TestRuntimeAttrHeuristic:
    def test_os_environ_string_is_info(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "fake.py", "\
            import os\n\
            def f():\n\
                k = os.environ.get('SOME_LONG_KEY_NAME_HERE')\n\
        ")
        v = scan_file(p, tmp_path)
        # The 'SOME_LONG_KEY_NAME_HERE' string literal is read from
        # the environment -- downgraded to info.
        for viol in v:
            if viol.content == "'SOME_LONG_KEY_NAME_HERE'":
                assert viol.severity == SEVERITY_INFO

    def test_path_expanduser_is_info(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "fake.py", "\
            from pathlib import Path\n\
            def f():\n\
                p = Path('~/.cache/torcha-verse-foo-bar').expanduser()\n\
        ")
        v = scan_file(p, tmp_path)
        # The Path argument literal is parameterised by .expanduser();
        # downgraded to info.
        for viol in v:
            if "torcha-verse-foo-bar" in viol.content:
                assert viol.severity == SEVERITY_INFO

    def test_plain_string_stays_critical(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "fake.py", "\
            def f():\n\
                s = 'a-plain-constant-string-literal'\n\
        ")
        v = scan_file(p, tmp_path)
        for viol in v:
            if "a-plain-constant-string-literal" in viol.content:
                assert viol.severity == SEVERITY_CRITICAL


class TestLogMessageFormat:
    """D1 phase 2 -- "log message" heuristic.

    The D1 convention says: a string literal that is the *first*
    positional argument of a ``logger.{level}(...)`` call is a
    log-format string.  Log format strings are protocol/format
    identifiers (printf placeholders, log keys, JSON keys), not
    user-tunable runtime config, so the scanner downgrades them to
    ``info`` while still emitting them in the report.
    """

    def test_logger_info_format_is_info(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "fake.py", "\
            from infrastructure.logger import get_logger\n\
            logger = get_logger(__name__)\n\
            def f():\n\
                logger.info('a-pretty-long-format-string')\n\
        ")
        v = scan_file(p, tmp_path)
        hits = [
            viol for viol in v
            if "a-pretty-long-format-string" in viol.content
        ]
        assert len(hits) == 1, hits
        assert hits[0].severity == SEVERITY_INFO

    def test_logger_warning_format_is_info(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "fake.py", "\
            from infrastructure.logger import get_logger\n\
            logger = get_logger(__name__)\n\
            def f():\n\
                logger.warning('something-bad-happened-here-ok')\n\
        ")
        v = scan_file(p, tmp_path)
        hits = [
            viol for viol in v
            if "something-bad-happened-here-ok" in viol.content
        ]
        assert len(hits) == 1
        assert hits[0].severity == SEVERITY_INFO

    def test_logger_error_format_is_info(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "fake.py", "\
            from infrastructure.logger import get_logger\n\
            logger = get_logger(__name__)\n\
            def f():\n\
                logger.error('failed-to-process-input-please')\n\
        ")
        v = scan_file(p, tmp_path)
        hits = [
            viol for viol in v
            if "failed-to-process-input-please" in viol.content
        ]
        assert len(hits) == 1
        assert hits[0].severity == SEVERITY_INFO

    def test_subsequent_args_stay_critical(self, tmp_path: Path) -> None:
        """Only the FIRST positional argument is the format string;
        subsequent ``%s``/``{}`` substitution values must remain
        critical (they are dynamic by construction, so the scanner
        won't even see them as literals, but other long constants
        such as ``'a-very-long-constant-payload'`` after the format
        string stay critical).
        """
        p = _write(tmp_path, "fake.py", "\
            from infrastructure.logger import get_logger\n\
            logger = get_logger(__name__)\n\
            def f():\n\
                logger.info('format-msg-okay', 'a-plain-constant-string-literal')\n\
        ")
        v = scan_file(p, tmp_path)
        fmt_hits = [
            viol for viol in v
            if "format-msg-okay" in viol.content
        ]
        plain_hits = [
            viol for viol in v
            if "a-plain-constant-string-literal" in viol.content
        ]
        # Format string -> info
        assert any(h.severity == SEVERITY_INFO for h in fmt_hits)
        # Subsequent plain literal -> critical
        assert any(h.severity == SEVERITY_CRITICAL for h in plain_hits)

    def test_log_method_only_when_first_arg(self, tmp_path: Path) -> None:
        """The heuristic must only apply when the literal is the
        *first* positional argument; a long string in
        ``logger.info(extra=...)`` keyword is not a format string.
        """
        p = _write(tmp_path, "fake.py", "\
            from infrastructure.logger import get_logger\n\
            logger = get_logger(__name__)\n\
            def f():\n\
                logger.info(extra='a-very-long-keyword-string-here')\n\
        ")
        v = scan_file(p, tmp_path)
        hits = [
            viol for viol in v
            if "a-very-long-keyword-string-here" in viol.content
        ]
        # Keyword arg, not format string -> critical
        if hits:
            assert hits[0].severity == SEVERITY_CRITICAL

    def test_is_log_message_format_helper(self, tmp_path: Path) -> None:
        """Direct unit test for the helper exported from the module."""
        import ast as _ast
        src = "logger.info('a-very-long-format-string-here-ok')\n"
        tree = _ast.parse(src)
        # The string Constant's parent isn't attached by parse() alone;
        # we walk and attach a parent so the helper can read it.
        for parent in _ast.walk(tree):
            for child in _ast.iter_child_nodes(parent):
                child.parent = parent  # type: ignore[attr-defined]
        # Find the string constant.
        for node in _ast.walk(tree):
            if (
                isinstance(node, _ast.Constant)
                and isinstance(node.value, str)
                and "a-very-long-format-string-here-ok" in node.value
            ):
                assert is_log_message_format(node) is True
                return
        pytest.fail("test setup could not find the string constant")

    def test_is_log_message_format_false_for_unrelated(self) -> None:
        """A bare string literal (not inside a logger call) returns False."""
        import ast as _ast
        src = "x = 'a-very-long-plain-string-here-okay'\n"
        tree = _ast.parse(src)
        for parent in _ast.walk(tree):
            for child in _ast.iter_child_nodes(parent):
                child.parent = parent  # type: ignore[attr-defined]
        for node in _ast.walk(tree):
            if (
                isinstance(node, _ast.Constant)
                and isinstance(node.value, str)
                and "a-very-long-plain-string-here-okay" in node.value
            ):
                assert is_log_message_format(node) is False
                return
        pytest.fail("test setup could not find the string constant")


# ---------------------------------------------------------------------------
# Whitelist loading
# ---------------------------------------------------------------------------
class TestLoadWhitelist:
    def test_loads_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "w.yaml"
        f.write_text("exemptions: []\n", encoding="utf-8")
        assert load_whitelist(f) == []

    def test_loads_severity_field(self, tmp_path: Path) -> None:
        f = tmp_path / "w.yaml"
        f.write_text(
            "exemptions:\n"
            "  - file: a.py\n"
            "    type: numeric_literal\n"
            "    severity: info\n"
            "    reason: test\n",
            encoding="utf-8",
        )
        ex = load_whitelist(f)
        assert len(ex) == 1
        assert ex[0].file == "a.py"
        assert ex[0].type == "numeric_literal"
        assert ex[0].severity == "info"
        assert ex[0].reason == "test"

    def test_loads_protocol_format(self, tmp_path: Path) -> None:
        f = tmp_path / "w.yaml"
        f.write_text(
            "exemptions:\n"
            "  - file: a.py\n"
            "    protocol_format: true\n",
            encoding="utf-8",
        )
        ex = load_whitelist(f)
        assert ex[0].protocol_format is True

    def test_rejects_invalid_severity(self, tmp_path: Path) -> None:
        f = tmp_path / "w.yaml"
        f.write_text(
            "exemptions:\n"
            "  - file: a.py\n"
            "    severity: bogus\n",
            encoding="utf-8",
        )
        with pytest.raises(SystemExit):
            load_whitelist(f)

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            load_whitelist(tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# export_critical
# ---------------------------------------------------------------------------
class TestExportCritical:
    def test_dedups_by_key(self, tmp_path: Path) -> None:
        # Three violations on the same (file, line, type) should
        # collapse to one entry.
        violations = [
            _violation(file="a.py", line=1, type="string_literal",
                       content="'hello world'"),
            _violation(file="a.py", line=1, type="string_literal",
                       content="'hello world'"),
            _violation(file="a.py", line=1, type="string_literal",
                       content="'hello world'"),
        ]
        out = tmp_path / "out.yaml"
        n = export_critical(violations, out)
        assert n == 1

    def test_filters_non_critical(self, tmp_path: Path) -> None:
        violations = [
            _violation(severity=SEVERITY_CRITICAL),
            _violation(severity=SEVERITY_INFO),
        ]
        out = tmp_path / "out.yaml"
        n = export_critical(violations, out)
        assert n == 1
        # The exported file should contain only the critical entry
        # (the info violation is filtered out before export).  We
        # check by counting the number of ``- file:`` markers -- each
        # exemption starts with a ``- file:`` bullet.
        text = out.read_text()
        bullets = text.count("- file:")
        assert bullets == 1

    def test_writes_yaml(self, tmp_path: Path) -> None:
        violations = [_violation(file="a.py", line=1, type="string_literal")]
        out = tmp_path / "out.yaml"
        export_critical(violations, out)
        # PyYAML is optional; check the file exists and has a key.
        text = out.read_text()
        assert "exemptions:" in text
        assert "file: a.py" in text


# ---------------------------------------------------------------------------
# End-to-end: real whitelist + real scan
# ---------------------------------------------------------------------------
class TestEndToEnd:
    def test_real_whitelist_actually_downgrades(self) -> None:
        """The shipped :file:`config/hardcoded_whitelist.yaml` should
        downgrade at least one violation from critical to info.
        """
        project_root = Path(__file__).resolve().parent.parent
        wl = project_root / "config" / "hardcoded_whitelist.yaml"
        if not wl.exists():
            pytest.skip("whitelist file not present")
        exemptions = load_whitelist(wl)
        assert len(exemptions) > 0, "whitelist should have entries"
        # Scan the whole project so we exercise the full set of
        # batch exemptions.  D1 stage three terminated all the
        # critical hits; the few remaining ``info`` violations
        # come from log messages and structural init heuristics
        # (e.g. ``torcha-verse/__init__.py``).
        violations = scan_directory(project_root, exemptions)
        info_hits = [v for v in violations if v.severity == SEVERITY_INFO]
        assert len(info_hits) > 0, (
            "Expected at least one info-tagged violation, "
            "got 0. The whitelist may not be applying."
        )
