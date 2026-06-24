"""Tests for v0.3.0 check_hardcoding.py scanner.

Verifies that the hardcoding scanner detects string literals, respects
the whitelist mechanism, and returns the correct exit codes when invoked
as a subprocess.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Make the scripts directory importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from check_hardcoding import (  # noqa: E402
    Exemption,
    Violation,
    scan_file,
    scan_directory,
    main,
)

_SCRIPT_PATH = _SCRIPTS_DIR / "check_hardcoding.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def bad_code_file(tmp_path):
    """Create a Python file with a long string literal inside a function."""
    src = tmp_path / "bad_code.py"
    src.write_text(
        "def my_function():\n"
        "    message = 'this is a very long hardcoded string literal'\n"
        "    return message\n"
    )
    return src


@pytest.fixture()
def clean_code_file(tmp_path):
    """Create a Python file with no hardcoding violations."""
    src = tmp_path / "clean_code.py"
    src.write_text(
        "def my_function():\n"
        "    x = 1 + 2\n"
        "    return x\n"
    )
    return src


# ---------------------------------------------------------------------------
# scan_file / scan_directory
# ---------------------------------------------------------------------------
class TestScanFile:
    """Direct scan_file() and scan_directory() tests."""

    def test_scan_file_detects_string_literal(self, bad_code_file, tmp_path):
        """scan_file() finds string_literal violations."""
        violations = scan_file(bad_code_file, tmp_path)
        assert len(violations) >= 1
        assert any(v.type == "string_literal" for v in violations)

    def test_scan_file_clean_code(self, clean_code_file, tmp_path):
        """scan_file() returns no violations for clean code."""
        violations = scan_file(clean_code_file, tmp_path)
        assert violations == []

    def test_scan_directory_finds_violations(self, bad_code_file, tmp_path):
        """scan_directory() finds violations across files."""
        violations = scan_directory(tmp_path)
        assert len(violations) >= 1

    def test_scan_directory_with_exemption(self, bad_code_file, tmp_path):
        """Exemptions suppress matching violations."""
        violations_before = scan_directory(tmp_path)
        assert len(violations_before) >= 1

        exemption = Exemption(file="*.py", type="*")
        violations_after = scan_directory(tmp_path, exemptions=[exemption])
        assert len(violations_after) == 0

    def test_violation_has_expected_fields(self, bad_code_file, tmp_path):
        """A Violation has file, line, col, type and content."""
        violations = scan_file(bad_code_file, tmp_path)
        v = violations[0]
        assert isinstance(v, Violation)
        assert v.file
        assert isinstance(v.line, int)
        assert isinstance(v.col, int)
        assert v.type
        assert v.content


# ---------------------------------------------------------------------------
# Exemption matching
# ---------------------------------------------------------------------------
class TestExemption:
    """Exemption.matches() logic."""

    def test_exemption_matches_wildcard_type(self):
        """A '*' type exemption matches any violation type."""
        v = Violation(file="src/foo.py", line=1, col=0, type="string_literal", content="x")
        ex = Exemption(file="*.py", type="*")
        assert ex.matches(v) is True

    def test_exemption_matches_specific_type(self):
        """A specific type exemption matches only that type."""
        v = Violation(file="src/foo.py", line=1, col=0, type="string_literal", content="x")
        ex = Exemption(file="*.py", type="numeric_literal")
        assert ex.matches(v) is False

    def test_exemption_matches_file_glob(self):
        """The file pattern is a glob."""
        v = Violation(file="src/foo.py", line=1, col=0, type="string_literal", content="x")
        ex = Exemption(file="src/*.py", type="*")
        assert ex.matches(v) is True

    def test_exemption_content_contains(self):
        """content_contains filters by substring."""
        v = Violation(file="a.py", line=1, col=0, type="string_literal", content="hello world")
        ex_match = Exemption(file="*.py", type="*", content_contains="hello")
        assert ex_match.matches(v) is True
        ex_no_match = Exemption(file="*.py", type="*", content_contains="goodbye")
        assert ex_no_match.matches(v) is False


# ---------------------------------------------------------------------------
# CLI / subprocess exit codes
# ---------------------------------------------------------------------------
class TestCLIExitCodes:
    """main() and subprocess exit-code behaviour."""

    def test_main_returns_1_on_violations(self, bad_code_file, tmp_path):
        """main() returns 1 when violations are found."""
        rc = main(["--path", str(tmp_path)])
        assert rc == 1

    def test_main_returns_0_on_clean(self, clean_code_file, tmp_path):
        """main() returns 0 when no violations are found."""
        rc = main(["--path", str(clean_code_file)])
        assert rc == 0

    def test_main_returns_2_on_missing_path(self, tmp_path):
        """main() returns 2 when the path does not exist."""
        rc = main(["--path", str(tmp_path / "nonexistent")])
        assert rc == 2

    def test_subprocess_clean_exit_code(self, clean_code_file):
        """Running the script via subprocess returns 0 for clean code."""
        result = subprocess.run(
            [sys.executable, str(_SCRIPT_PATH), "--path", str(clean_code_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_subprocess_violation_exit_code(self, bad_code_file):
        """Running the script via subprocess returns 1 for violations."""
        result = subprocess.run(
            [sys.executable, str(_SCRIPT_PATH), "--path", str(bad_code_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
