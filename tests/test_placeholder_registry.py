"""Tests for :mod:`infrastructure.placeholder_registry`.

These tests cover the v0.4.0 D3 work-stream: the machine-readable
companion to :doc:`/docs/placeholder_registry` that CI uses to verify
every ``pass`` / ``NotImplementedError`` is documented.

Coverage:

* :class:`PlaceholderCategory` enum completeness.
* :func:`load_registry` parser: happy path, malformed row, missing file.
* :func:`scan_source` scanner: ``pass`` lines, ``raise NotImplementedError``
  lines, ignored directories (tests / caches), inline ``# placeholder-
  registry: ignore`` marker, backtick-quoted docstring text.
* :func:`find_unregistered` set-diff between scanner and registry.
* :class:`PlaceholderEntry.matches` lookup.
* End-to-end: parsing the project registry + scanning the project root
  should produce **zero** unregistered hits (matches what
  ``scripts/check_placeholders.py`` reports in CI).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from infrastructure.placeholder_registry import (
    DEFAULT_REGISTRY_PATH,
    PlaceholderCategory,
    PlaceholderEntry,
    PlaceholderScannerError,
    ScanHit,
    SCAN_IGNORE_DIRS,
    find_unregistered,
    load_registry,
    registry_index,
    scan_source,
)


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.placeholder_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content), encoding="utf-8")


# ---------------------------------------------------------------------------
# Category
# ---------------------------------------------------------------------------
class TestCategory:
    def test_all_five_categories_present(self) -> None:
        names = {c.value for c in PlaceholderCategory}
        assert names == {
            "protocol", "tp_pp", "protocol_stub",
            "degrade_try_except", "degrade_noop",
        }

    def test_categories_are_strings(self) -> None:
        for c in PlaceholderCategory:
            assert isinstance(c.value, str)


# ---------------------------------------------------------------------------
# PlaceholderEntry
# ---------------------------------------------------------------------------
class TestPlaceholderEntry:
    def test_matches(self) -> None:
        e = PlaceholderEntry(
            file="foo/bar.py", line=42, category=PlaceholderCategory.PROTOCOL,
        )
        assert e.matches("foo/bar.py", 42)
        assert not e.matches("foo/bar.py", 41)
        assert not e.matches("foo/baz.py", 42)

    def test_frozen(self) -> None:
        e = PlaceholderEntry(
            file="x.py", line=1, category=PlaceholderCategory.TP_PP,
        )
        with pytest.raises(Exception):
            e.line = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------
class TestLoadRegistry:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_registry(tmp_path / "nope.md", project_root=tmp_path)

    def test_parses_basic_table(self, tmp_path: Path) -> None:
        md = tmp_path / "reg.md"
        md.write_text(
            textwrap.dedent(
                """\
                # Test registry

                ## 1. Section heading

                | # | 文件:行 | 类/函数 | 方法 | 说明 |
                |---:|---|---|---|---|
                | 1 | `models/x.py:10` | `Foo` | `bar` | reason one |
                | 2 | `models/y.py:20` | `Baz` | `qux` | reason two |
                """
            ),
            encoding="utf-8",
        )
        entries = load_registry(md, project_root=tmp_path)
        assert len(entries) == 2
        e0 = entries[0]
        assert e0.file == "models/x.py"
        assert e0.line == 10
        assert e0.category == PlaceholderCategory.DEGRADE_TRY_EXCEPT  # default
        assert "reason one" in e0.description
        e1 = entries[1]
        assert e1.file == "models/y.py"
        assert e1.line == 20

    def test_heading_drives_category(self, tmp_path: Path) -> None:
        md = tmp_path / "reg.md"
        md.write_text(
            textwrap.dedent(
                """\
                ## 1. Category

                ### 1.1 TP/PP placeholder

                | # | 文件:行 | 函数 | 说明 |
                |---:|---|---|---|
                | 1 | `infrastructure/device_manager.py:10` | `tp` | tbd |
                """
            ),
            encoding="utf-8",
        )
        entries = load_registry(md, project_root=tmp_path)
        assert len(entries) == 1
        assert entries[0].category == PlaceholderCategory.TP_PP

    def test_malformed_row_skipped_silently(self, tmp_path: Path) -> None:
        """Rows that don't match the strict regex are skipped, not raised."""
        md = tmp_path / "reg.md"
        md.write_text(
            textwrap.dedent(
                """\
                ## Section

                | 合计 | 5 |
                | valid | row |
                | 1 | `models/x.py:5` | `C` | `m` | desc |
                """
            ),
            encoding="utf-8",
        )
        entries = load_registry(md, project_root=tmp_path)
        # Only the row that contains a `path:line` token is parsed.
        assert len(entries) == 1
        assert entries[0].file == "models/x.py"


# ---------------------------------------------------------------------------
# Source scanner
# ---------------------------------------------------------------------------
class TestScanSource:
    def test_scans_pass_line(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "\
            def f():\n\
                pass  # noqa\n\
        ")
        hits = scan_source(tmp_path, project_root=tmp_path)
        assert len(hits) == 1
        assert hits[0].kind == "pass"
        assert hits[0].line == 2

    def test_scans_not_implemented_error(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "\
            def f():\n\
                raise NotImplementedError('todo')\n\
        ")
        hits = scan_source(tmp_path, project_root=tmp_path)
        assert len(hits) == 1
        assert hits[0].kind == "NotImplementedError"
        assert hits[0].line == 2

    def test_ignores_test_directory(self, tmp_path: Path) -> None:
        _write(tmp_path / "src.py", "def f():\n    pass\n")
        _write(tmp_path / "tests" / "test_x.py", "def g():\n    pass\n")
        hits = scan_source(tmp_path, project_root=tmp_path)
        # Only src.py's pass is reported; tests/ is excluded.
        files = {h.file for h in hits}
        assert "src.py" in str(files)
        assert not any("test_" in f for f in files)

    def test_ignores_pycache(self, tmp_path: Path) -> None:
        _write(tmp_path / "src.py", "def f():\n    pass\n")
        _write(tmp_path / "__pycache__" / "src.cpython-310.pyc", "def f():\n    pass\n")
        hits = scan_source(tmp_path, project_root=tmp_path)
        files = [h.file for h in hits]
        assert not any("__pycache__" in f for f in files)

    def test_inline_ignore_marker(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "\
            def f():\n\
                raise NotImplementedError  # placeholder-registry: ignore\n\
        ")
        hits = scan_source(tmp_path, project_root=tmp_path)
        assert hits == []

    def test_docstring_backtick_quoted_text_ignored(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.py", "\
            '''Scans for `pass` and `NotImplementedError` lines.'''\n\
            def f():\n\
                raise NotImplementedError('real')\n\
        ")
        hits = scan_source(tmp_path, project_root=tmp_path)
        # The docstring mentions both keywords but the only real
        # placeholder is the raise line; the docstring is not flagged.
        assert len(hits) == 1
        assert hits[0].line == 3

    def test_skip_dirs_constant_includes_tests(self) -> None:
        assert "tests" in SCAN_IGNORE_DIRS
        assert "__pycache__" in SCAN_IGNORE_DIRS

    def test_single_file_target(self, tmp_path: Path) -> None:
        f = tmp_path / "single.py"
        _write(f, "def g():\n    raise NotImplementedError\n")
        hits = scan_source(f, project_root=tmp_path)
        assert len(hits) == 1
        assert hits[0].file == "single.py"

    def test_nonexistent_target_returns_empty(self, tmp_path: Path) -> None:
        hits = scan_source(tmp_path / "nope.py", project_root=tmp_path)
        assert hits == []


# ---------------------------------------------------------------------------
# find_unregistered
# ---------------------------------------------------------------------------
class TestFindUnregistered:
    def test_returns_only_unregistered(self) -> None:
        hits = [
            ScanHit(file="a.py", line=1, kind="pass", text="pass"),
            ScanHit(file="b.py", line=2, kind="NotImplementedError", text="raise NotImplementedError"),
        ]
        registry = [
            PlaceholderEntry(file="a.py", line=1, category=PlaceholderCategory.DEGRADE_TRY_EXCEPT),
        ]
        unregistered = find_unregistered(hits, registry)
        assert len(unregistered) == 1
        assert unregistered[0].file == "b.py"

    def test_empty_inputs(self) -> None:
        assert find_unregistered([], []) == []


# ---------------------------------------------------------------------------
# registry_index
# ---------------------------------------------------------------------------
class TestRegistryIndex:
    def test_builds_lookup(self) -> None:
        e1 = PlaceholderEntry(
            file="x.py", line=1, category=PlaceholderCategory.PROTOCOL,
        )
        e2 = PlaceholderEntry(
            file="y.py", line=2, category=PlaceholderCategory.TP_PP,
        )
        idx = registry_index([e1, e2])
        assert idx[("x.py", 1)] is e1
        assert idx[("y.py", 2)] is e2
        assert ("z.py", 3) not in idx


# ---------------------------------------------------------------------------
# End-to-end: real project registry + scan
# ---------------------------------------------------------------------------
class TestEndToEnd:
    def test_project_root_scans_clean(self) -> None:
        """Parsing the project registry and scanning the project root
        should produce **zero** unregistered hits.

        This is the same check ``scripts/check_placeholders.py`` runs in
        CI -- it guarantees the registry is the single source of truth.
        """
        project_root = Path(__file__).resolve().parent.parent
        registry = load_registry(
            DEFAULT_REGISTRY_PATH, project_root=project_root,
        )
        assert len(registry) > 0, "registry should be non-empty"
        hits = scan_source(project_root, project_root=project_root)
        assert len(hits) > 0, "scanner should find at least one placeholder"
        unregistered = find_unregistered(hits, registry)
        assert unregistered == [], (
            "Found unregistered placeholders: {}".format(unregistered)
        )

    def test_all_categories_represented(self) -> None:
        """The real registry should cover all 5 categories (or note
        the empty ones explicitly)."""
        project_root = Path(__file__).resolve().parent.parent
        registry = load_registry(
            DEFAULT_REGISTRY_PATH, project_root=project_root,
        )
        cats = {e.category for e in registry}
        # At least the three non-empty categories are present.
        assert PlaceholderCategory.PROTOCOL in cats
        assert PlaceholderCategory.TP_PP in cats
        assert PlaceholderCategory.DEGRADE_TRY_EXCEPT in cats
