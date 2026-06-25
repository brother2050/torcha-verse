"""Smoke tests for ``examples/*.py``.

These tests only verify that every example:

* can be imported (no syntax errors, no top-level side effects that
  require network / GPU / model weights);
* exposes a ``main()`` callable (the canonical CLI entry point
  referenced by every example's docstring);
* can be invoked as a Python module without crashing the importer
  (``python -m examples.<name> --help`` is not necessary; we just
  confirm the function exists).

We deliberately **do not** run ``main()`` itself because several
examples spawn a real model forward pass and would take many
seconds.  The ``main()`` of each example is exercised by the v0.3
examples checklist (see ``docs/examples_catalog.md``).

Markers
-------
* ``examples`` -- not slow, not GPU-required.  Runs on the test
  default.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Make sure the project root is on sys.path so that
# ``import examples.<name>`` works (examples/ does that explicitly
# for itself, but the test runner doesn't follow that trick).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# All examples we expect to ship.  Keeping this in a list (rather
# than ``glob``) gives the test runner a clear, ordered, *exhaustive*
# list to assert against; if a new example is added, this test
# should be updated alongside the docs.
EXAMPLE_NAMES = [
    "agent_demo",
    "audio_tts",
    "basic_text_gen",
    "consistency_character",
    "dh_lipsync",
    "image_gen",
    "model_download",
    "rag_demo",
    "real_text_chat",
    "video_gen",
]


pytestmark = pytest.mark.examples


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(params=EXAMPLE_NAMES, ids=EXAMPLE_NAMES)
def example_name(request) -> str:
    """Yield each example's module name as a parameter."""
    return request.param


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestExamplesImport:
    """All examples can be imported and expose a ``main()`` callable."""

    def test_module_imports(self, example_name: str) -> None:
        mod = importlib.import_module(f"examples.{example_name}")
        assert mod is not None

    def test_module_has_main(self, example_name: str) -> None:
        mod = importlib.import_module(f"examples.{example_name}")
        assert hasattr(mod, "main"), (
            f"examples/{example_name}.py must define a top-level main() "
            f"function; the file should be importable and the entry "
            f"point callable from `python -m examples.{example_name}`."
        )
        main = getattr(mod, "main")
        assert callable(main)

    def test_module_has_docstring(self, example_name: str) -> None:
        """Every example must lead with a one-paragraph usage docstring."""
        mod = importlib.import_module(f"examples.{example_name}")
        assert mod.__doc__ is not None, (
            f"examples/{example_name}.py is missing a module docstring. "
            f"Add a one-paragraph description + a `Run with::` or "
            f"`Usage::` block so the docs catalog can extract the "
            f"command line."
        )
        # Both `Run with::` and `Usage::` are accepted conventions.
        assert ("Run with" in mod.__doc__) or ("Usage::" in mod.__doc__), (
            f"examples/{example_name}.py docstring must include either "
            f"a `Run with::` or `Usage::` block that shows the CLI "
            f"invocation."
        )

    def test_module_has_main_guard(self, example_name: str) -> None:
        """Every example must guard its body with
        ``if __name__ == "__main__":`` so it is import-safe."""
        source = (PROJECT_ROOT / "examples" / f"{example_name}.py").read_text(
            encoding="utf-8",
        )
        assert '__name__ == "__main__"' in source, (
            f"examples/{example_name}.py must guard its body with "
            f"`if __name__ == \"__main__\":` so importing the module "
            f"does not execute the demo logic."
        )

    def test_module_syntax_ok(self, example_name: str) -> None:
        """``python -m py_compile`` is a redundant safety net for the
        import-clean check above."""
        import py_compile
        path = PROJECT_ROOT / "examples" / f"{example_name}.py"
        py_compile.compile(str(path), doraise=True)


class TestExamplesCatalog:
    """Sanity checks for ``docs/examples_catalog.md`` and
    ``examples/README.md``."""

    def test_examples_readme_exists(self) -> None:
        readme = PROJECT_ROOT / "examples" / "README.md"
        assert readme.is_file(), (
            "examples/README.md must exist; it serves as the index for "
            "the 11 example scripts and lists their run commands."
        )

    def test_examples_catalog_exists(self) -> None:
        catalog = PROJECT_ROOT / "docs" / "examples_catalog.md"
        assert catalog.is_file(), (
            "docs/examples_catalog.md must exist; it gives a per-example "
            "breakdown (node mapping / dependencies / fall-through path)."
        )

    def test_examples_readme_lists_all(self) -> None:
        """The README must mention every example module by name."""
        readme = (PROJECT_ROOT / "examples" / "README.md").read_text(
            encoding="utf-8",
        )
        for name in EXAMPLE_NAMES:
            assert name in readme, (
                f"examples/README.md must reference examples/{name}.py. "
                f"Add it to the index table."
            )

    def test_examples_catalog_lists_all(self) -> None:
        """The catalog must give every example its own subsection."""
        catalog = (PROJECT_ROOT / "docs" / "examples_catalog.md").read_text(
            encoding="utf-8",
        )
        for name in EXAMPLE_NAMES:
            assert name in catalog, (
                f"docs/examples_catalog.md must have a section for "
                f"examples/{name}.py."
            )
