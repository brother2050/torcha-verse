"""R-18 lazy-import tests.

Verify that:
* :mod:`papers` is importable without triggering ``torch`` (or
  any of the bundled adapter modules).
* The bundled paper YAML specs are still loaded eagerly.
* The :class:`AdapterRegistry` resolves bundled adapter names on
  demand (``has`` is a name-only check, ``get`` actually loads).
* The PEP 562 lazy export (``papers.StableDiffusion3Adapter``
  etc.) resolves to the correct class.
* The :mod:`papers.adapters` sub-package follows the same lazy
  contract: ``import papers.adapters`` does **not** import either
  adapter module.
"""

from __future__ import annotations

import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ADAPTER_MODULES = (
    "papers.adapters._mmdit",
    "papers.adapters.stable_diffusion_3",
    "papers.adapters.hunyuan_dit",
)


def _purge_adapter_modules() -> None:
    """Remove the adapter modules from ``sys.modules`` and the
    in-process :data:`papers._loaded_adapters` cache.

    Needed when a previous test in the same process has already
    imported one of them -- we want to assert that *this* code
    path does not cause a re-import.
    """
    for mod in ADAPTER_MODULES:
        sys.modules.pop(mod, None)
    # Also clear the lazy cache on the papers package so the next
    # ``__getattr__`` triggers a fresh import.
    try:
        import papers  # noqa: F401
        papers._loaded_adapters.clear()
    except Exception:  # noqa: BLE001 - defensive
        pass


# ---------------------------------------------------------------------------
# import papers
# ---------------------------------------------------------------------------
class TestPapersImport:
    """R-18: ``import papers`` does not load any adapter modules."""

    def test_import_papers_does_not_load_torch_modules(self) -> None:
        """The 1,000+ line torch-backed adapter modules stay out of
        ``sys.modules`` after importing the package."""
        _purge_adapter_modules()
        import papers  # noqa: F401
        for mod in ADAPTER_MODULES:
            assert mod not in sys.modules, (
                "Adapter module {!r} was eagerly imported; "
                "R-18 requires lazy loading.".format(mod)
            )

    def test_bundled_papers_still_loaded_eagerly(self) -> None:
        """The YAML specs (cheap, dataclass-only) are still
        populated after import.

        Some tests in the suite call :meth:`PaperRegistry._reset`
        to start from a clean slate; we re-load the bundled
        directory in that case so the assertion remains
        deterministic.
        """
        from papers import PaperRegistry
        reg = PaperRegistry()
        if not reg.list():
            reg.load_bundled()
        # The bundled catalogue ships at least 2 papers
        # (sd3 + hunyuan-dit).  Use a soft lower bound.
        assert len(reg.list()) >= 2

    def test_default_registry_has_lazily_resolvable_names(self) -> None:
        """``has()`` is a name-only check and does not load torch."""
        _purge_adapter_modules()
        from papers import default_registry
        assert default_registry.has("stable-diffusion-3")
        assert default_registry.has("sd3")
        assert default_registry.has("hunyuan-dit")
        for mod in ADAPTER_MODULES:
            assert mod not in sys.modules


# ---------------------------------------------------------------------------
# AdapterRegistry lazy resolution
# ---------------------------------------------------------------------------
class TestAdapterRegistryLazy:
    """R-18: :meth:`AdapterRegistry.get` loads the class on first call."""

    def test_get_loads_module_and_returns_class(self) -> None:
        _purge_adapter_modules()
        from papers import default_registry
        cls = default_registry.get("stable-diffusion-3")
        assert cls.__name__ == "StableDiffusion3Adapter"
        # The module is now in sys.modules.
        assert "papers.adapters.stable_diffusion_3" in sys.modules

    def test_get_is_cached(self) -> None:
        """Second ``get`` call does not re-import the module."""
        from papers import default_registry
        cls1 = default_registry.get("hunyuan-dit")
        cls2 = default_registry.get("hunyuan-dit")
        assert cls1 is cls2

    def test_unknown_name_still_raises(self) -> None:
        from papers.adapter import AdapterNotFoundError, AdapterRegistry
        reg = AdapterRegistry()
        with pytest.raises(AdapterNotFoundError):
            reg.get("definitely-not-a-real-adapter-xyz")


# ---------------------------------------------------------------------------
# PEP 562 lazy export
# ---------------------------------------------------------------------------
class TestLazyExport:
    """R-18: ``from papers import StableDiffusion3Adapter`` works."""

    def test_attr_access_resolves_class(self) -> None:
        _purge_adapter_modules()
        import papers
        cls = papers.StableDiffusion3Adapter
        assert cls.__name__ == "StableDiffusion3Adapter"
        # Module should be loaded now.
        assert "papers.adapters.stable_diffusion_3" in sys.modules

    def test_attr_access_caches_in_loaded_adapters(self) -> None:
        """After the first access, the class is cached in
        :data:`papers._loaded_adapters` so a second access is
        served from the module-level cache (no re-import)."""
        _purge_adapter_modules()
        import papers
        cls1 = papers.HunyuanDiTAdapter
        # The module-level cache returns the same object.
        cls2 = papers._loaded_adapters["hunyuan-dit"]
        assert cls1 is cls2

    def test_unknown_attr_raises_attribute_error(self) -> None:
        import papers
        with pytest.raises(AttributeError):
            papers.not_a_real_attribute


# ---------------------------------------------------------------------------
# papers.adapters sub-package
# ---------------------------------------------------------------------------
class TestAdaptersSubpackageLazy:
    """R-18: ``import papers.adapters`` itself does not import the
    individual adapter modules."""

    def test_import_adapters_subpackage_does_not_load_modules(self) -> None:
        _purge_adapter_modules()
        import papers.adapters  # noqa: F401
        for mod in ADAPTER_MODULES:
            assert mod not in sys.modules, (
                "{!r} was eagerly imported from papers.adapters; "
                "R-18 requires lazy loading.".format(mod)
            )

    def test_paper_adapter_base_class_still_eager(self) -> None:
        """The base class is still importable eagerly (no torch)."""
        from papers.adapters import PaperAdapter
        assert hasattr(PaperAdapter, "load_model")  # abstract method
