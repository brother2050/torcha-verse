"""Tests for the paper integration system.

Covers :class:`PaperSpec` / :class:`ModelRef` creation and YAML
round-tripping, the :class:`PaperRegistry` singleton (register / get /
list / search / load_from_dir), the :class:`AdapterRegistry`, and the
:mod:`papers.cli` command surface.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Importing the papers package eagerly loads the bundled YAML specs.
import papers  # noqa: F401
from papers import (
    AdapterRegistry,
    ModelRef,
    PaperAdapter,
    PaperNotFoundError,
    PaperRegistry,
    PaperSpec,
)
from papers.adapter import AdapterNotFoundError
from papers.cli import (
    paper_benchmark,
    paper_info,
    paper_install,
    paper_list,
    paper_reproduce,
)

#: Absolute path to the bundled papers directory (sibling of this package).
PAPERS_DIR: Path = Path(__file__).resolve().parent.parent / "papers"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_registry():
    """Ensure a clean PaperRegistry singleton before and after each test.

    Importing ``papers`` eagerly loads the bundled specs into the
    singleton; resetting here gives every test a deterministic, empty
    starting point.  Tests that need the bundled papers load them
    explicitly via :meth:`PaperRegistry.load_from_dir`.
    """
    PaperRegistry.reset()
    yield
    PaperRegistry.reset()


@pytest.fixture()
def registry():
    """Return the (reset) PaperRegistry singleton."""
    return PaperRegistry()


@pytest.fixture()
def loaded_registry():
    """Return a PaperRegistry with the bundled YAML specs loaded."""
    reg = PaperRegistry()
    reg.load_from_dir(PAPERS_DIR)
    return reg


# ---------------------------------------------------------------------------
# PaperSpec / ModelRef creation
# ---------------------------------------------------------------------------
class TestPaperSpec:
    """PaperSpec and ModelRef dataclass behaviour."""

    def test_paper_spec_creation(self):
        """A PaperSpec populates every field with correct defaults."""
        spec = PaperSpec(
            name="test_paper",
            title="Test Paper",
            authors=["Alice", "Bob"],
            arxiv_id="1234.5678",
            github="https://github.com/example/test",
            license="MIT",
            published="2024-01",
            integration_type="node",
            node_type="dh_lip_sync",
            method="musetalk",
            category="digital_human",
            seed=123,
            deterministic=False,
        )

        assert spec.name == "test_paper"
        assert spec.title == "Test Paper"
        assert spec.authors == ["Alice", "Bob"]
        assert spec.arxiv_id == "1234.5678"
        assert spec.github == "https://github.com/example/test"
        assert spec.license == "MIT"
        assert spec.published == "2024-01"
        assert spec.integration_type == "node"
        assert spec.node_type == "dh_lip_sync"
        assert spec.method == "musetalk"
        assert spec.category == "digital_human"
        assert spec.seed == 123
        assert spec.deterministic is False

        # Defaults for unset fields.
        assert spec.models == []
        assert spec.config == {}
        assert spec.reference_impl == {}
        assert spec.min_torcha_verse == "0.3.1"
        assert spec.gpu_required is True
        assert spec.min_vram_gb == 4

    def test_model_ref_creation(self):
        """A ModelRef stores its fields and defaults dependencies to []."""
        model = ModelRef(
            name="musetalk",
            source="huggingface",
            repo="TMElyralab/MuseTalk",
            size_gb=2.5,
            vram_gb=4,
            dependencies=["mmcv", "diffusers"],
        )
        assert model.name == "musetalk"
        assert model.source == "huggingface"
        assert model.repo == "TMElyralab/MuseTalk"
        assert model.size_gb == 2.5
        assert model.vram_gb == 4
        assert model.dependencies == ["mmcv", "diffusers"]

        # Defaults.
        empty = ModelRef(name="empty")
        assert empty.source == ""
        assert empty.repo == ""
        assert empty.size_gb == 0.0
        assert empty.vram_gb == 0.0
        assert empty.dependencies == []

    def test_paper_spec_to_from_dict_roundtrip(self):
        """to_dict() / from_dict() round-trip a spec structurally."""
        original = PaperSpec(
            name="rt",
            title="Round Trip",
            authors=["A"],
            arxiv_id="0000.0000",
            integration_type="model",
            method="transformer",
            category="foundation",
            models=[ModelRef(name="m", source="github", repo="r", size_gb=1.0)],
            seed=7,
            deterministic=False,
            config={"k": "v"},
            reference_impl={"karpathy": "karpathy/nanoGPT"},
            min_torcha_verse="0.3.1",
            gpu_required=False,
            min_vram_gb=0,
        )
        data = original.to_dict()
        restored = PaperSpec.from_dict(data)
        assert restored.name == original.name
        assert restored.title == original.title
        assert restored.authors == original.authors
        assert restored.integration_type == original.integration_type
        assert restored.method == original.method
        assert len(restored.models) == 1
        assert restored.models[0].name == "m"
        assert restored.models[0].source == "github"
        assert restored.seed == 7
        assert restored.deterministic is False
        assert restored.config == {"k": "v"}
        assert restored.reference_impl == {"karpathy": "karpathy/nanoGPT"}
        assert restored.gpu_required is False
        assert restored.min_vram_gb == 0


# ---------------------------------------------------------------------------
# PaperRegistry singleton / register / get
# ---------------------------------------------------------------------------
class TestPaperRegistryRegisterGet:
    """PaperRegistry singleton, register and get surface."""

    def test_singleton_identity(self, registry):
        """PaperRegistry() always returns the same instance."""
        assert PaperRegistry() is PaperRegistry()

    def test_reset_creates_fresh_instance(self):
        """reset() drops the singleton so the next call is fresh."""
        b1 = PaperRegistry()
        b1.register(PaperSpec(name="x", title="X", authors=[]))
        assert b1.count() == 1

        PaperRegistry.reset()
        b2 = PaperRegistry()
        assert b2 is not b1
        assert b2.count() == 0

    def test_paper_registry_register_get(self, registry):
        """register() stores a spec and get() returns the same object."""
        spec = PaperSpec(
            name="test_paper",
            title="Test Paper",
            authors=["A", "B"],
            method="musetalk",
        )
        registry.register(spec)

        got = registry.get("test_paper")
        assert got is spec
        assert got.name == "test_paper"
        assert got.authors == ["A", "B"]

    def test_get_raises_for_unknown(self, registry):
        """get() raises PaperNotFoundError for an unknown paper."""
        with pytest.raises(PaperNotFoundError):
            registry.get("does_not_exist")

    def test_has(self, registry):
        """has() reflects registration state."""
        registry.register(PaperSpec(name="p", title="P", authors=[]))
        assert registry.has("p") is True
        assert registry.has("missing") is False

    def test_register_replaces_existing(self, registry):
        """Re-registering a name replaces the previous spec."""
        registry.register(PaperSpec(name="p", title="Old", authors=[]))
        registry.register(PaperSpec(name="p", title="New", authors=[]))
        assert registry.get("p").title == "New"
        assert registry.count() == 1

    def test_register_rejects_empty_name(self, registry):
        """register() rejects a spec with an empty name."""
        with pytest.raises(ValueError):
            registry.register(PaperSpec(name="  ", title="T", authors=[]))


# ---------------------------------------------------------------------------
# PaperRegistry list / search
# ---------------------------------------------------------------------------
class TestPaperRegistryListSearch:
    """PaperRegistry list() and search() behaviour."""

    def test_paper_registry_list(self, registry):
        """list() returns every registered spec sorted by name."""
        registry.register(PaperSpec(name="b", title="B", authors=[]))
        registry.register(PaperSpec(name="a", title="A", authors=[]))
        registry.register(PaperSpec(name="c", title="C", authors=[]))

        specs = registry.list()
        assert [s.name for s in specs] == ["a", "b", "c"]
        assert len(specs) == 3

    def test_list_empty_when_nothing_registered(self, registry):
        """list() on a fresh registry returns an empty list."""
        assert registry.list() == []

    def test_paper_registry_search(self, registry):
        """search() matches name, title, authors, method and category."""
        registry.register(
            PaperSpec(
                name="musetalk",
                title="MuseTalk Lip Synchronization",
                authors=["Tencent Music Lyra Lab"],
                method="musetalk",
                category="digital_human",
            )
        )
        registry.register(
            PaperSpec(
                name="attention_is_all_you_need",
                title="Attention Is All You Need",
                authors=["Vaswani"],
                method="transformer",
                category="foundation",
            )
        )

        # Match by title substring.
        hits = registry.search("lip sync")
        assert len(hits) == 1
        assert hits[0].name == "musetalk"

        # Match by author.
        hits = registry.search("vaswani")
        assert len(hits) == 1
        assert hits[0].name == "attention_is_all_you_need"

        # Match by method.
        hits = registry.search("transformer")
        assert len(hits) == 1
        assert hits[0].name == "attention_is_all_you_need"

        # Case-insensitive.
        hits = registry.search("MUSE")
        assert len(hits) == 1
        assert hits[0].name == "musetalk"

        # Empty query returns everything.
        assert len(registry.search("")) == 2

        # No match.
        assert registry.search("nonexistent_paper") == []


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------
class TestPaperYamlLoad:
    """Loading bundled YAML specs into the registry."""

    def test_paper_yaml_load(self, loaded_registry):
        """The bundled musetalk.yaml loads with the expected fields."""
        spec = loaded_registry.get("musetalk")
        assert spec.name == "musetalk"
        assert spec.title.startswith("MuseTalk")
        assert spec.arxiv_id == "2501.01895"
        assert spec.github == "https://github.com/TMElyralab/MuseTalk"
        assert spec.license == "CC-BY-NC-4.0"
        assert spec.published == "2025-01"
        assert spec.authors == ["Tencent Music Lyra Lab"]

        # Integration.
        assert spec.integration_type == "node"
        assert spec.node_type == "dh_lip_sync"
        assert spec.method == "musetalk"
        assert spec.category == "digital_human"

        # Models.
        assert len(spec.models) == 1
        model = spec.models[0]
        assert model.name == "musetalk"
        assert model.source == "huggingface"
        assert model.repo == "TMElyralab/MuseTalk"
        assert model.size_gb == 2.5
        assert model.vram_gb == 4
        assert "mmcv" in model.dependencies
        assert "diffusers" in model.dependencies

        # Reproducibility.
        assert spec.seed == 42
        assert spec.deterministic is True
        assert spec.config["face_region_size"] == 256
        assert spec.config["vae"] == "sd-vae-ft-mse"

        # Compatibility.
        assert spec.min_torcha_verse == "0.3.1"
        assert spec.gpu_required is True
        assert spec.min_vram_gb == 4

    def test_load_from_dir_returns_count(self, registry):
        """load_from_dir returns the number of papers loaded."""
        n = registry.load_from_dir(PAPERS_DIR)
        assert n == 5
        assert registry.count() == 5

    def test_load_from_dir_loads_all_five(self, loaded_registry):
        """All five bundled papers are registered."""
        names = {s.name for s in loaded_registry.list()}
        assert names == {
            "musetalk",
            "liveportrait",
            "sadtalker",
            "attention_is_all_you_need",
            "rag",
        }

    def test_liveportrait_maps_to_portrait_animate(self, loaded_registry):
        """liveportrait.yaml maps to the dh_portrait_animate node."""
        spec = loaded_registry.get("liveportrait")
        assert spec.node_type == "dh_portrait_animate"
        assert spec.method == "liveportrait"
        assert spec.category == "digital_human"

    def test_sadtalker_maps_to_talking_head(self, loaded_registry):
        """sadtalker.yaml maps to the dh_talking_head node."""
        spec = loaded_registry.get("sadtalker")
        assert spec.node_type == "dh_talking_head"
        assert spec.method == "sadtalker"

    def test_load_from_dir_missing_dir(self, registry, tmp_path):
        """load_from_dir raises FileNotFoundError for a missing dir."""
        missing = tmp_path / "nope"
        with pytest.raises(FileNotFoundError):
            registry.load_from_dir(missing)


# ---------------------------------------------------------------------------
# reference_impl
# ---------------------------------------------------------------------------
class TestPaperReferenceImpl:
    """reference_impl links to community implementation collections."""

    def test_paper_yaml_with_reference_impl(self, loaded_registry):
        """attention_is_all_you_need.yaml carries all four reference impls."""
        spec = loaded_registry.get("attention_is_all_you_need")
        assert spec.reference_impl["sutskever_30"] == (
            "yoko19191/sutskever-30-implementations-zhCN/13_transformer_attention"
        )
        assert spec.reference_impl["labml"] == (
            "labmlai/annotated_deep_learning_paper_implementations/transformers/basic"
        )
        assert spec.reference_impl["karpathy"] == "karpathy/nanoGPT"
        assert spec.reference_impl["lucidrains"] == "lucidrains/x-transformers"

    def test_rag_reference_impl_points_to_29(self, loaded_registry):
        """rag.yaml's sutskever_30 reference points at the 29_rag entry."""
        spec = loaded_registry.get("rag")
        assert spec.reference_impl["sutskever_30"].endswith("29_rag")
        assert spec.reference_impl["sutskever_30"] == (
            "yoko19191/sutskever-30-implementations-zhCN/29_rag"
        )

    def test_musetalk_reference_impl_empty(self, loaded_registry):
        """musetalk.yaml ships empty reference_impl entries."""
        spec = loaded_registry.get("musetalk")
        for key in ("sutskever_30", "labml", "karpathy", "lucidrains"):
            assert key in spec.reference_impl
            assert spec.reference_impl[key] == ""


# ---------------------------------------------------------------------------
# AdapterRegistry
# ---------------------------------------------------------------------------
class TestAdapterRegistry:
    """AdapterRegistry register / get / list behaviour."""

    def test_adapter_registry(self):
        """register/get/list round-trip a PaperAdapter subclass."""
        reg = AdapterRegistry()

        class MuseTalkAdapter(PaperAdapter):
            paper_name = "musetalk"
            node_type = "dh_lip_sync"

            def load_model(self, ctx):
                return {"loaded": True, "paper": self.paper_name}

            def infer(self, model, **kwargs):
                return {"output": model["paper"], "kwargs": kwargs}

        reg.register("musetalk", MuseTalkAdapter)

        assert reg.has("musetalk")
        assert "musetalk" in reg.list()
        assert reg.count() == 1

        cls = reg.get("musetalk")
        assert cls is MuseTalkAdapter

        adapter = cls()
        assert adapter.paper_name == "musetalk"
        assert adapter.node_type == "dh_lip_sync"
        model = adapter.load_model(ctx=None)
        assert model == {"loaded": True, "paper": "musetalk"}
        out = adapter.infer(model, audio="clip")
        assert out == {"output": "musetalk", "kwargs": {"audio": "clip"}}

    def test_adapter_get_raises_for_unknown(self):
        """get() raises AdapterNotFoundError for an unknown adapter."""
        reg = AdapterRegistry()
        with pytest.raises(AdapterNotFoundError):
            reg.get("missing")

    def test_adapter_register_rejects_non_adapter(self):
        """register() rejects classes that are not PaperAdapter subclasses."""
        reg = AdapterRegistry()
        with pytest.raises(TypeError):
            reg.register("bad", object)  # type: ignore[arg-type]

    def test_adapter_register_rejects_empty_name(self):
        """register() rejects an empty name."""
        reg = AdapterRegistry()

        class A(PaperAdapter):
            def load_model(self, ctx):
                return None

            def infer(self, model, **kwargs):
                return {}

        with pytest.raises(ValueError):
            reg.register("  ", A)

    def test_paper_adapter_is_abstract(self):
        """PaperAdapter cannot be instantiated without implementing methods."""
        with pytest.raises(TypeError):
            PaperAdapter()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------
class TestPaperCli:
    """papers.cli command surface."""

    def test_paper_cli_commands(self, loaded_registry):
        """All five CLI commands work against the loaded registry."""
        # paper_list returns the bundled specs.
        papers = paper_list()
        assert len(papers) == 5
        assert all(isinstance(p, PaperSpec) for p in papers)

        # paper_info returns detailed fields.
        info = paper_info("musetalk")
        assert info["name"] == "musetalk"
        assert info["title"].startswith("MuseTalk")
        assert info["node_type"] == "dh_lip_sync"
        assert info["method"] == "musetalk"
        assert len(info["models"]) == 1
        assert info["seed"] == 42
        assert info["gpu_required"] is True

        # paper_install returns an install plan.
        plan = paper_install("musetalk")
        assert plan["paper"] == "musetalk"
        assert plan["status"] == "planned"
        assert len(plan["models"]) == 1
        assert plan["models"][0]["repo"] == "TMElyralab/MuseTalk"
        assert "mmcv" in plan["dependencies"]
        assert plan["total_size_gb"] == 2.5
        assert plan["peak_vram_gb"] == 4

        # paper_reproduce returns the reproducibility report.
        repro = paper_reproduce("musetalk")
        assert repro["paper"] == "musetalk"
        assert repro["status"] == "ok"
        assert repro["seed"] == 42
        assert repro["deterministic"] is True
        assert repro["config"]["face_region_size"] == 256

        # paper_benchmark returns a benchmark report.
        bench = paper_benchmark("musetalk")
        assert bench["paper"] == "musetalk"
        assert bench["method"] == "musetalk"
        assert bench["node_type"] == "dh_lip_sync"
        assert bench["gpu_required"] is True
        assert bench["model_count"] == 1

    def test_paper_cli_unknown_paper_raises(self, registry):
        """CLI commands raise PaperNotFoundError for unknown papers."""
        with pytest.raises(PaperNotFoundError):
            paper_info("missing")
        with pytest.raises(PaperNotFoundError):
            paper_install("missing")
        with pytest.raises(PaperNotFoundError):
            paper_reproduce("missing")
        with pytest.raises(PaperNotFoundError):
            paper_benchmark("missing")

    def test_paper_install_deduplicates_dependencies(self, loaded_registry):
        """paper_install deduplicates shared dependencies across models."""
        # sadtalker has several deps; ensure the plan lists each once.
        plan = paper_install("sadtalker")
        deps = plan["dependencies"]
        assert len(deps) == len(set(deps))
        assert "torch" in deps
