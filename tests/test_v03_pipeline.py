"""Tests for v0.3.0 pipeline layer (DAG, PipelineBuilder, templates, prompt studio).

Covers DAG topological sort + cycle detection, DAG validation, the
PipelineBuilder fluent API, Pipeline YAML round-trip, the 12 built-in
templates, PromptEnhancer.enhance and SeedManager record / recall.
"""
from __future__ import annotations

import pytest

from pipeline.dag import DAG, DAGEdge, DAGNode
from pipeline.composer import Pipeline, PipelineBuilder, PipelineConfig
from pipeline.templates import TemplateRegistry
from pipeline.prompt_studio import PromptEnhancer, SeedManager


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------
class TestDAG:
    """DAG construction, topological sort, cycle detection, validation."""

    def test_topological_sort_linear(self):
        """A linear A -> B -> C graph sorts in dependency order."""
        dag = DAG()
        dag.add_node(DAGNode(id="a", node_type="text_chat"))
        dag.add_node(DAGNode(id="b", node_type="text_chat", dependencies=["a"]))
        dag.add_node(DAGNode(id="c", node_type="text_chat", dependencies=["b"]))
        order = dag.topological_sort()
        assert order.index("a") < order.index("b") < order.index("c")

    def test_topological_sort_cycle_raises(self):
        """A cyclic graph raises ValueError on topological_sort."""
        dag = DAG()
        dag.add_node(DAGNode(id="x", node_type="text_chat", dependencies=["y"]))
        dag.add_node(DAGNode(id="y", node_type="text_chat", dependencies=["x"]))
        with pytest.raises(ValueError):
            dag.topological_sort()

    def test_validate_empty_no_errors(self):
        """A well-formed DAG has no validation errors."""
        dag = DAG()
        dag.add_node(DAGNode(id="a", node_type="text_chat"))
        dag.add_node(DAGNode(id="b", node_type="text_chat", dependencies=["a"]))
        dag.add_edge(DAGEdge(from_node="a", to_node="b", output_key="text", input_key="prompt"))
        errors = dag.validate()
        assert errors == []

    def test_validate_missing_dependency(self):
        """A dependency on a non-existent node is reported."""
        dag = DAG()
        dag.add_node(DAGNode(id="a", node_type="text_chat", dependencies=["ghost"]))
        errors = dag.validate()
        assert any("ghost" in e for e in errors)

    def test_validate_dangling_edge(self):
        """An edge referencing a missing node is reported."""
        dag = DAG()
        dag.add_node(DAGNode(id="a", node_type="text_chat"))
        dag.add_edge(DAGEdge(from_node="a", to_node="ghost", output_key="text", input_key="prompt"))
        errors = dag.validate()
        assert any("ghost" in e for e in errors)

    def test_parallel_groups(self):
        """parallel_groups() partitions independent nodes into the same layer."""
        dag = DAG()
        dag.add_node(DAGNode(id="a", node_type="text_chat"))
        dag.add_node(DAGNode(id="b", node_type="text_chat"))
        dag.add_node(DAGNode(id="c", node_type="text_chat", dependencies=["a", "b"]))
        groups = dag.parallel_groups()
        assert len(groups) >= 2
        # 'a' and 'b' should be in the first group (no dependencies).
        assert set(groups[0]) == {"a", "b"}


# ---------------------------------------------------------------------------
# PipelineBuilder & Pipeline
# ---------------------------------------------------------------------------
class TestPipelineBuilder:
    """PipelineBuilder fluent API and Pipeline serialisation."""

    def test_builder_creates_pipeline(self):
        """PipelineBuilder.node().connect().build() returns a Pipeline."""
        pipeline = (
            PipelineBuilder("test_pipeline")
            .node("text_chat", id="step1", prompt="hello")
            .node("text_chat", id="step2", prompt="world")
            .connect("step1", "step2", output_key="text", input_key="prompt")
            .build()
        )
        assert isinstance(pipeline, Pipeline)
        assert pipeline.config.name == "test_pipeline"

    def test_pipeline_to_yaml_from_yaml(self, tmp_path):
        """Pipeline round-trips through YAML."""
        pipeline = (
            PipelineBuilder("yaml_test")
            .node("text_chat", id="n1", prompt="hello")
            .build()
        )
        yaml_path = tmp_path / "pipeline.yaml"
        pipeline.to_yaml(yaml_path)
        assert yaml_path.exists()

        loaded = Pipeline.from_yaml(yaml_path)
        assert loaded.config.name == "yaml_test"
        assert len(loaded.dag.node_ids) == 1

    def test_pipeline_validate(self):
        """A well-formed pipeline validates without errors."""
        pipeline = (
            PipelineBuilder("valid_pipe")
            .node("text_chat", id="a", prompt="hi")
            .node("text_chat", id="b", prompt="bye")
            .connect("a", "b", output_key="text", input_key="prompt")
            .build()
        )
        errors = pipeline.validate()
        assert errors == []

    def test_builder_requires_nodes(self):
        """build() raises ValueError when no nodes have been declared."""
        builder = PipelineBuilder("empty")
        with pytest.raises(ValueError):
            builder.build()


# ---------------------------------------------------------------------------
# TemplateRegistry
# ---------------------------------------------------------------------------
class TestTemplateRegistry:
    """Built-in template catalogue."""

    def test_list_returns_12_templates(self):
        """The registry ships with exactly 12 built-in templates."""
        reg = TemplateRegistry()
        assert reg.count() == 12
        assert len(reg.list()) == 12

    def test_get_template(self):
        """get() returns a named template."""
        reg = TemplateRegistry()
        templates = reg.list()
        name = templates[0].name
        tmpl = reg.get(name)
        assert tmpl.name == name

    def test_get_unknown_raises(self):
        """get() raises KeyError for an unknown template."""
        reg = TemplateRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent_template")

    def test_template_to_dag(self):
        """A template can be converted to a DAG."""
        reg = TemplateRegistry()
        tmpl = reg.list()[0]
        dag = tmpl.to_dag()
        assert isinstance(dag, DAG)
        assert len(dag.node_ids) >= 1

    def test_search(self):
        """search() returns matching templates."""
        reg = TemplateRegistry()
        results = reg.search("image")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# PromptEnhancer
# ---------------------------------------------------------------------------
class TestPromptEnhancer:
    """PromptEnhancer.enhance and style listing."""

    def test_enhance_appends_style_boost(self):
        """enhance() appends the style's positive vocabulary."""
        enhancer = PromptEnhancer()
        styles = enhancer.list_styles()
        assert len(styles) > 0

        enhanced = enhancer.enhance("a cat", styles[0])
        assert len(enhanced) >= len("a cat")

    def test_enhance_unknown_style_returns_prompt(self):
        """An unknown style leaves the prompt unchanged."""
        enhancer = PromptEnhancer()
        result = enhancer.enhance("a dog", "nonexistent_style")
        assert result == "a dog"

    def test_list_styles_returns_sorted(self):
        """list_styles() returns a sorted list of style names."""
        enhancer = PromptEnhancer()
        styles = enhancer.list_styles()
        assert styles == sorted(styles)


# ---------------------------------------------------------------------------
# SeedManager
# ---------------------------------------------------------------------------
class TestSeedManager:
    """SeedManager record / recall / prompt_hash."""

    def test_record_and_recall(self):
        """record() stores a seed; recall() retrieves it."""
        sm = SeedManager()
        sm.clear()
        sm.record("a beautiful sunset", seed=42, model="sdxl")
        records = sm.recall("a beautiful sunset")
        assert len(records) == 1
        assert records[0]["seed"] == 42
        assert records[0]["model"] == "sdxl"

    def test_recall_multiple_records(self):
        """Multiple records for the same prompt are all returned."""
        sm = SeedManager()
        sm.clear()
        sm.record("same prompt", seed=1, model="a")
        sm.record("same prompt", seed=2, model="b")
        records = sm.recall("same prompt")
        assert len(records) == 2

    def test_prompt_hash_is_hex(self):
        """prompt_hash() returns a 64-char hex digest."""
        h = SeedManager.prompt_hash("test prompt")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_recall_unknown_prompt_empty(self):
        """recall() for an unknown prompt returns an empty list."""
        sm = SeedManager()
        sm.clear()
        records = sm.recall("never recorded prompt")
        assert records == []

    def test_random_seed_non_negative(self):
        """random_seed() returns a non-negative integer."""
        sm = SeedManager()
        seed = sm.random_seed()
        assert isinstance(seed, int)
        assert seed >= 0
