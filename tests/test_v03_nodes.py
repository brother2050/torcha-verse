"""Tests for v0.3.0 node system (NodeRegistry + concrete nodes).

Verifies that the node catalogue has at least 15 registered nodes, that
``ImageTxt2ImgNode.validate_inputs`` enforces the width/height range,
that ``estimate_resources`` returns the expected fields, and that
``execute`` returns the correct output keys.
"""
from __future__ import annotations

import pytest

# Importing the nodes package triggers every @register_node decorator.
import nodes  # noqa: F401
from nodes import NodeRegistry
from nodes.base import NodeContext, NodeSpec
from nodes.image import ImageTxt2ImgNode
from nodes.text import TextNode
from nodes.type_system import TypeSystem, is_optional, unwrap_optional


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def registry():
    """Return a NodeRegistry backed by the global ModuleBus."""
    return NodeRegistry()


@pytest.fixture()
def ctx():
    """Return a minimal NodeContext for execute() calls."""
    return NodeContext()


# ---------------------------------------------------------------------------
# NodeRegistry
# ---------------------------------------------------------------------------
class TestNodeRegistry:
    """NodeRegistry discovery surface."""

    def test_list_returns_at_least_15_nodes(self, registry):
        """The catalogue contains >= 15 registered nodes."""
        specs = registry.list()
        assert len(specs) >= 15

    def test_list_returns_node_specs(self, registry):
        """Every entry in list() is a NodeSpec with a non-empty type."""
        specs = registry.list()
        for spec in specs:
            assert isinstance(spec, NodeSpec)
            assert spec.type
            assert spec.name

    def test_get_image_txt2img(self, registry):
        """get('image_txt2img') returns an ImageTxt2ImgNode instance."""
        node = registry.get("image_txt2img")
        assert isinstance(node, ImageTxt2ImgNode)

    def test_get_text_chat(self, registry):
        """get('text_chat') returns a TextNode instance."""
        node = registry.get("text_chat")
        assert isinstance(node, TextNode)

    def test_get_unknown_raises(self, registry):
        """get() raises KeyError for an unregistered node type."""
        with pytest.raises(KeyError):
            registry.get("nonexistent_node_type")

    def test_search(self, registry):
        """search() finds nodes by substring in type/name/description/tags."""
        results = registry.search("image")
        assert len(results) >= 1
        # At least one result should have "image" in its type.
        assert any("image" in s.type for s in results)

    def test_spec_inputs_outputs_are_strings(self, registry):
        """Every NodeSpec input/output value is a type string."""
        specs = registry.list()
        for spec in specs:
            for name, type_str in spec.inputs.items():
                assert isinstance(type_str, str), (
                    "Input {!r} of {!r} should be a str, got {!r}".format(
                        name, spec.type, type(type_str)
                    )
                )
                assert type_str, (
                    "Input {!r} of {!r} has an empty type string".format(
                        name, spec.type
                    )
                )
            for name, type_str in spec.outputs.items():
                assert isinstance(type_str, str), (
                    "Output {!r} of {!r} should be a str, got {!r}".format(
                        name, spec.type, type(type_str)
                    )
                )
                assert type_str, (
                    "Output {!r} of {!r} has an empty type string".format(
                        name, spec.type
                    )
                )

    def test_image_txt2img_uses_expected_type_strings(self, registry):
        """image_txt2img declares IMAGE/PROMPT/INT/FLOAT/SEED type strings."""
        spec = registry.get("image_txt2img").spec
        assert spec.inputs["prompt"] == "PROMPT"
        assert spec.inputs["width"] == "INT"
        assert spec.inputs["guidance_scale"] == "FLOAT"
        assert spec.inputs["seed"] == "Optional[SEED]"
        assert spec.outputs["image"] == "IMAGE"
        assert spec.outputs["seed"] == "SEED"


# ---------------------------------------------------------------------------
# TypeSystem
# ---------------------------------------------------------------------------
class TestTypeSystem:
    """TypeSystem compatibility matrix and helpers."""

    def test_self_compatibility(self):
        """A type is always compatible with itself."""
        for type_str in TypeSystem.all_types():
            assert TypeSystem.is_compatible(type_str, type_str)

    def test_image_to_latent(self):
        """IMAGE outputs can connect to LATENT inputs."""
        assert TypeSystem.is_compatible("IMAGE", "LATENT")

    def test_text_to_image_incompatible(self):
        """TEXT outputs cannot connect to IMAGE inputs."""
        assert not TypeSystem.is_compatible("TEXT", "IMAGE")

    def test_text_to_prompt(self):
        """TEXT outputs can connect to PROMPT inputs."""
        assert TypeSystem.is_compatible("TEXT", "PROMPT")

    def test_prompt_to_text(self):
        """PROMPT outputs can connect to TEXT inputs."""
        assert TypeSystem.is_compatible("PROMPT", "TEXT")

    def test_int_to_float_and_seed(self):
        """INT outputs can connect to FLOAT and SEED inputs."""
        assert TypeSystem.is_compatible("INT", "FLOAT")
        assert TypeSystem.is_compatible("INT", "SEED")

    def test_float_to_int_incompatible(self):
        """FLOAT outputs cannot connect to INT inputs."""
        assert not TypeSystem.is_compatible("FLOAT", "INT")

    def test_character_to_asset_ref(self):
        """CHARACTER outputs can connect to ASSET_REF inputs."""
        assert TypeSystem.is_compatible("CHARACTER", "ASSET_REF")

    def test_list_compatibility(self):
        """LIST[T] is compatible with LIST[T] and LIST[X] -> X is checked."""
        assert TypeSystem.is_compatible("LIST[IMAGE]", "LIST[IMAGE]")
        # LIST[IMAGE] -> IMAGE: inner IMAGE -> IMAGE is compatible.
        assert TypeSystem.is_compatible("LIST[IMAGE]", "IMAGE")
        # LIST[IMAGE] -> LIST[VIDEO]: inner IMAGE -> VIDEO is incompatible.
        assert not TypeSystem.is_compatible("LIST[IMAGE]", "LIST[VIDEO]")

    def test_optional_wrapper(self):
        """Optional[T] unwraps to T for compatibility checks."""
        assert is_optional("Optional[SEED]")
        assert not is_optional("SEED")
        assert unwrap_optional("Optional[SEED]") == "SEED"
        assert unwrap_optional("SEED") == "SEED"
        assert TypeSystem.is_compatible("SEED", "Optional[SEED]")
        assert TypeSystem.is_compatible("INT", "Optional[SEED]")

    def test_compatible_inputs_includes_self(self):
        """compatible_inputs() always includes the output type itself."""
        result = TypeSystem.compatible_inputs("IMAGE")
        assert "IMAGE" in result
        assert "LATENT" in result

    def test_all_types_non_empty(self):
        """all_types() returns a non-empty list of registered types."""
        types = TypeSystem.all_types()
        assert len(types) >= 10
        assert "IMAGE" in types
        assert "TEXT" in types


# ---------------------------------------------------------------------------
# ImageTxt2ImgNode validation
# ---------------------------------------------------------------------------
class TestImageTxt2ImgValidation:
    """validate_inputs() dimension and parameter checks."""

    def test_valid_inputs_no_errors(self):
        """A well-formed input dict produces no errors."""
        node = ImageTxt2ImgNode()
        errors = node.validate_inputs({
            "prompt": "a cat",
            "width": 512,
            "height": 512,
            "steps": 20,
            "guidance_scale": 7.5,
        })
        assert errors == []

    def test_width_too_small(self):
        """Width below 64 is rejected."""
        node = ImageTxt2ImgNode()
        errors = node.validate_inputs({
            "prompt": "test",
            "width": 32,
            "height": 512,
            "steps": 10,
            "guidance_scale": 7.0,
        })
        assert any("width" in e for e in errors)

    def test_width_too_large(self):
        """Width above 2048 is rejected."""
        node = ImageTxt2ImgNode()
        errors = node.validate_inputs({
            "prompt": "test",
            "width": 4096,
            "height": 512,
            "steps": 10,
            "guidance_scale": 7.0,
        })
        assert any("width" in e for e in errors)

    def test_height_out_of_range(self):
        """Height below 64 is rejected."""
        node = ImageTxt2ImgNode()
        errors = node.validate_inputs({
            "prompt": "test",
            "width": 512,
            "height": 16,
            "steps": 10,
            "guidance_scale": 7.0,
        })
        assert any("height" in e for e in errors)

    def test_steps_must_be_positive(self):
        """Non-positive steps are rejected."""
        node = ImageTxt2ImgNode()
        errors = node.validate_inputs({
            "prompt": "test",
            "width": 512,
            "height": 512,
            "steps": 0,
            "guidance_scale": 7.0,
        })
        assert any("steps" in e for e in errors)

    def test_empty_prompt_rejected(self):
        """An empty prompt string is rejected."""
        node = ImageTxt2ImgNode()
        errors = node.validate_inputs({
            "prompt": "   ",
            "width": 512,
            "height": 512,
            "steps": 10,
            "guidance_scale": 7.0,
        })
        assert any("prompt" in e for e in errors)


# ---------------------------------------------------------------------------
# estimate_resources
# ---------------------------------------------------------------------------
class TestEstimateResources:
    """estimate_resources() returns the three expected fields."""

    def test_image_node_resource_fields(self):
        """ImageTxt2ImgNode.estimate_resources returns vram/ram/time."""
        node = ImageTxt2ImgNode()
        est = node.estimate_resources({
            "width": 512,
            "height": 512,
            "steps": 20,
        })
        assert "vram_gb" in est
        assert "ram_gb" in est
        assert "time_s" in est
        assert est["vram_gb"] > 0
        assert est["ram_gb"] > 0
        assert est["time_s"] > 0

    def test_text_node_resource_fields(self):
        """TextNode.estimate_resources returns vram/ram/time."""
        node = TextNode()
        est = node.estimate_resources({"max_tokens": 256})
        assert "vram_gb" in est
        assert "ram_gb" in est
        assert "time_s" in est
        assert est["vram_gb"] > 0

    def test_resources_scale_with_resolution(self):
        """Higher resolution yields higher VRAM estimate."""
        node = ImageTxt2ImgNode()
        small = node.estimate_resources({
            "width": 256, "height": 256, "steps": 20,
        })
        large = node.estimate_resources({
            "width": 1024, "height": 1024, "steps": 20,
        })
        assert large["vram_gb"] > small["vram_gb"]
        assert large["time_s"] > small["time_s"]


# ---------------------------------------------------------------------------
# execute (placeholder)
# ---------------------------------------------------------------------------
class TestNodeExecute:
    """execute() returns the correct output keys (placeholder data)."""

    def test_image_txt2img_execute_returns_image_and_seed(self, ctx):
        """ImageTxt2ImgNode.execute returns 'image' and 'seed' keys."""
        node = ImageTxt2ImgNode()
        result = node.execute(
            ctx,
            prompt="a beautiful sunset",
            width=512,
            height=512,
            steps=20,
            guidance_scale=7.5,
        )
        assert "image" in result
        assert "seed" in result
        assert isinstance(result["seed"], int)

    def test_text_chat_execute_returns_text_and_usage(self, ctx):
        """TextNode.execute returns 'text' and 'usage' keys."""
        node = TextNode()
        result = node.execute(
            ctx,
            prompt="Hello, world!",
            max_tokens=128,
            temperature=0.7,
        )
        assert "text" in result
        assert "usage" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["usage"], dict)
