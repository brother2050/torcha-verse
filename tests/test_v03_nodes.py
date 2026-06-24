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
