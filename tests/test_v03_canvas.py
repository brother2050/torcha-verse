"""Tests for v0.3.0 canvas layer (Canvas, CanvasHistory, AutoDirector, ShareManager).

Covers Canvas add/remove/connect, Canvas-to-DAG conversion,
CanvasHistory commit/log, AutoDirector.generate and ShareManager.create_link.
"""
from __future__ import annotations

import pytest

from pipeline.dag import DAG
from pipeline.templates import TemplateRegistry
from canvas.canvas import Canvas, CanvasState
from canvas.versioning import CanvasHistory
from canvas.autodirector import AutoDirector
from canvas.sharing import ShareManager


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
class TestCanvas:
    """Canvas node/connection operations and DAG conversion."""

    def test_add_node(self):
        """add_node() returns a CanvasNode and the node appears in list_nodes."""
        canvas = Canvas("test")
        node = canvas.add_node("text_chat", id="n1", prompt="hello")
        assert node.id == "n1"
        assert node.type == "text_chat"
        assert len(canvas.list_nodes()) == 1

    def test_remove_node(self):
        """remove_node() removes the node and its connections."""
        canvas = Canvas("test")
        canvas.add_node("text_chat", id="n1", prompt="hello")
        assert canvas.remove_node("n1") is True
        assert len(canvas.list_nodes()) == 0
        assert canvas.remove_node("n1") is False

    def test_connect(self):
        """connect() creates a connection between two nodes."""
        canvas = Canvas("test")
        canvas.add_node("text_chat", id="a", prompt="hello")
        canvas.add_node("text_chat", id="b", prompt="world")
        conn = canvas.connect("a", "text", "b", "prompt")
        assert conn.from_node == "a"
        assert conn.to_node == "b"
        assert len(canvas.list_connections()) == 1

    def test_to_dag(self):
        """Canvas.state.to_dag() returns a pipeline.dag.DAG."""
        canvas = Canvas("test")
        canvas.add_node("text_chat", id="a", prompt="hello")
        canvas.add_node("text_chat", id="b", prompt="world")
        canvas.connect("a", "text", "b", "prompt")
        dag = canvas.state.to_dag()
        assert isinstance(dag, DAG)
        assert len(dag.node_ids) == 2
        # 'b' should depend on 'a'.
        order = dag.topological_sort()
        assert order.index("a") < order.index("b")

    def test_remove_node_cleans_connections(self):
        """Removing a node also removes connections referencing it."""
        canvas = Canvas("test")
        canvas.add_node("text_chat", id="a")
        canvas.add_node("text_chat", id="b")
        canvas.connect("a", "text", "b", "prompt")
        canvas.remove_node("a")
        assert len(canvas.list_connections()) == 0

    def test_canvas_name_validation(self):
        """An empty canvas name raises ValueError."""
        with pytest.raises(ValueError):
            Canvas("")


# ---------------------------------------------------------------------------
# CanvasHistory
# ---------------------------------------------------------------------------
class TestCanvasHistory:
    """Version history commit and log."""

    def test_commit_and_log(self):
        """commit() records a version; log() returns the history."""
        canvas = Canvas("history_test")
        history = CanvasHistory(canvas)
        canvas.add_node("text_chat", id="n1", prompt="v1")
        v1 = history.commit("first commit")
        assert v1.message == "first commit"
        assert v1.parent_id is None

        canvas.add_node("text_chat", id="n2", prompt="v2")
        v2 = history.commit("second commit")
        assert v2.parent_id == v1.version_id

        log = history.log()
        assert len(log) == 2
        assert log[0].version_id == v1.version_id
        assert log[1].version_id == v2.version_id

    def test_version_count(self):
        """version_count tracks the number of commits."""
        canvas = Canvas("count_test")
        history = CanvasHistory(canvas)
        assert history.version_count == 0
        history.commit("c1")
        assert history.version_count == 1
        history.commit("c2")
        assert history.version_count == 2

    def test_checkout(self):
        """checkout() returns the state of a specific version."""
        canvas = Canvas("checkout_test")
        history = CanvasHistory(canvas)
        canvas.add_node("text_chat", id="n1")
        v1 = history.commit("v1")
        canvas.add_node("text_chat", id="n2")
        history.commit("v2")

        state = history.checkout(v1.version_id)
        assert len(state.nodes) == 1

    def test_checkout_unknown_raises(self):
        """checkout() raises KeyError for an unknown version id."""
        canvas = Canvas("unknown_test")
        history = CanvasHistory(canvas)
        with pytest.raises(KeyError):
            history.checkout("nonexistent")


# ---------------------------------------------------------------------------
# AutoDirector
# ---------------------------------------------------------------------------
class TestAutoDirector:
    """AutoDirector intelligent canvas generation."""

    def test_generate_returns_canvas(self):
        """generate() returns a Canvas with nodes from a template."""
        registry = TemplateRegistry()
        director = AutoDirector(registry)
        canvas = director.generate("a cat playing piano")
        assert isinstance(canvas, Canvas)
        assert len(canvas.list_nodes()) >= 1

    def test_suggest_template(self):
        """suggest_template() returns a non-empty template name."""
        registry = TemplateRegistry()
        director = AutoDirector(registry)
        name = director.suggest_template("anime short film")
        assert isinstance(name, str)
        assert len(name) > 0

    def test_generate_with_custom_callback(self):
        """generate() works with a custom LLM callback."""
        registry = TemplateRegistry()

        def my_callback(prompt: str) -> str:
            return '{"prompt": "custom filled prompt"}'

        director = AutoDirector(registry, llm_callback=my_callback)
        canvas = director.generate("a beautiful landscape")
        assert isinstance(canvas, Canvas)


# ---------------------------------------------------------------------------
# ShareManager
# ---------------------------------------------------------------------------
class TestShareManager:
    """Share link creation and resolution."""

    def test_create_link(self):
        """create_link() returns a ShareLink for the canvas."""
        canvas = Canvas("share_test")
        canvas.add_node("text_chat", id="n1", prompt="hello")
        sm = ShareManager()
        link = sm.create_link(canvas)
        assert link.link_id
        assert link.view_only is True

    def test_resolve_link(self):
        """resolve_link() returns the canvas state."""
        canvas = Canvas("resolve_test")
        canvas.add_node("text_chat", id="n1", prompt="hello")
        sm = ShareManager()
        link = sm.create_link(canvas)
        state = sm.resolve_link(link.link_id)
        assert state is not None
        assert len(state.nodes) == 1

    def test_resolve_unknown_link_returns_none(self):
        """resolve_link() returns None for an unknown link id."""
        sm = ShareManager()
        assert sm.resolve_link("nonexistent-link-id") is None

    def test_revoke_link(self):
        """revoke_link() removes the link."""
        canvas = Canvas("revoke_test")
        sm = ShareManager()
        link = sm.create_link(canvas)
        assert sm.revoke_link(link.link_id) is True
        assert sm.resolve_link(link.link_id) is None
        assert sm.revoke_link(link.link_id) is False

    def test_create_link_with_password(self):
        """A password-protected link requires the correct password."""
        canvas = Canvas("pwd_test")
        sm = ShareManager()
        link = sm.create_link(canvas, password="secret")
        # Wrong password -> None.
        assert sm.resolve_link(link.link_id, password="wrong") is None
        # Correct password -> state.
        state = sm.resolve_link(link.link_id, password="secret")
        assert state is not None
