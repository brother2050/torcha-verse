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
# Canvas connection type validation
# ---------------------------------------------------------------------------
class TestCanvasTypeValidation:
    """connect() and validate() port-type and structural validation."""

    def test_connect_returns_connection_on_success(self):
        """A type-compatible connection returns a CanvasConnection."""
        from canvas.canvas import CanvasConnection

        canvas = Canvas("test")
        canvas.add_node("text_chat", id="a", prompt="hello")
        canvas.add_node("text_chat", id="b", prompt="world")
        conn = canvas.connect("a", "text", "b", "prompt")
        assert isinstance(conn, CanvasConnection)
        assert conn.from_node == "a"
        assert conn.to_node == "b"

    def test_connect_type_mismatch_returns_error(self):
        """Connecting IMAGE output to TEXT input raises ValueError."""
        canvas = Canvas("test")
        canvas.add_node("image_txt2img", id="img", prompt="cat")
        canvas.add_node("audio_tts", id="tts", text="hello", voice="v1")
        with pytest.raises(ValueError) as exc_info:
            canvas.connect("img", "image", "tts", "text")
        msg = str(exc_info.value)
        assert "类型不匹配" in msg or "不兼容" in msg
        # The connection must not have been created.
        assert len(canvas.list_connections()) == 0

    def test_connect_compatible_text_to_prompt(self):
        """TEXT output can connect to PROMPT input."""
        canvas = Canvas("test")
        canvas.add_node("text_chat", id="a", prompt="hello")
        canvas.add_node("image_txt2img", id="b", prompt="cat")
        conn = canvas.connect("a", "text", "b", "prompt")
        # TEXT -> PROMPT is compatible.
        assert not isinstance(conn, str)
        assert len(canvas.list_connections()) == 1

    def test_connect_image_to_image_compatible(self):
        """IMAGE output can connect to IMAGE input."""
        canvas = Canvas("test")
        canvas.add_node("image_txt2img", id="a", prompt="cat")
        canvas.add_node("image_upscale", id="b")
        conn = canvas.connect("a", "image", "b", "image")
        assert not isinstance(conn, str)
        assert len(canvas.list_connections()) == 1

    def test_connect_unknown_output_port_returns_error(self):
        """Connecting from a non-existent output port raises ValueError."""
        canvas = Canvas("test")
        canvas.add_node("text_chat", id="a", prompt="hello")
        canvas.add_node("text_chat", id="b", prompt="world")
        with pytest.raises(ValueError) as exc_info:
            canvas.connect("a", "nonexistent_port", "b", "prompt")
        assert "声明输出" in str(exc_info.value)
        assert len(canvas.list_connections()) == 0

    def test_connect_unknown_input_port_returns_error(self):
        """Connecting to a non-existent input port raises ValueError."""
        canvas = Canvas("test")
        canvas.add_node("text_chat", id="a", prompt="hello")
        canvas.add_node("text_chat", id="b", prompt="world")
        with pytest.raises(ValueError) as exc_info:
            canvas.connect("a", "text", "b", "nonexistent_port")
        assert "声明输入" in str(exc_info.value)
        assert len(canvas.list_connections()) == 0

    def test_connect_one_to_one_input_rejects_second(self):
        """An input port may receive at most one incoming connection."""
        canvas = Canvas("test")
        canvas.add_node("text_chat", id="a", prompt="hello")
        canvas.add_node("text_chat", id="b", prompt="world")
        canvas.add_node("text_chat", id="c", prompt="foo")
        # First connection to b.prompt is fine.
        conn1 = canvas.connect("a", "text", "b", "prompt")
        assert not isinstance(conn1, str)
        # Second connection to the same input should fail.
        with pytest.raises(ValueError) as exc_info:
            canvas.connect("c", "text", "b", "prompt")
        assert "already has an incoming" in str(exc_info.value)
        assert len(canvas.list_connections()) == 1

    def test_connect_duplicate_returns_error(self):
        """A duplicate connection raises ValueError."""
        canvas = Canvas("test")
        canvas.add_node("text_chat", id="a", prompt="hello")
        canvas.add_node("text_chat", id="b", prompt="world")
        canvas.connect("a", "text", "b", "prompt")
        with pytest.raises(ValueError) as exc_info:
            canvas.connect("a", "text", "b", "prompt")
        assert "重复" in str(exc_info.value)
        assert len(canvas.list_connections()) == 1

    def test_connect_self_loop_returns_error(self):
        """A self-loop connection raises ValueError."""
        canvas = Canvas("test")
        canvas.add_node("text_chat", id="a", prompt="hello")
        with pytest.raises(ValueError) as exc_info:
            canvas.connect("a", "text", "a", "prompt")
        assert "自环" in str(exc_info.value)
        assert len(canvas.list_connections()) == 0

    def test_connect_cycle_detection(self):
        """Adding an edge that closes a cycle is rejected."""
        canvas = Canvas("test")
        canvas.add_node("text_chat", id="a", prompt="1")
        canvas.add_node("text_chat", id="b", prompt="2")
        canvas.add_node("text_chat", id="c", prompt="3")
        # a -> b -> c  (all compatible: TEXT -> PROMPT)
        assert not isinstance(canvas.connect("a", "text", "b", "prompt"), str)
        assert not isinstance(canvas.connect("b", "text", "c", "prompt"), str)
        # c -> a would close the cycle a -> b -> c -> a.
        with pytest.raises(ValueError) as exc_info:
            canvas.connect("c", "text", "a", "prompt")
        assert "环" in str(exc_info.value)
        assert len(canvas.list_connections()) == 2

    def test_connect_missing_node_returns_error(self):
        """Connecting to a non-existent node raises ValueError."""
        canvas = Canvas("test")
        canvas.add_node("text_chat", id="a", prompt="hello")
        with pytest.raises(ValueError) as exc_info:
            canvas.connect("a", "text", "nonexistent", "prompt")
        assert "does not exist" in str(exc_info.value)

    def test_validate_detects_type_mismatch(self):
        """validate() reports type mismatches for pre-existing connections."""
        canvas = Canvas("test")
        canvas.add_node("image_txt2img", id="img", prompt="cat")
        canvas.add_node("audio_tts", id="tts", text="hello", voice="v1")
        # Manually inject an incompatible connection (bypassing connect()).
        from canvas.canvas import CanvasConnection
        from uuid import uuid4
        canvas._state.connections.append(
            CanvasConnection(
                id=str(uuid4()),
                from_node="img",
                from_port="image",
                to_node="tts",
                to_port="text",
            )
        )
        errors = canvas.validate()
        assert any("Type mismatch" in e or "not compatible" in e for e in errors)

    def test_validate_clean_canvas_no_errors(self):
        """A canvas with only compatible connections has no type errors."""
        canvas = Canvas("test")
        canvas.add_node("text_chat", id="a", prompt="hello")
        canvas.add_node("text_chat", id="b", prompt="world")
        canvas.connect("a", "text", "b", "prompt")
        errors = canvas.validate()
        assert errors == []


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
