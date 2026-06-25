"""Canvas.merge 合并测试。"""
import pytest
from canvas.canvas import Canvas

class TestCanvasMerge:
    """Canvas.merge 测试。"""

    def test_merge_empty_canvases(self):
        """合并两个空画布。"""
        c1 = Canvas("canvas1")
        c2 = Canvas("canvas2")
        merged = c1.merge(c2)
        assert len(merged.list_nodes()) == 0

    def test_merge_disjoint_nodes(self):
        """合并不相交的节点。"""
        c1 = Canvas("c1")
        c1.add_node("image_txt2img", id="node_a")
        c2 = Canvas("c2")
        c2.add_node("image_upscale", id="node_b")
        merged = c1.merge(c2)
        assert len(merged.list_nodes()) == 2
        node_ids = {n.id for n in merged.list_nodes()}
        assert "node_a" in node_ids
        assert "merged_node_b" in node_ids

    def test_merge_preserves_connections(self):
        """合并后保留连接（带前缀重映射）。"""
        c1 = Canvas("c1")
        c1.add_node("image_txt2img", id="src")
        c2 = Canvas("c2")
        c2.add_node("image_txt2img", id="a")
        c2.add_node("image_upscale", id="b")
        c2.connect("a", "image", "b", "image")
        merged = c1.merge(c2)
        # 合并后的连接应使用前缀
        conns = merged.list_connections()
        assert len(conns) == 1
        assert conns[0].from_node == "merged_a"
        assert conns[0].to_node == "merged_b"

    def test_merge_id_collision_avoided(self):
        """合并相同 id 的画布不冲突。"""
        c1 = Canvas("c1")
        c1.add_node("image_txt2img", id="shared")
        c2 = Canvas("c2")
        c2.add_node("image_upscale", id="shared")
        merged = c1.merge(c2)
        assert len(merged.list_nodes()) == 2

    def test_merge_does_not_modify_originals(self):
        """合并不修改原始画布。"""
        c1 = Canvas("c1")
        c1.add_node("image_txt2img", id="a")
        c2 = Canvas("c2")
        c2.add_node("image_upscale", id="b")
        c1.merge(c2)
        # 原始画布不变
        assert len(c1.list_nodes()) == 1
        assert len(c2.list_nodes()) == 1
