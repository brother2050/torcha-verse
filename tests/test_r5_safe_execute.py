"""_safe_execute 异常包装测试。"""
import pytest
from nodes.base import BaseNode, NodeContext, NodeSpec

class TestSafeExecute:
    """_safe_execute 异常包装测试。"""

    def test_safe_execute_success(self):
        """正常执行返回结果。"""
        class DummyNode(BaseNode):
            spec = NodeSpec(type="dummy", name="Dummy")
            def execute(self, ctx, **inputs):
                return {"output": "ok"}

        node = DummyNode()
        ctx = NodeContext()
        result = node._safe_execute(ctx, prompt="test")
        assert result == {"output": "ok"}

    def test_safe_execute_os_error_logged_and_reraised(self):
        """OSError 被记录后重新抛出。"""
        class FailingNode(BaseNode):
            spec = NodeSpec(type="failing", name="Failing")
            def execute(self, ctx, **inputs):
                raise OSError("disk full")

        node = FailingNode()
        ctx = NodeContext()
        with pytest.raises(OSError):
            node._safe_execute(ctx, prompt="test")

    def test_safe_execute_runtime_error_logged_and_reraised(self):
        """RuntimeError 被记录后重新抛出。"""
        class FailingNode(BaseNode):
            spec = NodeSpec(type="failing", name="Failing")
            def execute(self, ctx, **inputs):
                raise RuntimeError("model load failed")

        node = FailingNode()
        ctx = NodeContext()
        with pytest.raises(RuntimeError):
            node._safe_execute(ctx, prompt="test")

    def test_safe_execute_memory_error_logged_and_reraised(self):
        """MemoryError 被记录后重新抛出。"""
        class FailingNode(BaseNode):
            spec = NodeSpec(type="failing", name="Failing")
            def execute(self, ctx, **inputs):
                raise MemoryError("OOM")

        node = FailingNode()
        ctx = NodeContext()
        with pytest.raises(MemoryError):
            node._safe_execute(ctx, prompt="test")

    def test_safe_execute_value_error_not_caught(self):
        """ValueError 不被 _safe_execute 捕获（直接传播）。"""
        class FailingNode(BaseNode):
            spec = NodeSpec(type="failing", name="Failing")
            def execute(self, ctx, **inputs):
                raise ValueError("bad input")

        node = FailingNode()
        ctx = NodeContext()
        with pytest.raises(ValueError):
            node._safe_execute(ctx, prompt="test")
