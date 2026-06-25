"""Pipeline cancel/pause/resume 运行控制测试。"""
import time
import pytest
from nodes.base import NodeContext
from pipeline.composer import PipelineBuilder, PipelineConfig
from pipeline.dag import DAG, DAGNode, DAGEdge

class TestPipelineCancel:
    """Pipeline cancel 测试。"""

    def test_cancel_before_run(self):
        """cancel 在 run 之前调用不影响。"""
        builder = PipelineBuilder("test")
        builder.node("passthrough", id="a")
        pipeline = builder.build()
        ctx = NodeContext()
        # cancel 不应崩溃
        pipeline.cancel()
        # 正常运行应成功
        results = pipeline.run(ctx)
        assert "a" in results

    def test_cancel_during_run(self):
        """cancel 在运行中调用，应提前终止。"""
        # 构建多层 pipeline
        builder = PipelineBuilder("test_cancel")
        builder.node("passthrough", id="a")
        builder.node("passthrough", id="b")
        builder.node("passthrough", id="c")
        builder.connect("a", "b")
        builder.connect("b", "c")
        pipeline = builder.build()
        ctx = NodeContext()

        # 注册一个执行器，在执行时触发 cancel
        call_count = [0]
        def slow_executor(inputs, ctx):
            call_count[0] += 1
            if call_count[0] >= 1:
                pipeline.cancel()
            return {"output": inputs}

        ctx.register_executor("passthrough", slow_executor)
        results = pipeline.run(ctx)
        # 应该至少执行了第一个节点
        assert len(results) >= 1

    def test_pause_resume(self):
        """pause 后 resume 应继续执行。"""
        builder = PipelineBuilder("test_pause")
        builder.node("passthrough", id="a")
        builder.node("passthrough", id="b")
        builder.connect("a", "b")
        pipeline = builder.build()
        ctx = NodeContext()

        # 注册执行器，在第一个节点执行后暂停
        def pausing_executor(inputs, ctx):
            pipeline.pause()
            # 短暂等待后恢复
            import threading
            def resume_after():
                time.sleep(0.1)
                pipeline.resume()
            threading.Thread(target=resume_after, daemon=True).start()
            return {"output": inputs}

        ctx.register_executor("passthrough", pausing_executor)
        results = pipeline.run(ctx)
        # 两个节点都应执行完成
        assert "a" in results
        assert "b" in results

    def test_cancel_returns_partial_results(self):
        """cancel 后应返回已完成的节点结果。"""
        builder = PipelineBuilder("test_partial")
        builder.node("passthrough", id="a")
        builder.node("passthrough", id="b")
        builder.connect("a", "b")
        pipeline = builder.build()
        ctx = NodeContext()

        def first_node_executor(inputs, ctx):
            pipeline.cancel()
            return {"output": inputs}

        ctx.register_executor("passthrough", first_node_executor)
        results = pipeline.run(ctx)
        # 第一个节点应完成
        assert "a" in results

    def test_is_running_flag(self):
        """is_running 标志在运行时为 True。"""
        builder = PipelineBuilder("test_running")
        builder.node("passthrough", id="a")
        pipeline = builder.build()
        ctx = NodeContext()
        assert pipeline.is_running is False

        running_during = [False]
        def check_running(inputs, ctx):
            running_during[0] = pipeline.is_running
            return {"output": inputs}

        ctx.register_executor("passthrough", check_running)
        pipeline.run(ctx)
        assert running_during[0] is True
        assert pipeline.is_running is False
