"""Pipeline.run() 的集成测试 (S2-1)。

覆盖以下场景:

* 正常执行多节点管道
* 无执行器时的 passthrough 行为
* 节点失败时的部分结果保留 (R0-7)
* 空 DAG 返回空 dict
* 并发执行验证
"""

from __future__ import annotations

import threading
import time

import pytest

from nodes.base import NodeContext
from pipeline.composer import Pipeline, PipelineBuilder, PipelineConfig
from pipeline.dag import DAG


# ---------------------------------------------------------------------------
# 1. 正常执行多节点管道
# ---------------------------------------------------------------------------
def test_pipeline_run_normal(pipeline_ctx):
    """正常执行多节点管道,验证结果包含所有节点输出。"""
    pipeline = (
        PipelineBuilder("test_normal")
        .node("text_chat", id="a", prompt="hello")
        .node("text_chat", id="b")
        .connect("a", "b", output_key="text", input_key="prompt")
        .build()
    )
    results = pipeline.run(pipeline_ctx)

    assert set(results.keys()) == {"a", "b"}
    assert "text" in results["a"]
    assert "text" in results["b"]
    # 节点 b 的 prompt 输入应来自节点 a 的 text 输出
    assert pipeline_ctx.get_output("a", "text") == results["a"]["text"]


# ---------------------------------------------------------------------------
# 2. 无执行器时 passthrough 行为
# ---------------------------------------------------------------------------
def test_pipeline_run_passthrough():
    """无执行器且 strict_mode=False 时,节点回退到 passthrough(返回合并输入)。"""
    ctx = NodeContext()  # 无 executors,strict_mode 默认 False
    # 使用未注册的节点类型,确保 resolve_executor 返回 None 触发 passthrough。
    pipeline = (
        PipelineBuilder("test_passthrough")
        .node("unknown_type", id="a", prompt="hello", temperature=0.7)
        .build()
    )
    results = pipeline.run(ctx)

    # passthrough 返回合并后的输入字典
    assert results["a"]["prompt"] == "hello"
    assert results["a"]["temperature"] == 0.7


# ---------------------------------------------------------------------------
# 3. 节点失败时部分结果保留 (R0-7)
# ---------------------------------------------------------------------------
def test_pipeline_run_node_failure():
    """同层中一个节点失败时,其余已完成节点的结果应保留在 ctx 输出存储中。"""
    ctx = NodeContext(max_workers=4, strict_mode=False)

    def _ok_executor(inputs, ctx):
        return {"output": "ok_result"}

    def _bad_executor(inputs, ctx):
        raise RuntimeError("intentional failure")

    ctx.register_executor("ok_type", _ok_executor)
    ctx.register_executor("bad_type", _bad_executor)

    pipeline = (
        PipelineBuilder("test_failure")
        .node("ok_type", id="ok")
        .node("bad_type", id="bad")
        .build()
    )

    with pytest.raises(RuntimeError, match="failed"):
        pipeline.run(ctx)

    # 部分结果应保留在 ctx 的输出存储中
    assert ctx.has_output("ok")
    assert ctx.get_output("ok") == {"output": "ok_result"}


# ---------------------------------------------------------------------------
# 4. 空 DAG 返回空 dict
# ---------------------------------------------------------------------------
def test_pipeline_run_empty_dag():
    """空 DAG(无节点)执行后返回空字典。"""
    config = PipelineConfig(name="empty")
    dag = DAG()
    pipeline = Pipeline(config, dag)
    ctx = NodeContext()

    results = pipeline.run(ctx)
    assert results == {}


# ---------------------------------------------------------------------------
# 5. 并发执行验证
# ---------------------------------------------------------------------------
def test_pipeline_run_concurrent():
    """多个独立节点应并发执行,总耗时应远小于串行之和。"""
    ctx = NodeContext(max_workers=4)
    sleep_seconds = 0.15

    def _slow_executor(inputs, ctx):
        time.sleep(sleep_seconds)
        return {"output": "done"}

    ctx.register_executor("slow", _slow_executor)

    pipeline = (
        PipelineBuilder("test_concurrent")
        .node("slow", id="n1")
        .node("slow", id="n2")
        .node("slow", id="n3")
        .build()
    )

    start = time.monotonic()
    results = pipeline.run(ctx)
    elapsed = time.monotonic() - start

    assert set(results.keys()) == {"n1", "n2", "n3"}
    # 3 个各 sleep 0.15s 的节点并发执行,总耗时应远小于 3 * 0.15 = 0.45s。
    # 留足余量,避免 CI 环境抖动导致误报。
    assert elapsed < sleep_seconds * 2, (
        "并发执行总耗时 {:.3f}s 超过预期(应 < {:.3f}s)".format(
            elapsed, sleep_seconds * 2
        )
    )
