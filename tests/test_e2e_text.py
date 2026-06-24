"""端到端连通测试 -- 文本节点(text_chat / text_completion)。

覆盖单一功能正常路径:构建 Pipeline -> 执行 -> 验证输出键;以及
Pipeline 的 YAML 序列化往返与 :class:`SeedManager` 的记录/召回。
"""
from __future__ import annotations

from pipeline.composer import Pipeline, PipelineBuilder
from pipeline.prompt_studio import SeedManager


# ---------------------------------------------------------------------------
# text_chat
# ---------------------------------------------------------------------------
def test_text_chat_e2e(pipeline_ctx):
    """构建 text_chat Pipeline -> 执行 -> 验证输出含 text 与 usage 键。"""
    pipeline = (
        PipelineBuilder("text_chat_e2e")
        .node(
            "text_chat",
            id="chat",
            prompt="用一句话介绍数字人技术。",
            max_tokens=64,
            temperature=0.7,
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["chat"]

    assert "text" in out, "text_chat 输出应包含 'text' 键"
    assert "usage" in out, "text_chat 输出应包含 'usage' 键"
    assert isinstance(out["text"], str) and out["text"]


# ---------------------------------------------------------------------------
# text_completion
# ---------------------------------------------------------------------------
def test_text_completion_e2e(pipeline_ctx):
    """构建 text_completion Pipeline -> 执行 -> 验证输出含 text 与 usage 键。"""
    pipeline = (
        PipelineBuilder("text_completion_e2e")
        .node(
            "text_completion",
            id="complete",
            prompt="从前有座山,山上有个",
            max_tokens=32,
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["complete"]

    assert "text" in out, "text_completion 输出应包含 'text' 键"
    assert "usage" in out, "text_completion 输出应包含 'usage' 键"
    assert isinstance(out["text"], str) and out["text"]


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------
def test_text_pipeline_to_yaml(tmp_path, pipeline_ctx):
    """Pipeline -> YAML -> 反序列化 -> 执行 -> 验证输出。"""
    pipeline = (
        PipelineBuilder("text_yaml_roundtrip")
        .node(
            "text_chat",
            id="chat",
            prompt="解释什么是 DAG。",
            max_tokens=48,
            temperature=0.3,
        )
        .build()
    )

    yaml_path = tmp_path / "text_pipeline.yaml"
    written = pipeline.to_yaml(yaml_path)
    assert written.exists(), "to_yaml 应写出文件"

    restored = Pipeline.from_yaml(yaml_path)

    results = restored.run(pipeline_ctx)
    out = results["chat"]

    assert "text" in out, "反序列化后的 Pipeline 输出应包含 'text' 键"
    assert "usage" in out, "反序列化后的 Pipeline 输出应包含 'usage' 键"


# ---------------------------------------------------------------------------
# SeedManager record / recall
# ---------------------------------------------------------------------------
def test_text_with_seed(pipeline_ctx, mock_model):
    """验证 SeedManager 记录与召回,并跑通文本 Pipeline。"""
    seed_manager = SeedManager()
    prompt = "一只在弹钢琴的猫"
    seed = 12345
    model = mock_model.name

    record = seed_manager.record(prompt, seed, model, {"max_tokens": 64})
    assert record["seed"] == seed, "record 应返回所记录的 seed"

    recalled = seed_manager.recall(prompt)
    assert recalled, "recall 应能召回已记录的 seed"
    assert any(r["seed"] == seed for r in recalled), "召回结果应包含所记录的 seed"

    pipeline = (
        PipelineBuilder("text_with_seed")
        .node(
            "text_chat",
            id="chat",
            prompt=prompt,
            max_tokens=64,
            temperature=0.7,
        )
        .build()
    )
    results = pipeline.run(pipeline_ctx)
    assert "text" in results["chat"], "文本 Pipeline 应正常执行并返回 text"
