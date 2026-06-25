"""共享 pytest fixture,服务于端到端与跨模块连通测试套件。

本文件为 ``tests/test_e2e_*.py`` 与 ``tests/test_integration_combo.py`` 提供
可复用的 fixture:

* :func:`tmp_asset_store` -- 基于 ``tmp_path`` 的临时 :class:`AssetStore`。
* :func:`node_ctx` -- 预配置的 L4 :class:`~nodes.base.NodeContext`,含临时
  ``AssetStore`` 与默认模型配置,供节点 ``execute()`` 使用。
* :func:`mock_model` -- 返回确定性输出的假模型对象。
* :func:`pipeline_service` -- :class:`~serving.service.PipelineService` 实例。
* :func:`pipeline_ctx` -- 编排层 :class:`~pipeline.composer.NodeContext`,其
  ``executors`` 将每个节点类型派发到对应的 L4 节点(返回占位数据,无需 GPU)。

所有测试均不依赖 GPU:节点的 ``execute()`` 返回占位数据。
"""
from __future__ import annotations

import pytest

# 轻量导入(不依赖 torch)
from assets.store import AssetStore
from nodes.base import NodeContext, NodeRegistry


__all__ = [
    "tmp_asset_store",
    "node_ctx",
    "mock_model",
    "pipeline_service",
    "pipeline_ctx",
]


# ---------------------------------------------------------------------------
# AssetStore
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_asset_store(tmp_path):
    """基于 ``tmp_path`` 的临时 :class:`AssetStore`,测试结束自动关闭。"""
    store = AssetStore(base_dir=tmp_path / "assets")
    try:
        yield store
    finally:
        store.close()


# ---------------------------------------------------------------------------
# L4 NodeContext
# ---------------------------------------------------------------------------
@pytest.fixture
def node_ctx(tmp_asset_store):
    """预配置的 L4 :class:`NodeContext`,含临时 AssetStore 与默认模型配置。"""
    return NodeContext(
        assets=tmp_asset_store,
        config={
            "default_text_model": "mock-text-model",
            "default_image_model": "mock-image-model",
            "default_video_model": "mock-video-model",
            "default_tts_model": "mock-tts-model",
            "default_music_model": "mock-music-model",
            "default_translate_model": "mock-translate-model",
            "default_asr_model": "mock-asr-model",
            "default_upscale_model": "mock-upscale-model",
            "default_interpolate_model": "mock-interpolate-model",
            "default_depth_model": "mock-depth-model",
            "default_five_view_model": "mock-five-view-model",
            "default_tts_sample_rate": 22050,
            "default_voice_sample_rate": 24000,
        },
    )


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_model():
    """返回确定性输出的假模型对象。

    用于在不依赖真实推理后端的情况下,验证模型调用契约。
    """

    class _MockModel:
        name = "mock-deterministic-model"

        def generate(self, prompt, **kwargs):
            seed = kwargs.get("seed", 0)
            return {
                "text": "mock-output:" + str(prompt)[:32],
                "seed": seed,
                "deterministic": True,
            }

    return _MockModel()


# ---------------------------------------------------------------------------
# PipelineService
# ---------------------------------------------------------------------------
@pytest.fixture
def pipeline_service():
    """:class:`PipelineService` 实例(惰性导入,避免影响其余测试的收集速度)。"""
    from serving.service import PipelineService

    return PipelineService()


# ---------------------------------------------------------------------------
# Composer NodeContext with L4 executors
# ---------------------------------------------------------------------------
@pytest.fixture
def pipeline_ctx(node_ctx):
    """编排层 :class:`~nodes.base.NodeContext`(统一上下文)。

    为 ``NodeRegistry`` 中注册的每个节点类型安装一个执行器,执行器将输入
    派发到对应的 L4 节点 ``execute()``(返回占位数据,无需 GPU)。配合
    :class:`~pipeline.composer.PipelineBuilder` 构建的 ``Pipeline`` 使用:

    .. code-block:: python

        pipeline = PipelineBuilder("demo").node("text_chat", id="c", prompt="hi").build()
        results = pipeline.run(pipeline_ctx)
        assert "text" in results["c"]
    """
    registry = NodeRegistry()
    executors = {}

    def _make_executor(node_type: str):
        def _executor(inputs, ctx):
            node = registry.get(node_type)
            return node.execute(node_ctx, **inputs)

        return _executor

    for spec in registry.list():
        executors[spec.type] = _make_executor(spec.type)

    return NodeContext(executors=executors)
