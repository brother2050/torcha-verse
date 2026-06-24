"""端到端连通测试 -- 数字人节点(dh_lip_sync / talking_head / voice_clone /
face_enhance / portrait_animate)。

覆盖单一功能正常路径:构建 Pipeline -> 执行 -> 验证输出键。
所有节点返回占位数据,无需 GPU。
"""
from __future__ import annotations

from pipeline.composer import PipelineBuilder


# ---------------------------------------------------------------------------
# dh_lip_sync
# ---------------------------------------------------------------------------
def test_dh_lip_sync_e2e(pipeline_ctx):
    """构建 dh_lip_sync Pipeline -> 执行 -> 验证输出含 video 与 sync_score。"""
    pipeline = (
        PipelineBuilder("dh_lip_sync_e2e")
        .node(
            "dh_lip_sync",
            id="lip",
            video={"kind": "source_video"},
            audio={"kind": "driving_audio"},
            method="musetalk",
            face_region={"x": 10, "y": 10, "w": 128, "h": 128},
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["lip"]

    assert "video" in out, "dh_lip_sync 输出应包含 'video' 键"
    assert "sync_score" in out, "dh_lip_sync 输出应包含 'sync_score' 键"


# ---------------------------------------------------------------------------
# dh_talking_head
# ---------------------------------------------------------------------------
def test_dh_talking_head_e2e(pipeline_ctx):
    """构建 dh_talking_head Pipeline -> 执行 -> 验证输出含 video。"""
    pipeline = (
        PipelineBuilder("dh_talking_head_e2e")
        .node(
            "dh_talking_head",
            id="head",
            portrait_image={"kind": "portrait"},
            audio={"kind": "driving_audio"},
            method="sadtalker",
            enhance_resolution=True,
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["head"]

    assert "video" in out, "dh_talking_head 输出应包含 'video' 键"


# ---------------------------------------------------------------------------
# dh_voice_clone
# ---------------------------------------------------------------------------
def test_dh_voice_clone_e2e(pipeline_ctx):
    """构建 dh_voice_clone Pipeline -> 执行 -> 验证输出含 audio 与 sample_rate。"""
    pipeline = (
        PipelineBuilder("dh_voice_clone_e2e")
        .node(
            "dh_voice_clone",
            id="clone",
            reference_audio={"kind": "reference_clip"},
            text="这是一段用于语音克隆的测试文本。",
            language="zh",
            method="cosyvoice",
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["clone"]

    assert "audio" in out, "dh_voice_clone 输出应包含 'audio' 键"
    assert "sample_rate" in out, "dh_voice_clone 输出应包含 'sample_rate' 键"
    assert isinstance(out["sample_rate"], int) and out["sample_rate"] > 0


# ---------------------------------------------------------------------------
# dh_face_enhance
# ---------------------------------------------------------------------------
def test_dh_face_enhance_e2e(pipeline_ctx):
    """构建 dh_face_enhance Pipeline -> 执行 -> 验证输出含 video。"""
    pipeline = (
        PipelineBuilder("dh_face_enhance_e2e")
        .node(
            "dh_face_enhance",
            id="enhance",
            video={"kind": "source_video"},
            method="gfpgan",
            strength=0.75,
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["enhance"]

    assert "video" in out, "dh_face_enhance 输出应包含 'video' 键"


# ---------------------------------------------------------------------------
# dh_portrait_animate
# ---------------------------------------------------------------------------
def test_dh_portrait_animate_e2e(pipeline_ctx):
    """构建 dh_portrait_animate Pipeline -> 执行 -> 验证输出含 video。"""
    pipeline = (
        PipelineBuilder("dh_portrait_animate_e2e")
        .node(
            "dh_portrait_animate",
            id="animate",
            source_image={"kind": "portrait"},
            driving_signal={"kind": "driving_video"},
            method="liveportrait",
            stitching=True,
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["animate"]

    assert "video" in out, "dh_portrait_animate 输出应包含 'video' 键"
