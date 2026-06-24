"""端到端连通测试 -- 视频节点(video_txt2vid / video_interpolate / video_stitch)。

覆盖单一功能正常路径:构建 Pipeline -> 执行 -> 验证输出键;以及
txt2vid -> interpolate 的链式执行。所有节点返回占位数据,无需 GPU。
"""
from __future__ import annotations

from pipeline.composer import PipelineBuilder


# ---------------------------------------------------------------------------
# video_txt2vid
# ---------------------------------------------------------------------------
def test_video_txt2vid_e2e(pipeline_ctx):
    """构建 video_txt2vid Pipeline -> 执行 -> 验证输出含 video 与 seed。"""
    pipeline = (
        PipelineBuilder("video_txt2vid_e2e")
        .node(
            "video_txt2vid",
            id="gen",
            prompt="一只猫在草地上奔跑,电影感镜头",
            num_frames=16,
            width=512,
            height=512,
            steps=6,
            guidance_scale=7.0,
            seed=2024,
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["gen"]

    assert "video" in out, "video_txt2vid 输出应包含 'video' 键"
    assert "seed" in out, "video_txt2vid 输出应包含 'seed' 键"
    assert out["seed"] == 2024, "返回的 seed 应与输入一致"


# ---------------------------------------------------------------------------
# txt2vid -> interpolate chain
# ---------------------------------------------------------------------------
def test_video_interpolate_e2e(pipeline_ctx):
    """txt2vid -> interpolate 链式执行,验证两节点均产出 video。"""
    pipeline = (
        PipelineBuilder("video_interpolate_chain")
        .node(
            "video_txt2vid",
            id="gen",
            prompt="海浪拍打礁石",
            num_frames=16,
            width=512,
            height=512,
            steps=6,
            guidance_scale=7.0,
            seed=11,
        )
        .node("video_interpolate", id="interp", target_fps=60)
        .connect("gen", "interp", output_key="video", input_key="video")
        .build()
    )

    results = pipeline.run(pipeline_ctx)

    assert "video" in results["gen"], "txt2vid 应产出 video"
    assert "video" in results["interp"], "interpolate 应产出 video"


# ---------------------------------------------------------------------------
# video_stitch
# ---------------------------------------------------------------------------
def test_video_stitch_e2e(pipeline_ctx):
    """构建 video_stitch Pipeline -> 执行 -> 验证输出含 video。"""
    pipeline = (
        PipelineBuilder("video_stitch_e2e")
        .node(
            "video_stitch",
            id="stitch",
            videos=[
                {"kind": "clip_a"},
                {"kind": "clip_b"},
                {"kind": "clip_c"},
            ],
            transition="crossfade",
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["stitch"]

    assert "video" in out, "video_stitch 输出应包含 'video' 键"
