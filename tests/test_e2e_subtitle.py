"""端到端连通测试 -- 字幕节点(subtitle_generate / translate / burn / export)。

覆盖单一功能正常路径与 generate -> translate -> burn -> export 的链式执行。
所有节点返回占位数据,无需 GPU。
"""
from __future__ import annotations

from pipeline.composer import PipelineBuilder


# ---------------------------------------------------------------------------
# subtitle_generate
# ---------------------------------------------------------------------------
def test_subtitle_generate_e2e(pipeline_ctx):
    """构建 subtitle_generate Pipeline -> 执行 -> 验证输出含 subtitle_track。"""
    pipeline = (
        PipelineBuilder("subtitle_generate_e2e")
        .node(
            "subtitle_generate",
            id="gen",
            source="text",
            text="数字人技术正在改变内容创作方式。",
            language="zh",
            method="asr",
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["gen"]

    assert "subtitle_track" in out, "subtitle_generate 输出应包含 'subtitle_track' 键"
    track = out["subtitle_track"]
    assert "cues" in track and isinstance(track["cues"], list), "字幕轨道应含 cues 列表"


# ---------------------------------------------------------------------------
# subtitle_translate
# ---------------------------------------------------------------------------
def test_subtitle_translate_e2e(pipeline_ctx):
    """构建 subtitle_translate Pipeline -> 执行 -> 验证输出含 subtitle_track。"""
    source_track = {
        "language": "en",
        "method": "asr",
        "source": "text",
        "cues": [{"index": 1, "start": 0.0, "end": 2.0, "text": "Hello world"}],
    }
    pipeline = (
        PipelineBuilder("subtitle_translate_e2e")
        .node(
            "subtitle_translate",
            id="tr",
            subtitle_track=source_track,
            target_language="zh",
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["tr"]

    assert "subtitle_track" in out, "subtitle_translate 输出应包含 'subtitle_track' 键"
    assert out["subtitle_track"].get("language") == "zh", "翻译后语言应为目标语言"


# ---------------------------------------------------------------------------
# subtitle_burn
# ---------------------------------------------------------------------------
def test_subtitle_burn_e2e(pipeline_ctx):
    """构建 subtitle_burn Pipeline -> 执行 -> 验证输出含 video。"""
    source_track = {
        "language": "zh",
        "method": "asr",
        "source": "text",
        "cues": [{"index": 1, "start": 0.0, "end": 2.0, "text": "你好世界"}],
    }
    pipeline = (
        PipelineBuilder("subtitle_burn_e2e")
        .node(
            "subtitle_burn",
            id="burn",
            video={"kind": "source_video"},
            subtitle_track=source_track,
            style={"font_size": 24},
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["burn"]

    assert "video" in out, "subtitle_burn 输出应包含 'video' 键"


# ---------------------------------------------------------------------------
# subtitle_export
# ---------------------------------------------------------------------------
def test_subtitle_export_e2e(tmp_path, pipeline_ctx):
    """构建 subtitle_export Pipeline -> 执行 -> 验证输出含 path。"""
    source_track = {
        "language": "zh",
        "method": "asr",
        "source": "text",
        "cues": [{"index": 1, "start": 0.0, "end": 2.0, "text": "你好世界"}],
    }
    out_path = str(tmp_path / "subtitle.srt")
    pipeline = (
        PipelineBuilder("subtitle_export_e2e")
        .node(
            "subtitle_export",
            id="exp",
            subtitle_track=source_track,
            format="srt",
            path=out_path,
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["exp"]

    assert "path" in out, "subtitle_export 输出应包含 'path' 键"
    assert out["path"] == out_path, "导出路径应与输入一致"


# ---------------------------------------------------------------------------
# full chain: generate -> translate -> burn -> export
# ---------------------------------------------------------------------------
def test_subtitle_full_chain(tmp_path, pipeline_ctx):
    """generate -> translate -> burn -> export 链式执行,验证跨节点协同。"""
    out_path = str(tmp_path / "full_chain.srt")
    pipeline = (
        PipelineBuilder("subtitle_full_chain")
        .node(
            "subtitle_generate",
            id="gen",
            source="text",
            text="一段用于链式测试的旁白文本。",
            language="en",
            method="asr",
        )
        .node("subtitle_translate", id="tr", target_language="zh")
        .node("subtitle_burn", id="burn", video={"kind": "source_video"})
        .node(
            "subtitle_export",
            id="exp",
            format="srt",
            path=out_path,
        )
        .connect("gen", "tr", output_key="subtitle_track", input_key="subtitle_track")
        .connect("tr", "burn", output_key="subtitle_track", input_key="subtitle_track")
        .connect("tr", "exp", output_key="subtitle_track", input_key="subtitle_track")
        .build()
    )

    results = pipeline.run(pipeline_ctx)

    assert "subtitle_track" in results["gen"], "generate 应产出 subtitle_track"
    assert "subtitle_track" in results["tr"], "translate 应产出 subtitle_track"
    assert results["tr"]["subtitle_track"].get("language") == "zh", "translate 应转为中文"
    assert "video" in results["burn"], "burn 应产出 video"
    assert "path" in results["exp"], "export 应产出 path"
    assert results["exp"]["path"] == out_path, "export 路径应与预期一致"
