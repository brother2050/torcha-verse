"""跨模块组合连通测试。

验证多个子系统协同工作的端到端正常路径:

* 画布(Canvas) -> DAG -> Pipeline -> 执行 -> 资产落盘 AssetStore。
* 一致性生成 + 字幕 + 导出 的组合 Pipeline。
* AutoDirector -> 画布 -> 版本控制(CanvasHistory) -> 分享(ShareManager)。
* 数字人全链路:文本 -> 语音克隆 -> 说话头 -> 人脸增强 -> 字幕 -> 导出。

所有节点返回占位数据,无需 GPU。
"""
from __future__ import annotations

from assets.base import AssetRef
from assets.model_asset import ModelAsset
from assets.types import AssetType
from canvas.autodirector import AutoDirector
from canvas.canvas import Canvas
from canvas.sharing import ShareManager
from canvas.versioning import CanvasHistory
from pipeline.composer import PipelineBuilder
from pipeline.templates import TemplateRegistry


def _ref(asset_id: str, asset_type: AssetType) -> AssetRef:
    return AssetRef(
        asset_id=asset_id,
        asset_type=asset_type,
        revision="r1",
        content_hash="0" * 64,
    )


# ---------------------------------------------------------------------------
# 1. Canvas -> DAG -> Pipeline -> 执行 -> 资产落盘
# ---------------------------------------------------------------------------
def test_canvas_to_pipeline_to_assetstore(tmp_path, pipeline_ctx, tmp_asset_store):
    """画布 -> DAG -> Pipeline -> 执行 -> 资产落盘 AssetStore。"""
    canvas = Canvas("canvas_to_pipeline")
    canvas.add_node(
        "image_txt2img",
        id="gen",
        prompt="一只在弹钢琴的猫",
        width=512,
        height=512,
        steps=6,
        guidance_scale=7.0,
        seed=5,
    )
    canvas.add_node("image_upscale", id="up", scale=2)

    conn = canvas.connect("gen", "image", "up", "image")
    assert not isinstance(conn, str), "画布连线应成功: {}".format(conn)

    pipeline = canvas.to_pipeline()
    results = pipeline.run(pipeline_ctx)

    assert "image" in results["gen"], "画布转 Pipeline 后 txt2img 应产出 image"
    assert "image" in results["up"], "画布转 Pipeline 后 upscale 应产出 image"

    # 资产落盘到 AssetStore 并取回。
    content_file = tmp_path / "canvas_asset.bin"
    content_file.write_bytes(b"canvas-asset-bytes")
    asset = ModelAsset(
        id="canvas-asset-001",
        name="Canvas Asset",
        architecture="image",
        format="png",
        source="local",
    )
    ref = tmp_asset_store.put(asset, content_file)
    retrieved, retrieved_path = tmp_asset_store.get(ref)
    assert retrieved.id == "canvas-asset-001", "AssetStore 取回的资产 id 应一致"
    assert retrieved_path.exists(), "资产内容文件应落盘存在"


# ---------------------------------------------------------------------------
# 2. 一致性生成 + 字幕 + 导出
# ---------------------------------------------------------------------------
def test_consistency_subtitle_export(tmp_path, pipeline_ctx):
    """一致性生成 + 字幕 + 导出 的组合 Pipeline。"""
    character = _ref("char-combo", AssetType.CHARACTER)
    out_path = str(tmp_path / "combo_subtitle.srt")

    pipeline = (
        PipelineBuilder("consistency_subtitle_export")
        .node(
            "character_apply",
            id="char",
            character=character,
            prompt="一位穿着机甲的少女",
            width=512,
            height=768,
        )
        .node(
            "subtitle_generate",
            id="sub",
            source="text",
            text="机甲少女的独白。",
            language="zh",
            method="asr",
        )
        .node("subtitle_translate", id="tr", target_language="en")
        .node("subtitle_export", id="exp", format="srt", path=out_path)
        .connect("sub", "tr", output_key="subtitle_track", input_key="subtitle_track")
        .connect("tr", "exp", output_key="subtitle_track", input_key="subtitle_track")
        .build()
    )

    results = pipeline.run(pipeline_ctx)

    assert "image" in results["char"], "一致性节点应产出 image"
    assert "subtitle_track" in results["sub"], "字幕生成应产出 subtitle_track"
    assert "subtitle_track" in results["tr"], "字幕翻译应产出 subtitle_track"
    assert results["tr"]["subtitle_track"].get("language") == "en", "翻译后应为英文"
    assert "path" in results["exp"], "字幕导出应产出 path"
    assert results["exp"]["path"] == out_path, "导出路径应与预期一致"


# ---------------------------------------------------------------------------
# 3. AutoDirector -> 画布 -> 版本控制 -> 分享
# ---------------------------------------------------------------------------
def test_autodirector_versioning_sharing(tmp_path):
    """AutoDirector -> 画布 -> 版本控制 -> 分享。"""
    director = AutoDirector(TemplateRegistry())
    canvas = director.generate("a cat playing piano")
    assert isinstance(canvas, Canvas), "AutoDirector 应返回 Canvas"

    # 确保画布有可追踪内容。
    canvas.add_node(
        "text_chat",
        id="narration",
        prompt="为这段猫咪弹琴的画面写一句旁白。",
        max_tokens=32,
    )

    history = CanvasHistory(canvas)
    v1 = history.commit("初始版本", author="tester")
    assert history.version_count == 1, "首次提交后版本数应为 1"

    # 修改画布并再次提交,验证版本递增与 checkout。
    canvas.add_node(
        "image_txt2img",
        id="hero_shot",
        prompt="猫咪弹琴特写",
        width=512,
        height=512,
        seed=8,
    )
    v2 = history.commit("增加主视觉镜头", author="tester")
    assert history.version_count == 2, "二次提交后版本数应为 2"

    v1_state = history.checkout(v1.version_id)
    v2_state = history.checkout(v2.version_id)
    assert len(v2_state.nodes) > len(v1_state.nodes), "v2 节点数应多于 v1"

    # 分享:创建链接 -> 解析 -> 校验状态一致。
    sharer = ShareManager()
    link = sharer.create_link(canvas, view_only=True)
    resolved = sharer.resolve_link(link.link_id)
    assert resolved is not None, "分享链接应能被解析"
    assert len(resolved.nodes) == len(canvas.state.nodes), "解析出的画布节点数应一致"

    # 导出 / 导入 bundle 往返。
    bundle_path = tmp_path / "canvas.bundle.zip"
    sharer.export_bundle(canvas, bundle_path)
    assert bundle_path.exists(), "bundle 应落盘"
    imported = sharer.import_bundle(bundle_path)
    assert isinstance(imported, Canvas), "导入 bundle 应返回 Canvas"
    assert len(imported.list_nodes()) == len(canvas.list_nodes()), (
        "导入画布节点数应与原画布一致"
    )


# ---------------------------------------------------------------------------
# 4. 数字人全链路:文本 -> 语音克隆 -> 说话头 -> 人脸增强 -> 字幕 -> 导出
# ---------------------------------------------------------------------------
def test_digital_human_full_chain(tmp_path, pipeline_ctx):
    """文本 -> 语音克隆 -> 说话头 -> 人脸增强 -> 字幕 -> 导出 的全链路。"""
    out_path = str(tmp_path / "dh_full_chain.srt")

    pipeline = (
        PipelineBuilder("digital_human_full_chain")
        .node(
            "text_chat",
            id="text",
            prompt="写一段数字人自我介绍的台词。",
            max_tokens=64,
            temperature=0.7,
        )
        .node(
            "dh_voice_clone",
            id="voice",
            reference_audio={"kind": "reference_clip"},
            language="zh",
            method="cosyvoice",
        )
        .node(
            "dh_talking_head",
            id="head",
            portrait_image={"kind": "portrait"},
            method="sadtalker",
            enhance_resolution=True,
        )
        .node(
            "dh_face_enhance",
            id="enhance",
            method="gfpgan",
            strength=0.8,
        )
        .node(
            "subtitle_generate",
            id="sub",
            source="text",
            language="zh",
            method="asr",
        )
        .node("subtitle_export", id="exp", format="srt", path=out_path)
        .connect("text", "voice", output_key="text", input_key="text")
        .connect("voice", "head", output_key="audio", input_key="audio")
        .connect("head", "enhance", output_key="video", input_key="video")
        .connect("text", "sub", output_key="text", input_key="text")
        .connect("sub", "exp", output_key="subtitle_track", input_key="subtitle_track")
        .build()
    )

    results = pipeline.run(pipeline_ctx)

    assert "text" in results["text"], "文本节点应产出 text"
    assert "audio" in results["voice"], "语音克隆应产出 audio"
    assert "sample_rate" in results["voice"], "语音克隆应产出 sample_rate"
    assert "video" in results["head"], "说话头应产出 video"
    assert "video" in results["enhance"], "人脸增强应产出 video"
    assert "subtitle_track" in results["sub"], "字幕生成应产出 subtitle_track"
    assert "path" in results["exp"], "字幕导出应产出 path"
    assert results["exp"]["path"] == out_path, "导出路径应与预期一致"
