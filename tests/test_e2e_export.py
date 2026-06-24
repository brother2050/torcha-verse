"""端到端连通测试 -- 导出节点(export_image / export_video / export_audio)。

覆盖单一功能正常路径:构建 Pipeline -> 执行 -> 验证输出含 path 键。
其中图像导出额外验证资产可落盘到临时 :class:`AssetStore` 并被取回。
所有节点返回占位数据,无需 GPU。
"""
from __future__ import annotations

from assets.model_asset import ModelAsset
from pipeline.composer import PipelineBuilder


# ---------------------------------------------------------------------------
# export_image (+ AssetStore 落盘验证)
# ---------------------------------------------------------------------------
def test_export_image_e2e(tmp_path, pipeline_ctx, tmp_asset_store):
    """构建 export_image Pipeline -> 执行 -> 验证 path;并验证资产可落盘 AssetStore。"""
    out_path = tmp_path / "out.png"
    pipeline = (
        PipelineBuilder("export_image_e2e")
        .node(
            "export_image",
            id="exp",
            image={"kind": "placeholder_image"},
            path=str(out_path),
            format="png",
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["exp"]

    assert "path" in out, "export_image 输出应包含 'path' 键"
    assert out["path"] == str(out_path), "导出路径应与输入一致"

    # 验证资产可落盘到 AssetStore 并被取回(占位内容)。
    content_file = tmp_path / "image.bin"
    content_file.write_bytes(b"placeholder-image-bytes")
    asset = ModelAsset(
        id="exported-image-001",
        name="Exported Image",
        architecture="image",
        format="png",
        size_gb=0.0,
        source="local",
    )
    ref = tmp_asset_store.put(asset, content_file)
    retrieved_asset, retrieved_path = tmp_asset_store.get(ref)
    assert retrieved_asset.id == "exported-image-001", "取回的资产 id 应一致"
    assert retrieved_path.exists(), "资产内容文件应存在"


# ---------------------------------------------------------------------------
# export_video
# ---------------------------------------------------------------------------
def test_export_video_e2e(tmp_path, pipeline_ctx):
    """构建 export_video Pipeline -> 执行 -> 验证输出含 path。"""
    out_path = tmp_path / "out.mp4"
    pipeline = (
        PipelineBuilder("export_video_e2e")
        .node(
            "export_video",
            id="exp",
            video={"kind": "placeholder_video"},
            path=str(out_path),
            format="mp4",
            fps=30,
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["exp"]

    assert "path" in out, "export_video 输出应包含 'path' 键"
    assert out["path"] == str(out_path), "导出路径应与输入一致"


# ---------------------------------------------------------------------------
# export_audio
# ---------------------------------------------------------------------------
def test_export_audio_e2e(tmp_path, pipeline_ctx):
    """构建 export_audio Pipeline -> 执行 -> 验证输出含 path。"""
    out_path = tmp_path / "out.wav"
    pipeline = (
        PipelineBuilder("export_audio_e2e")
        .node(
            "export_audio",
            id="exp",
            audio={"kind": "placeholder_audio"},
            path=str(out_path),
            format="wav",
            sample_rate=22050,
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["exp"]

    assert "path" in out, "export_audio 输出应包含 'path' 键"
    assert out["path"] == str(out_path), "导出路径应与输入一致"
