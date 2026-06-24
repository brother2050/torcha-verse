"""端到端连通测试 -- 图像节点(image_txt2img / image_upscale / image_inpaint)。

覆盖单一功能正常路径:构建 Pipeline -> 执行 -> 验证输出键;以及
txt2img -> upscale 的链式执行与 YAML 往返。所有节点返回占位数据,无需 GPU。
"""
from __future__ import annotations

from pipeline.composer import Pipeline, PipelineBuilder


# ---------------------------------------------------------------------------
# image_txt2img
# ---------------------------------------------------------------------------
def test_image_txt2img_e2e(pipeline_ctx):
    """构建 image_txt2img Pipeline -> 执行 -> 验证输出含 image 与 seed。"""
    pipeline = (
        PipelineBuilder("image_txt2img_e2e")
        .node(
            "image_txt2img",
            id="gen",
            prompt="一只在弹钢琴的猫,赛博朋克风格",
            width=512,
            height=512,
            steps=10,
            guidance_scale=7.0,
            seed=42,
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["gen"]

    assert "image" in out, "image_txt2img 输出应包含 'image' 键"
    assert "seed" in out, "image_txt2img 输出应包含 'seed' 键"
    assert out["seed"] == 42, "返回的 seed 应与输入一致"


# ---------------------------------------------------------------------------
# txt2img -> upscale chain
# ---------------------------------------------------------------------------
def test_image_upscale_e2e(pipeline_ctx):
    """txt2img -> upscale 链式执行,验证两节点均产出 image。"""
    pipeline = (
        PipelineBuilder("image_upscale_chain")
        .node(
            "image_txt2img",
            id="gen",
            prompt="雪山日落",
            width=512,
            height=512,
            steps=8,
            guidance_scale=7.5,
            seed=7,
        )
        .node("image_upscale", id="up", scale=2)
        .connect("gen", "up", output_key="image", input_key="image")
        .build()
    )

    results = pipeline.run(pipeline_ctx)

    assert "image" in results["gen"], "txt2img 应产出 image"
    assert "image" in results["up"], "upscale 应产出 image"
    assert results["up"]["image"]["scale"] == 2, "upscale 应反映 scale 参数"


# ---------------------------------------------------------------------------
# image_inpaint
# ---------------------------------------------------------------------------
def test_image_inpaint_e2e(pipeline_ctx):
    """构建 image_inpaint Pipeline -> 执行 -> 验证输出含 image。"""
    pipeline = (
        PipelineBuilder("image_inpaint_e2e")
        .node(
            "image_inpaint",
            id="inp",
            image={"kind": "source_image"},
            mask={"kind": "binary_mask"},
            prompt="把背景换成星空",
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["inp"]

    assert "image" in out, "image_inpaint 输出应包含 'image' 键"


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------
def test_image_pipeline_yaml_roundtrip(tmp_path, pipeline_ctx):
    """to_yaml/from_yaml 往返后执行,验证输出键。"""
    pipeline = (
        PipelineBuilder("image_yaml_roundtrip")
        .node(
            "image_txt2img",
            id="gen",
            prompt="水墨山水画",
            width=512,
            height=512,
            steps=6,
            guidance_scale=8.0,
            seed=99,
        )
        .build()
    )

    yaml_path = tmp_path / "image_pipeline.yaml"
    pipeline.to_yaml(yaml_path)
    restored = Pipeline.from_yaml(yaml_path)

    results = restored.run(pipeline_ctx)
    out = results["gen"]

    assert "image" in out, "往返后 Pipeline 应产出 image"
    assert "seed" in out, "往返后 Pipeline 应产出 seed"
