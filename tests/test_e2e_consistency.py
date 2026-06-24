"""端到端连通测试 -- 一致性节点(character_apply / outfit_apply / scene_apply /
depth_condition / character_five_view)。

覆盖单一功能正常路径:构建 Pipeline -> 执行 -> 验证输出键。角色/衣装/场景
节点通过 :class:`AssetRef` 接收资产引用(占位,无需真实资产落盘)。
"""
from __future__ import annotations

from assets.base import AssetRef
from assets.types import AssetType
from pipeline.composer import PipelineBuilder


def _ref(asset_id: str, asset_type: AssetType) -> AssetRef:
    """构造一个占位 :class:`AssetRef`。"""
    return AssetRef(
        asset_id=asset_id,
        asset_type=asset_type,
        revision="r1",
        content_hash="0" * 64,
    )


# ---------------------------------------------------------------------------
# character_apply
# ---------------------------------------------------------------------------
def test_character_apply_e2e(pipeline_ctx):
    """构建 character_apply Pipeline -> 执行 -> 验证输出含 image。"""
    character = _ref("char-001", AssetType.CHARACTER)
    pipeline = (
        PipelineBuilder("character_apply_e2e")
        .node(
            "character_apply",
            id="apply",
            character=character,
            prompt="一位穿着未来感外套的少女",
            width=512,
            height=768,
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["apply"]

    assert "image" in out, "character_apply 输出应包含 'image' 键"
    assert out["image"]["character"] == "char-001", "输出应携带角色资产 id"


# ---------------------------------------------------------------------------
# outfit_apply
# ---------------------------------------------------------------------------
def test_outfit_apply_e2e(pipeline_ctx):
    """构建 outfit_apply Pipeline -> 执行 -> 验证输出含 image。"""
    outfit = _ref("outfit-001", AssetType.OUTFIT)
    pipeline = (
        PipelineBuilder("outfit_apply_e2e")
        .node(
            "outfit_apply",
            id="apply",
            image={"kind": "source_image"},
            outfit=outfit,
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["apply"]

    assert "image" in out, "outfit_apply 输出应包含 'image' 键"


# ---------------------------------------------------------------------------
# scene_apply
# ---------------------------------------------------------------------------
def test_scene_apply_e2e(pipeline_ctx):
    """构建 scene_apply Pipeline -> 执行 -> 验证输出含 image。"""
    scene = _ref("scene-001", AssetType.SCENE)
    pipeline = (
        PipelineBuilder("scene_apply_e2e")
        .node(
            "scene_apply",
            id="apply",
            image={"kind": "source_image"},
            scene=scene,
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["apply"]

    assert "image" in out, "scene_apply 输出应包含 'image' 键"


# ---------------------------------------------------------------------------
# depth_condition
# ---------------------------------------------------------------------------
def test_depth_condition_e2e(pipeline_ctx):
    """构建 depth_condition Pipeline -> 执行 -> 验证输出含 depth_map。"""
    pipeline = (
        PipelineBuilder("depth_condition_e2e")
        .node(
            "depth_condition",
            id="depth",
            image_or_scene={"kind": "source_image"},
            method="midas",
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["depth"]

    assert "depth_map" in out, "depth_condition 输出应包含 'depth_map' 键"


# ---------------------------------------------------------------------------
# character_five_view
# ---------------------------------------------------------------------------
def test_five_view_e2e(pipeline_ctx):
    """构建 character_five_view Pipeline -> 执行 -> 验证输出含 five_views(5 张)。"""
    pipeline = (
        PipelineBuilder("five_view_e2e")
        .node(
            "character_five_view",
            id="fv",
            reference_image={"kind": "portrait"},
            character_name="主角-艾莉",
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["fv"]

    assert "five_views" in out, "character_five_view 输出应包含 'five_views' 键"
    assert isinstance(out["five_views"], list), "five_views 应为列表"
    assert len(out["five_views"]) == 5, "应生成 5 个视角"
