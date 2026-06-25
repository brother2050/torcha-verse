"""Consistency pipeline for the TorchaVerse consistency framework
(v0.3.0).

This module provides :class:`ConsistencyPipeline`, the top-level
orchestrator that composes the three consistency engines (character /
outfit / scene) plus a depth conditioning input into a single generation
surface.  It is the "integration layer" of the consistency framework.

Capabilities:

* :meth:`ConsistencyPipeline.generate_via_pipeline` -- （推荐）通过 L5
  Pipeline 执行一致性生成，将流程编译为可执行的
  :class:`~pipeline.composer.Pipeline`。
* :meth:`ConsistencyPipeline.to_pipeline` -- 将一致性生成流程编译为
  L5 :class:`~pipeline.composer.Pipeline`。
* :meth:`ConsistencyPipeline.score` -- 评估生成输出的一致性。

The pipeline delegates the actual conditioning to the three engines
(:class:`~consistency.character.CharacterEngine`,
:class:`~consistency.outfit.OutfitEngine`,
:class:`~consistency.scene.SceneEngine`) and the scoring to
:class:`~consistency.score.ScoreCalculator`.  Depth is applied as a
conditioning input rather than via a dedicated engine.  The generation
itself is a placeholder that returns descriptor dictionaries; the full
interface is exercised so that the pipeline can be swapped for a real
generation backend without changing call sites.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L2 ``assets`` -- asset types.
* L6 ``consistency`` (this module) -- pipeline orchestration.

This module depends on :mod:`torch` (transitively, through the score
calculator) and the L1/L2/L6 layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from assets.model_asset import CharacterAsset, DepthAsset, OutfitAsset, SceneAsset
from assets.store import AssetStore
from infrastructure.defaults import DIFFUSION_STEPS, DIFFUSION_GUIDANCE_SCALE
from infrastructure.logger import get_logger

from .character import CharacterEngine
from .outfit import OutfitEngine
from .profile import ConsistencyProfile
from .scene import SceneEngine
from .score import ConsistencyScore, ScoreCalculator

if TYPE_CHECKING:
    # These imports are only needed for type annotations; importing them
    # at runtime would create circular dependencies (pipeline.composer ->
    # nodes -> ... -> consistency).
    from pipeline.composer import Pipeline
    from nodes.base import NodeContext

__all__ = ["ConsistencyPipeline"]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Default image width for generation.
_DEFAULT_WIDTH: int = 1024

#: Default image height for generation.
_DEFAULT_HEIGHT: int = 1024

#: Lower bound for image dimensions.
_DIM_MIN: int = 64

#: Upper bound for image dimensions.
_DIM_MAX: int = 2048

#: Module-level logger.
_logger = get_logger("consistency.pipeline")


# ---------------------------------------------------------------------------
# ConsistencyPipeline
# ---------------------------------------------------------------------------
class ConsistencyPipeline:
    """Top-level consistency pipeline composing the three engines.

    The pipeline is constructed with a :class:`ConsistencyProfile` and
    optional references to the four asset types (character / outfit /
    scene / depth).  When an :class:`~assets.store.AssetStore` is
    provided the three engines (character / outfit / scene) are created
    from it; depth is applied as a conditioning input without a dedicated
    engine.  Otherwise the engines default to ``None`` and only the
    scoring surface is available.

    Args:
        profile: The :class:`ConsistencyProfile` controlling per-axis
            weights and temporal-consistency knobs.
        character: Optional :class:`CharacterAsset` to condition on.
        outfit: Optional :class:`OutfitAsset` to condition on.
        scene: Optional :class:`SceneAsset` to condition on.
        depth: Optional :class:`DepthAsset` to condition on.
        asset_store: Optional :class:`AssetStore` used to construct the
            three engines (character / outfit / scene).  When ``None`` the
            engines are ``None`` and only :meth:`score` is functional.
        score_calculator: Optional pre-configured
            :class:`ScoreCalculator`.  When ``None`` a default
            calculator is created.
    """

    def __init__(
        self,
        profile: ConsistencyProfile,
        character: Optional[CharacterAsset] = None,
        outfit: Optional[OutfitAsset] = None,
        scene: Optional[SceneAsset] = None,
        depth: Optional[DepthAsset] = None,
        asset_store: Optional[AssetStore] = None,
        score_calculator: Optional[ScoreCalculator] = None,
    ) -> None:
        self._profile: ConsistencyProfile = profile
        self._character: Optional[CharacterAsset] = character
        self._outfit: Optional[OutfitAsset] = outfit
        self._scene: Optional[SceneAsset] = scene
        self._depth: Optional[DepthAsset] = depth

        self._store: Optional[AssetStore] = asset_store
        self._scorer: ScoreCalculator = (
            score_calculator if score_calculator is not None
            else ScoreCalculator()
        )

        self._character_engine: Optional[CharacterEngine] = None
        self._outfit_engine: Optional[OutfitEngine] = None
        self._scene_engine: Optional[SceneEngine] = None
        if asset_store is not None:
            self._character_engine = CharacterEngine(
                asset_store, score_calculator=self._scorer
            )
            self._outfit_engine = OutfitEngine(asset_store)
            self._scene_engine = SceneEngine(asset_store)

        self._logger = _logger

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def profile(self) -> ConsistencyProfile:
        """The active :class:`ConsistencyProfile`."""
        return self._profile

    @property
    def character(self) -> Optional[CharacterAsset]:
        """The configured :class:`CharacterAsset` (or ``None``)."""
        return self._character

    @property
    def outfit(self) -> Optional[OutfitAsset]:
        """The configured :class:`OutfitAsset` (or ``None``)."""
        return self._outfit

    @property
    def scene(self) -> Optional[SceneAsset]:
        """The configured :class:`SceneAsset` (or ``None``)."""
        return self._scene

    @property
    def depth(self) -> Optional[DepthAsset]:
        """The configured :class:`DepthAsset` (or ``None``)."""
        return self._depth

    @property
    def character_engine(self) -> Optional[CharacterEngine]:
        """The :class:`CharacterEngine` (or ``None`` when no store)."""
        return self._character_engine

    @property
    def outfit_engine(self) -> Optional[OutfitEngine]:
        """The :class:`OutfitEngine` (or ``None`` when no store)."""
        return self._outfit_engine

    @property
    def scene_engine(self) -> Optional[SceneEngine]:
        """The :class:`SceneEngine` (or ``None`` when no store)."""
        return self._scene_engine

    @property
    def score_calculator(self) -> ScoreCalculator:
        """The :class:`ScoreCalculator` used for consistency scoring."""
        return self._scorer

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self, output: Any) -> ConsistencyScore:
        """Evaluate the consistency of a generation output.

        Compares ``output`` against the configured reference assets
        (character / outfit / scene / depth) using the
        :class:`ScoreCalculator`.

        Args:
            output: The generation output to score (an image, a frame,
                or a result dictionary from
                :meth:`generate_via_pipeline`).

        Returns:
            A :class:`ConsistencyScore` with all six fields populated.
        """
        image = output
        if isinstance(output, dict) and "image" in output:
            image = output["image"]
        references = self._build_references()
        return self._scorer.calculate(image, references)

    # ------------------------------------------------------------------
    # L5 Pipeline 集成（推荐入口）
    # ------------------------------------------------------------------
    def to_pipeline(
        self,
        prompt: str,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        **kwargs: Any,
    ) -> "Pipeline":
        """构建 L5 Pipeline 用于一致性生成。

        将一致性生成流程编译为 L5 :class:`~pipeline.composer.Pipeline`：
        基础图像生成节点（``image_txt2img``）之后按权重链式连接
        character / outfit / scene / depth 一致性节点。只有已配置
        （非 ``None``）且权重 > 0 的轴才会被加入链中。

        Args:
            prompt: 文本提示词。
            width: 输出图像宽度（像素），默认 ``1024``。
            height: 输出图像高度（像素），默认 ``1024``。
            **kwargs: 传递给基础图像节点的额外参数（如 ``steps``、
                ``seed``、``guidance_scale``）。

        Returns:
            构建完成的 :class:`~pipeline.composer.Pipeline`。

        Raises:
            ValueError: 如果 ``width`` 或 ``height`` 超出
                ``[_DIM_MIN, _DIM_MAX]``。
        """
        from pipeline.composer import PipelineBuilder

        self._validate_dimensions(width, height)
        builder = PipelineBuilder("consistency_generate")

        # 基础图像生成节点（当 character 未激活时创建）。
        # 当 character 激活时，character_apply 直接作为链的起点，
        # 避免浪费一个 base 生成节点的输出（fix #18）。
        base_inputs: Dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
        }
        base_inputs.update(kwargs)
        # 为必填输入提供默认值，确保 validate_inputs 校验通过。
        # 优先使用 kwargs 传入值 > defaults 模块配置值 > 硬编码兜底值。
        base_inputs.setdefault("steps", DIFFUSION_STEPS)
        base_inputs.setdefault("guidance_scale", DIFFUSION_GUIDANCE_SCALE)

        character_active = (
            self._character is not None
            and self._profile.character_weight > 0.0
        )

        if character_active:
            # character_apply 直接作为链起点，不创建 base 节点。
            builder.node(
                "character_apply",
                id="char",
                character=self._character.id,
                prompt=prompt,
                width=width,
                height=height,
                character_weight=self._profile.character_weight,
            )
            prev_id, prev_key = "char", "image"
        else:
            builder.node("image_txt2img", id="base", **base_inputs)
            prev_id, prev_key = "base", "image"

        if (
            self._outfit is not None
            and self._profile.outfit_weight > 0.0
        ):
            builder.node(
                "outfit_apply",
                id="outfit",
                outfit=self._outfit.id,
                outfit_weight=self._profile.outfit_weight,
            )
            builder.connect(
                prev_id, "outfit", output_key=prev_key, input_key="image"
            )
            prev_id, prev_key = "outfit", "image"

        if (
            self._scene is not None
            and self._profile.scene_weight > 0.0
        ):
            builder.node(
                "scene_apply",
                id="scene",
                scene=self._scene.id,
                scene_weight=self._profile.scene_weight,
            )
            builder.connect(
                prev_id, "scene", output_key=prev_key, input_key="image"
            )
            prev_id, prev_key = "scene", "image"

        if (
            self._depth is not None
            and self._profile.depth_weight > 0.0
        ):
            builder.node(
                "depth_condition",
                id="depth",
                method="midas",
                depth_weight=self._profile.depth_weight,
            )
            builder.connect(
                prev_id, "depth", output_key=prev_key, input_key="image_or_scene"
            )
            # depth_condition 的输出是 depth_map（DEPTH 类型），不是图像。
            # 不将其作为链末端，保持 prev 指向上一个图像节点，
            # 这样链的最终输出仍是图像而非深度图（fix #17）。
            # depth 节点仍然执行（产生 depth_map），但其输出不作为
            # 后续节点的输入。

        self._logger.debug(
            "Built consistency pipeline (prompt=%r, %dx%d).",
            prompt[:48], width, height,
        )
        return builder.build()

    def generate_via_pipeline(
        self,
        prompt: str,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        ctx: Optional["NodeContext"] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """通过 L5 Pipeline 执行一致性生成。

        推荐的一致性生成入口：先将流程编译为 L5
        :class:`~pipeline.composer.Pipeline`，再通过
        :class:`~nodes.base.NodeContext` 执行。

        Args:
            prompt: 文本提示词。
            width: 输出图像宽度（像素），默认 ``1024``。
            height: 输出图像高度（像素），默认 ``1024``。
            ctx: 可选的 :class:`~nodes.base.NodeContext`。为 ``None``
                时创建默认上下文。
            **kwargs: 传递给 :meth:`to_pipeline` 的额外参数。

        Returns:
            :meth:`Pipeline.run` 的执行结果（节点 ID 到输出的映射）。
        """
        if ctx is None:
            from nodes.base import NodeContext

            ctx = NodeContext(assets=self._store)
        pipeline = self.to_pipeline(prompt, width, height, **kwargs)
        try:
            return pipeline.run(ctx)
        finally:
            pipeline.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_dimensions(width: int, height: int) -> None:
        """Validate that width and height are within bounds."""
        for name, value in (("width", width), ("height", height)):
            if value < _DIM_MIN or value > _DIM_MAX:
                raise ValueError(
                    "{} must be in [{}, {}], got {}.".format(
                        name, _DIM_MIN, _DIM_MAX, value
                    )
                )

    def _build_references(self) -> Dict[str, Any]:
        """Build the references dictionary for scoring.

        Returns:
            A dictionary mapping axis names to reference assets.
        """
        references: Dict[str, Any] = {}
        if self._character is not None:
            references["character"] = self._character
        if self._outfit is not None:
            references["outfit"] = self._outfit
        if self._scene is not None:
            references["scene"] = self._scene
        if self._depth is not None:
            references["depth"] = self._depth
        return references

    # ------------------------------------------------------------------
    # 资源释放与上下文管理器协议
    # ------------------------------------------------------------------
    def close(self) -> None:
        """释放底层资源（AssetStore 连接等）。"""
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                self._logger.debug(
                    "Error closing asset store", exc_info=True
                )

    def __enter__(self) -> "ConsistencyPipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            "ConsistencyPipeline(profile={!r}, character={}, "
            "outfit={}, scene={}, depth={})".format(
                self._profile,
                self._character is not None,
                self._outfit is not None,
                self._scene is not None,
                self._depth is not None,
            )
        )
