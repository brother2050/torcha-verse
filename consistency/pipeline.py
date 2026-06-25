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
* :meth:`ConsistencyPipeline.generate` -- （已弃用）直接生成单张图像。
* :meth:`ConsistencyPipeline.generate_batch` -- （已弃用）批量生成。
* :meth:`ConsistencyPipeline.generate_video` -- （已弃用）生成视频。
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

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

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

#: Default number of video frames.
_DEFAULT_NUM_FRAMES: int = 16

#: Lower bound for image dimensions.
_DIM_MIN: int = 64

#: Upper bound for image dimensions.
_DIM_MAX: int = 2048

#: Lower bound for the number of video frames.
_FRAMES_MIN: int = 1

#: Upper bound for the number of video frames.
_FRAMES_MAX: int = 1024

#: Default consistency seed for video generation.
_DEFAULT_VIDEO_SEED: int = 42

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
    # Single-image generation (已弃用，推荐使用 generate_via_pipeline)
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a single image with all consistency conditioning applied.

        .. deprecated::
            请改用 :meth:`generate_via_pipeline`，它通过 L5 Pipeline
            执行生成，是推荐的一致性生成入口。

        The conditioning signals (character / outfit / scene / depth)
        are applied according to the weights in the
        :class:`ConsistencyProfile`.  Only the assets that are
        configured (non-``None``) and have a non-zero weight are
        applied.

        Args:
            prompt: The text prompt describing the desired image.
            width: Output width in pixels.  Defaults to ``1024``.
            height: Output height in pixels.  Defaults to ``1024``.
            **kwargs: Additional generation parameters (e.g. ``seed``,
                ``steps``, ``guidance_scale``).

        Returns:
            A dictionary with keys:

            * ``image`` -- the generated image descriptor.
            * ``consistency_scores`` -- a :class:`ConsistencyScore`
              (as a dict) for the output.
            * ``metadata`` -- generation metadata (prompt, dimensions,
              applied conditioning, profile).

        Raises:
            ValueError: If ``width`` or ``height`` are outside
                ``[_DIM_MIN, _DIM_MAX]``.
        """
        warnings.warn(
            "ConsistencyPipeline.generate() 已弃用，请改用 "
            "generate_via_pipeline() 通过 L5 Pipeline 执行一致性生成。",
            DeprecationWarning,
            stacklevel=2,
        )
        self._validate_dimensions(width, height)
        seed = kwargs.get("seed", _DEFAULT_VIDEO_SEED)

        applied: Dict[str, Any] = {}
        image: Any = {
            "kind": "consistency_generated_image",
            "prompt": prompt,
            "width": width,
            "height": height,
            "seed": seed,
        }

        # --- Character conditioning ---
        if (
            self._character is not None
            and self._profile.character_weight > 0.0
        ):
            char_result = self._apply_character_conditioning(
                image, self._character, self._profile.character_weight
            )
            applied["character"] = char_result

        # --- Outfit conditioning ---
        if (
            self._outfit is not None
            and self._profile.outfit_weight > 0.0
        ):
            outfit_result = self._apply_outfit_conditioning(
                image, self._outfit, self._profile.outfit_weight
            )
            applied["outfit"] = outfit_result

        # --- Scene conditioning ---
        if (
            self._scene is not None
            and self._profile.scene_weight > 0.0
        ):
            scene_result = self._apply_scene_conditioning(
                image, self._scene, self._profile.scene_weight
            )
            applied["scene"] = scene_result

        # --- Depth conditioning ---
        if (
            self._depth is not None
            and self._profile.depth_weight > 0.0
        ):
            depth_result = self._apply_depth_conditioning(
                image, self._depth, self._profile.depth_weight
            )
            applied["depth"] = depth_result

        image["applied_conditioning"] = applied

        # --- Consistency scoring ---
        references = self._build_references()
        score = self._scorer.calculate(image, references)

        result = {
            "image": image,
            "consistency_scores": score.to_dict(),
            "metadata": {
                "prompt": prompt,
                "width": width,
                "height": height,
                "seed": seed,
                "profile": self._profile.to_dict(),
                "applied_conditioning": list(applied.keys()),
                "kwargs": dict(kwargs),
            },
        }
        self._logger.debug(
            "Generated image (prompt=%r, %dx%d, score=%s).",
            prompt[:48], width, height, score,
        )
        return result

    # ------------------------------------------------------------------
    # Batch generation (已弃用，推荐使用 generate_via_pipeline)
    # ------------------------------------------------------------------
    def generate_batch(
        self,
        prompts: Sequence[str],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generate a batch of images while maintaining consistency.

        .. deprecated::
            请改用 :meth:`generate_via_pipeline`，逐个提示词调用即可。

        Each prompt is generated independently but with the same
        conditioning assets and profile, so that identity / outfit /
        scene consistency is preserved across the batch.

        Args:
            prompts: A list of text prompts.
            **kwargs: Additional generation parameters forwarded to
                :meth:`generate`.

        Returns:
            A list of generation result dictionaries (one per prompt).
        """
        warnings.warn(
            "ConsistencyPipeline.generate_batch() 已弃用，请改用 "
            "generate_via_pipeline() 逐个执行。",
            DeprecationWarning,
            stacklevel=2,
        )
        results: List[Dict[str, Any]] = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        self._logger.info(
            "Generated batch of %d images.", len(results)
        )
        return results

    # ------------------------------------------------------------------
    # Video generation (已弃用)
    # ------------------------------------------------------------------
    def generate_video(
        self,
        prompt: str,
        num_frames: int = _DEFAULT_NUM_FRAMES,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a video with temporal consistency.

        .. deprecated::
            该方法仍保留但已弃用，未来将通过 L5 Pipeline 的视频节点
            提供等效功能。

        The generation enforces temporal consistency across frames:

        1. The consistency seed is locked to a single value for the
           entire video so that the base noise is correlated.
        2. Every ``reframe_interval`` frames the conditioning features
           are re-written to prevent drift.
        3. Frame-to-frame drift is detected via CLIP-I distance; when
           the drift exceeds :attr:`ConsistencyProfile.drift_threshold`
           the offending frame is locally re-generated.

        The current implementation is a placeholder that returns
        descriptor dictionaries for each frame, but the full temporal
        logic (seed locking, reframe intervals, drift detection +
        re-generation) is exercised.

        Args:
            prompt: The text prompt describing the desired video.
            num_frames: Number of frames to generate.  Defaults to
                ``16``.
            **kwargs: Additional generation parameters (e.g. ``seed``,
                ``width``, ``height``).

        Returns:
            A dictionary with keys:

            * ``frames`` -- a list of frame descriptors.
            * ``consistency_scores`` -- a :class:`ConsistencyScore`
              (as a dict) for the video.
            * ``metadata`` -- video generation metadata (prompt,
              num_frames, seed, drift_log, reframe_log).

        Raises:
            ValueError: If ``num_frames`` is outside
                ``[_FRAMES_MIN, _FRAMES_MAX]``.
        """
        warnings.warn(
            "ConsistencyPipeline.generate_video() 已弃用，未来将通过 "
            "L5 Pipeline 的视频节点提供等效功能。",
            DeprecationWarning,
            stacklevel=2,
        )
        if num_frames < _FRAMES_MIN or num_frames > _FRAMES_MAX:
            raise ValueError(
                "num_frames must be in [{}, {}], got {}.".format(
                    _FRAMES_MIN, _FRAMES_MAX, num_frames
                )
            )

        seed = kwargs.get("seed", _DEFAULT_VIDEO_SEED)
        width = kwargs.get("width", _DEFAULT_WIDTH)
        height = kwargs.get("height", _DEFAULT_HEIGHT)
        reframe_interval = self._profile.reframe_interval
        drift_threshold = self._profile.drift_threshold

        frames: List[Dict[str, Any]] = []
        drift_log: List[Dict[str, Any]] = []
        reframe_log: List[int] = []
        prev_frame: Optional[Dict[str, Any]] = None

        for frame_idx in range(num_frames):
            # --- Feature re-write every reframe_interval frames ---
            if frame_idx % reframe_interval == 0:
                reframe_log.append(frame_idx)
                self._reframe_features(frame_idx, seed)

            # --- Generate the frame (placeholder) ---
            frame = {
                "kind": "consistency_generated_frame",
                "prompt": prompt,
                "frame_index": frame_idx,
                "width": width,
                "height": height,
                "seed": seed,
            }

            # --- Apply conditioning ---
            if self._character is not None:
                frame["character"] = self._character.id
            if self._outfit is not None:
                frame["outfit"] = self._outfit.id
            if self._scene is not None:
                frame["scene"] = self._scene.id

            # --- Drift detection ---
            if prev_frame is not None:
                drift = self._detect_frame_drift(
                    prev_frame, frame, frame_idx
                )
                if drift["distance"] > drift_threshold:
                    drift_log.append(drift)
                    # --- Local re-generation ---
                    frame = self._regenerate_frame(
                        prompt, frame_idx, seed, width, height
                    )
                    drift["regenerated"] = True
                else:
                    drift["regenerated"] = False

            frames.append(frame)
            prev_frame = frame

        # --- Consistency scoring ---
        references = self._build_references()
        references["frames"] = frames
        score = self._scorer.calculate(
            frames[0] if frames else None, references
        )

        result = {
            "frames": frames,
            "consistency_scores": score.to_dict(),
            "metadata": {
                "prompt": prompt,
                "num_frames": num_frames,
                "seed": seed,
                "width": width,
                "height": height,
                "temporal_consistency": self._profile.temporal_consistency,
                "drift_threshold": drift_threshold,
                "reframe_interval": reframe_interval,
                "drift_log": drift_log,
                "reframe_log": reframe_log,
                "kwargs": dict(kwargs),
            },
        }
        self._logger.info(
            "Generated video (%d frames, %d drifts, %d reframes).",
            num_frames,
            len(drift_log),
            len(reframe_log),
        )
        return result

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
                or a result dictionary from :meth:`generate`).

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

    def _apply_character_conditioning(
        self,
        image: Dict[str, Any],
        character: CharacterAsset,
        weight: float,
    ) -> Dict[str, Any]:
        """Apply character conditioning to an image.

        Delegates to the :class:`CharacterEngine` when available;
        otherwise returns a descriptor dictionary.
        """
        if self._character_engine is not None:
            return self._character_engine.apply_character(
                image, character, weight=weight
            )
        return {
            "kind": "character_conditioned_image",
            "character_id": character.id,
            "weight": weight,
        }

    def _apply_outfit_conditioning(
        self,
        image: Dict[str, Any],
        outfit: OutfitAsset,
        weight: float,
    ) -> Dict[str, Any]:
        """Apply outfit conditioning to an image.

        Delegates to the :class:`OutfitEngine` when available;
        otherwise returns a descriptor dictionary.
        """
        if self._outfit_engine is not None:
            return self._outfit_engine.apply_outfit(
                image, outfit, weight=weight
            )
        return {
            "kind": "outfit_conditioned_image",
            "outfit_id": outfit.id,
            "weight": weight,
        }

    def _apply_scene_conditioning(
        self,
        image: Dict[str, Any],
        scene: SceneAsset,
        weight: float,
    ) -> Dict[str, Any]:
        """Apply scene conditioning to an image.

        Delegates to the :class:`SceneEngine` when available;
        otherwise returns a descriptor dictionary.
        """
        if self._scene_engine is not None:
            return self._scene_engine.apply_scene(
                image, scene, weight=weight
            )
        return {
            "kind": "scene_conditioned_image",
            "scene_id": scene.id,
            "weight": weight,
        }

    def _apply_depth_conditioning(
        self,
        image: Dict[str, Any],
        depth: DepthAsset,
        weight: float,
    ) -> Dict[str, Any]:
        """Apply depth conditioning to an image.

        Returns a descriptor dictionary encoding the depth asset id,
        method and weight.
        """
        return {
            "kind": "depth_conditioned_image",
            "depth_id": depth.id,
            "depth_method": depth.method,
            "weight": weight,
        }

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

    def _reframe_features(
        self, frame_idx: int, seed: int
    ) -> Dict[str, Any]:
        """Re-write conditioning features for temporal consistency.

        Args:
            frame_idx: The current frame index.
            seed: The locked consistency seed.

        Returns:
            A descriptor dictionary for the reframe operation.
        """
        self._logger.debug(
            "Reframing features at frame %d (seed=%d).",
            frame_idx, seed,
        )
        return {
            "kind": "feature_reframe",
            "frame_index": frame_idx,
            "seed": seed,
        }

    def _detect_frame_drift(
        self,
        prev_frame: Dict[str, Any],
        curr_frame: Dict[str, Any],
        frame_idx: int,
    ) -> Dict[str, Any]:
        """Detect drift between two consecutive frames.

        Uses a deterministic pseudo-distance derived from the frame
        indices as a placeholder for the real CLIP-I distance.

        Args:
            prev_frame: The previous frame descriptor.
            curr_frame: The current frame descriptor.
            frame_idx: The current frame index.

        Returns:
            A descriptor dictionary with keys ``frame_index``,
            ``distance`` and ``threshold``.
        """
        # Placeholder: deterministic pseudo-distance based on the
        # frame index so that the drift logic is exercised.
        # Values span [0, 0.18] so drift exceeds the default threshold
        # of 0.15 when frame_idx % 10 >= 8, triggering re-generation.
        distance = (frame_idx % 10) / 50.0
        return {
            "frame_index": frame_idx,
            "distance": distance,
            "threshold": self._profile.drift_threshold,
        }

    def _regenerate_frame(
        self,
        prompt: str,
        frame_idx: int,
        seed: int,
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        """Locally re-generate a frame that drifted.

        Args:
            prompt: The text prompt.
            frame_idx: The frame index to re-generate.
            seed: The locked consistency seed.
            width: Frame width.
            height: Frame height.

        Returns:
            A re-generated frame descriptor.
        """
        self._logger.debug(
            "Re-generating drifted frame %d.", frame_idx
        )
        return {
            "kind": "consistency_regenerated_frame",
            "prompt": prompt,
            "frame_index": frame_idx,
            "width": width,
            "height": height,
            "seed": seed,
            "regenerated": True,
        }

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
