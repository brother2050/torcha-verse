"""Consistency pipeline for the TorchaVerse consistency framework
(v0.3.0).

This module provides :class:`ConsistencyPipeline`, the top-level
orchestrator that composes the four consistency engines (character /
outfit / scene / depth) into a single generation surface.  It is the
"integration layer" of the consistency framework.

Capabilities:

* :meth:`ConsistencyPipeline.generate` -- generate a single image by
  applying all configured conditioning signals (character / outfit /
  scene / depth) according to the weights in the
  :class:`~consistency.profile.ConsistencyProfile`.
* :meth:`ConsistencyPipeline.generate_batch` -- generate a batch of
  images from a list of prompts while maintaining consistency across
  the batch.
* :meth:`ConsistencyPipeline.generate_video` -- generate a video
  (sequence of frames) with temporal consistency: the consistency seed
  is locked, features are re-written every ``reframe_interval`` frames,
  and drift is detected and corrected by local re-generation.
* :meth:`ConsistencyPipeline.score` -- evaluate the consistency of a
  generation output against the configured reference assets.

The pipeline delegates the actual conditioning to the four engines
(:class:`~consistency.character.CharacterEngine`,
:class:`~consistency.outfit.OutfitEngine`,
:class:`~consistency.scene.SceneEngine`) and the scoring to
:class:`~consistency.score.ScoreCalculator`.  The generation itself is
a placeholder that returns descriptor dictionaries; the full interface
is exercised so that the pipeline can be swapped for a real generation
backend without changing call sites.

Layering (L1 -> L4):

* L1 ``infrastructure`` -- logging.
* L2 ``assets`` -- asset types.
* L4 ``consistency`` (this module) -- pipeline orchestration.

This module depends on :mod:`torch` (transitively, through the score
calculator) and the L1/L2/L4 layers.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Sequence, Union

from assets.model_asset import CharacterAsset, DepthAsset, OutfitAsset, SceneAsset
from assets.store import AssetStore
from infrastructure.logger import get_logger

from .character import CharacterEngine
from .outfit import OutfitEngine
from .profile import ConsistencyProfile
from .scene import SceneEngine
from .score import ConsistencyScore, ScoreCalculator

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
_DIM_MAX: int = 4096

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
    """Top-level consistency pipeline composing the four engines.

    The pipeline is constructed with a :class:`ConsistencyProfile` and
    optional references to the four asset types (character / outfit /
    scene / depth).  When an :class:`~assets.store.AssetStore` is
    provided the four engines are created from it; otherwise the engines
    default to ``None`` and only the scoring surface is available.

    Args:
        profile: The :class:`ConsistencyProfile` controlling per-axis
            weights and temporal-consistency knobs.
        character: Optional :class:`CharacterAsset` to condition on.
        outfit: Optional :class:`OutfitAsset` to condition on.
        scene: Optional :class:`SceneAsset` to condition on.
        depth: Optional :class:`DepthAsset` to condition on.
        asset_store: Optional :class:`AssetStore` used to construct the
            four engines.  When ``None`` the engines are ``None`` and
            only :meth:`score` is functional.
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

        self._lock: threading.Lock = threading.Lock()
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
    # Single-image generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a single image with all consistency conditioning applied.

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
    # Batch generation
    # ------------------------------------------------------------------
    def generate_batch(
        self,
        prompts: Sequence[str],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generate a batch of images while maintaining consistency.

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
        results: List[Dict[str, Any]] = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        self._logger.info(
            "Generated batch of %d images.", len(results)
        )
        return results

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------
    def generate_video(
        self,
        prompt: str,
        num_frames: int = _DEFAULT_NUM_FRAMES,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a video with temporal consistency.

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
        image: Any,
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
        image: Any,
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
        image: Any,
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
        image: Any,
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
