"""v1.0.0 video node input validation / resource estimation tests.

Exercises :meth:`validate_inputs` and :meth:`estimate_resources`
on the three v0.4.x P0 video capability nodes:

* :class:`nodes.video.VideoTxt2VidNode` -- text-to-video generation.
* :class:`nodes.video.VideoInterpolateNode` -- frame interpolation.
* :class:`nodes.video.VideoStitchNode` -- video clip stitching.

The fourth test exercises :meth:`VideoTxt2VidNode.estimate_resources`
on a valid input dict and verifies the documented resource keys.

4 tests; all CPU-only.
"""
from __future__ import annotations

import pytest

from nodes.base import NodeContext
from nodes.video import (
    VideoInterpolateNode,
    VideoStitchNode,
    VideoTxt2VidNode,
)


def _make_ctx() -> NodeContext:
    """Build a minimal :class:`NodeContext` for the validation tests.

    The validation code paths are pure -- they do not touch the
    context's executors or backend -- so a stripped-down context
    is sufficient.
    """
    return NodeContext()


# ---------------------------------------------------------------------------
# Section 1 -- validate_inputs (3 tests)
# ---------------------------------------------------------------------------
class TestVideoNodeValidation:
    """``validate_inputs`` rejections on the three video nodes."""

    def test_video_txt2vid_validate_width_too_small_raises(self):
        """``VideoTxt2VidNode.validate_inputs(width=32, height=64, ...)``
        returns at least one error mentioning the small width."""
        node = VideoTxt2VidNode()
        errors = node.validate_inputs(
            {
                "prompt": "a cat",
                "width": 32,        # below the documented minimum (64)
                "height": 64,
                "num_frames": 16,
                "fps": 24,
                "steps": 20,
                "guidance_scale": 7.5,
            }
        )
        # At least one error must mention the width.
        assert any("width" in err for err in errors), (
            f"expected an error mentioning 'width', got {errors}"
        )
        # And the safe-execute wrapper raises ``ValueError`` when
        # the validation list is non-empty -- this is the contract
        # the pipeline layer relies on.
        ctx = _make_ctx()
        with pytest.raises(ValueError):
            node._safe_execute(
                ctx,
                prompt="a cat",
                width=32, height=64,
                num_frames=16, fps=24,
                steps=20, guidance_scale=7.5,
            )

    def test_video_interpolate_validate_target_fps_out_of_range(self):
        """``VideoInterpolateNode.validate_inputs(target_fps=0, ...)``
        returns at least one error mentioning ``target_fps``."""
        node = VideoInterpolateNode()
        errors = node.validate_inputs(
            {
                "video": "placeholder",
                "target_fps": 0,   # below the documented minimum (1)
            }
        )
        # At least one error must mention target_fps.
        assert any("target_fps" in err for err in errors), (
            f"expected an error mentioning 'target_fps', got {errors}"
        )
        # And the safe-execute wrapper raises ``ValueError``.
        ctx = _make_ctx()
        with pytest.raises(ValueError):
            node._safe_execute(
                ctx,
                video="placeholder",
                target_fps=0,
            )

    def test_video_stitch_validate_empty_videos(self):
        """``VideoStitchNode.validate_inputs(videos=[], ...)``
        returns at least one error mentioning ``videos``."""
        node = VideoStitchNode()
        errors = node.validate_inputs(
            {
                "videos": [],   # empty -> at least one clip required
            }
        )
        # At least one error must mention videos.
        assert any("videos" in err for err in errors), (
            f"expected an error mentioning 'videos', got {errors}"
        )
        # And the safe-execute wrapper raises ``ValueError``.
        ctx = _make_ctx()
        with pytest.raises(ValueError):
            node._safe_execute(ctx, videos=[])


# ---------------------------------------------------------------------------
# Section 2 -- estimate_resources (1 test)
# ---------------------------------------------------------------------------
class TestVideoNodeResourceEstimation:
    """``estimate_resources`` returns a dict with the documented keys."""

    def test_video_txt2vid_estimate_resources_returns_dict(self):
        """``VideoTxt2VidNode().estimate_resources(...)`` returns a dict
        containing ``vram_gb``, ``ram_gb`` and ``time_s`` keys (or
        the legacy ``vram_mb`` / ``time_s`` aliases -- we accept
        either set of documented names)."""
        node = VideoTxt2VidNode()
        est = node.estimate_resources(
            {
                "prompt": "a cat",
                "width": 512,
                "height": 512,
                "num_frames": 16,
                "fps": 24,
                "steps": 20,
            }
        )
        # The result must be a dict.
        assert isinstance(est, dict)
        # The v0.4.x P0 video node uses ``vram_gb`` / ``ram_gb`` /
        # ``time_s`` keys (see :meth:`VideoTxt2VidNode.estimate_resources`).
        # Some earlier / later revisions used ``vram_mb`` / ``time_s``;
        # we accept any of the documented variants.
        vram_keys = ("vram_gb", "vram_mb")
        time_keys = ("time_s", "time_ms")
        ram_keys = ("ram_gb", "ram_mb")
        assert any(k in est for k in vram_keys), (
            f"expected one of {vram_keys} in estimate, got {list(est)}"
        )
        assert any(k in est for k in time_keys), (
            f"expected one of {time_keys} in estimate, got {list(est)}"
        )
        assert any(k in est for k in ram_keys), (
            f"expected one of {ram_keys} in estimate, got {list(est)}"
        )
        # All values are numeric and non-negative.
        for k, v in est.items():
            assert isinstance(v, (int, float)), (
                f"estimate[{k!r}] = {v!r} is not a number"
            )
            assert v >= 0, f"estimate[{k!r}] = {v!r} is negative"
