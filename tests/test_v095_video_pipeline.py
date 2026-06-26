"""v0.9.5 video pipeline primitive tests.

Exercises the four v0.9.5 video-pipeline primitives that are
otherwise 100% untested in the current suite:

* :class:`models.video.motion_module.MotionModule` -- temporal
  attention + residual stack.
* :class:`models.video.motion_module.TemporalAttention` --
  constructor-time validation.
* :class:`models.video.frame_interpolator.FrameInterpolator`
  -- ``forward`` shape contract on a tiny volume.
* :func:`models.video.frame_interpolator.flow_warp` -- identity
  behaviour on zero flow + non-zero shift behaviour.
* :class:`models.providers.local_video.LocalTorchVideoProvider`
  -- ``from_file`` error path, ``VideoProviderConfig`` field
  contract, ``num_parameters`` introspection.

8 tests; all CPU-only.
"""
from __future__ import annotations

import pytest
import torch

from models.video.frame_interpolator import (
    FrameInterpolator,
    flow_warp,
)
from models.video.motion_module import MotionModule, TemporalAttention
from models.providers.local_video import (
    LocalTorchVideoProvider,
    TINY_VIDEO_CONFIG,
    VideoProviderConfig,
)


# ---------------------------------------------------------------------------
# Section 1 -- MotionModule + TemporalAttention (2 tests)
# ---------------------------------------------------------------------------
class TestMotionModule:
    """Tests for :class:`MotionModule` / :class:`TemporalAttention`."""

    def test_motion_module_forward_shape(self):
        """``MotionModule(hidden_size=64, num_heads=4, num_layers=2)``
        returns a tensor of the same shape as the input."""
        # tiny config: 64 channels, 2 attention layers, num_frames=16 default
        mm = MotionModule(hidden_size=64, num_heads=4, num_layers=2)
        mm.eval()
        # (B, C, T, H, W) = (2, 64, 8, 1, 1) -- channels-last-friendly
        x = torch.randn(2, 64, 8, 1, 1)
        with torch.no_grad():
            y = mm(x)
        # Shape is preserved through the residual stack.
        assert tuple(y.shape) == (2, 64, 8, 1, 1)

    def test_temporal_attention_raises_on_invalid_head_dim(self):
        """``TemporalAttention(hidden_size=63, num_heads=4)`` raises.

        ``63 % 4 != 0`` so the constructor must raise a clear
        error about the divisibility constraint.
        """
        with pytest.raises(ValueError) as exc_info:
            TemporalAttention(hidden_size=63, num_heads=4)
        # The error message should mention the divisibility constraint.
        msg = str(exc_info.value)
        assert "hidden_size" in msg
        assert "num_heads" in msg
        # And the word "divisible" / "must be" to convey the contract.
        assert "divisible" in msg or "must" in msg


# ---------------------------------------------------------------------------
# Section 2 -- FrameInterpolator / flow_warp (3 tests)
# ---------------------------------------------------------------------------
class TestFrameInterpolator:
    """Tests for :class:`FrameInterpolator` and :func:`flow_warp`."""

    def test_frame_interpolator_forward_at_half(self):
        """``FrameInterpolator()`` with default config returns a frame.

        We instantiate the interpolator and exercise its forward
        pass on a tiny 8x8 volume.  The shape contract is
        ``(B, C, H, W)``.
        """
        interp = FrameInterpolator()
        interp.eval()
        # Two deterministic frames so the result is reproducible.
        frame0 = torch.zeros(1, 3, 8, 8)
        frame1 = torch.ones(1, 3, 8, 8)
        with torch.no_grad():
            y = interp(frame0, frame1, t=0.5)
        # Shape matches the input layout (B, C, H, W).
        assert y.shape == (1, 3, 8, 8)
        # Output is bounded -- interpolator outputs a real-valued
        # tensor; we only check that it is finite (no NaN / Inf).
        assert torch.isfinite(y).all()

    def test_flow_warp_zero_is_identity(self):
        """``flow_warp(x, zero_flow) == x`` exactly (within fp tolerance)."""
        x = torch.randn(1, 3, 8, 8)
        zero_flow = torch.zeros(1, 2, 8, 8)
        y = flow_warp(x, zero_flow)
        # Identity (within float32 rounding).
        assert torch.allclose(x, y, atol=1e-6)

    def test_flow_warp_nonzero_shifts(self):
        """A non-zero flow shifts content to a new spatial position.

        We use a single white pixel on a black background and
        shift the sampling location by +2 in x.  ``flow_warp``
        interprets the flow as the displacement the sampler
        applies to the base grid: a positive flow of +2 moves the
        *source* coordinate to ``base + 2`` in pixel units, which
        in normalised coordinates shrinks toward -1 (i.e. toward
        the left edge).  We therefore assert the value moved to
        a *different* column (the +2 displacement) and that the
        source column went dark.
        """
        # 8x8 image with a single white pixel at (4, 4).
        x = torch.zeros(1, 1, 8, 8)
        x[0, 0, 4, 4] = 1.0
        # Shift by +2 pixels in x and 0 in y.
        flow = torch.zeros(1, 2, 8, 8)
        flow[0, 0] = 2.0
        y = flow_warp(x, flow)
        # The warped image must differ from the input (the
        # white pixel has moved) and the total energy must be
        # preserved (the shifted value is just relocated by
        # bilinear sampling).
        assert not torch.allclose(x, y), (
            "flow_warp with a non-zero flow must change the image"
        )
        # The white pixel has moved away from column 4.
        # The exact destination depends on the bilinear-sampling
        # convention; the safe invariant is that some other cell
        # in row 4 is now non-zero and the source cell is dark.
        assert y[0, 0, 4].max() < 1.0, (
            f"expected the source pixel (4, 4) to no longer be 1.0, "
            f"got row y[0,0,4,:] = {y[0, 0, 4, :]}"
        )
        assert y[0, 0, 4].sum() > 0.5, (
            f"expected the value to remain in row 4, got {y[0, 0, 4, :]}"
        )


# ---------------------------------------------------------------------------
# Section 3 -- LocalTorchVideoProvider / VideoProviderConfig (3 tests)
# ---------------------------------------------------------------------------
class TestLocalTorchVideoProvider:
    """Tests for the v0.4.x P0 :class:`LocalTorchVideoProvider`."""

    def test_local_video_provider_from_file_raises(self):
        """``from_file`` on a missing path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            LocalTorchVideoProvider.from_file("/nonexistent/path.safetensors")

    def test_video_provider_config_defaults(self):
        """``VideoProviderConfig()`` exposes the documented fields + to_dict."""
        cfg = VideoProviderConfig()
        # Required public fields are present and have the documented
        # defaults.
        assert cfg.name == "tiny"
        assert isinstance(cfg.dit_in_channels, int) and cfg.dit_in_channels > 0
        assert isinstance(cfg.dit_hidden_size, int) and cfg.dit_hidden_size > 0
        assert isinstance(cfg.dit_num_layers, int) and cfg.dit_num_layers > 0
        assert isinstance(cfg.dit_num_heads, int) and cfg.dit_num_heads > 0
        # patch_size is a tuple of three positive ints.
        assert len(cfg.dit_patch_size) == 3
        assert all(int(p) > 0 for p in cfg.dit_patch_size)
        # VAE + sampling defaults.
        assert isinstance(cfg.vae_in_channels, int) and cfg.vae_in_channels > 0
        assert (
            isinstance(cfg.vae_latent_channels, int)
            and cfg.vae_latent_channels > 0
        )
        assert isinstance(cfg.default_steps, int) and cfg.default_steps > 0
        assert isinstance(cfg.default_fps, int) and cfg.default_fps > 0
        assert isinstance(cfg.default_num_frames, int) and cfg.default_num_frames > 0
        assert isinstance(cfg.default_height, int) and cfg.default_height > 0
        assert isinstance(cfg.default_width, int) and cfg.default_width > 0
        # to_dict() returns a dict containing every field name.
        d = cfg.to_dict()
        assert isinstance(d, dict)
        for fname in (
            "name",
            "dit_in_channels",
            "dit_hidden_size",
            "dit_num_layers",
            "dit_num_heads",
            "dit_patch_size",
            "dit_num_frames",
            "dit_context_dim",
            "vae_in_channels",
            "vae_latent_channels",
            "vae_hidden_size",
            "vae_num_down_blocks",
            "vae_temporal_stride",
            "default_steps",
            "default_fps",
            "default_num_frames",
            "default_height",
            "default_width",
        ):
            assert fname in d, f"missing {fname!r} in to_dict() output"

    def test_local_video_provider_num_parameters_positive(self):
        """``LocalTorchVideoProvider.from_random(TINY_VIDEO_CONFIG)``
        has a positive parameter count."""
        provider = LocalTorchVideoProvider.from_random(TINY_VIDEO_CONFIG)
        n = provider.num_parameters()
        assert isinstance(n, int)
        assert n > 0, f"expected positive parameter count, got {n}"
