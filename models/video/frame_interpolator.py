"""Frame interpolation network.

This module implements a frame interpolation model that synthesises
intermediate frames between two given frames.  It estimates
bidirectional optical flow, warps the two input frames, and fuses the
warped frames with a residual network conditioned on the interpolation
time ``t``.

Key components:

* :class:`FlowEstimator` -- estimates bidirectional optical flow.
* :class:`FrameInterpolator` -- the full interpolation network.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel

__all__ = ["FlowEstimator", "FrameInterpolator", "flow_warp"]


def flow_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
) -> torch.Tensor:
    """Warp an image tensor according to an optical flow field.

    Uses bilinear sampling with ``grid_sample``.

    Args:
        x: Image tensor of shape ``(batch, channels, height, width)``.
        flow: Optical flow of shape ``(batch, 2, height, width)`` where
            channel 0 is the x-displacement and channel 1 the
            y-displacement (in pixels).

    Returns:
        Warped image of the same shape as ``x``.
    """
    batch, _, height, width = x.shape
    # Build a base grid of normalised coordinates in [-1, 1].
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=x.device),
        torch.linspace(-1.0, 1.0, width, device=x.device),
        indexing="ij",
    )
    base = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)  # (1, 2, H, W)

    # Convert pixel displacements to normalised displacements.
    flow_x = flow[:, 0] * (2.0 / max(width - 1, 1))
    flow_y = flow[:, 1] * (2.0 / max(height - 1, 1))
    flow_norm = torch.stack((flow_x, flow_y), dim=1)

    grid = base + flow_norm  # (batch, 2, H, W)
    grid = grid.permute(0, 2, 3, 1)  # (batch, H, W, 2) for grid_sample
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


class FlowEstimator(nn.Module):
    """Bidirectional optical flow estimator.

    Estimates the forward flow (frame0 -> frame1) and backward flow
    (frame1 -> frame0).

    Args:
        in_channels: Input image channels.
        hidden_size: Base channel width.
        num_layers: Number of conv layers.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 64,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        layers: list = []
        in_ch = in_channels * 2  # concatenated frames
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_ch, hidden_size, 3, padding=1))
            layers.append(nn.LeakyReLU(0.1))
            in_ch = hidden_size
        layers.append(nn.Conv2d(hidden_size, 4, 3, padding=1))  # 2 flows * 2 channels
        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate bidirectional flow.

        Args:
            frame0: First frame ``(batch, in_channels, H, W)``.
            frame1: Second frame ``(batch, in_channels, H, W)``.

        Returns:
            Flow tensor of shape ``(batch, 4, H, W)`` where the first
            two channels are the forward flow and the last two the
            backward flow.
        """
        x = torch.cat([frame0, frame1], dim=1)
        return self.net(x)


class FrameInterpolator(BaseModel):
    """Frame interpolation network.

    Estimates bidirectional optical flow, warps both input frames to the
    intermediate time ``t``, and fuses them with a residual network.

    Args:
        in_channels: Input image channels.
        hidden_size: Base channel width.
        num_layers: Number of fusion conv layers.
        config: Optional configuration dictionary.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 64,
        num_layers: int = 6,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            in_channels = config.get("in_channels", in_channels)
            hidden_size = config.get("hidden_size", hidden_size)
            num_layers = config.get("num_layers", num_layers)

        super().__init__(config=config)

        self.in_channels: int = in_channels
        self.hidden_size: int = hidden_size

        self.flow_estimator: FlowEstimator = FlowEstimator(
            in_channels=in_channels,
            hidden_size=hidden_size,
        )

        # Fusion network: input = warped0 + warped1 + difference + t.
        fusion_in = in_channels * 3 + 1
        layers: list = []
        in_ch = fusion_in
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_ch, hidden_size, 3, padding=1))
            layers.append(nn.LeakyReLU(0.1))
            in_ch = hidden_size
        layers.append(nn.Conv2d(hidden_size, in_channels, 3, padding=1))
        self.fusion: nn.Sequential = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def forward(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        t: float = 0.5,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Interpolate a frame between ``frame0`` and ``frame1``.

        Args:
            frame0: First frame ``(batch, in_channels, H, W)``.
            frame1: Second frame ``(batch, in_channels, H, W)``.
            t: Interpolation position in ``[0, 1]`` (``0`` = frame0,
                ``1`` = frame1).

        Returns:
            Interpolated frame ``(batch, in_channels, H, W)``.
        """
        batch, _, height, width = frame0.shape

        # Estimate bidirectional flow.
        flow = self.flow_estimator(frame0, frame1)
        flow_forward = flow[:, :2]   # frame0 -> frame1
        flow_backward = flow[:, 2:]  # frame1 -> frame0

        # Scale flows by the interpolation time.
        t_val = float(t) if isinstance(t, torch.Tensor) else t
        t_tensor = frame0.new_tensor(t_val)
        warped0 = flow_warp(frame0, flow_forward * t_tensor)
        warped1 = flow_warp(frame1, flow_backward * (1.0 - t_tensor))

        # Linear blend as a fallback.
        blend = warped0 * (1.0 - t_tensor) + warped1 * t_tensor

        # Residual fusion conditioned on t.
        t_map = torch.full((batch, 1, height, width), t_val, device=frame0.device, dtype=frame0.dtype)
        diff = warped0 - warped1
        fusion_input = torch.cat([warped0, warped1, diff, t_map], dim=1)
        residual = self.fusion(fusion_input)

        return blend + residual

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        t: float = 0.5,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate an interpolated frame (inference mode).

        Args:
            frame0: First frame.
            frame1: Second frame.
            t: Interpolation position in ``[0, 1]``.

        Returns:
            Interpolated frame.
        """
        self.eval()
        return self.forward(frame0, frame1, t=t)
