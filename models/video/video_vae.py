"""Spatiotemporal Variational Auto-Encoder for video.

This module implements a 3D convolutional VAE that compresses video
data (``[batch, channels, time, height, width]``) into a lower-
dimensional spatiotemporal latent space.  It is the component used by
video diffusion models (e.g. Sora, Stable Video Diffusion) to compress
video before diffusion.

Key components:

* :class:`ResBlock3D` -- 3D residual block with GroupNorm + SiLU.
* :class:`Encoder3D` -- 3D convolutional downsampling encoder.
* :class:`Decoder3D` -- 3D convolutional upsampling decoder.
* :class:`VideoVAE` -- the full spatiotemporal VAE.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model_registry import BaseModel

__all__ = ["ResBlock3D", "Encoder3D", "Decoder3D", "VideoVAE"]


def _num_groups(channels: int, groups: int = 32) -> int:
    """Return the largest divisor of ``channels`` that is ``<= groups``.

    ``nn.GroupNorm`` requires ``channels`` to be divisible by the number
    of groups.  When the channel count is small or not a multiple of
    ``groups`` we fall back to the largest valid divisor.
    """
    g = min(groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


def _conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Optional[int] = None,
) -> nn.Conv3d:
    """Helper to create a 3-D conv layer with same-padding."""
    if padding is None:
        padding = kernel_size // 2
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)


class ResBlock3D(nn.Module):
    """3-D residual block with GroupNorm and SiLU.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        groups: GroupNorm groups.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.norm1: nn.GroupNorm = nn.GroupNorm(_num_groups(in_channels, groups), in_channels)
        self.conv1: nn.Conv3d = _conv3d(in_channels, out_channels)
        self.norm2: nn.GroupNorm = nn.GroupNorm(_num_groups(out_channels, groups), out_channels)
        self.conv2: nn.Conv3d = _conv3d(out_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut: nn.Module = _conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the 3-D residual block.

        Args:
            x: Input tensor of shape ``(batch, in_channels, T, H, W)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, T, H, W)``.
        """
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class Encoder3D(nn.Module):
    """3-D convolutional downsampling encoder.

    Downsamples both spatially and temporally.

    Args:
        in_channels: Input video channels.
        hidden_size: Base channel width.
        latent_channels: Number of latent channels.
        num_res_blocks: Residual blocks per stage.
        num_down_blocks: Number of downsample stages.
        temporal_stride: Temporal stride of the final downsample.
        groups: GroupNorm groups.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 128,
        latent_channels: int = 16,
        num_res_blocks: int = 2,
        num_down_blocks: int = 3,
        temporal_stride: int = 1,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        self.hidden_size: int = hidden_size
        self.latent_channels: int = latent_channels
        self.num_down_blocks: int = num_down_blocks

        layers: List[nn.Module] = [_conv3d(in_channels, hidden_size)]

        channels = hidden_size
        for i in range(num_down_blocks):
            for _ in range(num_res_blocks):
                layers.append(ResBlock3D(channels, channels, groups))
            # Downsample: spatial stride 2, temporal stride on last stage.
            t_stride = temporal_stride if i == num_down_blocks - 1 else 1
            layers.append(_conv3d(channels, channels * 2, stride=(t_stride, 2, 2)))
            channels *= 2

        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(channels, channels, groups))

        layers.append(nn.GroupNorm(_num_groups(channels, groups), channels))
        layers.append(_conv3d(channels, 2 * latent_channels))

        self.model: nn.Sequential = nn.Sequential(*layers)
        self.out_channels: int = channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a video into latent distribution parameters.

        Args:
            x: Video tensor of shape ``(batch, in_channels, T, H, W)``.

        Returns:
            ``(mean, logvar)`` each of shape
            ``(batch, latent_channels, T', H', W')``.
        """
        h = self.model(x)
        mean, logvar = h.chunk(2, dim=1)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        return mean, logvar


class Decoder3D(nn.Module):
    """3-D convolutional upsampling decoder.

    Args:
        latent_channels: Number of latent channels.
        hidden_size: Base channel width.
        out_channels: Output video channels.
        num_res_blocks: Residual blocks per stage.
        num_up_blocks: Number of upsample stages.
        temporal_stride: Temporal upsample factor of the first stage.
        groups: GroupNorm groups.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        hidden_size: int = 128,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        num_up_blocks: int = 3,
        temporal_stride: int = 1,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.latent_channels: int = latent_channels
        self.hidden_size: int = hidden_size
        self.out_channels: int = out_channels
        self.num_up_blocks: int = num_up_blocks

        layers: List[nn.Module] = [_conv3d(latent_channels, hidden_size)]

        channels = hidden_size
        for i in range(num_up_blocks):
            for _ in range(num_res_blocks):
                layers.append(ResBlock3D(channels, channels, groups))
            # Upsample: spatial x2, temporal on first stage.
            t_stride = temporal_stride if i == 0 else 1
            layers.append(nn.Upsample(scale_factor=(t_stride, 2.0, 2.0), mode="trilinear", align_corners=False))
            layers.append(_conv3d(channels, channels // 2))
            channels //= 2

        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(channels, channels, groups))

        layers.append(nn.GroupNorm(_num_groups(channels, groups), channels))
        layers.append(_conv3d(channels, out_channels))

        self.model: nn.Sequential = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor back to video space.

        Args:
            z: Latent tensor of shape ``(batch, latent_channels, T', H', W')``.

        Returns:
            Reconstructed video of shape
            ``(batch, out_channels, T, H, W)``.
        """
        return self.model(z)


class VideoVAE(BaseModel):
    """Spatiotemporal Variational Auto-Encoder for video.

    Args:
        in_channels: Input video channels.
        latent_channels: Number of latent channels.
        hidden_size: Base channel width.
        num_res_blocks: Residual blocks per stage.
        num_down_blocks: Number of downsample stages.
        temporal_stride: Temporal compression factor.
        scaling_factor: Latent normalisation factor.
        config: Optional configuration dictionary.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        hidden_size: int = 128,
        num_res_blocks: int = 2,
        num_down_blocks: int = 3,
        temporal_stride: int = 1,
        scaling_factor: float = 0.18215,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            in_channels = config.get("in_channels", in_channels)
            latent_channels = config.get("latent_channels", latent_channels)
            hidden_size = config.get("hidden_size", hidden_size)
            num_res_blocks = config.get("num_res_blocks", num_res_blocks)
            num_down_blocks = config.get("num_down_blocks", num_down_blocks)
            temporal_stride = config.get("temporal_stride", temporal_stride)
            scaling_factor = config.get("scaling_factor", scaling_factor)

        super().__init__(config=config)

        self.in_channels: int = in_channels
        self.latent_channels: int = latent_channels
        self.hidden_size: int = hidden_size
        self.num_down_blocks: int = num_down_blocks
        self.scaling_factor: float = scaling_factor

        self.encoder: Encoder3D = Encoder3D(
            in_channels=in_channels,
            hidden_size=hidden_size,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            num_down_blocks=num_down_blocks,
            temporal_stride=temporal_stride,
        )
        self.decoder: Decoder3D = Decoder3D(
            latent_channels=latent_channels,
            hidden_size=self.encoder.out_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            num_up_blocks=num_down_blocks,
            temporal_stride=temporal_stride,
        )

    # ------------------------------------------------------------------
    def encode(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a video into ``(mean, logvar)``.

        Args:
            video: Video tensor of shape ``(batch, in_channels, T, H, W)``.

        Returns:
            ``(mean, logvar)`` latent distribution parameters.
        """
        return self.encoder(video)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor to a video.

        Args:
            z: Latent tensor (already scaled).

        Returns:
            Reconstructed video.
        """
        z = z / self.scaling_factor
        return self.decoder(z)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from the latent distribution via the reparameterisation trick.

        Args:
            mean: Mean of the latent distribution.
            logvar: Log-variance of the latent distribution.

        Returns:
            A sampled latent tensor.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(mean)
            return mean + eps * std
        return mean

    def forward(
        self,
        video: torch.Tensor,
        sample: bool = True,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full video VAE forward pass.

        Args:
            video: Input video of shape ``(batch, in_channels, T, H, W)``.
            sample: Whether to sample from the latent distribution.

        Returns:
            ``(recon, mean, logvar)``.
        """
        mean, logvar = self.encode(video)
        if sample:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        z = z * self.scaling_factor
        recon = self.decode(z)
        return recon, mean, logvar

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        latents: Optional[torch.Tensor] = None,
        shape: Optional[Tuple[int, ...]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate video from latents (or random noise).

        Args:
            latents: Optional pre-scaled latent tensor.
            shape: Shape of the random latents
                ``(batch, latent_channels, T, h, w)``.

        Returns:
            Generated video tensor.
        """
        self.eval()
        if latents is None:
            if shape is None:
                raise ValueError("Either latents or shape must be provided.")
            latents = torch.randn(shape, device=next(self.parameters()).device)
        return self.decode(latents)
