"""Variational Auto-Encoder (VAE) for image latent-space modelling.

This module implements a convolutional VAE that maps between pixel space
and a lower-dimensional latent space.  It is the component used by
latent diffusion models (e.g. Stable Diffusion) to compress images
before diffusion.

The encoder performs strided convolutions to downsample the spatial
resolution, while the decoder uses transposed convolutions (or
interpolation + conv) to upsample back to the original resolution.

Key components:

* :class:`ResBlock` -- residual block with GroupNorm + SiLU.
* :class:`Encoder` -- convolutional downsampling encoder.
* :class:`Decoder` -- convolutional upsampling decoder.
* :class:`VAE` -- the full variational auto-encoder.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel

__all__ = ["ResBlock", "Encoder", "Decoder", "VAE"]


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


def _conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: Optional[int] = None,
) -> nn.Conv2d:
    """Helper to create a conv layer with default same-padding."""
    if padding is None:
        padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)


class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        groups: Number of GroupNorm groups.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.norm1: nn.GroupNorm = nn.GroupNorm(_num_groups(in_channels, groups), in_channels)
        self.conv1: nn.Conv2d = _conv(in_channels, out_channels)
        self.norm2: nn.GroupNorm = nn.GroupNorm(_num_groups(out_channels, groups), out_channels)
        self.conv2: nn.Conv2d = _conv(out_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut: nn.Module = _conv(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block.

        Args:
            x: Input tensor of shape ``(batch, in_channels, H, W)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, H, W)``.
        """
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class Encoder(nn.Module):
    """Convolutional downsampling encoder.

    Applies a sequence of residual blocks followed by strided
    convolutions to downsample the spatial resolution by
    ``2 ** num_down_blocks``.

    Args:
        in_channels: Input image channels.
        hidden_size: Base channel width.
        latent_channels: Number of latent channels.
        num_res_blocks: Residual blocks per downsample stage.
        num_down_blocks: Number of downsample stages.
        groups: GroupNorm groups.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 512,
        latent_channels: int = 4,
        num_res_blocks: int = 2,
        num_down_blocks: int = 3,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        self.hidden_size: int = hidden_size
        self.latent_channels: int = latent_channels
        self.num_down_blocks: int = num_down_blocks

        # Initial conv.
        layers: List[nn.Module] = [_conv(in_channels, hidden_size)]

        channels = hidden_size
        for _ in range(num_down_blocks):
            for _ in range(num_res_blocks):
                layers.append(ResBlock(channels, channels, groups))
            # Downsample.
            layers.append(_conv(channels, channels * 2, stride=2))
            channels *= 2

        for _ in range(num_res_blocks):
            layers.append(ResBlock(channels, channels, groups))

        # Output norm + projection to (mean, logvar).
        layers.append(nn.GroupNorm(_num_groups(channels, groups), channels))
        layers.append(_conv(channels, 2 * latent_channels))

        self.model: nn.Sequential = nn.Sequential(*layers)
        self.out_channels: int = channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode an image into latent distribution parameters.

        Args:
            x: Image tensor of shape ``(batch, in_channels, H, W)``.

        Returns:
            A tuple ``(mean, logvar)`` each of shape
            ``(batch, latent_channels, H/2**n, W/2**n)``.
        """
        h = self.model(x)
        mean, logvar = h.chunk(2, dim=1)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        return mean, logvar


class Decoder(nn.Module):
    """Convolutional upsampling decoder.

    Args:
        latent_channels: Number of latent channels.
        hidden_size: Base channel width (should match the encoder's
            final channel width).
        out_channels: Output image channels.
        num_res_blocks: Residual blocks per upsample stage.
        num_up_blocks: Number of upsample stages.
        groups: GroupNorm groups.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        hidden_size: int = 512,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        num_up_blocks: int = 3,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.latent_channels: int = latent_channels
        self.hidden_size: int = hidden_size
        self.out_channels: int = out_channels
        self.num_up_blocks: int = num_up_blocks

        layers: List[nn.Module] = [_conv(latent_channels, hidden_size)]

        channels = hidden_size
        for _ in range(num_up_blocks):
            for _ in range(num_res_blocks):
                layers.append(ResBlock(channels, channels, groups))
            # Upsample.
            layers.append(nn.Upsample(scale_factor=2.0, mode="nearest"))
            layers.append(_conv(channels, channels // 2))
            channels //= 2

        for _ in range(num_res_blocks):
            layers.append(ResBlock(channels, channels, groups))

        layers.append(nn.GroupNorm(_num_groups(channels, groups), channels))
        layers.append(_conv(channels, out_channels))

        self.model: nn.Sequential = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor back to image space.

        Args:
            z: Latent tensor of shape ``(batch, latent_channels, h, w)``.

        Returns:
            Reconstructed image of shape
            ``(batch, out_channels, h*2**n, w*2**n)``.
        """
        return self.model(z)


class VAE(BaseModel):
    """Variational Auto-Encoder for images.

    Maps images to a latent distribution (encode) and back (decode),
    using the reparameterisation trick for differentiable sampling.

    Args:
        in_channels: Input image channels.
        latent_channels: Number of latent channels.
        hidden_size: Base channel width.
        num_res_blocks: Residual blocks per stage.
        num_down_blocks: Number of downsample stages (spatial
            downsampling factor is ``2 ** num_down_blocks``).
        groups: GroupNorm groups.
        scaling_factor: Latent normalisation factor (as in Stable
            Diffusion).  Latents are divided by this factor before
            decoding and multiplied after encoding.
        config: Optional configuration dictionary.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        hidden_size: int = 512,
        num_res_blocks: int = 2,
        num_down_blocks: int = 3,
        groups: int = 32,
        scaling_factor: float = 0.18215,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            in_channels = config.get("in_channels", in_channels)
            latent_channels = config.get("latent_channels", latent_channels)
            hidden_size = config.get("hidden_size", hidden_size)
            num_res_blocks = config.get("num_res_blocks", num_res_blocks)
            num_down_blocks = config.get("num_down_blocks", num_down_blocks)
            groups = config.get("groups", groups)
            scaling_factor = config.get("scaling_factor", scaling_factor)

        super().__init__(config=config)

        self.in_channels: int = in_channels
        self.latent_channels: int = latent_channels
        self.hidden_size: int = hidden_size
        self.num_down_blocks: int = num_down_blocks
        self.scaling_factor: float = scaling_factor

        self.encoder: Encoder = Encoder(
            in_channels=in_channels,
            hidden_size=hidden_size,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            num_down_blocks=num_down_blocks,
            groups=groups,
        )
        self.decoder: Decoder = Decoder(
            latent_channels=latent_channels,
            hidden_size=self.encoder.out_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            num_up_blocks=num_down_blocks,
            groups=groups,
        )

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode an image to ``(mean, logvar)``.

        Args:
            x: Image tensor of shape ``(batch, in_channels, H, W)``.

        Returns:
            ``(mean, logvar)`` latent distribution parameters.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor to an image.

        Args:
            z: Latent tensor (already scaled).

        Returns:
            Reconstructed image.
        """
        z = z / self.scaling_factor
        return self.decoder(z)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from the latent distribution via the reparameterisation trick.

        ``z = mean + std * eps`` where ``eps ~ N(0, I)`` and
        ``std = exp(0.5 * logvar)``.

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
        x: torch.Tensor,
        sample: bool = True,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full VAE forward pass.

        Args:
            x: Input image of shape ``(batch, in_channels, H, W)``.
            sample: Whether to sample from the latent distribution
                (training) or use the mean (inference).

        Returns:
            A tuple ``(recon, mean, logvar)``.
        """
        mean, logvar = self.encode(x)
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
        """Generate images from latents (or random noise).

        Args:
            latents: Optional pre-scaled latent tensor.  When ``None``
                random latents of the given ``shape`` are sampled.
            shape: Shape of the random latents
                ``(batch, latent_channels, h, w)`` (used when
                ``latents`` is ``None``).

        Returns:
            Generated image tensor.
        """
        self.eval()
        if latents is None:
            if shape is None:
                raise ValueError("Either latents or shape must be provided.")
            latents = torch.randn(shape, device=next(self.parameters()).device)
        return self.decode(latents)
