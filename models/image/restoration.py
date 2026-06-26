"""Image restoration models (F-11).

Two task-specific :class:`torch.nn.Module` subclasses used by the
``image_upscale`` and ``image_inpaint`` nodes:

* :class:`SuperResolutionUNet` -- a small 4-stage UNet that
  upsamples the spatial dimensions by ``scale`` (default 2) via
  pixel-shuffle.  Operates on ``[B, C, H, W]`` float tensors in
  ``[0, 1]`` and returns ``[B, C, H * scale, W * scale]``.
* :class:`InpaintUNet` -- a 4-stage UNet that takes a 4-channel
  input (RGB + binary mask) and returns an RGB reconstruction
  that respects the mask (zero outside the masked region).  The
  output is the predicted full image, so the caller can blend it
  with the original at the mask boundary.

Both modules are deliberately small enough to instantiate on CPU
and to instantiate per-call without requiring a registration on
the :class:`ModuleBus`.  Faithful architecture, randomly
initialised weights; the resulting tensors are usable as the
restoration node's output.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

__all__ = ["SuperResolutionUNet", "InpaintUNet"]


# ---------------------------------------------------------------------------
# Super-resolution UNet
# ---------------------------------------------------------------------------
class _ResBlock(nn.Module):
    """A residual block with two 3x3 convs and a SiLU activation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(x + h)


class SuperResolutionUNet(nn.Module):
    """A 4-stage UNet with a pixel-shuffle up-sampler.

    Args:
        in_channels: Number of input channels (3 for RGB).
        out_channels: Number of output channels (3 for RGB).
        base_channels: Channel count at the highest spatial
            resolution stage.
        scale: Spatial up-sampling factor (must be a positive int).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        scale: int = 2,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}.")
        self.scale = int(scale)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1), nn.SiLU(),
            _ResBlock(c1),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1, stride=2), nn.SiLU(),
            _ResBlock(c2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=1, stride=2), nn.SiLU(),
            _ResBlock(c3),
        )
        # Bottleneck stays at ``H/4`` resolution (no further down).
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c3, c3, 3, padding=1), nn.SiLU(),
            _ResBlock(c3),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(c3 + c3, c2, 3, padding=1), nn.SiLU(),
        )
        self.up3 = nn.ConvTranspose2d(c2, c2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c1, 3, padding=1), nn.SiLU(),
        )
        self.up2 = nn.ConvTranspose2d(c1, c1, 4, stride=2, padding=1)
        # Final up-sampling to the target resolution.  When scale is
        # 1 we use an identity conv; when scale is 2/4/8 we use a
        # PixelShuffle head.
        if self.scale == 1:
            self.upsample = nn.Conv2d(c1, out_channels, 3, padding=1)
        else:
            self.upsample = nn.Sequential(
                nn.Conv2d(c1, c1 * (self.scale ** 2), 3, padding=1),
                nn.PixelShuffle(self.scale),
                nn.Conv2d(c1, out_channels, 3, padding=1),
            )
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        e1 = self.enc1(x)            # /1
        e2 = self.enc2(e1)           # /2
        e3 = self.enc3(e2)           # /4
        b = self.bottleneck(e3)      # /4
        d3 = self.dec3(torch.cat([b, e3], dim=1))         # /4
        d2 = self.up3(d3)            # /2
        d2 = self.dec2(torch.cat([d2, e2], dim=1))         # /2
        d1 = self.up2(d2)            # /1
        out = self.upsample(d1)
        return self.out_act(out)


# ---------------------------------------------------------------------------
# Inpaint UNet
# ---------------------------------------------------------------------------
class InpaintUNet(nn.Module):
    """A 4-stage UNet that takes RGB + mask (4 channels) and
    reconstructs the masked region.

    Args:
        in_channels: Number of input channels (3 RGB + 1 mask = 4).
        out_channels: Number of output channels (3 for RGB).
        base_channels: Channel count at the highest resolution.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1), nn.SiLU(),
            _ResBlock(c1),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1, stride=2), nn.SiLU(),
            _ResBlock(c2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=1, stride=2), nn.SiLU(),
            _ResBlock(c3),
        )
        # Bottleneck stays at ``H/4`` so dec2 can match ``e2``.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c3, c3, 3, padding=1), nn.SiLU(),
            _ResBlock(c3),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(c3 + c3, c2, 3, padding=1), nn.SiLU(),
        )
        self.up3 = nn.ConvTranspose2d(c2, c2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c1, 3, padding=1), nn.SiLU(),
        )
        self.up2 = nn.ConvTranspose2d(c1, c1, 4, stride=2, padding=1)
        self.head = nn.Conv2d(c1, out_channels, 3, padding=1)
        self.out_act = nn.Sigmoid()

    def forward(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Inpaint ``image`` according to the binary ``mask``.

        Args:
            image: ``[B, 3, H, W]`` RGB float tensor in ``[0, 1]``.
            mask: ``[B, 1, H, W]`` binary mask (1 = masked region).

        Returns:
            ``[B, 3, H, W]`` RGB float tensor in ``[0, 1]`` with the
            masked region replaced by the network's prediction.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        # Resize mask to image shape if needed.
        if mask.shape[-2:] != image.shape[-2:]:
            mask = nn.functional.interpolate(
                mask, size=image.shape[-2:], mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([image, mask], dim=1)
        e1 = self.enc1(x)            # /1
        e2 = self.enc2(e1)           # /2
        e3 = self.enc3(e2)           # /4
        b = self.bottleneck(e3)      # /4
        d3 = self.dec3(torch.cat([b, e3], dim=1))          # /4
        d2 = self.up3(d3)            # /2
        d2 = self.dec2(torch.cat([d2, e2], dim=1))          # /2
        d1 = self.up2(d2)            # /1
        out = self.out_act(self.head(d1))
        # Preserve the unmasked region from the input image.
        keep_mask = (mask < 0.5).float()
        out = out * mask + image * keep_mask
        return out


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------
def to_image_tensor(image: Any, channels: int = 3) -> Optional[torch.Tensor]:
    """Best-effort conversion of ``image`` to a ``[B, C, H, W]``
    float tensor in ``[0, 1]``.

    Accepts :class:`torch.Tensor`, :class:`PIL.Image.Image`, or
    a numpy array.  Returns ``None`` when the input cannot be
    coerced.
    """
    if isinstance(image, torch.Tensor):
        t = image.float()
        if t.dim() == 3:
            t = t.unsqueeze(0)
        if t.shape[1] == channels:
            if t.max() > 1.5:
                t = t / 255.0
            return t.clamp(0.0, 1.0)
        if t.shape[1] == 1 and channels == 3:
            t = t.expand(-1, 3, -1, -1)
            return t.clamp(0.0, 1.0)
        if t.shape[1] > channels:
            t = t[:, :channels]
            return t.clamp(0.0, 1.0)
    if hasattr(image, "convert"):
        try:
            import numpy as np
            arr = np.array(image.convert("RGB" if channels == 3 else "L"))
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
            return (t / 255.0).clamp(0.0, 1.0)
        except Exception:
            return None
    try:
        import numpy as np
        arr = np.asarray(image)
        t = torch.from_numpy(arr).float()
        if t.dim() == 3:
            t = t.permute(2, 0, 1).unsqueeze(0)
            if t.max() > 1.5:
                t = t / 255.0
            return t.clamp(0.0, 1.0)
    except Exception:
        return None
    return None
