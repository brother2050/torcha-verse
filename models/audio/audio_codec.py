"""Neural audio codec with Residual Vector Quantisation (RVQ).

This module implements an EnCodec-style audio codec that compresses a
raw waveform into discrete tokens via a convolutional encoder /
decoder and a stack of residual vector quantisers (RVQ).

Key components:

* :class:`Encoder` -- strided 1-D convolution encoder.
* :class:`Decoder` -- transposed 1-D convolution decoder.
* :class:`ResidualVectorQuantizer` -- multi-codebook RVQ.
* :class:`AudioCodec` -- the full codec model.

Reference:
    Défossez et al., "High Fidelity Neural Audio Compression" (2022).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model_registry import BaseModel

__all__ = ["Encoder", "Decoder", "ResidualVectorQuantizer", "AudioCodec"]


class _ConvBlock(nn.Module):
    """A Conv1d + GroupNorm + ELU block used by the encoder/decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv: nn.Conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.norm: nn.GroupNorm = nn.GroupNorm(1, out_channels)
        self.act: nn.Module = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Encoder(nn.Module):
    """Strided 1-D convolution encoder.

    Downsamples the waveform by the product of all strides.

    Args:
        in_channels: Input audio channels (1 = mono).
        hidden_size: Base channel width.
        latent_size: Output (latent) channels.
        strides: List of strides for each down-sampling stage.
        kernel_size: Convolution kernel size.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_size: int = 64,
        latent_size: int = 32,
        strides: Optional[List[int]] = None,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if strides is None:
            strides = [2, 4, 5, 8]
        self.strides: List[int] = strides
        self.downsample_factor: int = 1
        for s in strides:
            self.downsample_factor *= s

        layers: List[nn.Module] = [_ConvBlock(in_channels, hidden_size, kernel_size)]
        channels = hidden_size
        for stride in strides:
            out_ch = min(channels * 2, hidden_size * 4)
            layers.append(_ConvBlock(channels, out_ch, kernel_size, stride=stride))
            channels = out_ch
        layers.append(nn.Conv1d(channels, latent_size, 1))
        self.model: nn.Sequential = nn.Sequential(*layers)
        self.out_channels: int = latent_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a waveform into a continuous latent.

        Args:
            x: Waveform of shape ``(batch, in_channels, time)``.

        Returns:
            Latent of shape ``(batch, latent_size, time / downsample_factor)``.
        """
        return self.model(x)


class Decoder(nn.Module):
    """Transposed 1-D convolution decoder.

    Args:
        latent_size: Input (latent) channels.
        hidden_size: Base channel width.
        out_channels: Output audio channels.
        strides: List of strides (reversed for up-sampling).
        kernel_size: Convolution kernel size.
    """

    def __init__(
        self,
        latent_size: int = 32,
        hidden_size: int = 64,
        out_channels: int = 1,
        strides: Optional[List[int]] = None,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if strides is None:
            strides = [2, 4, 5, 8]
        self.strides: List[int] = list(reversed(strides))

        # Reverse the channel progression of the encoder.
        channels = hidden_size * (2 ** len(self.strides))
        channels = min(channels, hidden_size * 4)
        layers: List[nn.Module] = [nn.Conv1d(latent_size, channels, 1)]
        for stride in self.strides:
            out_ch = max(channels // 2, hidden_size)
            layers.append(nn.ConvTranspose1d(
                channels, out_ch, kernel_size, stride=stride,
                padding=kernel_size // 2, output_padding=stride - 1,
            ))
            layers.append(nn.GroupNorm(1, out_ch))
            layers.append(nn.ELU())
            channels = out_ch
        layers.append(nn.Conv1d(channels, out_channels, kernel_size, padding=kernel_size // 2))
        self.model: nn.Sequential = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent back to a waveform.

        Args:
            z: Latent of shape ``(batch, latent_size, time)``.

        Returns:
            Waveform of shape ``(batch, out_channels, time * upsample_factor)``.
        """
        return self.model(z)


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantisation (RVQ).

    A stack of ``num_quantizers`` vector quantisers, each operating on
    the residual of the previous one.  This allows a trade-off between
    bitrate and quality by using more or fewer codebooks.

    Args:
        latent_size: Dimension of the vectors to quantise.
        num_quantizers: Number of codebooks (quantisation stages).
        codebook_size: Number of entries per codebook.
        commitment_weight: Weight of the commitment loss.
    """

    def __init__(
        self,
        latent_size: int = 32,
        num_quantizers: int = 4,
        codebook_size: int = 1024,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.latent_size: int = latent_size
        self.num_quantizers: int = num_quantizers
        self.codebook_size: int = codebook_size
        self.commitment_weight: float = commitment_weight

        self.codebooks: nn.Parameter = nn.Parameter(
            torch.randn(num_quantizers, codebook_size, latent_size) * 0.02
        )

    # ------------------------------------------------------------------
    def _quantize_single(
        self,
        x: torch.Tensor,
        codebook: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantise ``x`` with a single codebook.

        Args:
            x: Vectors of shape ``(batch, dim, time)``.
            codebook: Codebook of shape ``(codebook_size, dim)``.

        Returns:
            ``(quantized, indices, commit_loss)``.
        """
        # x: (batch, dim, time) -> (batch*time, dim)
        batch, dim, time = x.shape
        x_flat = x.permute(0, 2, 1).reshape(-1, dim)

        # Compute distances to each codebook entry.
        dist = (
            x_flat.pow(2).sum(-1, keepdim=True)
            - 2 * x_flat @ codebook.t()
            + codebook.pow(2).sum(-1)
        )
        indices = dist.argmin(dim=-1)
        quantized = codebook[indices]  # (batch*time, dim)

        # Straight-through estimator.
        commit_loss = F.mse_loss(x_flat, quantized.detach())
        codebook_loss = F.mse_loss(quantized, x_flat.detach())
        quantized = x_flat + (quantized - x_flat).detach()

        quantized = quantized.reshape(batch, time, dim).permute(0, 2, 1)
        indices = indices.reshape(batch, time)
        loss = codebook_loss + self.commitment_weight * commit_loss
        return quantized, indices, loss

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        num_quantizers: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run residual vector quantisation.

        Args:
            x: Latent of shape ``(batch, latent_size, time)``.
            num_quantizers: Number of codebooks to use (defaults to all).

        Returns:
            ``(quantized, indices, commit_loss)`` where ``indices`` has
            shape ``(batch, num_quantizers, time)``.
        """
        n = num_quantizers or self.num_quantizers
        residual = x
        quantized_total = torch.zeros_like(x)
        all_indices: List[torch.Tensor] = []
        total_loss = x.new_zeros(1).squeeze()

        for i in range(n):
            codebook = self.codebooks[i]
            quantized, indices, loss = self._quantize_single(residual, codebook)
            residual = residual - quantized
            quantized_total = quantized_total + quantized
            all_indices.append(indices)
            total_loss = total_loss + loss

        indices = torch.stack(all_indices, dim=1)  # (batch, n, time)
        return quantized_total, indices, total_loss / max(n, 1)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert token indices back to a continuous latent.

        Args:
            indices: Token indices of shape ``(batch, num_quantizers, time)``.

        Returns:
            Latent of shape ``(batch, latent_size, time)``.
        """
        batch, n, time = indices.shape
        latent = torch.zeros(
            batch, self.latent_size, time, device=indices.device, dtype=self.codebooks.dtype
        )
        for i in range(n):
            codebook = self.codebooks[i]
            idx = indices[:, i, :]
            latent = latent + codebook[idx].permute(0, 2, 1)
        return latent


class AudioCodec(BaseModel):
    """EnCodec-style neural audio codec.

    Args:
        in_channels: Input audio channels.
        hidden_size: Base channel width.
        latent_size: Latent dimension.
        num_quantizers: Number of RVQ codebooks.
        codebook_size: Codebook entries per stage.
        strides: Encoder strides.
        config: Optional configuration dictionary.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_size: int = 64,
        latent_size: int = 32,
        num_quantizers: int = 4,
        codebook_size: int = 1024,
        strides: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            in_channels = config.get("in_channels", in_channels)
            hidden_size = config.get("hidden_size", hidden_size)
            latent_size = config.get("latent_size", latent_size)
            num_quantizers = config.get("num_quantizers", num_quantizers)
            codebook_size = config.get("codebook_size", codebook_size)
            strides = config.get("strides", strides)

        super().__init__(config=config)

        self.in_channels: int = in_channels
        self.latent_size: int = latent_size
        self.num_quantizers: int = num_quantizers

        self.encoder: Encoder = Encoder(
            in_channels=in_channels,
            hidden_size=hidden_size,
            latent_size=latent_size,
            strides=strides,
        )
        self.decoder: Decoder = Decoder(
            latent_size=latent_size,
            hidden_size=hidden_size,
            out_channels=in_channels,
            strides=strides,
        )
        self.quantizer: ResidualVectorQuantizer = ResidualVectorQuantizer(
            latent_size=latent_size,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
        )

    # ------------------------------------------------------------------
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode a waveform into discrete tokens.

        Args:
            waveform: Audio of shape ``(batch, in_channels, time)``.

        Returns:
            Token indices of shape ``(batch, num_quantizers, time / factor)``.
        """
        z = self.encoder(waveform)
        _, indices, _ = self.quantizer(z)
        return indices

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode discrete tokens back to a waveform.

        Args:
            tokens: Token indices of shape ``(batch, num_quantizers, time)``.

        Returns:
            Reconstructed waveform.
        """
        z = self.quantizer.dequantize(tokens)
        return self.decoder(z)

    def forward(
        self,
        waveform: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full codec forward pass.

        Args:
            waveform: Audio of shape ``(batch, in_channels, time)``.

        Returns:
            ``(recon, tokens, commit_loss)``.
        """
        z = self.encoder(waveform)
        quantized, tokens, commit_loss = self.quantizer(z)
        recon = self.decoder(quantized)
        return recon, tokens, commit_loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        tokens: Optional[torch.Tensor] = None,
        shape: Optional[Tuple[int, ...]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate audio from tokens (or random tokens).

        Args:
            tokens: Optional token indices.  When ``None`` random tokens
                of the given ``shape`` are sampled.
            shape: Shape of random tokens
                ``(batch, num_quantizers, time)``.

        Returns:
            Generated waveform.
        """
        self.eval()
        if tokens is None:
            if shape is None:
                raise ValueError("Either tokens or shape must be provided.")
            tokens = torch.randint(
                0, self.quantizer.codebook_size, shape, device=next(self.parameters()).device
            )
        return self.decode(tokens)
