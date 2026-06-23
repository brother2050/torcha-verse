"""HiFi-GAN neural vocoder.

This module implements a HiFi-GAN vocoder that converts a mel
spectrogram into a high-fidelity waveform using a generator based on
transposed convolutions and multi-receptive-field (MRF) residual
blocks.

Key components:

* :class:`MultiReceptiveField` -- multi-receptive-field residual block.
* :class:`Generator` -- the HiFi-GAN generator.
* :class:`HiFiGAN` -- the full vocoder (with optional streaming mode).

Reference:
    Kong et al., "HiFi-GAN: Generative Adversarial Networks for
    Efficient and High Fidelity Speech Synthesis" (2020).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model_registry import BaseModel

__all__ = ["MultiReceptiveField", "Generator", "HiFiGAN"]


class _ResidualBlock(nn.Module):
    """A single residual block with dilated convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1: nn.Conv1d = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding
        )
        self.conv2: nn.Conv1d = nn.Conv1d(
            channels, channels, kernel_size, dilation=1, padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(x, 0.1)
        h = self.conv1(h)
        h = F.leaky_relu(h, 0.1)
        h = self.conv2(h)
        return x + h


class MultiReceptiveField(nn.Module):
    """Multi-Receptive-Field (MRF) residual block.

    Runs several residual blocks with different kernel sizes / dilations
    in parallel and averages their outputs.

    Args:
        channels: Channel width.
        kernel_sizes: List of kernel sizes for the parallel branches.
        dilation_sizes: List of dilation lists for each branch.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: List[int],
        dilation_sizes: List[List[int]],
    ) -> None:
        super().__init__()
        self.num_branches: int = len(kernel_sizes)
        self.branches: nn.ModuleList = nn.ModuleList()
        for k, dilations in zip(kernel_sizes, dilation_sizes):
            layers: List[nn.Module] = []
            for d in dilations:
                layers.append(_ResidualBlock(channels, k, d))
            self.branches.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MRF block.

        Args:
            x: Input of shape ``(batch, channels, time)``.

        Returns:
            Output of the same shape, averaged across branches.
        """
        if self.num_branches == 1:
            return self.branches[0](x)
        outputs = [branch(x) for branch in self.branches]
        return sum(outputs) / self.num_branches


class Generator(nn.Module):
    """HiFi-GAN generator.

    Args:
        in_channels: Mel-spectrogram input channels.
        hidden_size: Channel width.
        upsample_rates: Upsampling factors per stage.
        upsample_kernel_sizes: Kernel sizes per upsampling stage.
        upsample_hidden: Hidden channels after upsampling.
        resblock_kernel_sizes: Kernel sizes for the MRF branches.
        resblock_dilation_sizes: Dilation lists for the MRF branches.
    """

    def __init__(
        self,
        in_channels: int = 80,
        hidden_size: int = 256,
        upsample_rates: Optional[List[int]] = None,
        upsample_kernel_sizes: Optional[List[int]] = None,
        upsample_hidden: Optional[List[int]] = None,
        resblock_kernel_sizes: Optional[List[int]] = None,
        resblock_dilation_sizes: Optional[List[List[int]]] = None,
    ) -> None:
        super().__init__()
        if upsample_rates is None:
            upsample_rates = [8, 8, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 16, 4, 4]
        if upsample_hidden is None:
            upsample_hidden = [128, 128, 256, 256]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.input_conv: nn.Conv1d = nn.Conv1d(in_channels, hidden_size, 7, padding=3)

        self.upsample_rates: List[int] = list(upsample_rates)
        self.ups: nn.ModuleList = nn.ModuleList()
        self.mrfs: nn.ModuleList = nn.ModuleList()
        channels = hidden_size
        for rate, kernel, out_ch in zip(upsample_rates, upsample_kernel_sizes, upsample_hidden):
            self.ups.append(nn.ConvTranspose1d(
                channels, out_ch, kernel, stride=rate, padding=(kernel - rate) // 2
            ))
            self.mrfs.append(MultiReceptiveField(out_ch, resblock_kernel_sizes, resblock_dilation_sizes))
            channels = out_ch

        self.output_conv: nn.Conv1d = nn.Conv1d(channels, 1, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a waveform from a mel spectrogram.

        Args:
            x: Mel spectrogram of shape ``(batch, in_channels, time)``.

        Returns:
            Waveform of shape ``(batch, 1, time * prod(upsample_rates))``.
        """
        x = F.leaky_relu(self.input_conv(x), 0.1)
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(up(x), 0.1)
            x = mrf(x)
        x = torch.tanh(self.output_conv(x))
        return x


class HiFiGAN(BaseModel):
    """HiFi-GAN neural vocoder.

    Args:
        in_channels: Mel-spectrogram input channels.
        upsample_rates: Upsampling factors per stage.
        upsample_kernel_sizes: Kernel sizes per upsampling stage.
        hidden_size: Generator hidden width.
        config: Optional configuration dictionary.
    """

    def __init__(
        self,
        in_channels: int = 80,
        upsample_rates: Optional[List[int]] = None,
        upsample_kernel_sizes: Optional[List[int]] = None,
        hidden_size: int = 256,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            in_channels = config.get("in_channels", in_channels)
            upsample_rates = config.get("upsample_rates", upsample_rates)
            upsample_kernel_sizes = config.get("upsample_kernel_sizes", upsample_kernel_sizes)
            hidden_size = config.get("hidden_size", hidden_size)

        super().__init__(config=config)

        self.in_channels: int = in_channels
        self.generator: Generator = Generator(
            in_channels=in_channels,
            hidden_size=hidden_size,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
        )
        self._stream_buffer: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Convert a mel spectrogram to a waveform.

        Args:
            mel_spectrogram: Mel spectrogram of shape
                ``(batch, in_channels, time)``.

        Returns:
            Waveform of shape ``(batch, 1, time * hop_length)``.
        """
        return self.generator(mel_spectrogram)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        mel_spectrogram: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate a waveform from a mel spectrogram.

        Args:
            mel_spectrogram: Mel spectrogram.

        Returns:
            Generated waveform.
        """
        self.eval()
        return self.generator(mel_spectrogram)

    # ------------------------------------------------------------------
    def reset_streaming(self) -> None:
        """Reset the streaming internal buffer."""
        self._stream_buffer = None

    @torch.no_grad()
    def forward_streaming(
        self,
        mel_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Process a chunk of mel frames in streaming mode.

        Maintains an internal buffer to handle the receptive field
        overlap between consecutive chunks.

        Args:
            mel_chunk: A chunk of mel frames
                ``(batch, in_channels, chunk_frames)``.

        Returns:
            Waveform chunk ``(batch, 1, chunk_frames * hop_length)``.
        """
        self.eval()
        if self._stream_buffer is None:
            self._stream_buffer = mel_chunk
        else:
            self._stream_buffer = torch.cat([self._stream_buffer, mel_chunk], dim=-1)

        # Run the generator on the buffered mel.
        waveform = self.generator(self._stream_buffer)

        # Compute how many output samples correspond to the new chunk.
        hop = 1
        for rate in self.generator.upsample_rates:
            hop *= rate
        chunk_samples = mel_chunk.shape[-1] * hop

        # Keep only the new samples.
        if waveform.shape[-1] > chunk_samples:
            waveform = waveform[..., -chunk_samples:]
            # Trim the buffer to the receptive field overlap.
            buffer_frames = self._stream_buffer.shape[-1] - mel_chunk.shape[-1]
            if buffer_frames > 0:
                self._stream_buffer = self._stream_buffer[..., -buffer_frames:]
            else:
                self._stream_buffer = None
        return waveform
