"""Vocoder management for TorchaVerse.

This module provides :class:`VocoderManager`, a central registry for
audio vocoders that convert mel-spectrograms into waveforms.  It ships
with a simplified HiFi-GAN style implementation built from
``torch.nn.ConvTranspose1d`` upsampling blocks, and supports both
batch and streaming (frame-by-frame) synthesis.

The :class:`BaseVocoder` abstract base class defines the contract that
all vocoders must honour: a :meth:`synthesize` method that maps a
mel-spectrogram to a waveform.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Generator, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

__all__ = [
    "BaseVocoder",
    "HiFiGANVocoder",
    "VocoderManager",
]


# ---------------------------------------------------------------------------
# BaseVocoder
# ---------------------------------------------------------------------------
class BaseVocoder(nn.Module, abc.ABC):
    """Abstract base class for all vocoders.

    A vocoder converts a mel-spectrogram (or other acoustic features)
    into a time-domain audio waveform.

    Args:
        sample_rate: Output audio sample rate in Hz.
        n_mels: Number of mel bins in the input spectrogram.
        device: Device for the vocoder parameters.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 80,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.sample_rate: int = sample_rate
        self.n_mels: int = n_mels
        self._device_manager: DeviceManager = DeviceManager()
        self._target_device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device or self._device_manager.get_device()
        )

    @abc.abstractmethod
    def synthesize(
        self,
        mel_spectrogram: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Convert a mel-spectrogram into a waveform.

        Args:
            mel_spectrogram: Input mel-spectrogram of shape
                ``(batch, n_mels, time)`` or ``(n_mels, time)``.

        Returns:
            Waveform tensor of shape ``(batch, samples)`` or
            ``(samples,)``.
        """
        ...

    def to_device(self, device: Union[str, torch.device]) -> "BaseVocoder":
        """Move the vocoder to ``device``."""
        self._target_device = torch.device(device) if isinstance(device, str) else device
        self.to(self._target_device)
        return self

    @property
    def device(self) -> torch.device:
        """The device on which the vocoder parameters reside."""
        return self._target_device


# ---------------------------------------------------------------------------
# HiFi-GAN Vocoder (simplified)
# ---------------------------------------------------------------------------
class _UpsampleBlock(nn.Module):
    """A single upsampling block for the HiFi-GAN generator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_factor: int,
        kernel_size: int = 8,
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=upsample_factor,
            padding=(kernel_size - upsample_factor) // 2,
        )
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu(self.conv(x))


class _ResidualBlock(nn.Module):
    """A multi-receptive-field residual block (simplified MRF)."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: tuple = (3, 7, 11),
        dilations: tuple = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for ks in kernel_sizes:
            for d in dilations:
                self.blocks.append(
                    nn.Sequential(
                        nn.LeakyReLU(0.1),
                        nn.Conv1d(
                            channels, channels,
                            kernel_size=ks, dilation=d,
                            padding=(ks - 1) * d // 2,
                        ),
                        nn.LeakyReLU(0.1),
                        nn.Conv1d(
                            channels, channels,
                            kernel_size=1, dilation=1, padding=0,
                        ),
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: sum of all residual branches."""
        out = x
        for block in self.blocks:
            out = out + block(x)
        return out / len(self.blocks)


class HiFiGANVocoder(BaseVocoder):
    """Simplified HiFi-GAN vocoder.

    A generative adversarial network vocoder that uses transposed
    convolutions for upsampling and multi-receptive-field (MRF) residual
    blocks for refinement.  This is a simplified, self-contained
    implementation suitable for testing and lightweight deployment.

    Args:
        sample_rate: Output sample rate.
        n_mels: Number of input mel bins.
        upsample_rates: List of upsampling factors per stage.
        upsample_initial_channel: Number of channels after the first
            projection.
        upsample_kernel_sizes: Kernel sizes for each upsampling stage.
        resblock_kernel_sizes: Kernel sizes for the residual blocks.
        resblock_dilations: Dilation rates for the residual blocks.
        device: Device for the vocoder.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 80,
        upsample_rates: Optional[List[int]] = None,
        upsample_initial_channel: int = 256,
        upsample_kernel_sizes: Optional[List[int]] = None,
        resblock_kernel_sizes: Optional[tuple] = None,
        resblock_dilations: Optional[tuple] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(sample_rate=sample_rate, n_mels=n_mels, device=device)

        self.upsample_rates: List[int] = upsample_rates or [8, 8, 2, 2]
        self.upsample_initial_channel: int = upsample_initial_channel
        self.upsample_kernel_sizes: List[int] = upsample_kernel_sizes or [16, 16, 4, 4]
        self.resblock_kernel_sizes: tuple = resblock_kernel_sizes or (3, 7, 11)
        self.resblock_dilations: tuple = resblock_dilations or (1, 3, 5)

        # Input projection.
        self.conv_pre = nn.Conv1d(n_mels, upsample_initial_channel, kernel_size=7, padding=3)

        # Upsampling layers and residual blocks (created together so that
        # each residual block matches the output channels of its stage).
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        in_ch = upsample_initial_channel
        for rate, ksize in zip(self.upsample_rates, self.upsample_kernel_sizes):
            out_ch = in_ch // 2
            self.ups.append(_UpsampleBlock(in_ch, out_ch, rate, kernel_size=ksize))
            self.resblocks.append(
                _ResidualBlock(out_ch, self.resblock_kernel_sizes, self.resblock_dilations)
            )
            in_ch = out_ch

        # Output projection.
        self.conv_post = nn.Conv1d(in_ch, 1, kernel_size=7, padding=3)
        self.activation = nn.Tanh()

        self.to(self._target_device)

    # ------------------------------------------------------------------
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Forward pass: mel-spectrogram -> waveform.

        Args:
            mel: Mel-spectrogram of shape ``(batch, n_mels, time)``.

        Returns:
            Waveform of shape ``(batch, 1, samples)``.
        """
        mel = mel.to(self._target_device)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        x = self.conv_pre(mel)
        for up, resblock in zip(self.ups, self.resblocks):
            x = up(x)
            x = resblock(x)

        x = self.conv_post(x)
        return self.activation(x)

    # ------------------------------------------------------------------
    def synthesize(
        self,
        mel_spectrogram: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Convert a mel-spectrogram into a waveform.

        Args:
            mel_spectrogram: Input of shape ``(batch, n_mels, time)``
                or ``(n_mels, time)``.

        Returns:
            Waveform of shape ``(batch, samples)`` or ``(samples,)``.
        """
        squeeze_output = mel_spectrogram.dim() == 2
        waveform = self.forward(mel_spectrogram)

        # Remove the channel dimension.
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        if squeeze_output and waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        return waveform

    # ------------------------------------------------------------------
    def synthesize_streaming(
        self,
        mel_spectrogram: torch.Tensor,
        hop_frames: int = 16,
        overlap_frames: int = 4,
    ) -> Generator[torch.Tensor, None, None]:
        """Stream waveform generation frame-by-frame.

        Splits the mel-spectrogram into overlapping chunks and yields
        the corresponding waveform segments.  This enables real-time
        audio streaming.

        Args:
            mel_spectrogram: Input of shape ``(n_mels, time)`` or
                ``(batch, n_mels, time)``.
            hop_frames: Number of mel frames per chunk.
            overlap_frames: Overlap between consecutive chunks (for
                smooth crossfading).

        Yields:
            Waveform segments of shape ``(samples,)``.
        """
        if mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)

        batch_size, n_mels, total_frames = mel_spectrogram.shape
        mel_spectrogram = mel_spectrogram.to(self._target_device)

        step = hop_frames - overlap_frames
        if step <= 0:
            raise ValueError("hop_frames must be > overlap_frames.")

        pos = 0
        while pos < total_frames:
            end = min(pos + hop_frames, total_frames)
            chunk = mel_spectrogram[:, :, pos:end]
            waveform = self.synthesize(chunk)
            yield waveform.squeeze(0) if waveform.dim() == 2 else waveform
            pos += step


# ---------------------------------------------------------------------------
# VocoderManager
# ---------------------------------------------------------------------------
class VocoderManager:
    """Central registry for audio vocoders.

    Manages the registration, retrieval, and instantiation of vocoders.
    The built-in :class:`HiFiGANVocoder` is registered automatically.

    Example:
        >>> manager = VocoderManager()
        >>> vocoder = manager.get_vocoder("hifi-gan", sample_rate=22050)
        >>> waveform = vocoder.synthesize(mel_spec)
    """

    _instance: Optional["VocoderManager"] = None
    _initialized: bool = False

    #: Built-in vocoder name to class mapping.
    _BUILTIN: Dict[str, Type[BaseVocoder]] = {
        "hifi-gan": HiFiGANVocoder,
    }

    def __new__(cls, *args: Any, **kwargs: Any) -> "VocoderManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._registry: Dict[str, Type[BaseVocoder]] = {}
        self._instances: Dict[str, BaseVocoder] = {}
        self._logger = get_logger(self.__class__.__name__)
        self._device_manager: DeviceManager = DeviceManager()

        for name, cls in self._BUILTIN.items():
            self.register_vocoder(name, cls)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_vocoder(
        self,
        name: str,
        vocoder_class: Type[BaseVocoder],
    ) -> None:
        """Register a vocoder class under ``name``.

        Args:
            name: Unique identifier (e.g. ``"hifi-gan"``).
            vocoder_class: A subclass of :class:`BaseVocoder`.

        Raises:
            TypeError: If ``vocoder_class`` is not a subclass of
                :class:`BaseVocoder`.
        """
        if not (isinstance(vocoder_class, type) and issubclass(vocoder_class, BaseVocoder)):
            raise TypeError(
                f"vocoder_class must be a subclass of BaseVocoder, got "
                f"{vocoder_class!r}."
            )
        key = name.strip().lower()
        self._registry[key] = vocoder_class
        self._instances.pop(key, None)
        self._logger.debug("Registered vocoder '%s' -> %s", key, vocoder_class.__name__)

    def unregister_vocoder(self, name: str) -> bool:
        """Remove a vocoder from the registry."""
        key = name.strip().lower()
        removed = self._registry.pop(key, None) is not None
        self._instances.pop(key, None)
        return removed

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def get_vocoder(self, name: str, **kwargs: Any) -> BaseVocoder:
        """Retrieve (or create) a vocoder instance.

        When ``kwargs`` are provided a new instance is always created.
        Otherwise a cached instance is returned if available.

        Args:
            name: Registered vocoder name.
            **kwargs: Constructor arguments forwarded to the vocoder.

        Returns:
            A :class:`BaseVocoder` instance.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        key = name.strip().lower()
        if key not in self._registry:
            raise KeyError(
                f"Vocoder '{name}' is not registered. "
                f"Available: {', '.join(self.list_available()) or '(none)'}."
            )

        if kwargs:
            vocoder = self._registry[key](**kwargs)
            self._instances[key] = vocoder
            return vocoder

        if key not in self._instances:
            self._instances[key] = self._registry[key]()
        return self._instances[key]

    def list_available(self) -> List[str]:
        """Return a sorted list of registered vocoder names."""
        return sorted(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Return ``True`` if ``name`` is a registered vocoder."""
        return name.strip().lower() in self._registry

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def synthesize(
        self,
        mel_spectrogram: torch.Tensor,
        vocoder_name: str = "hifi-gan",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Convenience method: get a vocoder and synthesize in one call.

        Args:
            mel_spectrogram: Input mel-spectrogram.
            vocoder_name: Name of the vocoder to use.
            **kwargs: Additional arguments forwarded to ``synthesize``.

        Returns:
            The synthesized waveform.
        """
        vocoder = self.get_vocoder(vocoder_name)
        return vocoder.synthesize(mel_spectrogram, **kwargs)

    # ------------------------------------------------------------------
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
        cls._initialized = False
