"""Local-torch audio provider for the v0.4.x P0 multi-modal milestone.

This module wires the project-owned
:mod:`models.audio.tts_transformer` (text-to-mel) and
:mod:`models.audio.hifi_gan` (mel-to-waveform) into the
:class:`models.interfaces.media_providers.AudioProvider` protocol
so that the v0.4.x P0 audio nodes / examples can be exercised
**end-to-end with a real neural network** (no echo, no passthrough)
while still being *pure torch, zero external dependencies*.

The class is intentionally small:

* it owns a :class:`TTSTransformer` + :class:`HiFiGAN` pair loaded
  from a single ``.pt`` file (or constructed in memory from a
  :class:`AudioProviderConfig`);
* it implements :meth:`generate` (the only
  :class:`AudioProvider` method exercised by ``call_audio_backend``)
  and a few introspection helpers used by the v0.4.x P0 demo /
  tests;
* it is **thread-safe** (a single re-entrant lock guards the
  forward pass so concurrent :meth:`generate` calls serialise on
  the same model).

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L4 ``models.audio`` -- real components (TTS / HiFi-GAN).
* L6 ``models.providers`` (this module) -- real audio provider.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from infrastructure.logger import get_logger

from ..interfaces.media_providers import AudioProvider
from ..audio import TTSTransformer, HiFiGAN

__all__ = [
    "LocalTorchAudioProvider",
    "AudioProviderConfig",
    "TINY_AUDIO_CONFIG",
    "SMALL_AUDIO_CONFIG",
]


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger = get_logger("models.providers.local_audio")


# ---------------------------------------------------------------------------
# Config presets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AudioProviderConfig:
    """Hyperparameter bundle for :class:`LocalTorchAudioProvider`.

    Defaults produce a tiny model whose full
    TTS-Transformer -> HiFi-GAN forward pass runs in well under
    a second on a single CPU thread; that is what the v0.4.x P0
    demo / CI smoke tests rely on to keep the milestone
    dependency-free.
    """

    name: str = "tiny"
    # TTS
    tts_vocab_size: int = 256          # byte-level
    tts_hidden_size: int = 64
    tts_num_layers: int = 2
    tts_num_heads: int = 4
    tts_mel_channels: int = 32
    tts_max_text_len: int = 32
    tts_max_mel_len: int = 64
    # HiFi-GAN
    hifigan_in_channels: int = 32
    hifigan_hidden_size: int = 32
    hifigan_upsample_rates: tuple = (4, 4)  # 16x total upsample
    hifigan_upsample_kernel_sizes: tuple = (8, 8)
    # Sampling
    default_sample_rate: int = 16000
    default_max_mel_len: int = 32
    default_speed: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        for k in (
            "hifigan_upsample_rates",
            "hifigan_upsample_kernel_sizes",
        ):
            d[k] = list(getattr(self, k))
        return d


TINY_AUDIO_CONFIG = AudioProviderConfig(name="tiny")
SMALL_AUDIO_CONFIG = AudioProviderConfig(
    name="small",
    tts_hidden_size=128,
    tts_num_layers=4,
    tts_mel_channels=64,
    hifigan_hidden_size=64,
    hifigan_upsample_rates=(4, 4),
    hifigan_upsample_kernel_sizes=(8, 8),
    default_max_mel_len=64,
)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
class LocalTorchAudioProvider(AudioProvider):
    """A real, project-owned :class:`AudioProvider` backed by ``torch``.

    The provider is **stateless at the framework level** -- it
    holds a single :class:`TTSTransformer` + :class:`HiFiGAN` pair
    and serialises concurrent calls behind a lock.  All forward
    passes run in ``torch.no_grad`` mode so inference does not
    allocate autograd graphs.

    Args:
        tts: A pre-built :class:`TTSTransformer`.  When ``None`` a
            fresh one is built from ``config``.
        vocoder: A pre-built :class:`HiFiGAN`.  When ``None`` a
            fresh one is built from ``config``.
        config: The :class:`AudioProviderConfig` that was used
            to build the models.  When ``None`` the
            :data:`TINY_AUDIO_CONFIG` is used.
        device: Device to run the model on.  Defaults to CPU so
            the provider is portable across CI environments.
    """

    def __init__(
        self,
        tts: Optional[nn.Module] = None,
        vocoder: Optional[nn.Module] = None,
        config: Optional[AudioProviderConfig] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        if config is None:
            config = TINY_AUDIO_CONFIG
        if tts is None:
            tts = TTSTransformer(
                vocab_size=config.tts_vocab_size,
                hidden_size=config.tts_hidden_size,
                num_layers=config.tts_num_layers,
                num_heads=config.tts_num_heads,
                mel_channels=config.tts_mel_channels,
                max_text_len=config.tts_max_text_len,
                max_mel_len=config.tts_max_mel_len,
            )
        if vocoder is None:
            vocoder = HiFiGAN(
                in_channels=config.hifigan_in_channels,
                upsample_rates=list(config.hifigan_upsample_rates),
                upsample_kernel_sizes=list(config.hifigan_upsample_kernel_sizes),
                hidden_size=config.hifigan_hidden_size,
            )

        self._tts: nn.Module = tts.to(device)
        self._vocoder: nn.Module = vocoder.to(device)
        for m in (self._tts, self._vocoder):
            m.eval()

        self._config: AudioProviderConfig = config
        self._device: torch.device = (
            torch.device(device)
            if not isinstance(device, torch.device)
            else device
        )
        self._lock: threading.RLock = threading.RLock()
        self._logger = _logger

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_random(
        cls,
        config: Optional[AudioProviderConfig] = None,
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> "LocalTorchAudioProvider":
        """Construct a provider with freshly initialised models."""
        return cls(config=config, device=device)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> "LocalTorchAudioProvider":
        """Load a provider from a ``.pt`` file produced by :meth:`save`."""
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError("audio provider file not found: {}".format(p))
        payload = torch.load(p, map_location=device, weights_only=False)
        cfg_dict = payload.get("config", {})
        if not isinstance(cfg_dict, dict):
            raise TypeError("payload['config'] must be a dict")
        for k in (
            "hifigan_upsample_rates",
            "hifigan_upsample_kernel_sizes",
        ):
            if k in cfg_dict:
                cfg_dict[k] = tuple(cfg_dict[k])
        config = AudioProviderConfig(**cfg_dict)
        provider = cls(config=config, device=device)
        if "tts" in payload:
            provider._tts.load_state_dict(payload["tts"], strict=False)
        if "vocoder" in payload:
            provider._vocoder.load_state_dict(payload["vocoder"], strict=False)
        return provider

    def save(self, path: Union[str, Path]) -> Path:
        """Persist the provider to ``path`` (a ``.pt`` file)."""
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self._config.to_dict(),
            "tts": self._tts.state_dict(),
            "vocoder": self._vocoder.state_dict(),
        }
        torch.save(payload, out)
        return out

    # ------------------------------------------------------------------
    # AudioProvider interface
    # ------------------------------------------------------------------
    def generate(
        self,
        text: str,
        *,
        sample_rate: Optional[int] = None,
        duration_s: Optional[float] = None,
        max_mel_len: Optional[int] = None,
        speed: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a waveform from ``text``.

        The pipeline is a minimal TTS loop:

        1. Byte-level tokenise ``text``.
        2. Run the :class:`TTSTransformer` in inference mode
           (``tts.generate``) to produce a mel spectrogram of
           shape ``(1, mel_len, mel_channels)``.
        3. Run the :class:`HiFiGAN` vocoder to turn the mel into
           a waveform of shape ``(1, num_samples)``.

        Args:
            text: Text to synthesise.
            sample_rate: Override the output sample rate (returned
                as metadata only -- the vocoder is configured for
                the rate implied by its upsample kernel sizes).
            duration_s: Optional target duration.  When given, the
                mel length is chosen to be at least
                ``ceil(duration_s * sample_rate / total_upsample)``.
                Ignored when ``max_mel_len`` is given.
            max_mel_len: Override the maximum mel length.
            speed: TTS speed factor (``> 1`` = faster).
            **kwargs: Ignored.  Forwarded for forward-compat with
                the v0.4.x P0 node kwargs.

        Returns:
            A dict with at least:

            * ``"waveform"`` -- a ``torch.Tensor`` of shape
              ``(1, num_samples)`` in ``[-1, 1]`` (clamped).
            * ``"sample_rate"`` -- the (effective) sample rate.
            * ``"duration_s"`` -- the actual duration in seconds.
            * ``"text"`` -- the (truncated) input text.
            * ``"mel_shape"`` -- the shape of the generated mel.
        """
        cfg = self._config
        sr = int(sample_rate or cfg.default_sample_rate)
        sp = float(speed or cfg.default_speed)
        total_upsample = 1
        for r in cfg.hifigan_upsample_rates:
            total_upsample *= int(r)
        if duration_s is not None and duration_s > 0:
            target_samples = int(float(duration_s) * sr)
            target_mel_len = max(target_samples // total_upsample, 1)
            mel_len = min(
                max(int(max_mel_len or 0), target_mel_len),
                cfg.tts_max_mel_len,
            )
        else:
            mel_len = int(max_mel_len or cfg.default_max_mel_len)
            mel_len = min(mel_len, cfg.tts_max_mel_len)
        mel_len = max(mel_len, 1)

        token_ids = self._byte_tokenize(text, cfg.tts_max_text_len)
        token_tensor = torch.tensor(
            [token_ids], dtype=torch.long, device=self._device,
        )

        with self._lock:
            with torch.no_grad():
                # 1. text -> mel
                mel = self._tts.generate(
                    token_tensor, max_mel_len=mel_len, speed=sp,
                )  # (1, mel_len, mel_channels)
                # The HiFi-GAN expects (batch, in_channels, time).
                mel_t = mel.transpose(1, 2)  # (1, mel_channels, mel_len)
                # 2. mel -> waveform
                waveform = self._vocoder(mel_t)  # (1, 1, num_samples)
                waveform = waveform.squeeze(1)  # (1, num_samples)
                waveform = waveform.clamp(-1.0, 1.0)

        num_samples = int(waveform.shape[-1])
        dur = num_samples / max(sr, 1)
        return {
            "waveform": waveform.cpu(),
            "sample_rate": sr,
            "duration_s": dur,
            "text": text[:64],
            "mel_shape": tuple(mel.shape[1:]),
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def config(self) -> AudioProviderConfig:
        """The provider config (read-only)."""
        return self._config

    @property
    def device(self) -> torch.device:
        """The device the models are bound to."""
        return self._device

    def num_parameters(self) -> int:
        """Total parameter count across TTS + HiFi-GAN."""
        return sum(
            sum(p.numel() for p in m.parameters())
            for m in (self._tts, self._vocoder)
        )

    @staticmethod
    def _byte_tokenize(text: str, max_len: int) -> List[int]:
        if not text:
            return [0] * max_len
        ids = list(text.encode("utf-8")[:max_len])
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        return ids

    def __repr__(self) -> str:
        return (
            "LocalTorchAudioProvider(name={!r}, params={}, device={!r})".format(
                self._config.name, self.num_parameters(), self._device,
            )
        )
