"""Audio models for TorchaVerse.

This sub-package contains audio generation and processing models:
the neural audio codec (RVQ), the HiFi-GAN vocoder, and the TTS
Transformer.
"""

from __future__ import annotations

from .audio_codec import AudioCodec, Decoder as AudioDecoder, Encoder as AudioEncoder, ResidualVectorQuantizer
from .hifi_gan import Generator, HiFiGAN, MultiReceptiveField
from .tts_transformer import AcousticDecoder, DurationPredictor, TextEncoder, TTSTransformer

__all__ = [
    # audio_codec
    "AudioCodec",
    "AudioEncoder",
    "AudioDecoder",
    "ResidualVectorQuantizer",
    # hifi_gan
    "HiFiGAN",
    "Generator",
    "MultiReceptiveField",
    # tts
    "TTSTransformer",
    "TextEncoder",
    "DurationPredictor",
    "AcousticDecoder",
]
