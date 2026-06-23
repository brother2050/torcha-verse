"""TorchaVerse model layer.

This package contains all neural network models for the TorchaVerse
framework, organised by modality:

* :mod:`components` -- reusable building blocks (RMSNorm, SwiGLU, RoPE,
  LoRA).
* :mod:`text` -- decoder-only Transformer and Mixture-of-Experts
  language models.
* :mod:`image` -- VAE, U-Net, DiT, and CLIP text encoder.
* :mod:`audio` -- neural audio codec, HiFi-GAN vocoder, and TTS
  Transformer.
* :mod:`video` -- spatiotemporal VAE, video DiT, motion module, and
  frame interpolator.
* :mod:`multimodal` -- vision-language and omni-modal models.

All models are implemented with native ``torch.nn`` modules and inherit
from :class:`BaseModel` (registerable with the
:class:`ModelRegistry`) where appropriate.
"""

from __future__ import annotations

from . import audio, components, image, multimodal, text, video

__all__ = [
    "components",
    "text",
    "image",
    "audio",
    "video",
    "multimodal",
]
