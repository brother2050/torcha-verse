"""TorchaVerse local model providers (v0.4.x P0 multi-modal).

This subpackage implements the v0.4.x P0 "real model" milestone
*without* introducing any external dependencies (no
``transformers``, no ``diffusers``, no ``tokenizers``, no
``safetensors``).  Everything is built on top of the project's
existing :mod:`models.text` / :mod:`models.image` / :mod:`models.audio`
/ :mod:`models.video` / :mod:`models.multimodal` modules.

Modules
-------

* :mod:`tiny_transformer` -- :class:`TinyTransformerConfig`,
  :class:`ByteTokenizer`, ``SMALL_CONFIG`` / ``TINY_CONFIG``
  presets, and the ``torch.save`` / ``torch.load`` round-trip
  helpers.
* :mod:`local_text` -- :class:`LocalTorchTextProvider`, an
  :class:`LLMProvider` that wraps a :class:`TransformerDecoder`
  with KV-cache-aware :meth:`generate` and a chat helper.
* :mod:`local_image` -- :class:`LocalTorchImageProvider`, an
  :class:`ImageProvider` that wires UNet + VAE + CLIP into a
  DDPM-style 2D image generation loop.
* :mod:`local_audio` -- :class:`LocalTorchAudioProvider`, an
  :class:`AudioProvider` that wires TTS-Transformer + HiFi-GAN
  into a text -> waveform pipeline.
* :mod:`local_video` -- :class:`LocalTorchVideoProvider`, a
  :class:`VideoProvider` that wires VideoDiT + VideoVAE into a
  3D diffusion loop.
* :mod:`local_multimodal` -- :class:`LocalTorchMultimodalProvider`,
  a :class:`MultimodalProvider` that wires OmniModel (vision +
  audio towers) plus a standalone causal LM for text generation.
* :mod:`factory` -- :func:`fetch_and_load_text` / ``fetch_and_load_image`` /
  ``fetch_and_load_audio`` / ``fetch_and_load_video`` /
  ``fetch_and_load_omni`` that tie the
  :mod:`models.source` cache and the project-owned provider
  together.
* :mod:`pretrain_tiny` -- :func:`train_tiny_transformer` /
  ``python -m models.providers.pretrain_tiny`` CLI used to
  produce a ``.pt`` checkpoint that exercises the end-to-end
  pipeline.

The headline entry points are :func:`fetch_and_load_*`::
"""

from __future__ import annotations

from .factory import (
    fetch_and_load_audio,
    fetch_and_load_image,
    fetch_and_load_text,
    fetch_and_load_video,
    fetch_and_load_omni,
    get_default_audio_provider,
    get_default_image_provider,
    get_default_provider,
    get_default_video_provider,
    get_default_omni_provider,
    publish_tiny_transformer,
    resolve_config_by_name,
)
from .local_audio import (
    AudioProviderConfig,
    LocalTorchAudioProvider,
    SMALL_AUDIO_CONFIG,
    TINY_AUDIO_CONFIG,
)
from .local_image import (
    ImageProviderConfig,
    LocalTorchImageProvider,
    SMALL_IMAGE_CONFIG,
    TINY_IMAGE_CONFIG,
)
from .local_multimodal import (
    LocalTorchMultimodalProvider,
    MultimodalProviderConfig,
    SMALL_MULTIMODAL_CONFIG,
    TINY_MULTIMODAL_CONFIG,
)
from .local_text import GenerationConfig, LocalTorchTextProvider
from .local_video import (
    LocalTorchVideoProvider,
    SMALL_VIDEO_CONFIG,
    TINY_VIDEO_CONFIG,
    VideoProviderConfig,
)
from .pretrain_tiny import (
    DEFAULT_CORPUS,
    TinyCorpus,
    TrainConfig,
    train_tiny_transformer,
)
from .tiny_transformer import (
    DEFAULT_VOCAB_SIZE,
    SMALL_CONFIG,
    TINY_CONFIG,
    ByteTokenizer,
    TinyTransformerConfig,
    build_tiny_transformer,
    load_tiny_transformer,
    save_tiny_transformer,
)

__all__ = [
    # tiny_transformer
    "ByteTokenizer",
    "TinyTransformerConfig",
    "SMALL_CONFIG",
    "TINY_CONFIG",
    "DEFAULT_VOCAB_SIZE",
    "build_tiny_transformer",
    "save_tiny_transformer",
    "load_tiny_transformer",
    # local_text
    "LocalTorchTextProvider",
    "GenerationConfig",
    # local_image
    "LocalTorchImageProvider",
    "ImageProviderConfig",
    "TINY_IMAGE_CONFIG",
    "SMALL_IMAGE_CONFIG",
    # local_audio
    "LocalTorchAudioProvider",
    "AudioProviderConfig",
    "TINY_AUDIO_CONFIG",
    "SMALL_AUDIO_CONFIG",
    # local_video
    "LocalTorchVideoProvider",
    "VideoProviderConfig",
    "TINY_VIDEO_CONFIG",
    "SMALL_VIDEO_CONFIG",
    # local_multimodal
    "LocalTorchMultimodalProvider",
    "MultimodalProviderConfig",
    "TINY_MULTIMODAL_CONFIG",
    "SMALL_MULTIMODAL_CONFIG",
    # factory
    "fetch_and_load_text",
    "fetch_and_load_image",
    "fetch_and_load_audio",
    "fetch_and_load_video",
    "fetch_and_load_omni",
    "publish_tiny_transformer",
    "resolve_config_by_name",
    "get_default_provider",
    "get_default_image_provider",
    "get_default_audio_provider",
    "get_default_video_provider",
    "get_default_omni_provider",
    # pretrain_tiny
    "TinyCorpus",
    "TrainConfig",
    "train_tiny_transformer",
    "DEFAULT_CORPUS",
]


__version__ = "0.4.0"
