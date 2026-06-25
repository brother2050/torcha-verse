"""TorchaVerse local model providers (v0.4.0 P0).

This subpackage implements the v0.4.0 P0 "real model" milestone
*without* introducing any external dependencies (no
``transformers``, no ``diffusers``, no ``tokenizers``, no
``safetensors``).  Everything is built on top of the project's
existing :class:`models.text.transformer.TransformerDecoder`.

Modules
-------

* :mod:`tiny_transformer` -- :class:`TinyTransformerConfig`,
  :class:`ByteTokenizer`, ``SMALL_CONFIG`` / ``TINY_CONFIG``
  presets, and the ``torch.save`` / ``torch.load`` round-trip
  helpers.
* :mod:`local_text` -- :class:`LocalTorchTextProvider`, an
  :class:`LLMProvider` that wraps a :class:`TransformerDecoder`
  with KV-cache-aware :meth:`generate` and a chat helper.
* :mod:`factory` -- :func:`fetch_and_load_text` /
  :func:`publish_tiny_transformer` that tie the
  :mod:`models.source` cache and the project-owned provider
  together.
* :mod:`pretrain_tiny` -- :func:`train_tiny_transformer` /
  ``python -m models.providers.pretrain_tiny`` CLI used to
  produce a ``.pt`` checkpoint that exercises the end-to-end
  pipeline.

The headline entry point is :func:`fetch_and_load_text`::

    from models.providers import fetch_and_load_text
    provider = fetch_and_load_text(
        "torcha-verse/tiny-transformer-small",
        config_name="small",
    )
    reply = provider.chat(
        [{"role": "user", "content": "Hello, who are you?"}],
    )
"""

from __future__ import annotations

from .factory import (
    fetch_and_load_text,
    get_default_provider,
    publish_tiny_transformer,
    resolve_config_by_name,
)
from .local_text import GenerationConfig, LocalTorchTextProvider
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
    # factory
    "fetch_and_load_text",
    "publish_tiny_transformer",
    "resolve_config_by_name",
    "get_default_provider",
    # pretrain_tiny
    "TinyCorpus",
    "TrainConfig",
    "train_tiny_transformer",
    "DEFAULT_CORPUS",
]


__version__ = "0.4.0"
