"""Image models for TorchaVerse.

This sub-package contains image generation and understanding models:
the VAE, the U-Net denoising network, the Diffusion Transformer (DiT),
and the CLIP text encoder.
"""

from __future__ import annotations

from .clip_encoder import CLIPTextEncoder, CLIPEncoderLayer
from .dit import DiT, DiTBlock, PatchEmbed
from .unet import (
    CrossAttentionBlock,
    DownBlock,
    MidBlock,
    ResBlock,
    SelfAttentionBlock,
    TimestepEmbedding,
    UNet,
    UpBlock,
)
from .vae import Decoder, Encoder, ResBlock as VAEResBlock, VAE

__all__ = [
    # vae
    "VAE",
    "Encoder",
    "Decoder",
    "VAEResBlock",
    # unet
    "UNet",
    "TimestepEmbedding",
    "ResBlock",
    "SelfAttentionBlock",
    "CrossAttentionBlock",
    "DownBlock",
    "MidBlock",
    "UpBlock",
    # dit
    "DiT",
    "DiTBlock",
    "PatchEmbed",
    # clip
    "CLIPTextEncoder",
    "CLIPEncoderLayer",
]
