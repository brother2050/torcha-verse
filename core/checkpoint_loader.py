"""Checkpoint loading entry point (v0.8.0).

The actual implementation lives in :mod:`models.base` (the
:class:`models.base.ModelMixin` exposes the full diffusers-style
``from_pretrained`` / ``save_pretrained`` surface area).  This
module is a thin public alias that re-exports those helpers under
the :mod:`core` namespace, which is the conventional home for
loader / runtime utilities in the rest of the project.

The split keeps :mod:`core` as the user-facing "what do I use to
load a model?" answer while still allowing the model layer to own
the implementation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from models.base import (
    ModelMixin,
    load_safetensors,
    load_state_dict_with_renames,
    save_safetensors,
    transform_checkpoint_dict_key,
)

__all__ = [
    "ModelMixin",
    "load_safetensors",
    "save_safetensors",
    "transform_checkpoint_dict_key",
    "load_state_dict_with_renames",
    "HUNYUAN_DIT_KEY_MAP",
    "load_hunyuan_dit",
]


# ---------------------------------------------------------------------------
# v0.8.0 — first upstream key-rename table
# ---------------------------------------------------------------------------
# The HunyuanDiT v1.2 DiT blocks use the upstream HunyuanDiT naming
# scheme.  When we load a checkpoint from a real upstream model we
# need to rewrite the keys into the local :mod:`models.image.dit`
# layout.  The table below is the canonical source of truth for
# those rewrites and is consumed by
# :func:`load_hunyuan_dit` (and by integration tests).
HUNYUAN_DIT_KEY_MAP: Dict[str, str] = {
    # Patch / token embed
    "img_in.proj.weight": "patch_embed.proj.weight",
    "img_in.proj.bias":   "patch_embed.proj.bias",
    "x_embedder.weight":  "x_embedder.weight",
    "x_embedder.bias":    "x_embedder.bias",
    # Time embedding
    "time_in.mlp.0.weight": "time_embed.0.weight",
    "time_in.mlp.0.bias":   "time_embed.0.bias",
    "time_in.mlp.2.weight": "time_embed.2.weight",
    "time_in.mlp.2.bias":   "time_embed.2.bias",
    # Vector (pooled) embedding
    "vector_in.proj.weight": "pooled_embed.proj.weight",
    "vector_in.proj.bias":   "pooled_embed.proj.bias",
    # Style / size / RoPE embeds (1:1)
    "style_embedder.weight": "style_embed.weight",
    "size_embedder.weight":  "size_embed.weight",
    "rope.freqs":            "rope_freqs",
    # Per-block DiT attention / MLP / AdaLN params
    "blocks.{i}.attn.qkv.weight": "blocks.{i}.attn.qkv.weight",
    "blocks.{i}.attn.qkv.bias":   "blocks.{i}.attn.qkv.bias",
    "blocks.{i}.attn.proj.weight": "blocks.{i}.attn.out_proj.weight",
    "blocks.{i}.attn.proj.bias":   "blocks.{i}.attn.out_proj.bias",
    "blocks.{i}.mlp.fc1.weight":   "blocks.{i}.mlp.fc1.weight",
    "blocks.{i}.mlp.fc1.bias":     "blocks.{i}.mlp.fc1.bias",
    "blocks.{i}.mlp.fc2.weight":   "blocks.{i}.mlp.fc2.weight",
    "blocks.{i}.mlp.fc2.bias":     "blocks.{i}.mlp.fc2.bias",
    "blocks.{i}.adaln_modulation.0.weight":
        "blocks.{i}.adaln_modulation.weight",
    "blocks.{i}.adaln_modulation.0.bias":
        "blocks.{i}.adaln_modulation.bias",
    # Final layer
    "final_layer.adaLN_modulation.0.weight":
        "final_layer.adaln_modulation.weight",
    "final_layer.adaLN_modulation.0.bias":
        "final_layer.adaln_modulation.bias",
    "final_layer.linear.weight":   "final_layer.out_proj.weight",
    "final_layer.linear.bias":     "final_layer.out_proj.bias",
    "final_layer.norm_final.weight": "final_layer.norm.weight",
    "final_layer.norm_final.bias":   "final_layer.norm.bias",
    # VAE (the smaller ``AutoencoderKL``) — 1:1 rename
    "decoder.conv_in.weight":     "vae.decoder.conv_in.weight",
    "decoder.conv_in.bias":       "vae.decoder.conv_in.bias",
    "decoder.conv_out.weight":    "vae.decoder.conv_out.weight",
    "decoder.conv_out.bias":      "vae.decoder.conv_out.bias",
    "encoder.conv_in.weight":     "vae.encoder.conv_in.weight",
    "encoder.conv_in.bias":       "vae.encoder.conv_in.bias",
    "encoder.conv_out.weight":    "vae.encoder.conv_out.weight",
    "encoder.conv_out.bias":      "vae.encoder.conv_out.bias",
    "quant_conv.weight":          "vae.quant_conv.weight",
    "quant_conv.bias":            "vae.quant_conv.bias",
    "post_quant_conv.weight":     "vae.post_quant_conv.weight",
    "post_quant_conv.bias":       "vae.post_quant_conv.bias",
    # CLIP text encoder (the 1.x release uses a CLIP-L instance)
    "token_embedding.weight":     "text_encoder.token_embedding.weight",
    "positional_embedding":       "text_encoder.positional_embedding",
    "ln_1.weight":                "text_encoder.ln_1.weight",
    "ln_1.bias":                  "text_encoder.ln_1.bias",
    "ln_2.weight":                "text_encoder.ln_2.weight",
    "ln_2.bias":                  "text_encoder.ln_2.bias",
}


def _materialise_per_block_map(num_blocks: int) -> Dict[str, str]:
    """Expand the ``{i}`` placeholders in :data:`HUNYUAN_DIT_KEY_MAP`
    so that ``load_state_dict_with_renames`` can apply them verbatim.
    """
    out: Dict[str, str] = {}
    for k, v in HUNYUAN_DIT_KEY_MAP.items():
        if "{i}" in k:
            for i in range(num_blocks):
                out[k.format(i=i)] = v.format(i=i)
        else:
            out[k] = v
    return out


def load_hunyuan_dit(
    weights_path: Union[str, "Path"],
    *,
    torch_dtype: Optional[torch.dtype] = None,
    device: Union[str, torch.device, None] = None,
    num_blocks: int = 20,
    strict: bool = False,
) -> "ModelMixin":
    """Load a :class:`models.image.dit.HunyuanDiT` from a real upstream
    checkpoint.

    This is the v0.8.0 "真大模型" entry point — it reuses
    :meth:`models.base.ModelMixin.from_pretrained` and applies
    :data:`HUNYUAN_DIT_KEY_MAP` to translate the upstream naming
    scheme into the local one.

    Args:
        weights_path: Path to a directory that contains a
            ``diffusion_pytorch_model.safetensors`` file, or a direct
            path to a ``.safetensors`` file.
        torch_dtype: Optional dtype cast applied to every tensor
            after load.
        device: Pin the model to a single device after load.
        num_blocks: Number of DiT blocks the model was instantiated
            with (controls the per-block key-rewrite expansion).
        strict: Forwarded to :func:`load_state_dict_with_renames`.

    Returns:
        An instantiated, ``eval()``-mode HunyuanDiT model.

    Raises:
        FileNotFoundError: When ``weights_path`` cannot be resolved.
        RuntimeError: When :class:`models.image.dit.HunyuanDiT` is
            not importable (the model class is implemented in
            :mod:`models.image.dit`).
    """
    try:
        from models.image.dit import HunyuanDiT
    except ImportError as exc:  # pragma: no cover - hard dep
        raise RuntimeError(
            "models.image.dit.HunyuanDiT is unavailable; "
            "ensure the image DiT module is on the import path.",
        ) from exc
    key_map = _materialise_per_block_map(num_blocks)
    return HunyuanDiT.from_pretrained(
        weights_path,
        torch_dtype=torch_dtype,
        device=device,
        key_renames=key_map,
        strict=strict,
    )
