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
    # v1.0.0 — model matrix key maps.
    "FLUX_KEY_MAP",
    "SD3_KEY_MAP",
    "WAN2_KEY_MAP",
    "MUSICGEN_KEY_MAP",
    "load_flux",
    "load_sd3",
    "load_wan2",
    "load_musicgen",
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


# ---------------------------------------------------------------------------
# v1.0.0 — model matrix key maps (FLUX / SD3 / Wan2.1 / MusicGen).
#
# Each table is the canonical source of truth for the upstream -> local
# rewrite used by :func:`load_state_dict_with_renames`.  The local layout
# follows the v0.6-v0.9 in-house ``models.<modality>.<name>`` module
# naming so all four model families share a single
# ``ModelMixin.from_pretrained(weights_path, key_renames=KEY_MAP)``
# entry point.
# ---------------------------------------------------------------------------
FLUX_KEY_MAP: Dict[str, str] = {
    # Double block (joint image+text attention).
    "double_blocks.{i}.img_attn.qkv.weight":      "double_blocks.{i}.img_attn.qkv.weight",
    "double_blocks.{i}.img_attn.qkv.bias":        "double_blocks.{i}.img_attn.qkv.bias",
    "double_blocks.{i}.img_attn.norm.query_norm.scale": "double_blocks.{i}.img_attn.norm.query_norm.scale",
    "double_blocks.{i}.img_attn.norm.k_norm.scale":     "double_blocks.{i}.img_attn.norm.k_norm.scale",
    "double_blocks.{i}.img_attn.proj.weight":     "double_blocks.{i}.img_attn.proj.weight",
    "double_blocks.{i}.img_attn.proj.bias":       "double_blocks.{i}.img_attn.proj.bias",
    "double_blocks.{i}.txt_attn.qkv.weight":      "double_blocks.{i}.txt_attn.qkv.weight",
    "double_blocks.{i}.txt_attn.qkv.bias":        "double_blocks.{i}.txt_attn.qkv.bias",
    "double_blocks.{i}.txt_attn.norm.query_norm.scale": "double_blocks.{i}.txt_attn.norm.query_norm.scale",
    "double_blocks.{i}.txt_attn.norm.k_norm.scale":     "double_blocks.{i}.txt_attn.norm.k_norm.scale",
    "double_blocks.{i}.txt_attn.proj.weight":     "double_blocks.{i}.txt_attn.proj.weight",
    "double_blocks.{i}.txt_attn.proj.bias":       "double_blocks.{i}.txt_attn.proj.bias",
    "double_blocks.{i}.img_mlp.0.weight":         "double_blocks.{i}.img_mlp.0.weight",
    "double_blocks.{i}.img_mlp.0.bias":           "double_blocks.{i}.img_mlp.0.bias",
    "double_blocks.{i}.img_mlp.2.weight":         "double_blocks.{i}.img_mlp.2.weight",
    "double_blocks.{i}.img_mlp.2.bias":           "double_blocks.{i}.img_mlp.2.bias",
    "double_blocks.{i}.txt_mlp.0.weight":         "double_blocks.{i}.txt_mlp.0.weight",
    "double_blocks.{i}.txt_mlp.0.bias":           "double_blocks.{i}.txt_mlp.0.bias",
    "double_blocks.{i}.txt_mlp.2.weight":         "double_blocks.{i}.txt_mlp.2.weight",
    "double_blocks.{i}.txt_mlp.2.bias":           "double_blocks.{i}.txt_mlp.2.bias",
    # Single block (image-only attention).
    "single_blocks.{i}.norm1.scale":              "single_blocks.{i}.norm1.scale",
    "single_blocks.{i}.norm1.shift":              "single_blocks.{i}.norm1.shift",
    "single_blocks.{i}.norm1.weight":             "single_blocks.{i}.norm1.weight",
    "single_blocks.{i}.linear1.weight":           "single_blocks.{i}.linear1.weight",
    "single_blocks.{i}.linear1.bias":             "single_blocks.{i}.linear1.bias",
    "single_blocks.{i}.norm2.scale":              "single_blocks.{i}.norm2.scale",
    "single_blocks.{i}.norm2.shift":              "single_blocks.{i}.norm2.shift",
    "single_blocks.{i}.linear2.weight":           "single_blocks.{i}.linear2.weight",
    "single_blocks.{i}.linear2.bias":             "single_blocks.{i}.linear2.bias",
    # Embedders / final.
    "img_in.weight":                              "img_in.weight",
    "img_in.bias":                                "img_in.bias",
    "time_in.in_layer.weight":                    "time_embed.0.weight",
    "time_in.in_layer.bias":                      "time_embed.0.bias",
    "time_in.out_layer.weight":                   "time_embed.2.weight",
    "time_in.out_layer.bias":                     "time_embed.2.bias",
    "vector_in.in_layer.weight":                  "vector_embed.0.weight",
    "vector_in.in_layer.bias":                    "vector_embed.0.bias",
    "vector_in.out_layer.weight":                 "vector_embed.2.weight",
    "vector_in.out_layer.bias":                   "vector_embed.2.bias",
    "guidance_in.in_layer.weight":                "guidance_embed.0.weight",
    "guidance_in.in_layer.bias":                  "guidance_embed.0.bias",
    "guidance_in.out_layer.weight":               "guidance_embed.2.weight",
    "guidance_in.out_layer.bias":                 "guidance_embed.2.bias",
    "txt_in.weight":                              "txt_in.weight",
    "txt_in.bias":                                "txt_in.bias",
    "final_layer.adaLN_modulation.1.weight":      "final_layer.adaln_modulation.weight",
    "final_layer.adaLN_modulation.1.bias":        "final_layer.adaln_modulation.bias",
    "final_layer.linear.weight":                  "final_layer.out_proj.weight",
    "final_layer.linear.bias":                    "final_layer.out_proj.bias",
}

SD3_KEY_MAP: Dict[str, str] = {
    # Joint MMDiT blocks.
    "joint_transformer_blocks.{i}.x_block.attn.qkv.weight":  "joint_blocks.{i}.x_attn.qkv.weight",
    "joint_transformer_blocks.{i}.x_block.attn.qkv.bias":    "joint_blocks.{i}.x_attn.qkv.bias",
    "joint_transformer_blocks.{i}.x_block.attn.proj.weight": "joint_blocks.{i}.x_attn.proj.weight",
    "joint_transformer_blocks.{i}.x_block.attn.proj.bias":   "joint_blocks.{i}.x_attn.proj.bias",
    "joint_transformer_blocks.{i}.x_block.mlp.fc1.weight":  "joint_blocks.{i}.x_mlp.fc1.weight",
    "joint_transformer_blocks.{i}.x_block.mlp.fc1.bias":    "joint_blocks.{i}.x_mlp.fc1.bias",
    "joint_transformer_blocks.{i}.x_block.mlp.fc2.weight":  "joint_blocks.{i}.x_mlp.fc2.weight",
    "joint_transformer_blocks.{i}.x_block.mlp.fc2.bias":    "joint_blocks.{i}.x_mlp.fc2.bias",
    "joint_transformer_blocks.{i}.context_block.attn.qkv.weight":  "joint_blocks.{i}.c_attn.qkv.weight",
    "joint_transformer_blocks.{i}.context_block.attn.qkv.bias":    "joint_blocks.{i}.c_attn.qkv.bias",
    "joint_transformer_blocks.{i}.context_block.attn.proj.weight": "joint_blocks.{i}.c_attn.proj.weight",
    "joint_transformer_blocks.{i}.context_block.attn.proj.bias":   "joint_blocks.{i}.c_attn.proj.bias",
    "joint_transformer_blocks.{i}.context_block.mlp.fc1.weight":  "joint_blocks.{i}.c_mlp.fc1.weight",
    "joint_transformer_blocks.{i}.context_block.mlp.fc1.bias":    "joint_blocks.{i}.c_mlp.fc1.bias",
    "joint_transformer_blocks.{i}.context_block.mlp.fc2.weight":  "joint_blocks.{i}.c_mlp.fc2.weight",
    "joint_transformer_blocks.{i}.context_block.mlp.fc2.bias":    "joint_blocks.{i}.c_mlp.fc2.bias",
    # Single-stream blocks.
    "single_transformer_blocks.{i}.attn.qkv.weight":   "single_blocks.{i}.attn.qkv.weight",
    "single_transformer_blocks.{i}.attn.qkv.bias":     "single_blocks.{i}.attn.qkv.bias",
    "single_transformer_blocks.{i}.attn.proj.weight":  "single_blocks.{i}.attn.proj.weight",
    "single_transformer_blocks.{i}.attn.proj.bias":    "single_blocks.{i}.attn.proj.bias",
    "single_transformer_blocks.{i}.mlp.fc1.weight":    "single_blocks.{i}.mlp.fc1.weight",
    "single_transformer_blocks.{i}.mlp.fc1.bias":      "single_blocks.{i}.mlp.fc1.bias",
    "single_transformer_blocks.{i}.mlp.fc2.weight":    "single_blocks.{i}.mlp.fc2.weight",
    "single_transformer_blocks.{i}.mlp.fc2.bias":      "single_blocks.{i}.mlp.fc2.bias",
    # Time / pooled / label / final.
    "time_embedding.linear_1.weight":    "time_embed.0.weight",
    "time_embedding.linear_1.bias":      "time_embed.0.bias",
    "time_embedding.linear_2.weight":    "time_embed.2.weight",
    "time_embedding.linear_2.bias":      "time_embed.2.bias",
    "pooled_text_embedding.linear_1.weight": "pooled_embed.0.weight",
    "pooled_text_embedding.linear_1.bias":   "pooled_embed.0.bias",
    "pooled_text_embedding.linear_2.weight": "pooled_embed.2.weight",
    "pooled_text_embedding.linear_2.bias":   "pooled_embed.2.bias",
    "label_embedding.embedding_table.weight": "label_embed.weight",
    "proj_out.weight":  "final_layer.out_proj.weight",
    "proj_out.bias":    "final_layer.out_proj.bias",
    "norm_out.weight":  "final_layer.norm.weight",
    "norm_out.bias":    "final_layer.norm.bias",
}

WAN2_KEY_MAP: Dict[str, str] = {
    # Patch embed (3D).
    "patch_embedding.weight":            "patch_embed.proj.weight",
    "patch_embedding.bias":              "patch_embed.proj.bias",
    # Time embed.
    "time_embedding.0.weight":           "time_embed.0.weight",
    "time_embedding.0.bias":             "time_embed.0.bias",
    "time_embedding.2.weight":           "time_embed.2.weight",
    "time_embedding.2.bias":             "time_embed.2.bias",
    "time_projection.1.weight":          "time_proj.1.weight",
    "time_projection.1.bias":            "time_proj.1.bias",
    # Text embed.
    "text_embedding.0.weight":           "text_embed.0.weight",
    "text_embedding.0.bias":             "text_embed.0.bias",
    "text_embedding.2.weight":           "text_embed.2.weight",
    "text_embedding.2.bias":             "text_embed.2.bias",
    # Cross-attention blocks (per layer).
    "blocks.{i}.cross_attn.q.weight":    "blocks.{i}.cross_attn.q.weight",
    "blocks.{i}.cross_attn.q.bias":      "blocks.{i}.cross_attn.q.bias",
    "blocks.{i}.cross_attn.k.weight":    "blocks.{i}.cross_attn.k.weight",
    "blocks.{i}.cross_attn.k.bias":      "blocks.{i}.cross_attn.k.bias",
    "blocks.{i}.cross_attn.v.weight":    "blocks.{i}.cross_attn.v.weight",
    "blocks.{i}.cross_attn.v.bias":      "blocks.{i}.cross_attn.v.bias",
    "blocks.{i}.cross_attn.o.weight":    "blocks.{i}.cross_attn.out_proj.weight",
    "blocks.{i}.cross_attn.o.bias":      "blocks.{i}.cross_attn.out_proj.bias",
    "blocks.{i}.cross_attn.norm_q.weight": "blocks.{i}.cross_attn.norm_q.weight",
    "blocks.{i}.cross_attn.norm_k.weight": "blocks.{i}.cross_attn.norm_k.weight",
    # Self-attention.
    "blocks.{i}.self_attn.qkv.weight":   "blocks.{i}.self_attn.qkv.weight",
    "blocks.{i}.self_attn.qkv.bias":     "blocks.{i}.self_attn.qkv.bias",
    "blocks.{i}.self_attn.proj.weight":  "blocks.{i}.self_attn.out_proj.weight",
    "blocks.{i}.self_attn.proj.bias":    "blocks.{i}.self_attn.out_proj.bias",
    # FFN.
    "blocks.{i}.ffn.net.0.proj.weight":  "blocks.{i}.mlp.fc1.weight",
    "blocks.{i}.ffn.net.0.proj.bias":    "blocks.{i}.mlp.fc1.bias",
    "blocks.{i}.ffn.net.2.weight":       "blocks.{i}.mlp.fc2.weight",
    "blocks.{i}.ffn.net.2.bias":         "blocks.{i}.mlp.fc2.bias",
    # Norms + modulation.
    "blocks.{i}.scale_shift_table":      "blocks.{i}.scale_shift_table",
    "blocks.{i}.modulation":             "blocks.{i}.modulation",
    # Final.
    "head.head.weight":                  "final_layer.proj.weight",
    "head.head.bias":                    "final_layer.proj.bias",
    "head.norm.weight":                  "final_layer.norm.weight",
    "head.norm.bias":                    "final_layer.norm.bias",
}

MUSICGEN_KEY_MAP: Dict[str, str] = {
    # Text encoder.
    "text_encoder.transformer.Layers.{i}.self_attn.k_proj.weight":     "text_encoder.layers.{i}.self_attn.k_proj.weight",
    "text_encoder.transformer.Layers.{i}.self_attn.k_proj.bias":       "text_encoder.layers.{i}.self_attn.k_proj.bias",
    "text_encoder.transformer.Layers.{i}.self_attn.v_proj.weight":     "text_encoder.layers.{i}.self_attn.v_proj.weight",
    "text_encoder.transformer.Layers.{i}.self_attn.v_proj.bias":       "text_encoder.layers.{i}.self_attn.v_proj.bias",
    "text_encoder.transformer.Layers.{i}.self_attn.q_proj.weight":     "text_encoder.layers.{i}.self_attn.q_proj.weight",
    "text_encoder.transformer.Layers.{i}.self_attn.q_proj.bias":       "text_encoder.layers.{i}.self_attn.q_proj.bias",
    "text_encoder.transformer.Layers.{i}.self_attn.out_proj.weight":   "text_encoder.layers.{i}.self_attn.out_proj.weight",
    "text_encoder.transformer.Layers.{i}.self_attn.out_proj.bias":     "text_encoder.layers.{i}.self_attn.out_proj.bias",
    "text_encoder.transformer.Layers.{i}.fc1.weight":                  "text_encoder.layers.{i}.fc1.weight",
    "text_encoder.transformer.Layers.{i}.fc1.bias":                    "text_encoder.layers.{i}.fc1.bias",
    "text_encoder.transformer.Layers.{i}.fc2.weight":                  "text_encoder.layers.{i}.fc2.weight",
    "text_encoder.transformer.Layers.{i}.fc2.bias":                    "text_encoder.layers.{i}.fc2.bias",
    "text_encoder.transformer.Layers.{i}.self_attn_layer_norm.weight": "text_encoder.layers.{i}.self_attn_layer_norm.weight",
    "text_encoder.transformer.Layers.{i}.self_attn_layer_norm.bias":   "text_encoder.layers.{i}.self_attn_layer_norm.bias",
    "text_encoder.transformer.Layers.{i}.final_layer_norm.weight":     "text_encoder.layers.{i}.final_layer_norm.weight",
    "text_encoder.transformer.Layers.{i}.final_layer_norm.bias":       "text_encoder.layers.{i}.final_layer_norm.bias",
    # Audio decoder.
    "audio_encoder.transformer.Layers.{i}.self_attn.k_proj.weight":     "audio_decoder.layers.{i}.self_attn.k_proj.weight",
    "audio_encoder.transformer.Layers.{i}.self_attn.v_proj.weight":     "audio_decoder.layers.{i}.self_attn.v_proj.weight",
    "audio_encoder.transformer.Layers.{i}.self_attn.q_proj.weight":     "audio_decoder.layers.{i}.self_attn.q_proj.weight",
    "audio_encoder.transformer.Layers.{i}.self_attn.out_proj.weight":   "audio_decoder.layers.{i}.self_attn.out_proj.weight",
    "audio_encoder.transformer.Layers.{i}.fc1.weight":                  "audio_decoder.layers.{i}.fc1.weight",
    "audio_encoder.transformer.Layers.{i}.fc2.weight":                  "audio_decoder.layers.{i}.fc2.weight",
    "audio_encoder.transformer.Layers.{i}.self_attn_layer_norm.weight": "audio_decoder.layers.{i}.self_attn_layer_norm.weight",
    "audio_encoder.transformer.Layers.{i}.final_layer_norm.weight":     "audio_decoder.layers.{i}.final_layer_norm.weight",
    # Conditioning projection.
    "conditioning_provider.conditioners.text.conditioning_encoder.0.weight": "text_conditioning.0.weight",
    "conditioning_provider.conditioners.text.conditioning_encoder.0.bias":   "text_conditioning.0.bias",
    "conditioning_provider.conditioners.text.conditioning_encoder.2.weight": "text_conditioning.2.weight",
    "conditioning_provider.conditioners.text.conditioning_encoder.2.bias":   "text_conditioning.2.bias",
    "conditioning_provider.text_projector.weight":   "text_projector.weight",
    "conditioning_provider.text_projector.bias":     "text_projector.bias",
    # Final projection.
    "output_proj.weight":   "output_proj.weight",
    "output_proj.bias":     "output_proj.bias",
}


def _materialise_per_block_map(
    num_blocks: int,
    key_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Expand the ``{i}`` placeholders in a per-block key-rename table
    so that :func:`load_state_dict_with_renames` can apply them
    verbatim.

    Args:
        num_blocks: Number of blocks the model was instantiated with.
        key_map: Optional explicit table.  Defaults to
            :data:`HUNYUAN_DIT_KEY_MAP` for backwards compatibility.
    """
    src = key_map if key_map is not None else HUNYUAN_DIT_KEY_MAP
    out: Dict[str, str] = {}
    for k, v in src.items():
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


# ---------------------------------------------------------------------------
# v1.0.0 — Public loaders for the four model-matrix families.
#
# Each loader is a thin shim over ``ModelMixin.from_pretrained`` that
# applies the matching ``*_KEY_MAP`` plus the per-block placeholder
# expansion.  The target model class is best-effort: if it is not yet
# implemented locally the call falls back to the diffusers-compatible
# ``ModelMixin`` with the rename table applied to the supplied state
# dict.
# ---------------------------------------------------------------------------
def _resolve_class(modpath: str, name: str):
    try:
        mod = __import__(modpath, fromlist=[name])
        return getattr(mod, name, None)
    except Exception:
        return None


def _load_with_keymap(
    cls_candidates: List[Tuple[str, str]],
    weights_path: Union[str, "Path"],
    key_map: Dict[str, str],
    *,
    num_blocks: int,
    torch_dtype: Optional[torch.dtype],
    device: Union[str, torch.device, None],
    strict: bool,
) -> "ModelMixin":
    """Shared implementation for the v1.0.0 loaders.

    ``cls_candidates`` is a list of ``(module_path, class_name)``
    fallbacks — the first one that imports wins; if none import we
    fall back to :class:`models.base.ModelMixin` (still uses the
    key_renames table verbatim).
    """
    for modpath, name in cls_candidates:
        cls = _resolve_class(modpath, name)
        if cls is not None:
            target_cls = cls
            break
    else:
        target_cls = ModelMixin
    expanded = _materialise_per_block_map(num_blocks, key_map)
    return target_cls.from_pretrained(
        weights_path,
        torch_dtype=torch_dtype,
        device=device,
        key_renames=expanded,
        strict=strict,
    )


def load_flux(
    weights_path: Union[str, "Path"],
    *,
    torch_dtype: Optional[torch.dtype] = None,
    device: Union[str, torch.device, None] = None,
    num_blocks: int = 19,
    strict: bool = False,
) -> "ModelMixin":
    """Load a FLUX.1-dev model using :data:`FLUX_KEY_MAP`."""
    return _load_with_keymap(
        cls_candidates=[("models.image.flux", "Flux")],
        weights_path=weights_path,
        key_map=FLUX_KEY_MAP,
        num_blocks=num_blocks,
        torch_dtype=torch_dtype,
        device=device,
        strict=strict,
    )


def load_sd3(
    weights_path: Union[str, "Path"],
    *,
    torch_dtype: Optional[torch.dtype] = None,
    device: Union[str, torch.device, None] = None,
    num_blocks: int = 24,
    strict: bool = False,
) -> "ModelMixin":
    """Load a StableDiffusion-3 Medium model using :data:`SD3_KEY_MAP`."""
    return _load_with_keymap(
        cls_candidates=[("models.image.sd3", "SD3")],
        weights_path=weights_path,
        key_map=SD3_KEY_MAP,
        num_blocks=num_blocks,
        torch_dtype=torch_dtype,
        device=device,
        strict=strict,
    )


def load_wan2(
    weights_path: Union[str, "Path"],
    *,
    torch_dtype: Optional[torch.dtype] = None,
    device: Union[str, torch.device, None] = None,
    num_blocks: int = 40,
    strict: bool = False,
) -> "ModelMixin":
    """Load a Wan-2.1 video model using :data:`WAN2_KEY_MAP`."""
    return _load_with_keymap(
        cls_candidates=[("models.video.wan2", "Wan2")],
        weights_path=weights_path,
        key_map=WAN2_KEY_MAP,
        num_blocks=num_blocks,
        torch_dtype=torch_dtype,
        device=device,
        strict=strict,
    )


def load_musicgen(
    weights_path: Union[str, "Path"],
    *,
    torch_dtype: Optional[torch.dtype] = None,
    device: Union[str, torch.device, None] = None,
    num_blocks: int = 24,
    strict: bool = False,
) -> "ModelMixin":
    """Load a MusicGen audio model using :data:`MUSICGEN_KEY_MAP`."""
    return _load_with_keymap(
        cls_candidates=[("models.audio.musicgen", "MusicGen")],
        weights_path=weights_path,
        key_map=MUSICGEN_KEY_MAP,
        num_blocks=num_blocks,
        torch_dtype=torch_dtype,
        device=device,
        strict=strict,
    )
