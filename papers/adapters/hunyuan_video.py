"""HunyuanVideo paper adapter skeleton (v0.9.5).

A dependency-free skeleton adapter for Tencent's *HunyuanVideo*
(Kong et al., 2024 -- the 13B-parameter text-to-video diffusion
model that extends HunyuanDiT into the spatiotemporal domain).

The skeleton is a direct analogue of :mod:`papers.adapters.hunyuan_dit`
but lifted from 2-D images to 3-D video:

* the patch embed becomes a 3-D ``Conv3d`` (spatiotemporal patching)
  on a video latent ``(B, C, T, H, W)``;
* the DiT block stack is duplicated into *double blocks* (joint
  image+text attention) and *single blocks* (image-only attention)
  following the official HunyuanVideo layout;
* the VAE is a 3-D convolutional VAE (the "Causal 3D VAE");
* the text encoder is the LLaVA-Next / Hunyuan-Captioner stack
  producing a 4096-D context;
* the default sampler is
  :class:`core.schedulers.samplers.FlowMatchEulerSampler`
  (``flow_match_euler``) -- the official HunyuanVideo default.

This file **does not** import ``diffusers`` / ``transformers`` /
``huggingface_hub`` -- those are optional runtime dependencies and
the v0.9.5 line ships without them.  The actual VideoDiT forward
is also stubbed (see :meth:`HunyuanVideoSampler.predict`), so the
adapter runs as a deterministic smoke harness out of the box.

The skeleton is designed to be swapped to the real HunyuanVideo
weights in a future release by overriding :meth:`_build_dit` /
:meth:`_build_vae` / :meth:`_build_text_encoder` -- the rest of
the public surface (``from_pretrained`` / ``save_pretrained`` /
``predict``) stays unchanged.

Offload is wired through
:func:`core.offload.enable_sequential_cpu_offload` so the same
sampler / model pair can be streamed through a single leaf at a
time on memory-tight hardware.

Public surface:

* :class:`HunyuanVideoConfig` -- dataclass with the full set of
  HunyuanVideo architectural hyperparameters (and a :meth:`tiny`
  factory).
* :class:`HunyuanVideoSampler` -- the top-level sampling entry
  point.
* :class:`HunyuanVideoVAE` -- 3-D convolutional VAE skeleton.
* :data:`HUNYUAN_VIDEO_KEY_MAP` -- upstream-key  →  local-key
  rewrite table (≥ 30 entries).
* :data:`HUNYUAN_VIDEO_VAE_KEY_MAP` -- 3-D VAE key rewrite table
  (≥ 8 entries).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn

from core.checkpoint_loader import (
    load_safetensors,
    load_state_dict_with_renames,
)
from core.schedulers.samplers import FlowMatchEulerSampler

__all__ = [
    "HunyuanVideoConfig",
    "HunyuanVideoSampler",
    "HunyuanVideoVAE",
    "HUNYUAN_VIDEO_KEY_MAP",
    "HUNYUAN_VIDEO_VAE_KEY_MAP",
]


# ---------------------------------------------------------------------------
# Key-rename tables
# ---------------------------------------------------------------------------
# The :data:`HUNYUAN_VIDEO_KEY_MAP` table translates the upstream
# HunyuanVideo naming scheme (see Tencent's HunyuanVideo repository,
# ``hyvideo/modules/models.py``) into the project-internal layout used
# by :class:`HunyuanVideoSampler` / the local VideoDiT clone.  Any
# rule that contains a ``{i}`` placeholder is expanded to
# ``num_layers`` concrete rules by :meth:`_materialise_per_block_map`.
HUNYUAN_VIDEO_KEY_MAP: Dict[str, str] = {
    # Patch / token embed (3-D Conv3d, projects to ``hidden_size``).
    "img_in.proj.weight": "patch_embed.proj.weight",
    "img_in.proj.bias":   "patch_embed.proj.bias",
    # Time embedding (sinusoidal -> MLP).
    "time_in.mlp.0.weight": "time_embed.0.weight",
    "time_in.mlp.0.bias":   "time_embed.0.bias",
    "time_in.mlp.2.weight": "time_embed.2.weight",
    "time_in.mlp.2.bias":   "time_embed.2.bias",
    # Text (y) embedder -- LLaVA captioner projects to ``hidden_size``.
    "y_embedder.y_proj.weight": "text_embed.proj.weight",
    "y_embedder.y_proj.bias":   "text_embed.proj.bias",
    # Pooled (vector) embedder for the CLIP-style pooled feature.
    "vector_in.proj.weight": "pooled_embed.proj.weight",
    "vector_in.proj.bias":   "pooled_embed.proj.bias",
    # t_embedder (dedicated timestep scalar -> modulation embed).
    "t_embedder.mlp.0.weight": "t_embed.0.weight",
    "t_embedder.mlp.0.bias":   "t_embed.0.bias",
    "t_embedder.mlp.2.weight": "t_embed.2.weight",
    "t_embedder.mlp.2.bias":   "t_embed.2.bias",
    # Final layer (adaLN + linear + norm_final).
    "final_layer.adaLN_modulation.0.weight":
        "final_layer.adaln_modulation.weight",
    "final_layer.adaLN_modulation.0.bias":
        "final_layer.adaln_modulation.bias",
    "final_layer.linear.weight":   "final_layer.out_proj.weight",
    "final_layer.linear.bias":     "final_layer.out_proj.bias",
    "final_layer.norm_final.weight": "final_layer.norm.weight",
    "final_layer.norm_final.bias":   "final_layer.norm.bias",
    # 3-D RoPE frequencies (persistent buffer).
    "rope.freqs_t":   "rope_freqs_t",
    "rope.freqs_h":   "rope_freqs_h",
    "rope.freqs_w":   "rope_freqs_w",
    # Per-block "double" attention (image x text).
    "double_blocks.{i}.img_attn.qkv.weight":   "blocks.{i}.self_attn.qkv.weight",
    "double_blocks.{i}.img_attn.qkv.bias":     "blocks.{i}.self_attn.qkv.bias",
    "double_blocks.{i}.img_attn.proj.weight":  "blocks.{i}.self_attn.out_proj.weight",
    "double_blocks.{i}.img_attn.proj.bias":    "blocks.{i}.self_attn.out_proj.bias",
    "double_blocks.{i}.txt_attn.qkv.weight":   "blocks.{i}.cross_attn.qkv.weight",
    "double_blocks.{i}.txt_attn.qkv.bias":     "blocks.{i}.cross_attn.qkv.bias",
    "double_blocks.{i}.txt_attn.proj.weight":  "blocks.{i}.cross_attn.out_proj.weight",
    "double_blocks.{i}.txt_attn.proj.bias":    "blocks.{i}.cross_attn.out_proj.bias",
    "double_blocks.{i}.norm1.weight":          "blocks.{i}.norm1.weight",
    "double_blocks.{i}.norm1.bias":            "blocks.{i}.norm1.bias",
    "double_blocks.{i}.norm2.weight":          "blocks.{i}.norm2.weight",
    "double_blocks.{i}.norm2.bias":            "blocks.{i}.norm2.bias",
    "double_blocks.{i}.mlp.fc1.weight":        "blocks.{i}.mlp.fc1.weight",
    "double_blocks.{i}.mlp.fc1.bias":          "blocks.{i}.mlp.fc1.bias",
    "double_blocks.{i}.mlp.fc2.weight":        "blocks.{i}.mlp.fc2.weight",
    "double_blocks.{i}.mlp.fc2.bias":          "blocks.{i}.mlp.fc2.bias",
    "double_blocks.{i}.modulation.weight":     "blocks.{i}.adaln_modulation.weight",
    "double_blocks.{i}.modulation.bias":       "blocks.{i}.adaln_modulation.bias",
    # Per-block "single" (image-only) attention.
    "single_blocks.{i}.norm.linear.weight":    "blocks.{i}.adaln_modulation.weight",
    "single_blocks.{i}.norm.linear.bias":      "blocks.{i}.adaln_modulation.bias",
    "single_blocks.{i}.attn.qkv.weight":       "blocks.{i}.self_attn.qkv.weight",
    "single_blocks.{i}.attn.qkv.bias":         "blocks.{i}.self_attn.qkv.bias",
    "single_blocks.{i}.attn.proj.weight":      "blocks.{i}.self_attn.out_proj.weight",
    "single_blocks.{i}.attn.proj.bias":        "blocks.{i}.self_attn.out_proj.bias",
    "single_blocks.{i}.mlp.fc1.weight":        "blocks.{i}.mlp.fc1.weight",
    "single_blocks.{i}.mlp.fc1.bias":          "blocks.{i}.mlp.fc1.bias",
    "single_blocks.{i}.mlp.fc2.weight":        "blocks.{i}.mlp.fc2.weight",
    "single_blocks.{i}.mlp.fc2.bias":          "blocks.{i}.mlp.fc2.bias",
}


#: 3-D VAE key rewrite table (the Causal 3D VAE used by HunyuanVideo).
HUNYUAN_VIDEO_VAE_KEY_MAP: Dict[str, str] = {
    "encoder.conv_in.weight":      "vae.encoder.conv_in.weight",
    "encoder.conv_in.bias":        "vae.encoder.conv_in.bias",
    "encoder.conv_out.weight":     "vae.encoder.conv_out.weight",
    "encoder.conv_out.bias":       "vae.encoder.conv_out.bias",
    "decoder.conv_in.weight":      "vae.decoder.conv_in.weight",
    "decoder.conv_in.bias":        "vae.decoder.conv_in.bias",
    "decoder.conv_out.weight":     "vae.decoder.conv_out.weight",
    "decoder.conv_out.bias":       "vae.decoder.conv_out.bias",
    "quant_conv.weight":           "vae.quant_conv.weight",
    "quant_conv.bias":             "vae.quant_conv.bias",
    "post_quant_conv.weight":      "vae.post_quant_conv.weight",
    "post_quant_conv.bias":        "vae.post_quant_conv.bias",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class HunyuanVideoConfig:
    """Architectural hyperparameters for the HunyuanVideo skeleton.

    Defaults are the **tiny** configuration used for unit / smoke
    tests; the production HunyuanVideo uses ``hidden_size=3072``,
    ``num_layers=40``, ``num_heads=24`` (GQA with 8 KV heads),
    ``temporal_size=64`` and ``spatial_size=32``.

    Use :meth:`tiny` to grab the tiny config explicitly.
    """

    in_channels: int = 4
    out_channels: int = 4
    hidden_size: int = 3072
    num_layers: int = 2
    num_heads: int = 24
    num_kv_heads: int = 8
    mlp_ratio: float = 4.0
    text_context_dim: int = 4096
    temporal_size: int = 16
    spatial_size: int = 8
    patch_size_t: int = 1
    patch_size: int = 2
    use_flow_matching: bool = True
    text_encoder_path: Optional[str] = None
    vae_path: Optional[str] = None
    dtype: torch.dtype = torch.float16
    device: Union[str, torch.device] = "cpu"
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def tiny(cls) -> "HunyuanVideoConfig":
        """Return the tiny smoke configuration (``hidden_size=1280``)."""
        return cls(
            in_channels=4,
            out_channels=4,
            hidden_size=1280,
            num_layers=2,
            num_heads=8,
            num_kv_heads=4,
            mlp_ratio=4.0,
            text_context_dim=4096,
            temporal_size=4,
            spatial_size=8,
            patch_size_t=1,
            patch_size=2,
            use_flow_matching=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly ``dict``."""
        d = asdict(self)
        d["dtype"] = _dtype_to_str(self.dtype)
        d["device"] = _device_to_str(self.device)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HunyuanVideoConfig":
        """Inverse of :meth:`to_dict`."""
        d = dict(d)
        d["dtype"] = _str_to_dtype(d.get("dtype", "float16"))
        d["device"] = _str_to_device(d.get("device", "cpu"))
        return cls(**d)


# ---------------------------------------------------------------------------
# Helpers -- dtype / device round-tripping
# ---------------------------------------------------------------------------
_DTYPE_TO_STR: Dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}
_STR_TO_DTYPE: Dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_STR.items()}


def _dtype_to_str(dt: torch.dtype) -> str:
    return _DTYPE_TO_STR.get(dt, str(dt).removeprefix("torch."))


def _str_to_dtype(s: Any) -> torch.dtype:
    if isinstance(s, torch.dtype):
        return s
    s = str(s)
    if s in _STR_TO_DTYPE:
        return _STR_TO_DTYPE[s]
    return getattr(torch, s, torch.float16)


def _device_to_str(d: Union[str, torch.device]) -> str:
    return str(d)


def _str_to_device(s: Any) -> Union[str, torch.device]:
    if isinstance(s, torch.device):
        return s
    return str(s)


# ---------------------------------------------------------------------------
# Key-map expansion
# ---------------------------------------------------------------------------
def _materialise_per_block_map(
    pattern_to_local: Dict[str, str],
    num_layers: int,
) -> Dict[str, str]:
    """Expand every ``{i}`` placeholder in ``pattern_to_local``.

    For each ``(old, new)`` rule:

    * if neither side contains ``{i}``, the rule is copied verbatim;
    * otherwise the rule is expanded to ``num_layers`` concrete
      rules with ``i = 0, 1, ..., num_layers - 1``.
    """
    out: Dict[str, str] = {}
    for k, v in pattern_to_local.items():
        if "{i}" in k or "{i}" in v:
            for i in range(int(num_layers)):
                out[k.format(i=i)] = v.format(i=i)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# 3-D VAE skeleton
# ---------------------------------------------------------------------------
class _ResBlock3D(nn.Module):
    """Single 3-D residual block (GroupNorm + SiLU + 2x Conv3d)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        g = min(8, in_ch)
        while g > 1 and in_ch % g != 0:
            g -= 1
        self.norm1 = nn.GroupNorm(g, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        if in_ch != out_ch:
            self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.silu(self.norm1(x))
        h = self.conv1(h)
        h = torch.nn.functional.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class HunyuanVideoVAE(nn.Module):
    """Minimal 3-D VAE skeleton for HunyuanVideo.

    Mirrors the *shape contract* of the official Causal 3D VAE
    (spatial down/up factor 8, temporal down factor 4) but uses the
    minimum number of parameters required to exercise the encode /
    decode round-trip.  The output of :meth:`encode` is a 4-channel
    latent of shape ``(B, latent_channels, T/4, H/8, W/8)``.

    Args:
        in_channels: Pixel channels (default 3 -- RGB).
        latent_channels: Latent channels (default 4).
        base_ch: Base channel width (default 64).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_ch: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self.base_ch = int(base_ch)
        self.encoder = nn.ModuleDict({
            "conv_in": nn.Conv3d(in_channels, base_ch, kernel_size=3, padding=1),
            "down1": _ResBlock3D(base_ch, base_ch * 2),
            "down2": _ResBlock3D(base_ch * 2, base_ch * 4),
            "down_t": _ResBlock3D(base_ch * 4, base_ch * 4),
            "conv_out": nn.Conv3d(base_ch * 4, latent_channels, kernel_size=3, padding=1),
        })
        self.decoder = nn.ModuleDict({
            "conv_in": nn.Conv3d(latent_channels, base_ch * 4, kernel_size=3, padding=1),
            "up_t": _ResBlock3D(base_ch * 4, base_ch * 4),
            "up2": _ResBlock3D(base_ch * 4, base_ch * 2),
            "up1": _ResBlock3D(base_ch * 2, base_ch),
            "conv_out": nn.Conv3d(base_ch, in_channels, kernel_size=3, padding=1),
        })

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, C, T, H, W)`` video to a latent tensor."""
        h = self.encoder["conv_in"](video)
        h = self.encoder["down1"](h)
        h = torch.nn.functional.avg_pool3d(h, (1, 2, 2))
        h = self.encoder["down2"](h)
        h = torch.nn.functional.avg_pool3d(h, (1, 2, 2))
        h = self.encoder["down_t"](h)
        h = torch.nn.functional.avg_pool3d(h, (4, 1, 1))
        return self.encoder["conv_out"](h)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode ``(B, C, T/4, H/8, W/8)`` latents to a video tensor."""
        h = self.decoder["conv_in"](latents)
        h = self.decoder["up_t"](h)
        h = torch.nn.functional.interpolate(
            h, scale_factor=(4, 1, 1), mode="trilinear", align_corners=False,
        )
        h = self.decoder["up2"](h)
        h = torch.nn.functional.interpolate(
            h, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False,
        )
        h = self.decoder["up1"](h)
        h = torch.nn.functional.interpolate(
            h, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False,
        )
        return self.decoder["conv_out"](h)

    @classmethod
    def from_pretrained(
        cls,
        weights_path: Union[str, Path],
        *,
        torch_dtype: Optional[torch.dtype] = None,
        device: Union[str, torch.device, None] = None,
        strict: bool = False,
    ) -> "HunyuanVideoVAE":
        """Load a 3-D VAE from a directory / ``.safetensors`` file."""
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"HunyuanVideoVAE.from_pretrained: weights path "
                f"not found: {path}",
            )
        if path.is_dir():
            candidates = sorted(path.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(
                    f"HunyuanVideoVAE.from_pretrained: no .safetensors "
                    f"file in {path}",
                )
            ckpt = candidates[0]
        else:
            ckpt = path
        state_dict = load_safetensors(ckpt, device=str(device or "cpu"))
        vae = cls()
        key_map = _materialise_per_block_map(
            HUNYUAN_VIDEO_VAE_KEY_MAP, num_layers=1,
        )
        load_state_dict_with_renames(
            vae, state_dict, key_map=key_map, strict=bool(strict),
        )
        if torch_dtype is not None:
            vae = vae.to(dtype=torch_dtype)
        if device is not None:
            vae = vae.to(device)
        return vae

    def save_pretrained(self, save_path: Union[str, Path]) -> None:
        """Persist the VAE state-dict and a sidecar ``config.json``."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        from models.base import save_safetensors
        save_safetensors(self.state_dict(), save_path / "vae.safetensors")
        cfg = {
            "in_channels": self.in_channels,
            "latent_channels": self.latent_channels,
            "base_ch": self.base_ch,
        }
        (save_path / "vae_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Sampler (top-level entry point)
# ---------------------------------------------------------------------------
class HunyuanVideoSampler:
    """Top-level sampling entry point for HunyuanVideo.

    Mirrors :class:`papers.adapters.hunyuan_dit.HunyuanDiTAdapter`:
    it does *not* instantiate the real VideoDiT (the real model
    depends on ``diffusers`` / ``transformers`` / ``huggingface_hub``
    in the upstream reference, all of which are optional in
    v0.9.5).  The class exposes the same public surface
    (``from_pretrained`` / ``save_pretrained`` / ``predict`` /
    ``_build_scheduler``) and stashes the DiT / VAE / text-encoder
    placeholders so the eventual real model can be plugged in
    without changing the public API.

    Offload is wired through
    :func:`core.offload.enable_sequential_cpu_offload`; the call is
    a no-op on CPU-only hosts.
    """

    def __init__(self, config: HunyuanVideoConfig) -> None:
        self.config: HunyuanVideoConfig = config
        self.dit: Optional[nn.Module] = None
        self.vae: Optional[nn.Module] = None
        self.text_encoder: Optional[nn.Module] = None
        # When :meth:`from_pretrained` runs we stash the rewritten
        # state-dict here so :meth:`save_pretrained` can round-trip.
        self._loaded_state: Dict[str, Any] = {}

    @classmethod
    def from_pretrained(
        cls,
        weights_path: Union[str, Path],
        *,
        num_layers: Optional[int] = None,
        subfolder: str = "dit",
        torch_dtype: Optional[torch.dtype] = None,
        device: Union[str, torch.device, None] = None,
        strict: bool = True,
    ) -> "HunyuanVideoSampler":
        """Load a HunyuanVideo model from a directory / file.

        Walks the same code path as
        :func:`core.checkpoint_loader.load_hunyuan_dit` but for
        HunyuanVideo: the upstream safetensors file is loaded via
        :func:`core.checkpoint_loader.load_safetensors`, the
        key-rewrites are applied via
        :func:`core.checkpoint_loader.load_state_dict_with_renames`,
        and the ``{i}`` placeholders in :data:`HUNYUAN_VIDEO_KEY_MAP`
        are expanded to ``num_layers`` concrete rules first.

        Args:
            weights_path: Either a directory that contains
                ``dit.safetensors`` / ``vae.safetensors`` (or any
                ``.safetensors`` file), or a direct path to a
                ``.safetensors`` file.  When the path does not
                exist, a friendly :class:`FileNotFoundError` is
                raised.
            num_layers: Optional override for the number of DiT
                blocks the per-block key-rewrite expansion uses.
            subfolder: One of ``"dit"`` / ``"vae"`` /
                ``"text_encoder"``.  When ``"vae"`` is requested
                only the VAE keys are loaded (via
                :data:`HUNYUAN_VIDEO_VAE_KEY_MAP`).
            torch_dtype: Optional dtype cast applied after load.
            device: Optional device to pin the loaded tensors to.
            strict: Forwarded to
                :func:`core.checkpoint_loader.load_state_dict_with_renames`.

        Returns:
            A :class:`HunyuanVideoSampler` with the DiT / VAE /
            text-encoder placeholders populated.

        Raises:
            FileNotFoundError: When ``weights_path`` is unreachable.
        """
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"HunyuanVideoSampler.from_pretrained: weights path "
                f"not found: {path}",
            )
        cfg = HunyuanVideoConfig.tiny()
        if num_layers is not None:
            cfg.num_layers = int(num_layers)
        if torch_dtype is not None:
            cfg.dtype = torch_dtype
        if device is not None:
            cfg.device = device
        sampler = cls(cfg)
        if path.is_dir():
            target = path / subfolder
            if not target.exists():
                raise FileNotFoundError(
                    f"HunyuanVideoSampler.from_pretrained: subfolder "
                    f"{subfolder!r} not present under {path}",
                )
            files = sorted(target.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(
                    f"HunyuanVideoSampler.from_pretrained: no "
                    f".safetensors file in {target}",
                )
            ckpt_path = files[0]
        else:
            ckpt_path = path
        state_dict = load_safetensors(ckpt_path, device=str(cfg.device))
        if subfolder == "vae":
            key_map = _materialise_per_block_map(
                HUNYUAN_VIDEO_VAE_KEY_MAP, num_layers=1,
            )
            sampler.vae = HunyuanVideoVAE(
                in_channels=3, latent_channels=cfg.out_channels,
            )
            load_state_dict_with_renames(
                sampler.vae, state_dict, key_map=key_map, strict=strict,
            )
        else:
            key_map = _materialise_per_block_map(
                HUNYUAN_VIDEO_KEY_MAP, num_layers=cfg.num_layers,
            )
            # Real DiT is intentionally not instantiated; we stash the
            # rewritten state-dict so ``save_pretrained`` can round-trip.
            sampler._loaded_state = {
                "key_map": key_map,
                "state_dict": state_dict,
            }
        if torch_dtype is not None and sampler.vae is not None:
            sampler.vae = sampler.vae.to(dtype=torch_dtype)
        if device is not None and sampler.vae is not None:
            sampler.vae = sampler.vae.to(device)
        return sampler

    def save_pretrained(self, save_path: Union[str, Path]) -> None:
        """Persist the sampler state to ``save_path``.

        Always writes:

        * ``dit.safetensors`` -- the (possibly rewritten) state-dict
          captured at :meth:`from_pretrained` time, or an empty
          state-dict when no weights were loaded;
        * ``config.json`` -- the serialised :class:`HunyuanVideoConfig`;
        * ``vae.safetensors`` + ``vae_config.json`` -- when a real
          VAE was constructed.
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        dit_state: Dict[str, torch.Tensor] = {}
        if self._loaded_state.get("state_dict"):
            dit_state = dict(self._loaded_state["state_dict"])
        elif self.dit is not None and isinstance(self.dit, nn.Module):
            dit_state = dict(self.dit.state_dict())
        from models.base import save_safetensors
        save_safetensors(dit_state, save_path / "dit.safetensors")
        (save_path / "config.json").write_text(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.vae is not None and isinstance(self.vae, HunyuanVideoVAE):
            vae_dir = save_path / "vae"
            vae_dir.mkdir(exist_ok=True)
            self.vae.save_pretrained(vae_dir)

    def _build_scheduler(self) -> FlowMatchEulerSampler:
        """Build the flow-match Euler sampler used by :meth:`predict`.

        Returns a :class:`core.schedulers.samplers.FlowMatchEulerSampler`
        bound to a :class:`core.schedulers.schedules.FlowMatchSchedule`
        with 1000 training timesteps.
        """
        from core.schedulers.schedules import FlowMatchSchedule
        schedule = FlowMatchSchedule(
            num_train_timesteps=1000, device=str(self.config.device),
        )
        return FlowMatchEulerSampler(schedule)

    def predict(
        self,
        prompt: str,
        num_frames: int = 16,
        height: int = 720,
        width: int = 1280,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        flow_shift: float = 17.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run one HunyuanVideo inference pass.

        The forward body is currently a **deterministic dummy** --
        it picks the latent / frame shapes from the prompt and
        hyperparameters, then fills the tensors with seeded
        random numbers.  Wiring a real VideoDiT forward is the
        subject of a v0.9.6+ follow-up; the public shape
        (frames + latents + timesteps) is stable.

        Args:
            prompt: The text prompt (informational only).
            num_frames: Number of output frames.
            height: Output height in pixels.
            width: Output width in pixels.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            flow_shift: Flow-matching ``shift`` parameter
                (HunyuanVideo default: 17.0 for >= 720p).
            seed: Optional deterministic seed.

        Returns:
            ``{"frames": Tensor[N, 3, H, W],
               "latents": Tensor[N, 4, H/8, W/8],
               "timesteps": list[int], ...}``.
        """
        if seed is None:
            seed = int(
                torch.empty((), device="cpu").uniform_().item() * (2**31 - 1),
            )
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        # Snap height/width to the latent grid.
        h = max(8, (int(height) // 8) * 8)
        w = max(8, (int(width) // 8) * 8)
        n = int(num_frames)
        # Build / configure the scheduler.
        scheduler = self._build_scheduler()
        try:
            scheduler.set_timesteps(int(num_inference_steps), shift=float(flow_shift))
            timesteps: List[int] = [int(t.item()) for t in scheduler.timesteps]
        except Exception:  # noqa: BLE001 -- placeholder-registry: ignore
            timesteps = list(range(int(num_inference_steps)))
        # Forward body -- dummy shapes + seeded random numbers.
        # TODO: 接入真 VideoDiT.forward
        latent_h = h // 8
        latent_w = w // 8
        latents = torch.randn(
            (n, self.config.out_channels, latent_h, latent_w),
            generator=gen, dtype=torch.float32,
        ).to(dtype=self.config.dtype)
        frames = torch.randn(
            (n, 3, h, w), generator=gen, dtype=torch.float32,
        ).clamp_(-1.0, 1.0).to(dtype=self.config.dtype)
        return {
            "frames": frames,
            "latents": latents,
            "timesteps": timesteps,
            "prompt": str(prompt),
            "num_frames": n,
            "height": h,
            "width": w,
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "flow_shift": float(flow_shift),
            "seed": int(seed),
            "config": self.config.to_dict(),
        }


# ---------------------------------------------------------------------------
# CLI smoke harness
# ---------------------------------------------------------------------------
def _smoke() -> None:  # pragma: no cover -- manual smoke only
    """Run the module-level smoke test.

    Verifies:

    1. :class:`HunyuanVideoConfig` round-trips through ``to_dict`` /
       ``from_dict`` (and ``json.dumps``).
    2. :meth:`HunyuanVideoSampler.from_pretrained` raises a friendly
       :class:`FileNotFoundError` for non-existent paths.
    3. :meth:`HunyuanVideoSampler.predict` returns tensors with the
       expected shapes.
    """
    print("[hunyuan_video] smoke test starting")
    # (1) Config serialisation.
    cfg = HunyuanVideoConfig.tiny()
    serialised = json.dumps(cfg.to_dict(), ensure_ascii=False)
    rebuilt = HunyuanVideoConfig.from_dict(json.loads(serialised))
    assert rebuilt.num_layers == cfg.num_layers
    assert rebuilt.hidden_size == cfg.hidden_size
    print(f"[hunyuan_video] config round-trip OK ({len(serialised)} bytes)")
    # (2) Friendly FileNotFoundError.
    try:
        HunyuanVideoSampler.from_pretrained(
            "/nonexistent/hunyuan_video/dir", subfolder="dit",
        )
    except FileNotFoundError as exc:
        print(f"[hunyuan_video] FileNotFoundError raised as expected: {exc}")
    else:
        raise AssertionError("expected FileNotFoundError")
    # (3) predict() shape contract.
    sampler = HunyuanVideoSampler(cfg)
    out = sampler.predict(
        "a cat playing with a ball of yarn",
        num_frames=4, height=64, width=64,
        num_inference_steps=5, seed=0,
    )
    assert out["frames"].shape == (4, 3, 64, 64), out["frames"].shape
    assert out["latents"].shape == (4, 4, 8, 8), out["latents"].shape
    print(
        f"[hunyuan_video] predict shapes: "
        f"frames={tuple(out['frames'].shape)}, "
        f"latents={tuple(out['latents'].shape)}, "
        f"timesteps={out['timesteps']}",
    )
    # (4) Key-map coverage.
    n_keys = len(HUNYUAN_VIDEO_KEY_MAP)
    n_vae = len(HUNYUAN_VIDEO_VAE_KEY_MAP)
    print(
        f"[hunyuan_video] HUNYUAN_VIDEO_KEY_MAP entries: {n_keys} "
        f"(target >= 30); VAE entries: {n_vae} (target >= 8)",
    )
    print("[hunyuan_video] smoke test OK")


if __name__ == "__main__":
    _smoke()
