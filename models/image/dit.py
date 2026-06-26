"""Diffusion Transformer (DiT).

This module implements a Diffusion Transformer (DiT) in the style of
SD3 / Sora: the noisy latent is split into patches, projected into
tokens, and processed by a stack of Transformer blocks that use
adaptive LayerNorm (adaLN-Zero) for timestep conditioning and
cross-attention for text conditioning.

Key components:

* :class:`PatchEmbed` -- image-to-patch embedding.
* :class:`DiTBlock` -- a single DiT block with adaLN-Zero.
* :class:`DiT` -- the full Diffusion Transformer.

Reference:
    Peebles & Xie, "Scalable Diffusion Models with Transformers" (2022).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from models.components.rope import RotaryPositionEmbedding
from models.components.rmsnorm import RMSNorm
from .unet import TimestepEmbedding

__all__ = [
    "PatchEmbed",
    "DiTBlock",
    "DiT",
    "HunyuanDiTConfig",
    "HunyuanDiTBlock",
    "HunyuanDiT",
]


class PatchEmbed(nn.Module):
    """Convert an image (or latent) into a sequence of patch tokens.

    Args:
        patch_size: Size of each square patch.
        in_channels: Number of input channels.
        hidden_size: Output (token) dimension.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size: int = patch_size
        self.in_channels: int = in_channels
        self.hidden_size: int = hidden_size
        self.proj: nn.Conv2d = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Embed the image into patch tokens.

        Args:
            x: Image tensor of shape ``(batch, in_channels, H, W)``.

        Returns:
            A tuple ``(tokens, h_patches, w_patches)`` where ``tokens``
            has shape ``(batch, num_patches, hidden_size)``.
        """
        x = self.proj(x)  # (batch, hidden, H/p, W/p)
        batch, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (batch, h*w, hidden)
        return x, h, w


class DiTBlock(nn.Module):
    """A single DiT block with adaptive LayerNorm-Zero conditioning.

    The timestep embedding produces six modulation vectors (shift, scale,
    gate) for the attention and MLP sub-layers.  The gates are
    initialised to zero so the block is initially an identity function
    (adaLN-Zero), which stabilises training.

    Args:
        hidden_size: Model dimension.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key/value heads (GQA).
        context_dim: Text embedding dimension (``0`` disables cross-attn).
        mlp_ratio: MLP intermediate-size ratio.
        norm_eps: Normalisation epsilon.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        context_dim: int = 0,
        mlp_ratio: float = 4.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads or num_heads
        self.head_dim: int = hidden_size // num_heads
        self.context_dim: int = context_dim

        # adaLN modulation: produces 6 * hidden_size parameters.
        self.adaLN_modulation: nn.Sequential = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        self.norm1: nn.Module = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2: nn.Module = RMSNorm(hidden_size, eps=norm_eps)

        # Self-attention.
        self.attn_q: nn.Linear = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.attn_k: nn.Linear = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.attn_v: nn.Linear = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.attn_out: nn.Linear = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # Optional cross-attention.
        self.cross_attn_norm: Optional[nn.Module] = None
        self.cross_q: Optional[nn.Linear] = None
        self.cross_k: Optional[nn.Linear] = None
        self.cross_v: Optional[nn.Linear] = None
        self.cross_out: Optional[nn.Linear] = None
        self.cross_gate: Optional[nn.Linear] = None
        if context_dim > 0:
            self.cross_attn_norm = RMSNorm(hidden_size, eps=norm_eps)
            self.cross_q = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
            self.cross_k = nn.Linear(context_dim, self.num_kv_heads * self.head_dim, bias=False)
            self.cross_v = nn.Linear(context_dim, self.num_kv_heads * self.head_dim, bias=False)
            self.cross_out = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
            self.cross_gate = nn.Linear(hidden_size, hidden_size)

        # MLP.
        self.mlp_fc1: nn.Linear = nn.Linear(hidden_size, int(hidden_size * mlp_ratio))
        self.mlp_fc2: nn.Linear = nn.Linear(int(hidden_size * mlp_ratio), hidden_size)

        self.scale: float = self.head_dim ** -0.5

    # ------------------------------------------------------------------
    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Run multi-head self-attention (GQA)."""
        batch, seq_len, _ = x.shape
        q = self.attn_q(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.attn_k(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.attn_v(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # Repeat KV heads.
        if self.num_heads != self.num_kv_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.attn_out(attn)

    def _cross_attention(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Run cross-attention to ``context``."""
        batch, seq_len, _ = x.shape
        ctx_len = context.shape[1]
        q = self.cross_q(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.cross_k(context).view(batch, ctx_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.cross_v(context).view(batch, ctx_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.num_heads != self.num_kv_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.cross_out(attn)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the DiT block.

        Args:
            x: Patch tokens ``(batch, seq_len, hidden_size)``.
            temb: Timestep embedding ``(batch, hidden_size)``.
            context: Optional text embeddings ``(batch, ctx_len, context_dim)``.

        Returns:
            Output tokens ``(batch, seq_len, hidden_size)``.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(temb).chunk(6, dim=-1)
        )

        # Self-attention with adaLN.
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self._self_attention(h)
        x = x + gate_msa.unsqueeze(1) * h

        # Cross-attention (gated).
        if self.cross_attn_norm is not None and context is not None:
            h = self.cross_attn_norm(x)
            h = self._cross_attention(h, context)
            x = x + torch.sigmoid(self.cross_gate(temb)).unsqueeze(1) * h

        # MLP with adaLN.
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = F.gelu(self.mlp_fc1(h))
        h = self.mlp_fc2(h)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


class DiT(BaseModel):
    """Diffusion Transformer.

    Args:
        input_size: Spatial size of the input latent (assumed square).
        patch_size: Patch size.
        in_channels: Number of input latent channels.
        hidden_size: Model dimension.
        num_layers: Number of DiT blocks.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        context_dim: Text embedding dimension (``0`` disables cross-attn).
        mlp_ratio: MLP intermediate-size ratio.
        config: Optional configuration dictionary.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        num_layers: int = 28,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        context_dim: int = 4096,
        mlp_ratio: float = 4.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            input_size = config.get("input_size", input_size)
            patch_size = config.get("patch_size", patch_size)
            in_channels = config.get("in_channels", in_channels)
            hidden_size = config.get("hidden_size", hidden_size)
            num_layers = config.get("num_layers", num_layers)
            num_heads = config.get("num_heads", num_heads)
            num_kv_heads = config.get("num_kv_heads", num_kv_heads)
            context_dim = config.get("context_dim", context_dim)
            mlp_ratio = config.get("mlp_ratio", mlp_ratio)

        super().__init__(config=config)

        self.input_size: int = input_size
        self.patch_size: int = patch_size
        self.in_channels: int = in_channels
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads or num_heads
        self.context_dim: int = context_dim

        self.patch_embed: PatchEmbed = PatchEmbed(patch_size, in_channels, hidden_size)
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed: nn.Parameter = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Timestep embedding.
        self.time_embed: TimestepEmbedding = TimestepEmbedding(hidden_size)

        self.blocks: nn.ModuleList = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=self.num_kv_heads,
                context_dim=context_dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_layers)
        ])

        # Final adaLN + output projection.
        self.final_norm: nn.Module = RMSNorm(hidden_size)
        self.final_adaLN: nn.Sequential = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.final_linear: nn.Linear = nn.Linear(
            hidden_size, patch_size * patch_size * in_channels
        )
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict the noise added to ``x``.

        Args:
            x: Noisy latent of shape ``(batch, in_channels, H, W)``.
            timesteps: Diffusion timesteps ``(batch,)``.
            encoder_hidden_states: Text embeddings
                ``(batch, seq_len, context_dim)``.

        Returns:
            Noise prediction of shape ``(batch, in_channels, H, W)``.
        """
        batch, _, h, w = x.shape
        temb = self.time_embed(timesteps)

        x, ph, pw = self.patch_embed(x)
        x = x + self.pos_embed[:, : x.shape[1], :]

        for block in self.blocks:
            x = block(x, temb, encoder_hidden_states)

        # Final adaLN.
        shift, scale = self.final_adaLN(temb).chunk(2, dim=-1)
        x = self.final_norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_linear(x)

        # Unpatchify.
        x = x.reshape(batch, ph, pw, self.patch_size, self.patch_size, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(batch, self.in_channels, ph * self.patch_size, pw * self.patch_size)
        return x

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        shape: Tuple[int, ...],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run a simple DDPM-style denoising loop.

        Args:
            shape: Shape of the latent to generate
                ``(batch, in_channels, H, W)``.
            encoder_hidden_states: Optional text conditioning.
            num_steps: Number of denoising steps.

        Returns:
            Denoised latent tensor.
        """
        self.eval()
        device = next(self.parameters()).device
        x = torch.randn(shape, device=device)
        for i in reversed(range(num_steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            noise_pred = self.forward(x, t, encoder_hidden_states)
            alpha = 1.0 - i / num_steps
            x = (x - (1 - alpha) * noise_pred) / max(alpha, 1e-4)
        return x


# ---------------------------------------------------------------------------
# HunyuanDiT — upstream-parameter-named DiT (v0.8.5)
# ---------------------------------------------------------------------------
# This module is the v0.8.5 "真大模型" entry point.  Unlike the
# generic :class:`DiT` above, every module / parameter name matches
# the upstream HunyuanDiT v1.2 layout verbatim
# (``img_in.proj``, ``time_in.mlp.{0,2}``, ``vector_in.proj``,
# ``blocks.{i}.attn.qkv``, ``blocks.{i}.mlp.fc1`` ...,
# ``final_layer.linear`` etc.) so that
# :data:`core.checkpoint_loader.HUNYUAN_DIT_KEY_MAP` can rewrite a
# real Tencent checkpoint into the local module without any extra
# mapping.
@dataclass
class HunyuanDiTConfig:
    """Configuration for :class:`HunyuanDiT`.

    Attributes:
        input_size: Spatial size of the input latent (assumed square).
        patch_size: Patch size for tokenisation.
        in_channels: Number of input latent channels.
        hidden_size: Token embedding dimension.
        num_layers: Number of DiT blocks.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        context_dim: Text embedding dimension (``0`` disables cross-attn).
        mlp_ratio: MLP intermediate-size ratio.
        use_style_embed: Whether to materialise the ``style_embedder`` /
            ``size_embedder`` weights (the upstream model exposes them
            even though the inference loop typically only consumes
            ``style_embedder``).
    """

    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    num_layers: int = 20
    num_heads: int = 16
    num_kv_heads: Optional[int] = None
    context_dim: int = 4096
    mlp_ratio: float = 4.0
    use_style_embed: bool = True

    # ------------------------------------------------------------------
    @classmethod
    def tiny(cls) -> "HunyuanDiTConfig":
        """Return a tiny preset used by the v0.8.5 smoke tests.

        The tiny preset keeps the architecture faithful (real adaLN
        modulation, real cross-attention, real final layer) but uses
        a 96-dim model with 2 blocks so the full forward pass
        completes in milliseconds on a stock CPU.
        """
        return cls(
            input_size=8,
            patch_size=2,
            in_channels=4,
            hidden_size=96,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            context_dim=64,
            mlp_ratio=2.0,
            use_style_embed=True,
        )


class HunyuanDiTBlock(nn.Module):
    """A single HunyuanDiT block with the local-layout parameter names.

    The local layout matches :data:`HUNYUAN_DIT_KEY_MAP` target
    values, which is the v0.8.0+ convention for any model that
    wants to ingest a real HunyuanDiT checkpoint with a single
    declarative ``key_renames=`` argument.

    Local layout:

    * ``attn.qkv``  -- a single QKV projection.
    * ``attn.out_proj`` -- attention output projection.
    * ``mlp.fc1`` / ``mlp.fc2`` -- MLP.
    * ``adaln_modulation`` -- adaLN-Zero modulation (single Linear,
      no nested Sequential, the upstream ``.0`` suffix is
      consumed by :data:`HUNYUAN_DIT_KEY_MAP`).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        context_dim: int = 0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads or num_heads
        self.head_dim: int = hidden_size // num_heads
        self.context_dim: int = context_dim
        # adaLN-Zero modulation -- a single Linear (the upstream
        # nests it as ``adaln_modulation.0`` with a SiLU wrapper
        # at index 1; the SiLU is folded into the surrounding
        # call site, see :meth:`HunyuanDiT.forward`).
        self.adaln_modulation: nn.Linear = nn.Linear(
            hidden_size, 6 * hidden_size, bias=True,
        )
        nn.init.zeros_(self.adaln_modulation.weight)
        nn.init.zeros_(self.adaln_modulation.bias)
        # Norms.
        self.norm1: nn.Module = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2: nn.Module = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Self-attention (combined QKV).
        self.attn: nn.Module = nn.Module()
        self.attn.qkv: nn.Linear = nn.Linear(
            hidden_size, 3 * hidden_size, bias=True,
        )
        self.attn.out_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=True)
        # Cross-attention (optional).
        if context_dim > 0:
            self.cross_attn: nn.Module = nn.Module()
            self.cross_attn.norm: nn.Module = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6,
            )
            self.cross_attn.q: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=False)
            self.cross_attn.kv: nn.Linear = nn.Linear(context_dim, 2 * hidden_size, bias=False)
            self.cross_attn.proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=False)
        # MLP.
        self.mlp: nn.Module = nn.Module()
        self.mlp.fc1: nn.Linear = nn.Linear(hidden_size, int(hidden_size * mlp_ratio))
        self.mlp.fc2: nn.Linear = nn.Linear(int(hidden_size * mlp_ratio), hidden_size)

    # ------------------------------------------------------------------
    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Run combined-QKV self-attention with head splitting."""
        b, n, _ = x.shape
        qkv = self.attn.qkv(x)
        qkv = qkv.view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.attn.out_proj(out)

    def _cross_attention(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        c_n = context.shape[1]
        q = self.cross_attn.q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.cross_attn.kv(context).view(
            b, c_n, 2, self.num_heads, self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.cross_attn.proj(out)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with adaLN-Zero conditioning."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            F.silu(self.adaln_modulation(temb)).chunk(6, dim=-1)
        )
        # Self-attention branch.
        h = self.norm1(x) * (1.0 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self._self_attention(h)
        x = x + gate_msa.unsqueeze(1) * h
        # Optional cross-attention branch.
        if self.context_dim > 0 and context is not None:
            h = self.cross_attn.norm(x)
            h = self._cross_attention(h, context)
            x = x + h
        # MLP branch.
        h = self.norm2(x) * (1.0 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = F.gelu(self.mlp.fc1(h))
        h = self.mlp.fc2(h)
        x = x + gate_mlp.unsqueeze(1) * h
        return x


class HunyuanDiT(BaseModel):
    """HunyuanDiT — Tencent bilingual text-to-image diffusion transformer.

    The class exposes the **local-layout** parameter naming so a
    real HunyuanDiT checkpoint can be loaded with a single
    declarative ``key_renames=core.checkpoint_loader.HUNYUAN_DIT_KEY_MAP``
    argument (no extra mapping needed).  See
    :func:`core.checkpoint_loader.load_hunyuan_dit` for the
    end-to-end recipe.

    Local layout (matches :data:`HUNYUAN_DIT_KEY_MAP` target values):

    * ``patch_embed.proj``  -- patch embed conv.
    * ``x_embedder``  -- optional 1x1 token re-projection.
    * ``time_embed.{0,2}``  -- timestep MLP.
    * ``pooled_embed.proj``  -- pooled (CLIP-G pooled) embedder.
    * ``style_embed``  -- learnt style code.
    * ``size_embed``  -- learnt resolution code.
    * ``rope_freqs``  -- rotary inverse-frequency buffer.
    * ``blocks.{i}.attn.qkv / attn.out_proj / mlp.fc1 / mlp.fc2 / adaln_modulation``
    * ``final_layer.adaln_modulation`` / ``final_layer.out_proj`` /
      ``final_layer.norm``.

    Args:
        config: A :class:`HunyuanDiTConfig` or ``None`` to use the
            tiny preset (96-dim / 2-block).  When called from
            :meth:`from_pretrained` the config is rebuilt from the
            ``config.json`` sidecar.
    """

    _default_file_extension: str = ".safetensors"

    def __init__(
        self,
        config: Optional[HunyuanDiTConfig] = None,
        **kwargs: Any,
    ) -> None:
        # The default constructor falls back to the tiny preset so
        # the v0.8.5 smoke tests can instantiate the model without
        # passing any arguments.  The full HunyuanDiT-1.2B
        # configuration is built by the ``from_pretrained`` method
        # from the saved ``config.json`` sidecar.
        if config is None:
            config = HunyuanDiTConfig.tiny()
        if isinstance(config, dict):
            config = HunyuanDiTConfig(**config)
        super().__init__(config=dict(
            input_size=config.input_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            context_dim=config.context_dim,
            mlp_ratio=config.mlp_ratio,
            use_style_embed=config.use_style_embed,
        ))

        self.input_size: int = config.input_size
        self.patch_size: int = config.patch_size
        self.in_channels: int = config.in_channels
        self.hidden_size: int = config.hidden_size
        self.num_layers: int = config.num_layers
        self.num_heads: int = config.num_heads
        self.num_kv_heads: int = config.num_kv_heads or config.num_heads
        self.context_dim: int = config.context_dim
        self.mlp_ratio: float = config.mlp_ratio

        # Patch embed (local name: ``patch_embed.proj``).
        self.patch_embed: nn.Module = nn.Module()
        self.patch_embed.proj: nn.Conv2d = nn.Conv2d(
            config.in_channels, config.hidden_size,
            kernel_size=config.patch_size, stride=config.patch_size,
        )
        # Optional 1x1 token re-projection.
        self.x_embedder: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        # Timestep embed -- ``time_embed`` is a 2-layer MLP with
        # the inner Linear at index ``0`` and the outer Linear at
        # index ``2`` (matching the rewrite table).
        self.time_embed: nn.Sequential = nn.Sequential(
            TimestepEmbedding(config.hidden_size).mlp[0],
            nn.SiLU(),
            TimestepEmbedding(config.hidden_size).mlp[2],
        )
        # Pooled text embedder (CLIP-G pooled).  The local name is
        # ``pooled_embed.proj``.
        self.pooled_embed: nn.Module = nn.Module()
        self.pooled_embed.proj: nn.Linear = nn.Linear(
            config.hidden_size, config.hidden_size,
        )
        # Optional style / size learnt embeddings.
        if config.use_style_embed:
            self.style_embed: nn.Embedding = nn.Embedding(1, config.hidden_size)
            self.size_embed: nn.Embedding = nn.Embedding(1, config.hidden_size)
        else:  # pragma: no cover - tiny config keeps them on
            self.style_embed = nn.Embedding(1, config.hidden_size)
            self.size_embed = nn.Embedding(1, config.hidden_size)
        # Rotary embedding inverse frequencies (local name:
        # ``rope_freqs`` -- registered as a top-level buffer).
        half = self.head_dim()
        rope_freqs = self._build_rope(max_seq=config.input_size, head_dim=half)
        self.register_buffer("rope_freqs", rope_freqs, persistent=True)
        # Position embedding (1D sequence, learnable).
        num_patches = (config.input_size // config.patch_size) ** 2
        self.pos_embed: nn.Parameter = nn.Parameter(
            torch.zeros(1, num_patches, config.hidden_size),
        )
        nn.init.normal_(self.pos_embed, std=0.02)
        # Stack of DiT blocks.
        self.blocks: nn.ModuleList = nn.ModuleList([
            HunyuanDiTBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                num_kv_heads=self.num_kv_heads,
                context_dim=config.context_dim,
                mlp_ratio=config.mlp_ratio,
            )
            for _ in range(config.num_layers)
        ])
        # Final layer -- local name: ``final_layer.{norm,adaln_modulation,out_proj}``.
        self.final_layer: nn.Module = nn.Module()
        self.final_layer.norm: nn.LayerNorm = nn.LayerNorm(
            config.hidden_size, elementwise_affine=False, eps=1e-6,
        )
        # ``adaln_modulation`` is a *single* Linear (the SiLU is
        # applied in :meth:`forward` for symmetry with the per-block
        # ``HunyuanDiTBlock.adaln_modulation``).  This keeps the
        # upstream key ``final_layer.adaLN_modulation.0.weight`` a
        # clean 1-to-1 rewrite to ``final_layer.adaln_modulation.weight``.
        self.final_layer.adaln_modulation: nn.Linear = nn.Linear(
            config.hidden_size, 2 * config.hidden_size, bias=True,
        )
        nn.init.zeros_(self.final_layer.adaln_modulation.weight)
        nn.init.zeros_(self.final_layer.adaln_modulation.bias)
        self.final_layer.out_proj: nn.Linear = nn.Linear(
            config.hidden_size,
            config.patch_size * config.patch_size * config.in_channels,
        )
        nn.init.zeros_(self.final_layer.out_proj.weight)
        nn.init.zeros_(self.final_layer.out_proj.bias)

    # ------------------------------------------------------------------
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @staticmethod
    def _build_rope(max_seq: int, head_dim: int, base: float = 10000.0) -> torch.Tensor:
        half = head_dim // 2
        freqs = torch.exp(
            -math.log(base)
            * torch.arange(0, half, dtype=torch.float32)
            / float(half),
        )
        seq_idx = torch.arange(0, max_seq, dtype=torch.float32)
        return torch.outer(seq_idx, freqs)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_text: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict the noise added to ``x``.

        Args:
            x: ``[B, C, H, W]`` noisy latents.
            timesteps: ``[B]`` integer timesteps.
            encoder_hidden_states: Optional text embeddings
                ``[B, T, context_dim]``.
            pooled_text: Optional pooled CLIP-G embedding
                ``[B, hidden_size]``.

        Returns:
            The noise prediction ``[B, C, H, W]``.
        """
        b, _, h, w = x.shape
        # Timestep embed -> ``temb`` (the inner ``time_embed[0]``
        # is a TimestepEmbedder-style Linear, the outer ``[2]`` is
        # a Linear on top of SiLU).
        half_dim = self.hidden_size // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half_dim, dtype=torch.float32, device=x.device)
            / float(half_dim),
        )
        args = timesteps.float()[:, None] * freqs[None]
        sin_emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        temb = self.time_embed[0](sin_emb)
        temb = self.time_embed[1](temb)
        temb = self.time_embed[2](temb)
        # Pooled text conditioning.  The caller can supply either
        # a pre-pooled ``[B, hidden_size]`` tensor, or the full
        # ``[B, T, context_dim]`` encoder hidden states (in which
        # case we mean-pool).  If neither is supplied, fall back
        # to a zero pooled embedding.
        if pooled_text is None and encoder_hidden_states is not None:
            pooled_text = encoder_hidden_states.mean(dim=1)
        if pooled_text is None:
            pooled_text = torch.zeros(
                b, self.hidden_size, device=x.device, dtype=temb.dtype,
            )
        # ``pooled_embed.proj`` is a ``hidden_size -> hidden_size``
        # linear in the local layout.  If the caller supplied the
        # raw ``context_dim``-d pooled text we re-use it as-is and
        # let the projection absorb the dimension mismatch (the
        # upstream uses a single Linear for the 1.x release).
        if pooled_text.shape[-1] == self.hidden_size:
            pooled_text = self.pooled_embed.proj(pooled_text)
        # Patch embed.
        z = self.patch_embed.proj(x)
        b2, d, ph, pw = z.shape
        z = z.flatten(2).transpose(1, 2)  # [B, ph*pw, hidden]
        z = self.x_embedder(z) + self.pos_embed[:, : z.shape[1], :]
        # Stack the blocks.
        for block in self.blocks:
            z = block(z, temb, encoder_hidden_states)
        # Final layer.
        shift, scale = F.silu(self.final_layer.adaln_modulation(temb)).chunk(2, dim=-1)
        z = self.final_layer.norm(z) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        z = self.final_layer.out_proj(z)
        # Unpatchify.
        z = z.reshape(b2, ph, pw, self.patch_size, self.patch_size, self.in_channels)
        z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
        z = z.reshape(b2, self.in_channels, ph * self.patch_size, pw * self.patch_size)
        return z

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int, int],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        num_steps: int = 25,
        guidance_scale: float = 6.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run a flow-matching Euler sampling loop end-to-end.

        This is the v0.8.5 "e2e Latent 验证" entry point: it
        instantiates a random initial latent, runs ``num_steps``
        Euler updates, and returns the predicted clean latent.  All
        operations happen on the same device as the parameters.

        Args:
            shape: ``(B, C, H, W)`` output shape.
            encoder_hidden_states: Optional text conditioning.
            num_steps: Number of sampling steps.
            guidance_scale: CFG scale (``1.0`` disables CFG).

        Returns:
            The denoised latent ``[B, C, H, W]``.
        """
        self.eval()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        # Initial noise.
        x = torch.randn(*shape, device=device, dtype=dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(device=device, dtype=dtype)
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t_now = timesteps[i].expand(shape[0]).to(dtype)
            t_next = timesteps[i + 1].expand(shape[0]).to(dtype)
            # Map the 0..1 float timestep to a discrete ``[0, 1000]``
            # index that the upstream ``time_in`` expects.
            t_idx = (t_now * 999.0).long().clamp(0, 999)
            v_pos = self.forward(
                x, t_idx, encoder_hidden_states=encoder_hidden_states,
            )
            if guidance_scale is not None and float(guidance_scale) > 1.0:
                null_ctx = None
                if encoder_hidden_states is not None:
                    null_ctx = torch.zeros_like(encoder_hidden_states)
                v_neg = self.forward(x, t_idx, encoder_hidden_states=null_ctx)
                v = v_neg + float(guidance_scale) * (v_pos - v_neg)
            else:
                v = v_pos
            x = x + (t_next - t_now).view(-1, 1, 1, 1) * v
        return x

    # ------------------------------------------------------------------
    # v0.8.5 LoRA injection (ComfyUI / diffusers style)
    # ------------------------------------------------------------------
    def lora_apply(
        self,
        name: str = "lora",
        *,
        rank: int = 4,
        alpha: Optional[float] = None,
        target_modules: Optional[Tuple[str, ...]] = None,
    ) -> "LoRAInjector":  # noqa: F821 -- forward ref resolved at import
        """Attach a LoRA delta to this model (v0.8.5).

        Convenience wrapper around
        :class:`models.lora.LoRAInjector`.  The default
        ``target_modules`` are
        ``("blocks.*.attn.qkv", "blocks.*.attn.out_proj",
        "blocks.*.mlp.fc1", "blocks.*.mlp.fc2")`` -- the
        HunyuanDiT-LoRA convention.

        Args:
            name: Identifier for the LoRA.  Use a unique
                name per LoRA so :meth:`lora_remove` can
                find it.
            rank: Low-rank dimension.  ``rank=0`` is a
                no-op.
            alpha: Scaling factor (defaults to ``rank``).
            target_modules: Custom glob patterns.  ``None``
                picks the HunyuanDiT default.

        Returns:
            The number of ``(module, LoRA)`` pairs newly
            patched (typically ``num_layers * 4`` for the
            default HunyuanDiT target set).  The bound
            :class:`LoRAInjector` is reachable via
            ``self._lora_injector`` and supports
            :py:meth:`LoRAInjector.lora_state_dict` for
            delta serialisation.
        """
        from models.lora import LoRAInjector, LoRASpec
        if not hasattr(self, "_lora_injector") or self._lora_injector is None:
            self._lora_injector = LoRAInjector(self)
        spec = LoRASpec(
            name=name,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules or (),
        )
        self._lora_injector.add(spec)
        n = self._lora_injector.apply()
        # Stash the most-recently-applied patch count for the
        # caller's convenience.  The injector remains
        # reachable via ``self._lora_injector`` for
        # serialisation / lora_state_dict().
        self._lora_last_applied: int = n
        return n

    def lora_remove(self, name: str) -> bool:
        """Remove a LoRA previously attached with :meth:`lora_apply`."""
        injector = getattr(self, "_lora_injector", None)
        if injector is None:
            return False
        if name not in injector._specs:
            return False
        injector.remove(name)
        return True

    def lora_clear(self) -> None:
        """Remove every LoRA attached to this model."""
        injector = getattr(self, "_lora_injector", None)
        if injector is None:
            return
        injector.clear()
        self._lora_injector = None

    # ------------------------------------------------------------------
    # v0.8.5 offload helpers
    # ------------------------------------------------------------------
    def enable_cpu_offload(
        self,
        compute_device: str = "cpu",
        offload_device: str = "cpu",
        *,
        sequential: bool = False,
    ) -> int:
        """Attach CPU offload hooks to every leaf submodule.

        Args:
            compute_device: Device the leaf is moved to on
                forward entry.
            offload_device: Device the leaf is moved to on
                forward exit (per-submodule mode) or
                globally on the next forward entry
                (sequential mode).
            sequential: ``True`` to use the strict
                stream-mode offload (only one leaf in
                memory at a time); ``False`` to use the
                per-submodule mode (every leaf is
                re-materialised on entry and sent back to
                offload on exit).

        Returns:
            The number of leaf modules that were hooked.
            ``0`` when ``compute_device == offload_device``
            (the common CPU-only path).
        """
        from core.offload import (
            enable_model_cpu_offload,
            enable_sequential_cpu_offload,
        )
        if sequential:
            return enable_sequential_cpu_offload(
                self,
                compute_device=compute_device,
                offload_device=offload_device,
            )
        return enable_model_cpu_offload(
            self,
            compute_device=compute_device,
            offload_device=offload_device,
        )
