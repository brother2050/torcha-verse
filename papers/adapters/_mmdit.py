"""Shared MM-DiT denoiser for the v0.5.x paper adapters.

Both Stable Diffusion 3 and HunyuanDiT use a *multimodal*
diffusion-transformer denoiser ("MM-DiT" block) that jointly attends
to the text and image tokens.  This module ships a tiny but
architecturally faithful clone of that block, so the two paper
adapters in :mod:`papers.adapters` can produce real (deterministic)
images end-to-end without taking on a multi-gigabyte third-party
weight dependency.

The module exposes two top-level symbols:

* :class:`MMDiTDenoiser` -- a stack of MM-DiT blocks with
  adaLN-zero conditioning, qk-norm and RoPE positional encoding.
* :func:`rectified_flow_sample` -- the rectified-flow sampling
  loop used by both SD3 and HunyuanDiT (the two papers share the
  same training objective, only the text encoder and rotary scheme
  differ).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_2tuple(x: Any) -> Tuple[Any, Any]:
    if isinstance(x, (tuple, list)):
        return (x[0], x[1] if len(x) > 1 else x[0])
    return (x, x)


def _rope_theta(max_seq: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    """Compute the inverse-frequency table for rotary embeddings.

    Args:
        max_seq: Maximum sequence length.
        dim: Head dimension.
        base: Exponential base for the RoPE frequencies.

    Returns:
        A ``[max_seq, dim/2]`` tensor of inverse frequencies.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(base)
        * torch.arange(0, half, dtype=torch.float32)
        / float(half)
    )
    seq_idx = torch.arange(0, max_seq, dtype=torch.float32)
    return torch.outer(seq_idx, freqs)


def _apply_rope(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings to ``x``.

    Args:
        x: A ``[B, H, N, D]`` tensor of token embeddings.
        freqs_cis: A ``[N, D/2]`` tensor of inverse frequencies.

    Returns:
        The rotated tensor with the same shape as ``x``.
    """
    # x: [B, H, N, D]
    n = x.shape[-2]
    f = freqs_cis[:n].to(x.device, x.dtype)
    cos = f.cos()[None, None, :, :]
    sin = f.sin()[None, None, :, :]
    x1, x2 = x.chunk(2, dim=-1)
    out_a = x1 * cos - x2 * sin
    out_b = x1 * sin + x2 * cos
    return torch.cat([out_a, out_b], dim=-1)


# ---------------------------------------------------------------------------
# AdaLN-Zero + QK-Norm
# ---------------------------------------------------------------------------
class AdaLNZero(nn.Module):
    """Adaptive layer-norm zero (adaLN-zero) modulation.

    Given a per-token conditioning vector ``c`` of shape ``[B, D]``,
    returns ``(shift, scale, gate)`` -- each of shape ``[B, D]``.
    When ``gate`` is zero, the residual branch is fully suppressed
    (the "zero" half of adaLN-zero).  The output affine parameters
    are produced by a two-layer MLP with SiLU non-linearity.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, 3 * self.dim, bias=True),
        )
        # Initialise the gate to zero so the residual branch starts
        # out empty, as recommended in the original adaLN-zero
        # paper (Peebles & Xie, 2022).
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.proj(c)
        return out.chunk(3, dim=-1)


class QKNorm(nn.Module):
    """Per-head L2 normalisation of the Q / K projections.

    The MM-DiT paper applies :math:`q = q / \\|q\\|` and
    :math:`k = k / \\|k\\|` after the QKV linear, before the
    scaled-dot-product attention.  This stabilises training at
    high resolutions and is now the de-facto standard for DiT
    models.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, N, D]
        return F.normalize(x, dim=-1, eps=1e-6)


# ---------------------------------------------------------------------------
# MM-DiT block
# ---------------------------------------------------------------------------
class MMDiTBlock(nn.Module):
    """A single MM-DiT block.

    Two parallel residual branches (text + image) each go through
    LayerNorm -> adaLN-zero modulation -> self-attention with QK-norm
    -> cross-attention with the other branch -> MLP.  The text and
    image branches are joined via a *joint* attention block (the
    "multimodal" half of MM-DiT).

    Args:
        dim: Token embedding dimension (shared by both branches).
        heads: Number of attention heads.
        mlp_ratio: Hidden-dim ratio for the MLP branch.
        max_seq: Maximum token sequence length (used to pre-compute
            the RoPE inverse-frequency table).
        text_seq: Default text-token sequence length.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        max_seq: int = 256,
        text_seq: int = 64,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        assert self.head_dim * self.heads == self.dim
        self.text_seq = int(text_seq)
        self.max_seq = int(max_seq)

        # Joint attention: text + image tokens are concatenated and
        # attend to each other in a single softmax.
        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.qk_norm = QKNorm()
        self.proj = nn.Linear(self.dim, self.dim, bias=False)
        # adaLN-zero (per-modality) for the joint attention output.
        self.adaln_attn = AdaLNZero(self.dim)
        # adaLN-zero (per-modality) for the MLP branch.
        self.adaln_mlp = AdaLNZero(self.dim)
        hidden = int(self.dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.dim),
        )
        # RoPE frequencies -- cached at construction time.
        self.register_buffer(
            "rope_freqs",
            _rope_theta(max_seq, self.head_dim),
            persistent=False,
        )

    def _attn(
        self, tokens: torch.Tensor
    ) -> torch.Tensor:
        # tokens: [B, N, D]
        b, n, _ = tokens.shape
        qkv = self.qkv(tokens)
        qkv = qkv.reshape(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: [B, H, N, D]
        q = self.qk_norm(q)
        k = self.qk_norm(k)
        q = _apply_rope(q, self.rope_freqs)
        k = _apply_rope(k, self.rope_freqs)
        out = F.scaled_dot_product_attention(q, k, v)  # [B, H, N, D]
        out = out.transpose(1, 2).reshape(b, n, self.dim)
        return self.proj(out)

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one MM-DiT block on a (text, image) token pair.

        Args:
            text: ``[B, T, D]`` text tokens.
            image: ``[B, N, D]`` image (latent) tokens.
            cond: ``[B, D]`` adaLN-zero conditioning vector.

        Returns:
            A tuple ``(text_out, image_out)`` of the same shapes.
        """
        # Concatenate text + image for the joint attention.
        joint = torch.cat([text, image], dim=1)
        attn_out = self._attn(joint)
        shift_a, scale_a, gate_a = self.adaln_attn(cond)
        # Split the joint attention output back into text / image.
        t_n = text.shape[1]
        attn_text = attn_out[:, :t_n, :]
        attn_image = attn_out[:, t_n:, :]
        # adaLN-zero: out = x + gate * (shift + scale * x).
        text = text + gate_a[:, None, :] * (
            attn_text * (1.0 + scale_a[:, None, :]) + shift_a[:, None, :]
        )
        image = image + gate_a[:, None, :] * (
            attn_image * (1.0 + scale_a[:, None, :]) + shift_a[:, None, :]
        )
        # MLP branch with its own adaLN-zero.
        shift_m, scale_m, gate_m = self.adaln_mlp(cond)
        text = text + gate_m[:, None, :] * (
            self.mlp(text) * (1.0 + scale_m[:, None, :]) + shift_m[:, None, :]
        )
        image = image + gate_m[:, None, :] * (
            self.mlp(image) * (1.0 + scale_m[:, None, :]) + shift_m[:, None, :]
        )
        return text, image


# ---------------------------------------------------------------------------
# Denoiser stack
# ---------------------------------------------------------------------------
@dataclass
class MMDiTConfig:
    """Configuration for :class:`MMDiTDenoiser`.

    Attributes:
        latent_channels: Number of latent channels (SD3 / Hunyuan
            both use 16 by default; we keep 4 by default for the
            tiny default preset).
        latent_size: Spatial size of the latent grid.
        text_seq: Default text-token sequence length.
        dim: Token embedding dimension.
        depth: Number of MM-DiT blocks.
        heads: Number of attention heads.
        mlp_ratio: Hidden-dim ratio for the MLP branch.
    """

    latent_channels: int = 4
    latent_size: int = 8
    text_seq: int = 64
    dim: int = 192
    depth: int = 4
    heads: int = 4
    mlp_ratio: float = 4.0

    @classmethod
    def tiny(cls) -> "MMDiTConfig":
        """Return a tiny preset used by the paper-adapter smoke tests."""
        # ``text_seq`` matches the text encoder's ``max_len`` in the
        # paper adapters (64 for SD3, 64 for HunyuanDiT).  Keeping
        # the two numbers aligned is required for the RoPE inverse
        # frequency cache to cover the entire joint attention
        # sequence.
        return cls(
            latent_channels=4,
            latent_size=8,
            text_seq=64,
            dim=96,
            depth=2,
            heads=2,
            mlp_ratio=2.0,
        )


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep -> MLP -> embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / float(half)
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class MMDiTDenoiser(nn.Module):
    """A minimal-but-faithful MM-DiT denoiser.

    The architecture is intentionally small: the project-internal
    default preset is :meth:`MMDiTConfig.tiny`, which is 4-block /
    96-dim / 2-head.  This is enough to faithfully reproduce the
    MM-DiT block design (joint attention, qk-norm, RoPE,
    adaLN-zero) end-to-end, while keeping the weights small
    enough to fit in a smoke test.

    Args:
        config: A :class:`MMDiTConfig` instance.
    """

    def __init__(self, config: MMDiTConfig) -> None:
        super().__init__()
        self.config = config
        # Per-token projections.
        self.text_proj = nn.Linear(config.dim, config.dim)
        self.image_proj = nn.Conv2d(
            in_channels=config.latent_channels,
            out_channels=config.dim,
            kernel_size=1,
        )
        self.unproj = nn.Conv2d(
            in_channels=config.dim,
            out_channels=config.latent_channels,
            kernel_size=1,
        )
        # Timestep + conditioning.
        self.t_emb = TimestepEmbedder(config.dim)
        # Stack of MM-DiT blocks.
        max_seq = config.text_seq + config.latent_size * config.latent_size
        self.blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    dim=config.dim,
                    heads=config.heads,
                    mlp_ratio=config.mlp_ratio,
                    max_seq=max_seq,
                    text_seq=config.text_seq,
                )
                for _ in range(config.depth)
            ]
        )
        # Final adaLN-zero: a single shared "out" conditioning.
        self.final_adaln = AdaLNZero(config.dim)
        self.final_norm = nn.LayerNorm(config.dim, elementwise_affine=False)
        # Init the gate to zero so the residual starts empty.
        nn.init.zeros_(self.final_adaln.proj[-1].weight)
        nn.init.zeros_(self.final_adaln.proj[-1].bias)

    # ------------------------------------------------------------------
    def _tokenise_latents(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, H*W, D]
        h = self.image_proj(x)  # [B, D, H, W]
        b, d, hh, ww = h.shape
        return h.flatten(2).transpose(1, 2)

    def _detokenise_latents(
        self, tokens: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        # tokens: [B, H*W, D] -> [B, C, H, W]
        b, n, d = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, d, h, w)
        return self.unproj(x)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the velocity field at time ``t``.

        Args:
            x: ``[B, C, H, W]`` noisy latents.
            t: ``[B]`` normalised timestep (``0`` = clean,
                ``1`` = fully noised).
            text_tokens: ``[B, T, D]`` pre-projected text tokens.

        Returns:
            The velocity prediction, same shape as ``x``.
        """
        b, c, hh, ww = x.shape
        if t.dim() == 0:
            t = t.expand(b)
        # Project image latents into the token space.
        img_tokens = self._tokenise_latents(x)
        # Project text tokens.
        text = self.text_proj(text_tokens)
        # Timestep conditioning.
        cond = self.t_emb(t)
        for block in self.blocks:
            text, img_tokens = block(text, img_tokens, cond)
        # Final adaLN-zero.
        shift, scale, gate = self.final_adaln(cond)
        out = img_tokens * (1.0 + scale[:, None, :]) + shift[:, None, :]
        out = out * gate[:, None, :]
        return self._detokenise_latents(out, hh, ww)


# ---------------------------------------------------------------------------
# Rectified-flow sampler
# ---------------------------------------------------------------------------
@torch.no_grad()
def rectified_flow_sample(
    model: MMDiTDenoiser,
    shape: Tuple[int, int, int, int],
    text_tokens: torch.Tensor,
    *,
    num_steps: int = 25,
    cfg_scale: float = 7.0,
    null_tokens: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample from a rectified-flow model with classifier-free guidance.

    Args:
        model: An :class:`MMDiTDenoiser` instance.
        shape: ``(B, C, H, W)`` output latent shape.
        text_tokens: ``[B, T, D]`` positive-prompt tokens.
        num_steps: Number of sampling steps.
        cfg_scale: Classifier-free guidance scale (>= 1).
        null_tokens: ``[B, T, D]`` negative / null-prompt tokens.
            When ``None`` a zero tensor with the same shape as
            ``text_tokens`` is used.
        device: Sampling device (defaults to ``text_tokens.device``).
        dtype: Sampling dtype (defaults to ``text_tokens.dtype``).
        seed: Optional RNG seed for reproducibility.

    Returns:
        The sampled latents ``[B, C, H, W]``.
    """
    if device is None:
        device = text_tokens.device
    if dtype is None:
        dtype = text_tokens.dtype
    if null_tokens is None:
        null_tokens = torch.zeros_like(text_tokens)
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        x = torch.randn(*shape, generator=gen, dtype=torch.float32)
    else:
        x = torch.randn(*shape, dtype=torch.float32)
    x = x.to(device=device, dtype=dtype)

    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        t_now = timesteps[i].expand(shape[0])
        t_next = timesteps[i + 1].expand(shape[0])
        v_pos = model(x, t_now, text_tokens)
        if cfg_scale is not None and cfg_scale > 1.0:
            v_neg = model(x, t_now, null_tokens)
            v = v_neg + cfg_scale * (v_pos - v_neg)
        else:
            v = v_pos
        x = x + (t_next - t_now)[:, None, None, None] * v
    return x


# ---------------------------------------------------------------------------
# Latent -> image decoder
# ---------------------------------------------------------------------------
class LatentDecoder(nn.Module):
    """Tiny latent -> image decoder.

    This is **not** a VAE -- it is a single transposed-conv
    stack that lifts the small latent grid to a ``3 x H*8 x W*8``
    image.  The output is the deterministic tanh of the
    projection, which is good enough for smoke tests.  The
    production VAE (e.g. the SD3 / Hunyuan autoencoders) is a
    v0.6.x follow-up.
    """

    def __init__(self, in_channels: int, scale_factor: int = 8) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.scale_factor = int(scale_factor)
        layers: List[nn.Module] = []
        c_in = self.in_channels
        c_out = max(8, c_in // 2)
        # 3 upsamples: in 8x8 -> 64x64, in 16x16 -> 128x128, ...
        for _ in range(int(math.log2(self.scale_factor))):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        c_in, c_out, kernel_size=4, stride=2, padding=1
                    ),
                    nn.GroupNorm(min(8, c_out), c_out),
                    nn.SiLU(),
                ]
            )
            c_in = c_out
            c_out = max(8, c_out // 2)
        layers.append(nn.Conv2d(c_in, 3, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return torch.tanh(out)
