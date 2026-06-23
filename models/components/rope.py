"""Rotary Position Embedding (RoPE).

RoPE encodes absolute positional information with rotation matrices and
naturally incorporates explicit relative position dependency into the
attention computation.  It is the de-facto positional encoding for
modern decoder-only language models.

The module precomputes the inverse-frequency ``inv_freq`` buffer and
exposes :meth:`forward` which returns the rotated tensor.  Several
length-extension scaling strategies are supported:

* ``"linear"`` -- Linear interpolation scaling.
* ``"ntk-aware"`` -- NTK-aware interpolation that rescales the
  frequencies non-uniformly.
* ``"dynamic"`` -- Dynamically rescaled RoPE (recomputed per forward).

References:
    Su et al., "RoFormer: Enhanced Transformer with Rotary Position
    Embedding" (2021).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["RotaryPositionEmbedding", "rotate_half"]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension.

    Given ``x`` of shape ``(..., d)`` (with ``d`` even), split it into
    two halves ``x1 = x[..., :d/2]`` and ``x2 = x[..., d/2:]`` and
    return ``concat(-x2, x1, dim=-1)``.

    Args:
        x: Input tensor whose last dimension is even.

    Returns:
        The rotated tensor of the same shape as ``x``.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Args:
        q: Query tensor of shape ``(batch, heads, seq_len, head_dim)``.
        k: Key tensor of the same shape (or with fewer KV heads).
        cos: Cosine table of shape ``(seq_len, head_dim)`` or broadcastable.
        sin: Sine table of the same shape as ``cos``.

    Returns:
        A tuple ``(q_rotated, k_rotated)``.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding module.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length for which to precompute the
            cos/sin tables.
        theta: Base frequency of the inverse-frequency computation.
        rope_scaling: Optional configuration dictionary for length
            extension.  Recognised keys:

            * ``"type"``: one of ``"linear"``, ``"ntk-aware"``,
              ``"dynamic"``.
            * ``"factor"``: scaling factor (e.g. ``4.0`` for 4x context).

            When ``None`` no scaling is applied.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even for RoPE, got {dim}.")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}.")

        self.dim: int = dim
        self.max_seq_len: int = max_seq_len
        self.theta: float = theta
        self.rope_scaling: Optional[Dict[str, Any]] = rope_scaling

        # Compute the (possibly scaled) inverse frequencies.
        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin tables for the static strategies.
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        if not self._is_dynamic():
            self._build_cache(max_seq_len)

    # ------------------------------------------------------------------
    def _is_dynamic(self) -> bool:
        """Return ``True`` when dynamic RoPE scaling is requested."""
        return (
            self.rope_scaling is not None
            and self.rope_scaling.get("type") == "dynamic"
        )

    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute the inverse frequencies, applying scaling if requested."""
        half = self.dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, half, dtype=torch.float32) / half))

        if self.rope_scaling is not None:
            scaling_type = self.rope_scaling.get("type")
            factor = float(self.rope_scaling.get("factor", 1.0))

            if scaling_type == "linear":
                # Linear scaling only stretches the sequence, not the
                # frequencies.  The cos/sin are interpolated at query time.
                pass
            elif scaling_type == "ntk-aware":
                # NTK-aware: rescale the base frequency.
                base = self.theta * (factor ** (self.dim / (self.dim - 2)))
                inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
            elif scaling_type == "dynamic":
                # Dynamic NTK: frequencies are recomputed per forward
                # based on the current sequence length.
                pass
            elif scaling_type is not None:
                raise ValueError(f"Unknown rope_scaling type: {scaling_type!r}.")

        return inv_freq

    def _build_cache(self, seq_len: int) -> None:
        """Precompute the cos/sin tables up to ``seq_len``."""
        device = self.inv_freq.device
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Duplicate to match the full head_dim for the rotate_half layout.
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()

    # ------------------------------------------------------------------
    def _get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the cos/sin tables for ``seq_len`` positions.

        For dynamic scaling the tables are recomputed on the fly.
        """
        if self._is_dynamic():
            factor = float(self.rope_scaling.get("factor", 1.0)) if self.rope_scaling else 1.0
            # Dynamic NTK: scale the base frequency by the current ratio.
            base = self.theta * (
                (factor * seq_len / self.max_seq_len) ** (self.dim / (self.dim - 2))
            )
            half = self.dim // 2
            inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos(), emb.sin()

        if self._cos_cached is None or self._sin_cached is None or seq_len > self._cos_cached.shape[0]:
            self._build_cache(max(seq_len, self.max_seq_len))

        cos = self._cos_cached[:seq_len]  # type: ignore[index]
        sin = self._sin_cached[:seq_len]  # type: ignore[index]

        # Linear scaling: interpolate the positions.
        if self.rope_scaling is not None and self.rope_scaling.get("type") == "linear":
            factor = float(self.rope_scaling.get("factor", 1.0))
            scaled_len = int(seq_len / factor)
            cos = cos[:scaled_len]
            sin = sin[:scaled_len]
            # Interpolate to the original length.
            cos = torch.nn.functional.interpolate(
                cos.transpose(0, 1).unsqueeze(0), size=seq_len, mode="linear", align_corners=False
            ).squeeze(0).transpose(0, 1)
            sin = torch.nn.functional.interpolate(
                sin.transpose(0, 1).unsqueeze(0), size=seq_len, mode="linear", align_corners=False
            ).squeeze(0).transpose(0, 1)

        return cos.to(device), sin.to(device)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply rotary embeddings to ``x``.

        ``x`` is expected to be a query or key tensor of shape
        ``(batch, heads, seq_len, head_dim)``.  The rotation is applied
        along the last (``head_dim``) dimension.

        Args:
            x: Input tensor of shape ``(batch, heads, seq_len, head_dim)``.
            seq_len: Sequence length to use.  When ``None`` it is inferred
                from ``x``.

        Returns:
            The rotated tensor with the same shape as ``x``.
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        cos, sin = self._get_cos_sin(seq_len, x.device)
        # Broadcast cos/sin to ``(1, 1, seq_len, head_dim)``.
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return (x * cos) + (rotate_half(x) * sin)

    def get_cos_sin(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the cos/sin tables for external use.

        Args:
            seq_len: Number of positions.
            device: Target device.

        Returns:
            A tuple ``(cos, sin)`` each of shape ``(seq_len, head_dim)``.
        """
        if device is None:
            device = self.inv_freq.device
        return self._get_cos_sin(seq_len, device)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, max_seq_len={self.max_seq_len}, "
            f"theta={self.theta}, rope_scaling={self.rope_scaling}"
        )
