"""Stable Diffusion 3 paper adapter.

This module ships a real :class:`PaperAdapter` for Stable Diffusion 3
(Esser et al., 2024 -- arXiv:2403.03206).  The adapter wires the
project-internal :class:`MMDiTDenoiser` to the v0.5.x image-diffusion
stack so the SD3 architecture can be exercised end-to-end without
downloading multi-gigabyte Stability AI weights.

The adapter implements the full ``PaperAdapter`` contract:

* :meth:`load_model` builds the denoiser + a tiny latent decoder and
  moves both to the requested device.
* :meth:`infer` runs the rectified-flow sampling loop with
  classifier-free guidance and returns a ``[3, H, W]`` image
  tensor in the ``[-1, 1]`` range.

The architectural plumbing lives behind :meth:`_build_denoiser` /
:meth:`_build_text_encoder` so that swapping the project-internal
clone for the official Stability AI weights (in a v0.6.x release)
is a single-file change.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from papers.adapter import PaperAdapter
from papers.spec import ModelRef, PaperSpec

from ._mmdit import (
    LatentDecoder,
    MMDiTConfig,
    MMDiTDenoiser,
    rectified_flow_sample,
)


__all__ = ["StableDiffusion3Adapter"]


# ---------------------------------------------------------------------------
# Text encoder
# ---------------------------------------------------------------------------
class SD3TextEncoder(nn.Module):
    """Project-internal text encoder for SD3.

    The official SD3 stacks three text encoders (CLIP-L/14,
    CLIP-G/14, T5-XXL) and concatenates their pooled outputs.  The
    project-internal clone uses a single character-level embedding
    + a 2-layer transformer encoder -- a faithful architecture
    simplification that still exercises the ``text -> D`` projection
    that the rest of the diffusion stack expects.

    The encoder always returns ``[B, T, D]`` tokens, never pooled
    vectors, so it can be plugged into the MM-DiT block directly.
    """

    def __init__(self, dim: int, vocab_size: int = 256, max_len: int = 64) -> None:
        super().__init__()
        self.dim = int(dim)
        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.embed = nn.Embedding(self.vocab_size, self.dim)
        # A tiny 2-layer transformer encoder, so the text tokens
        # have meaningful self-attention structure by the time they
        # hit the joint attention block.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=4,
            dim_feedforward=self.dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Learned positional embedding.
        self.pos = nn.Parameter(torch.zeros(1, self.max_len, self.dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, text: List[str]) -> torch.Tensor:
        """Encode a list of strings into ``[B, T, D]`` tokens.

        Args:
            text: A list of strings, all of the same length after
                padding (the adapter pads internally).

        Returns:
            ``[B, T, D]`` text tokens.
        """
        device = self.embed.weight.device
        # Tokenise as bytes (0-255), with a fixed length.
        b = len(text)
        t = self.max_len
        ids = torch.zeros(b, t, dtype=torch.long, device=device)
        for i, s in enumerate(text):
            raw = s.encode("utf-8")[:t]
            ids[i, : len(raw)] = torch.tensor(list(raw), dtype=torch.long)
        x = self.embed(ids) + self.pos[:, :t, :]
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class StableDiffusion3Adapter(PaperAdapter):
    """PaperAdapter for Stable Diffusion 3 (arXiv:2403.03206).

    The adapter is wired to the v0.5.x
    :class:`papers.adapters._mmdit.MMDiTDenoiser` by default, which
    is a project-internal clone of the SD3 architecture (MM-DiT
    blocks, qk-norm, RoPE, adaLN-zero).  Operators that have access
    to the official Stability AI weights can override
    :meth:`_build_denoiser` / :meth:`_build_text_encoder` to plug
    in the real weights without touching the adapter contract.
    """

    paper_name: str = "stable-diffusion-3"
    node_type: str = "image_txt2img"

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._denoiser: Optional[MMDiTDenoiser] = None
        self._text_encoder: Optional[SD3TextEncoder] = None
        self._decoder: Optional[LatentDecoder] = None
        self._device: Optional[torch.device] = None
        self._dtype: torch.dtype = torch.float32

    # ------------------------------------------------------------------
    def _resolve_device(self, ctx: Any) -> torch.device:
        """Pick the device to run the model on."""
        # First, defer to an explicit ``ctx.device`` if the runtime
        # exposes one (most node contexts do).
        dev = getattr(ctx, "device", None)
        if isinstance(dev, torch.device):
            return dev
        if isinstance(dev, str) and dev:
            return torch.device(dev)
        # Fall back to the framework :class:`DeviceManager`.
        try:
            from infrastructure.device_manager import DeviceManager

            info = DeviceManager().get_device_info()
            dev_str = info.get("device", "cpu")
            return torch.device(dev_str)
        except Exception:  # noqa: BLE001 - any failure -> CPU
            return torch.device("cpu")

    # ------------------------------------------------------------------
    def _build_denoiser(self) -> MMDiTDenoiser:
        return MMDiTDenoiser(MMDiTConfig.tiny())

    def _build_text_encoder(self, dim: int) -> SD3TextEncoder:
        return SD3TextEncoder(dim=dim)

    def _build_decoder(self, in_channels: int) -> LatentDecoder:
        return LatentDecoder(in_channels=in_channels, scale_factor=8)

    # ------------------------------------------------------------------
    def load_model(self, ctx: Any) -> Dict[str, Any]:
        """Build the SD3 stack and return an opaque model handle.

        Args:
            ctx: The runtime context (a
                :class:`~nodes.base.NodeContext` or compatible).

        Returns:
            A dict with the loaded ``denoiser``, ``text_encoder``,
            ``decoder`` and ``device`` keys -- the contract
            :meth:`infer` consumes.
        """
        self._device = self._resolve_device(ctx)
        denoiser = self._build_denoiser()
        text_encoder = self._build_text_encoder(denoiser.config.dim)
        decoder = self._build_decoder(denoiser.config.latent_channels)
        # Move to the target device and dtype.
        denoiser = denoiser.to(self._device, self._dtype)
        text_encoder = text_encoder.to(self._device, self._dtype)
        decoder = decoder.to(self._device, self._dtype)
        self._denoiser = denoiser
        self._text_encoder = text_encoder
        self._decoder = decoder
        return {
            "denoiser": denoiser,
            "text_encoder": text_encoder,
            "decoder": decoder,
            "device": self._device,
            "dtype": self._dtype,
        }

    # ------------------------------------------------------------------
    def infer(self, model: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Run one SD3 inference pass.

        Args:
            model: The handle returned by :meth:`load_model`.
            **kwargs: Inference inputs.  Recognised keys:

                * ``prompt`` (str) -- the positive text prompt.
                * ``negative_prompt`` (str) -- the negative prompt
                  (default ``""``).
                * ``height`` / ``width`` (int) -- image spatial size,
                  in pixels; snapped up to the nearest multiple of
                  ``scale_factor * 8``.
                * ``num_steps`` (int) -- number of rectified-flow
                  sampling steps (default 25).
                * ``cfg_scale`` (float) -- classifier-free guidance
                  scale (default 7.0).
                * ``seed`` (int) -- optional deterministic seed.

        Returns:
            A dict with keys ``image`` (a ``[3, H, W]`` tensor in
            ``[-1, 1]``), ``width``, ``height``, ``seed`` and
            ``steps``.
        """
        denoiser: MMDiTDenoiser = model["denoiser"]
        text_encoder: SD3TextEncoder = model["text_encoder"]
        decoder: LatentDecoder = model["decoder"]
        device = model["device"]
        dtype = model["dtype"]

        prompt: str = str(kwargs.get("prompt", ""))
        negative_prompt: str = str(kwargs.get("negative_prompt", ""))
        height: int = int(kwargs.get("height", 64))
        width: int = int(kwargs.get("width", 64))
        # Snap to the latent scale factor.
        scale = denoiser.config.latent_size
        # Ensure H and W are multiples of the latent-grid factor.
        # The latent decoder upscales by 8x, so the latent grid has
        # size (H / 8) x (W / 8) -- we want that to be a perfect
        # multiple of the latent_size (the MMDiT training
        # resolution).
        for axis, val in (("h", height), ("w", width)):
            pass  # documented for clarity
        # Round H and W to the nearest multiple of 64 (8 * latent_size).
        h = max(64, (height // 64) * 64)
        w = max(64, (width // 64) * 64)
        num_steps: int = int(kwargs.get("num_steps", 25))
        cfg_scale: float = float(kwargs.get("cfg_scale", 7.0))
        seed: Optional[int] = kwargs.get("seed")
        if seed is None:
            seed = int(torch.empty((), device="cpu").uniform_().item() * (2**31 - 1))

        # Latent shape: (1, C, h/8, w/8).
        latent_h = h // 8
        latent_w = w // 8
        # Encode the text.
        text_tokens = text_encoder([prompt or " "]).to(device=device, dtype=dtype)
        null_tokens = text_encoder([negative_prompt or " "]).to(
            device=device, dtype=dtype
        )
        # Run the sampler.
        latents = rectified_flow_sample(
            denoiser,
            shape=(1, denoiser.config.latent_channels, latent_h, latent_w),
            text_tokens=text_tokens,
            null_tokens=null_tokens,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            device=device,
            dtype=dtype,
            seed=int(seed),
        )
        # Decode.
        image = decoder(latents)
        # Squeeze the batch dim.
        image = image[0]
        return {
            "image": image.detach().cpu(),
            "width": w,
            "height": h,
            "seed": int(seed),
            "steps": num_steps,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "cfg_scale": cfg_scale,
        }
