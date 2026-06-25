"""HunyuanDiT paper adapter.

This module ships a real :class:`PaperAdapter` for Tencent
HunyuanDiT (Li et al., 2024 -- arXiv:2405.08748).  HunyuanDiT is a
bilingual (English / Chinese) text-to-image diffusion transformer
that mirrors the SD3 architecture with three additions:

* a multilingual text encoder (CLIP + mT5-XXL);
* rotary-2D positional encoding for arbitrary resolution;
* class-label-free guidance with token-drop for bilingual prompts.

The v0.5.x line ships a project-internal, dependency-free
re-implementation of the HunyuanDiT denoiser (the underlying
:class:`MMDiTDenoiser` is shared with the SD3 adapter) plus a
bilingual character-level text encoder that handles ASCII via the
same byte-level path and falls back to a deterministic
UTF-8-byte encoder for CJK input.

Plugging the official Tencent weights is a v0.6.x follow-up; the
architectural plumbing lives behind
:meth:`_build_denoiser` / :meth:`_build_text_encoder` so the swap
is a single-file change.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from papers.adapter import PaperAdapter

from ._mmdit import (
    LatentDecoder,
    MMDiTConfig,
    MMDiTDenoiser,
    rectified_flow_sample,
)


__all__ = ["HunyuanDiTAdapter"]


# ---------------------------------------------------------------------------
# Bilingual text encoder
# ---------------------------------------------------------------------------
class HunyuanTextEncoder(nn.Module):
    """Bilingual text encoder for HunyuanDiT.

    The official HunyuanDiT uses a CLIP-encoder for English + a
    mT5-XXL encoder for Chinese, fused by concatenation.  The
    project-internal clone uses a single 384-dim byte-level
    embedding + a 2-layer transformer encoder -- a faithful
    architecture simplification that exercises the bilingual code
    path without depending on mT5.

    Chinese characters are encoded as their UTF-8 byte sequence
    (3 bytes per CJK ideograph in the basic plane).  English
    characters are encoded as their ASCII byte value.  No external
    tokeniser is required.
    """

    def __init__(self, dim: int = 384, max_len: int = 64) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_len = int(max_len)
        self.embed = nn.Embedding(256, self.dim)
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
        # A language-id embedding: 0 = English, 1 = Chinese.
        self.lang_embed = nn.Embedding(2, self.dim)
        nn.init.trunc_normal_(self.lang_embed.weight, std=0.02)

    def _detect_language(self, text: str) -> int:
        """Return ``0`` for English / ASCII, ``1`` for CJK.

        The detection is a single-pass over the UTF-8 codepoints:
        any character in the CJK Unified Ideographs block (U+4E00
        .. U+9FFF) flags the text as Chinese.  Mixed-language
        inputs default to Chinese (the dominant CJK path is the
        harder one to get right).
        """
        for ch in text:
            cp = ord(ch)
            if 0x4E00 <= cp <= 0x9FFF:
                return 1
            if 0x3400 <= cp <= 0x4DBF:
                return 1
        return 0

    def _byte_tokenise(self, text: str) -> List[int]:
        # Pad / truncate to max_len bytes; UTF-8 bytes are 0..255.
        raw = text.encode("utf-8")[: self.max_len]
        return list(raw)

    def forward(self, text: List[str]) -> torch.Tensor:
        device = self.embed.weight.device
        b = len(text)
        t = self.max_len
        ids = torch.zeros(b, t, dtype=torch.long, device=device)
        lang_ids = torch.zeros(b, dtype=torch.long, device=device)
        for i, s in enumerate(text):
            lang_ids[i] = self._detect_language(s)
            raw = self._byte_tokenise(s or " ")
            ids[i, : len(raw)] = torch.tensor(raw, dtype=torch.long)
        x = self.embed(ids) + self.pos[:, :t, :]
        x = x + self.lang_embed(lang_ids)[:, None, :]
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class HunyuanDiTAdapter(PaperAdapter):
    """PaperAdapter for HunyuanDiT (arXiv:2405.08748)."""

    paper_name: str = "hunyuan-dit"
    node_type: str = "image_txt2img"

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._denoiser: Optional[MMDiTDenoiser] = None
        self._text_encoder: Optional[HunyuanTextEncoder] = None
        self._decoder: Optional[LatentDecoder] = None
        self._device: Optional[torch.device] = None
        self._dtype: torch.dtype = torch.float32

    # ------------------------------------------------------------------
    def _resolve_device(self, ctx: Any) -> torch.device:
        dev = getattr(ctx, "device", None)
        if isinstance(dev, torch.device):
            return dev
        if isinstance(dev, str) and dev:
            return torch.device(dev)
        try:
            from infrastructure.device_manager import DeviceManager

            info = DeviceManager().get_device_info()
            return torch.device(info.get("device", "cpu"))
        except Exception:  # noqa: BLE001
            return torch.device("cpu")

    # ------------------------------------------------------------------
    def _build_denoiser(self) -> MMDiTDenoiser:
        return MMDiTDenoiser(MMDiTConfig.tiny())

    def _build_text_encoder(self, dim: int) -> HunyuanTextEncoder:
        return HunyuanTextEncoder(dim=dim, max_len=64)

    def _build_decoder(self, in_channels: int) -> LatentDecoder:
        return LatentDecoder(in_channels=in_channels, scale_factor=8)

    # ------------------------------------------------------------------
    def load_model(self, ctx: Any) -> Dict[str, Any]:
        self._device = self._resolve_device(ctx)
        denoiser = self._build_denoiser()
        text_encoder = self._build_text_encoder(denoiser.config.dim)
        decoder = self._build_decoder(denoiser.config.latent_channels)
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
        denoiser: MMDiTDenoiser = model["denoiser"]
        text_encoder: HunyuanTextEncoder = model["text_encoder"]
        decoder: LatentDecoder = model["decoder"]
        device = model["device"]
        dtype = model["dtype"]

        prompt: str = str(kwargs.get("prompt", ""))
        negative_prompt: str = str(kwargs.get("negative_prompt", ""))
        height: int = int(kwargs.get("height", 64))
        width: int = int(kwargs.get("width", 64))
        h = max(64, (height // 64) * 64)
        w = max(64, (width // 64) * 64)
        num_steps: int = int(kwargs.get("num_steps", 25))
        cfg_scale: float = float(kwargs.get("cfg_scale", 6.0))
        seed: Optional[int] = kwargs.get("seed")
        if seed is None:
            seed = int(
                torch.empty((), device="cpu").uniform_().item() * (2**31 - 1)
            )

        latent_h = h // 8
        latent_w = w // 8
        text_tokens = text_encoder([prompt or " "]).to(device=device, dtype=dtype)
        null_tokens = text_encoder([negative_prompt or " "]).to(
            device=device, dtype=dtype
        )
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
        image = decoder(latents)
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
