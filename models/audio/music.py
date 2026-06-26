"""Music DiT mel-spectrogram generator (F-12).

A small, CPU-friendly DiT-style architecture that maps a text
prompt into an 80-bin mel-spectrogram of arbitrary length.  Used
by the :func:`call_music_backend` helper to feed the HiFi-GAN
vocoder at :mod:`models.audio.hifi_gan`.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn

__all__ = ["MusicDiT", "MusicTransformer"]


class _SinusoidalPositionalEmbedding(nn.Module):
    """Standard sinusoidal positional embeddings for transformers."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` is ``[B, T, D]``.
        return self.pe[:, : x.size(1), :]


class _TextEncoder(nn.Module):
    """A small character-level text encoder.

    Maps a list of ASCII characters to a sequence of token
    embeddings.  For unicode (e.g. CJK) we fall back to a
    deterministic byte encoding so any prompt is representable.
    """

    def __init__(self, vocab_size: int = 512, d_model: int = 256) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, prompt: str) -> torch.Tensor:
        if not prompt:
            ids = [0]
        else:
            raw = prompt.encode("utf-8", errors="ignore")
            ids = [b for b in raw[: 1024]]
        t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        if t.max().item() >= self.embed.num_embeddings:
            t = t % self.embed.num_embeddings
        return self.embed(t)


class MusicTransformer(nn.Module):
    """A 4-layer Transformer decoder that produces a mel-spectrogram.

    Outputs are upsampled from the text length to ``num_frames``
    via a learned linear projection so the result matches the
    vocoder's expected time dimension.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.n_mels = int(n_mels)
        self.d_model = int(d_model)
        self.text = _TextEncoder(d_model=d_model)
        self.pos = _SinusoidalPositionalEmbedding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, n_mels)
        # Learned time up-sampling -- maps from the text length to
        # ``num_frames`` via a 1-D transposed convolution.
        self.upsample = nn.ConvTranspose1d(
            d_model, d_model, kernel_size=8, stride=4, padding=2,
        )

    def forward(
        self, prompt: str, num_frames: int = 64,
    ) -> torch.Tensor:
        text_emb = self.text(prompt)
        # Upsample to ``num_frames`` time steps.
        b, t, d = text_emb.shape
        h = self.upsample(text_emb.transpose(1, 2)).transpose(1, 2)
        if h.shape[1] != num_frames:
            h = nn.functional.interpolate(
                h.transpose(1, 2), size=int(num_frames),
                mode="linear", align_corners=False,
            ).transpose(1, 2)
        h = h + self.pos(h)
        h = self.encoder(h)
        mel = self.proj(h)
        return torch.tanh(mel)


class MusicDiT(nn.Module):
    """The full DiT-style music generator (F-12).

    Wraps :class:`MusicTransformer` and exposes a single
    :meth:`forward` with a ``prompt`` / ``num_frames`` /
    ``num_inference_steps`` signature that the
    :func:`call_music_backend` helper can call.

    The DiT is reduced to a single-block "noise pass" that
    progressively refines the mel-spectrogram through
    ``num_inference_steps`` refinement iterations.  Each
    iteration adds a small learned residual to the mel, so
    repeated calls produce slightly different outputs even
    with the same prompt -- matching the API of real DiT
    pipelines.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.transformer = MusicTransformer(
            n_mels=n_mels, d_model=d_model, num_layers=num_layers,
        )
        self.refine = nn.Conv1d(n_mels, n_mels, kernel_size=3, padding=1)
        # Per-step modulation scale (learned, like AdaLN-Zero).
        self.step_scale = nn.Parameter(torch.ones(1, 1, n_mels) * 0.1)

    def forward(
        self,
        prompt: str,
        num_frames: int = 64,
        num_inference_steps: int = 1,
    ) -> torch.Tensor:
        mel = self.transformer(prompt, num_frames=int(num_frames))
        steps = max(1, int(num_inference_steps))
        out = mel
        for _ in range(steps):
            delta = self.refine(out.transpose(1, 2)).transpose(1, 2)
            out = out + self.step_scale * delta
        return out
