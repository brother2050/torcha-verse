"""Text-to-Speech (TTS) Transformer.

This module implements an end-to-end TTS model with the classic
encoder-decoder architecture used by models such as FastSpeech:

* :class:`TextEncoder` -- encodes text token ids into hidden states.
* :class:`DurationPredictor` -- predicts the duration (in mel frames)
  of each text token, used for non-autoregressive alignment.
* :class:`AcousticDecoder` -- decodes the length-aligned hidden states
  into a mel spectrogram.
* :class:`TTSTransformer` -- the full TTS model.

During training (when ``mel_target`` and durations are provided) the
model learns to predict mel frames aligned by the ground-truth
durations.  During inference the :meth:`generate` method uses the
duration predictor to expand the text encoding to the target length.

Reference:
    Ren et al., "FastSpeech: Fast, Robust and Controllable Text to
    Speech" (2019).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel

__all__ = ["TextEncoder", "DurationPredictor", "AcousticDecoder", "TTSTransformer"]


def _get_sinusoidal_pos(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """Return sinusoidal positional embeddings ``(seq_len, dim)``."""
    pos = torch.arange(seq_len, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TextEncoder(nn.Module):
    """Text encoder for TTS.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Model dimension.
        num_layers: Number of encoder layers.
        num_heads: Number of attention heads.
        max_seq_len: Maximum text length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, hidden_size)
        self.scale: float = math.sqrt(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.max_seq_len: int = max_seq_len

    def forward(
        self,
        text_ids: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text token ids.

        Args:
            text_ids: Token ids of shape ``(batch, text_len)``.
            src_key_padding_mask: Optional padding mask (``True`` = pad).

        Returns:
            Encoded hidden states of shape ``(batch, text_len, hidden_size)``.
        """
        x = self.embedding(text_ids) * self.scale
        x = x + _get_sinusoidal_pos(x.shape[1], self.hidden_size, x.device)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x


class DurationPredictor(nn.Module):
    """Duration predictor for non-autoregressive alignment.

    Predicts the number of mel frames each text token should occupy.

    Args:
        hidden_size: Input dimension.
        num_layers: Number of conv layers.
        kernel_size: Convolution kernel size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        layers: list = []
        in_ch = hidden_size
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_ch, hidden_size, kernel_size, padding=kernel_size // 2))
            layers.append(nn.GroupNorm(1, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = hidden_size
        self.conv_layers: nn.Sequential = nn.Sequential(*layers)
        self.linear: nn.Linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict log-durations for each text token.

        Args:
            x: Encoded text of shape ``(batch, text_len, hidden_size)``.

        Returns:
            Log-durations of shape ``(batch, text_len)``.
        """
        # (batch, text_len, hidden) -> (batch, hidden, text_len)
        h = x.transpose(1, 2)
        h = self.conv_layers(h)
        # (batch, hidden, text_len) -> (batch, text_len, hidden)
        h = h.transpose(1, 2)
        return self.linear(h).squeeze(-1)


class AcousticDecoder(nn.Module):
    """Acoustic decoder that produces a mel spectrogram.

    A non-autoregressive transformer (self-attention) over the
    length-aligned text encoding, followed by a linear projection to
    mel bins.

    Args:
        hidden_size: Model dimension.
        mel_channels: Number of mel bins.
        num_layers: Number of decoder layers.
        num_heads: Number of attention heads.
        max_seq_len: Maximum mel length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        mel_channels: int = 80,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.mel_channels: int = mel_channels
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder: nn.TransformerEncoder = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers
        )
        self.mel_proj: nn.Linear = nn.Linear(hidden_size, mel_channels)
        self.max_seq_len: int = max_seq_len

    def forward(
        self,
        memory: torch.Tensor,
        mel_target: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode the aligned text encoding into a mel spectrogram.

        Args:
            memory: Encoded (and length-aligned) text ``(batch, mel_len, hidden)``.
            mel_target: Optional target mel (unused for decoding; kept
                for API compatibility).
            tgt_key_padding_mask: Optional padding mask (``True`` = pad).

        Returns:
            Mel spectrogram of shape ``(batch, mel_len, mel_channels)``.
        """
        x = memory + _get_sinusoidal_pos(memory.shape[1], self.hidden_size, memory.device)
        x = self.decoder(x, src_key_padding_mask=tgt_key_padding_mask)
        mel = self.mel_proj(x)
        return mel


class TTSTransformer(BaseModel):
    """End-to-end Text-to-Speech Transformer.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Model dimension.
        num_layers: Number of encoder/decoder layers.
        num_heads: Number of attention heads.
        mel_channels: Number of mel bins.
        max_text_len: Maximum text length.
        max_mel_len: Maximum mel length.
        config: Optional configuration dictionary.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        mel_channels: int = 80,
        max_text_len: int = 512,
        max_mel_len: int = 2048,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            vocab_size = config.get("vocab_size", vocab_size)
            hidden_size = config.get("hidden_size", hidden_size)
            num_layers = config.get("num_layers", num_layers)
            num_heads = config.get("num_heads", num_heads)
            mel_channels = config.get("mel_channels", mel_channels)
            max_text_len = config.get("max_text_len", max_text_len)
            max_mel_len = config.get("max_mel_len", max_mel_len)

        super().__init__(config=config)

        self.vocab_size: int = vocab_size
        self.hidden_size: int = hidden_size
        self.mel_channels: int = mel_channels

        self.text_encoder: TextEncoder = TextEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_text_len,
        )
        self.duration_predictor: DurationPredictor = DurationPredictor(
            hidden_size=hidden_size,
        )
        self.acoustic_decoder: AcousticDecoder = AcousticDecoder(
            hidden_size=hidden_size,
            mel_channels=mel_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_mel_len,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _length_regulator(
        encoded: torch.Tensor,
        durations: torch.Tensor,
        max_mel_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand the text encoding by the predicted durations.

        Args:
            encoded: ``(batch, text_len, hidden)``.
            durations: ``(batch, text_len)`` integer durations.
            max_mel_len: Optional maximum output length.

        Returns:
            ``(aligned, mask)`` where ``aligned`` is
            ``(batch, mel_len, hidden)`` and ``mask`` is a padding mask.
        """
        batch, text_len, hidden = encoded.shape
        aligned_list: list = []
        max_len = 0
        for b in range(batch):
            frames: list = []
            for t in range(text_len):
                d = int(durations[b, t].item())
                if d > 0:
                    frames.append(encoded[b, t].unsqueeze(0).repeat(d, 1))
            if frames:
                frame = torch.cat(frames, dim=0)
            else:
                frame = encoded[b, :1].repeat(1, 1)
            aligned_list.append(frame)
            max_len = max(max_len, frame.shape[0])

        if max_mel_len is not None:
            max_len = min(max_len, max_mel_len)

        aligned = torch.zeros(batch, max_len, hidden, device=encoded.device, dtype=encoded.dtype)
        mask = torch.ones(batch, max_len, device=encoded.device, dtype=torch.bool)  # True = pad
        for b in range(batch):
            n = min(aligned_list[b].shape[0], max_len)
            aligned[b, :n] = aligned_list[b][:n]
            mask[b, :n] = False
        return aligned, mask

    # ------------------------------------------------------------------
    def forward(
        self,
        text_ids: torch.Tensor,
        mel_target: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the TTS forward pass.

        Args:
            text_ids: Text token ids ``(batch, text_len)``.
            mel_target: Optional target mel spectrogram
                ``(batch, mel_len, mel_channels)`` for teacher forcing.
            durations: Optional ground-truth durations
                ``(batch, text_len)``.  When ``None`` the duration
                predictor is used.
            text_lengths: Optional text lengths for masking.

        Returns:
            Predicted mel spectrogram ``(batch, mel_len, mel_channels)``.
        """
        src_key_padding_mask = None
        if text_lengths is not None:
            src_key_padding_mask = torch.arange(text_ids.shape[1], device=text_ids.device)[None, :] >= text_lengths[:, None]

        encoded = self.text_encoder(text_ids, src_key_padding_mask=src_key_padding_mask)

        # Predict or use provided durations.
        if durations is None:
            log_durations = self.duration_predictor(encoded)
            durations = torch.clamp(torch.round(torch.exp(log_durations) - 1), min=1).long()

        max_mel_len = mel_target.shape[1] if mel_target is not None else None
        aligned, mel_mask = self._length_regulator(encoded, durations, max_mel_len)

        mel = self.acoustic_decoder(aligned, mel_target=mel_target, tgt_key_padding_mask=mel_mask)
        return mel

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        text_ids: torch.Tensor,
        max_mel_len: int = 1000,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate a mel spectrogram from text.

        Uses the duration predictor to align the text encoding, then
        decodes non-autoregressively.

        Args:
            text_ids: Text token ids ``(batch, text_len)``.
            max_mel_len: Maximum mel length.
            speed: Speed factor (``> 1`` = faster).

        Returns:
            Generated mel spectrogram ``(batch, mel_len, mel_channels)``.
        """
        self.eval()
        encoded = self.text_encoder(text_ids)
        log_durations = self.duration_predictor(encoded)
        durations = torch.clamp(
            torch.round(torch.exp(log_durations) - 1) / max(speed, 1e-3), min=1
        ).long()
        aligned, mel_mask = self._length_regulator(encoded, durations, max_mel_len)
        mel = self.acoustic_decoder(aligned, mel_target=None, tgt_key_padding_mask=mel_mask)
        return mel
