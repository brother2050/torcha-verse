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

        Vectorised implementation: ``torch.repeat_interleave`` expands
        each token by its duration in a single GPU call, avoiding the
        previous Python double-loop and ``.item()`` host syncs.

        Args:
            encoded: ``(batch, text_len, hidden)``.
            durations: ``(batch, text_len)`` integer durations.
            max_mel_len: Optional maximum output length.

        Returns:
            ``(aligned, mask)`` where ``aligned`` is
            ``(batch, mel_len, hidden)`` and ``mask`` is a padding mask
            (``True`` = pad).
        """
        # repeat_interleave on durations broadcasts (batch, text_len) to
        # (batch, mel_len) by repeating each token index the right number
        # of times.  Negative durations are clipped to 0 for safety.
        #
        # We cannot use the bare ``torch.repeat_interleave(input, repeats)``
        # helper with a 2-D ``repeats`` tensor (PyTorch requires 1-D).
        # Instead, the per-sample method is vectorised via flattening:
        #   1. repeat_interleave in 1-D over a flattened token-index stream,
        #   2. rebuild the (batch, mel_len) tensor with right-padding.
        safe_durations = durations.clamp_min(0).long()
        text_len = encoded.shape[1]
        batch_size = safe_durations.shape[0]
        # Per-sample mel lengths; the maximum drives the padded length.
        per_sample_mel = safe_durations.sum(dim=1)
        max_mel_len_observed = int(per_sample_mel.max().item())  # single sync
        # Flattened token index stream: [0,1,...,T-1] repeated batch_size times.
        flat_index = (
            torch.arange(text_len, device=encoded.device)
            .repeat(batch_size)
        )
        # Element-wise repeat: ``safe_durations.view(-1)`` is
        # (batch * text_len,) with row-major order matching ``flat_index``.
        flat_repeated = torch.repeat_interleave(flat_index, safe_durations.view(-1))
        # ``flat_repeated`` is a 1-D stream of total length
        # ``sum(per_sample_mel)``.  The per-sample slices are not
        # contiguous, so we build a (batch, max_mel_len_observed) padded
        # index tensor by gathering with row offsets.
        offsets = torch.zeros(batch_size, dtype=torch.long, device=encoded.device)
        offsets[1:] = per_sample_mel.cumsum(0)[:-1]
        # Column positions within each row, clipped to the row's mel_len
        # for masking.
        col = torch.arange(max_mel_len_observed, device=encoded.device).unsqueeze(0)
        valid = col < per_sample_mel.unsqueeze(1)  # (batch, max_mel_len)
        # Index of the i-th element of each row in ``flat_repeated``:
        # ``offsets[b] + i`` for i in [0, per_sample_mel[b]).
        gather_idx = offsets.unsqueeze(1) + col  # (batch, max_mel_len_observed)
        gather_idx = gather_idx.clamp_max(flat_repeated.shape[0] - 1)
        token_index = torch.where(
            valid, flat_repeated[gather_idx], torch.zeros_like(gather_idx)
        )
        mel_len = max_mel_len_observed
        # Padding mask: True where the position is past the row's mel_len.
        mask = ~valid
        aligned = torch.gather(
            encoded, 1, token_index.unsqueeze(-1).expand(-1, -1, encoded.shape[-1])
        )
        if max_mel_len is not None and mel_len > max_mel_len:
            aligned = aligned[:, :max_mel_len]
            mask = mask[:, :max_mel_len]
            mel_len = max_mel_len
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
