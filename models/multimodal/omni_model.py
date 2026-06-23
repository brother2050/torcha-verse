"""Omni-modal model.

This module implements an omni-modal model that can process and
generate text, image, and audio jointly through a shared Transformer
backbone.  Each modality has a dedicated encoder that projects its
features into the shared embedding space; the projected embeddings are
concatenated along the sequence dimension and processed by the shared
backbone.

Key components:

* :class:`AudioEncoder` -- encodes a mel spectrogram into tokens.
* :class:`OmniModel` -- the full omni-modal model.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model_registry import BaseModel
from models.text.transformer import TransformerDecoder, _top_p_filter
from .vision_language import Projector, VisionEncoder

__all__ = ["AudioEncoder", "OmniModel"]


class AudioEncoder(nn.Module):
    """Encodes an audio mel spectrogram into a sequence of tokens.

    Uses a convolutional front-end to downsample the mel frames, then
    projects to the shared embedding dimension.

    Args:
        mel_channels: Number of mel bins.
        hidden_size: Encoder hidden size.
        output_dim: Shared embedding dimension.
        num_layers: Number of conv layers.
        stride: Convolution stride (frame downsampling).
    """

    def __init__(
        self,
        mel_channels: int = 80,
        hidden_size: int = 256,
        output_dim: int = 1024,
        num_layers: int = 3,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.mel_channels: int = mel_channels
        self.hidden_size: int = hidden_size
        self.output_dim: int = output_dim

        layers: list = []
        in_ch = mel_channels
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_ch, hidden_size, 3, stride=stride, padding=1))
            layers.append(nn.GroupNorm(1, hidden_size))
            layers.append(nn.GELU())
            in_ch = hidden_size
        self.conv: nn.Sequential = nn.Sequential(*layers)
        self.proj: nn.Linear = nn.Linear(hidden_size, output_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode a mel spectrogram into tokens.

        Args:
            mel: Mel spectrogram of shape ``(batch, mel_channels, time)``.

        Returns:
            Tokens of shape ``(batch, time / stride**num_layers, output_dim)``.
        """
        h = self.conv(mel)  # (batch, hidden, time')
        h = h.transpose(1, 2)  # (batch, time', hidden)
        return self.proj(h)


class OmniModel(BaseModel):
    """Omni-modal understanding and generation model.

    A shared decoder-only Transformer backbone processes the
    concatenated embeddings from modality-specific encoders (text,
    vision, audio).

    Args:
        text_config: Configuration for the language backbone
            (passed to :class:`TransformerDecoder`).
        vision_config: Configuration for the vision encoder.
        audio_config: Configuration for the audio encoder.
        config: Optional top-level configuration dictionary.
    """

    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        vision_config: Optional[Dict[str, Any]] = None,
        audio_config: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            text_config = config.get("text_config", text_config)
            vision_config = config.get("vision_config", vision_config)
            audio_config = config.get("audio_config", audio_config)

        super().__init__(config=config)

        if text_config is None:
            text_config = {}
        if vision_config is None:
            vision_config = {}
        if audio_config is None:
            audio_config = {}

        language_hidden = text_config.get("hidden_size", 1024)

        # Shared backbone.
        self.language_model: TransformerDecoder = TransformerDecoder(**text_config)

        # Vision encoder + projector.
        self.vision_encoder: VisionEncoder = VisionEncoder(
            image_size=vision_config.get("image_size", 224),
            patch_size=vision_config.get("patch_size", 16),
            in_channels=vision_config.get("in_channels", 3),
            hidden_size=vision_config.get("hidden_size", 768),
            num_layers=vision_config.get("num_layers", 6),
            num_heads=vision_config.get("num_heads", 12),
        )
        self.vision_projector: Projector = Projector(
            vision_hidden_size=self.vision_encoder.hidden_size,
            language_hidden_size=language_hidden,
        )

        # Audio encoder.
        self.audio_encoder: AudioEncoder = AudioEncoder(
            mel_channels=audio_config.get("mel_channels", 80),
            hidden_size=audio_config.get("hidden_size", 256),
            output_dim=language_hidden,
            num_layers=audio_config.get("num_layers", 3),
            stride=audio_config.get("stride", 2),
        )

    # ------------------------------------------------------------------
    def encode_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a multimodal input dictionary into a unified embedding.

        The embeddings are concatenated in the order: vision, audio,
        text.  An attention mask is built marking all modality tokens as
        valid (text padding is respected).

        Args:
            inputs: Dictionary with optional keys ``"image"``,
                ``"audio"``, ``"text_ids"``, and ``"attention_mask"``.

        Returns:
            ``(inputs_embeds, attention_mask)``.
        """
        embeds_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []

        # Vision tokens.
        if "image" in inputs and inputs["image"] is not None:
            vision_features, _ = self.vision_encoder(inputs["image"])
            vision_embeds = self.vision_projector(vision_features)
            embeds_list.append(vision_embeds)
            mask_list.append(
                torch.ones(vision_embeds.shape[:2], device=vision_embeds.device, dtype=torch.long)
            )

        # Audio tokens.
        if "audio" in inputs and inputs["audio"] is not None:
            audio_embeds = self.audio_encoder(inputs["audio"])
            embeds_list.append(audio_embeds)
            mask_list.append(
                torch.ones(audio_embeds.shape[:2], device=audio_embeds.device, dtype=torch.long)
            )

        # Text tokens.
        if "text_ids" in inputs and inputs["text_ids"] is not None:
            text_ids = inputs["text_ids"]
            text_embeds = self.language_model.embed_tokens(text_ids)
            embeds_list.append(text_embeds)
            if "attention_mask" in inputs and inputs["attention_mask"] is not None:
                text_mask = inputs["attention_mask"].long()
            else:
                text_mask = torch.ones(text_ids.shape, device=text_ids.device, dtype=torch.long)
            mask_list.append(text_mask)

        if not embeds_list:
            raise ValueError("inputs must contain at least one modality.")

        inputs_embeds = torch.cat(embeds_list, dim=1)
        attention_mask = torch.cat(mask_list, dim=1)
        return inputs_embeds, attention_mask

    # ------------------------------------------------------------------
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the omni-modal forward pass.

        Args:
            inputs: Dictionary with optional keys ``"image"``,
                ``"audio"``, ``"text_ids"``, and ``"attention_mask"``.

        Returns:
            Logits of shape ``(batch, total_seq_len, vocab_size)``.
        """
        inputs_embeds, attention_mask = self.encode_inputs(inputs)
        return self.language_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate text conditioned on multimodal inputs.

        Args:
            inputs: Dictionary with optional keys ``"image"``,
                ``"audio"``, and ``"text_ids"`` (the prompt).
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling threshold.
            eos_token_id: Optional EOS token id.

        Returns:
            Generated token ids.
        """
        self.eval()
        # Encode the conditioning modalities (without text).
        cond_inputs = {k: v for k, v in inputs.items() if k != "text_ids" and k != "attention_mask"}
        cond_embeds_list: List[torch.Tensor] = []
        if "image" in cond_inputs and cond_inputs["image"] is not None:
            vision_features, _ = self.vision_encoder(cond_inputs["image"])
            cond_embeds_list.append(self.vision_projector(vision_features))
        if "audio" in cond_inputs and cond_inputs["audio"] is not None:
            cond_embeds_list.append(self.audio_encoder(cond_inputs["audio"]))

        text_ids = inputs.get("text_ids")
        if text_ids is None:
            raise ValueError("inputs must contain 'text_ids' as the prompt.")

        generated = text_ids
        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.language_model.num_layers
        use_cache = True

        for step in range(max_tokens):
            if step == 0:
                text_embeds = self.language_model.embed_tokens(generated)
                embeds = cond_embeds_list + [text_embeds]
                inputs_embeds = torch.cat(embeds, dim=1)
                logits = self.language_model(inputs_embeds=inputs_embeds, kv_cache=kv_cache, use_cache=use_cache)
                next_logits = logits[:, -1, :]
            else:
                last_token = generated[:, -1:].contiguous()
                logits = self.language_model(input_ids=last_token, kv_cache=kv_cache, use_cache=use_cache)
                next_logits = logits[:, -1, :]

            if temperature > 0:
                next_logits = next_logits / temperature
                if top_k > 0:
                    top_k = min(top_k, next_logits.size(-1))
                    values, _ = torch.topk(next_logits, top_k)
                    min_values = values[:, -1, None]
                    next_logits = next_logits.masked_fill(
                        next_logits < min_values, float("-inf")
                    )
                if top_p < 1.0:
                    next_logits = _top_p_filter(next_logits, top_p)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(next_logits, dim=-1)

            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
