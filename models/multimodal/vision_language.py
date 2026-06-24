"""Vision-Language Model (VLM).

This module implements a vision-language understanding model that
combines a vision encoder (a simplified ViT) with a decoder-only
language model.  Visual features are projected into the language
embedding space and prepended to the text token embeddings, allowing the
language model to condition its generation on the image content.

Key components:

* :class:`VisionEncoder` -- a simplified Vision Transformer (ViT).
* :class:`Projector` -- projects vision features into the language
  embedding space.
* :class:`VisionLanguageModel` -- the full vision-language model.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from models.text.transformer import TransformerDecoder, _top_p_filter

__all__ = ["VisionEncoder", "Projector", "VisionLanguageModel"]


class VisionEncoder(nn.Module):
    """A simplified Vision Transformer (ViT) encoder.

    Splits an image into patches, adds a ``[CLS]`` token and sinusoidal
    positional embeddings, and processes the sequence with a stack of
    Transformer encoder layers.

    Args:
        image_size: Expected (square) image size.
        patch_size: Patch size.
        in_channels: Number of image channels.
        hidden_size: Model dimension.
        num_layers: Number of encoder layers.
        num_heads: Number of attention heads.
        mlp_ratio: MLP intermediate-size ratio.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})."
            )
        self.image_size: int = image_size
        self.patch_size: int = patch_size
        self.in_channels: int = in_channels
        self.hidden_size: int = hidden_size
        self.num_patches: int = (image_size // patch_size) ** 2

        self.patch_embed: nn.Conv2d = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed: nn.Parameter = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, hidden_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode an image into patch tokens.

        Args:
            image: Image tensor of shape ``(batch, in_channels, H, W)``.

        Returns:
            A tuple ``(sequence, pooled)`` where ``sequence`` has shape
            ``(batch, num_patches + 1, hidden_size)`` (including the
            ``[CLS]`` token) and ``pooled`` is the ``[CLS]`` representation.
        """
        batch = image.shape[0]
        x = self.patch_embed(image)  # (batch, hidden, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, hidden)

        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (batch, num_patches+1, hidden)
        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.norm(x)
        pooled = x[:, 0]  # [CLS] token
        return x, pooled


class Projector(nn.Module):
    """Projects vision features into the language embedding space.

    A two-layer MLP with GELU activation.

    Args:
        vision_hidden_size: Vision encoder output dimension.
        language_hidden_size: Language model dimension.
        mlp_hidden: Intermediate dimension (defaults to
            ``language_hidden_size``).
    """

    def __init__(
        self,
        vision_hidden_size: int,
        language_hidden_size: int,
        mlp_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = language_hidden_size
        self.fc1: nn.Linear = nn.Linear(vision_hidden_size, mlp_hidden)
        self.fc2: nn.Linear = nn.Linear(mlp_hidden, language_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project vision features.

        Args:
            x: Vision features of shape ``(batch, seq_len, vision_hidden_size)``.

        Returns:
            Projected features of shape
            ``(batch, seq_len, language_hidden_size)``.
        """
        return self.fc2(F.gelu(self.fc1(x)))


class VisionLanguageModel(BaseModel):
    """Vision-Language understanding model.

    Encodes an image with a ViT, projects the features into the
    language embedding space, and prepends them to the text token
    embeddings before running the shared decoder-only language model.

    Args:
        vision_config: Configuration dictionary for the vision encoder
            with keys ``image_size``, ``patch_size``, ``in_channels``,
            ``hidden_size``, ``num_layers``, ``num_heads``.
        language_config: Configuration dictionary for the language model
            (passed to :class:`TransformerDecoder`).
        config: Optional top-level configuration dictionary.
    """

    def __init__(
        self,
        vision_config: Optional[Dict[str, Any]] = None,
        language_config: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            vision_config = config.get("vision_config", vision_config)
            language_config = config.get("language_config", language_config)

        super().__init__(config=config)

        if vision_config is None:
            vision_config = {}
        if language_config is None:
            language_config = {}

        self.vision_encoder: VisionEncoder = VisionEncoder(
            image_size=vision_config.get("image_size", 224),
            patch_size=vision_config.get("patch_size", 16),
            in_channels=vision_config.get("in_channels", 3),
            hidden_size=vision_config.get("hidden_size", 768),
            num_layers=vision_config.get("num_layers", 12),
            num_heads=vision_config.get("num_heads", 12),
        )
        self.projector: Projector = Projector(
            vision_hidden_size=self.vision_encoder.hidden_size,
            language_hidden_size=language_config.get("hidden_size", 4096),
        )
        self.language_model: TransformerDecoder = TransformerDecoder(**language_config)

    # ------------------------------------------------------------------
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the vision-language forward pass.

        Args:
            image: Optional image of shape ``(batch, in_channels, H, W)``.
            text_ids: Optional text token ids ``(batch, seq_len)``.
            attention_mask: Optional text padding mask.
            inputs_embeds: Optional precomputed embeddings.  When
                provided, ``image`` and ``text_ids`` are ignored.

        Returns:
            Logits of shape ``(batch, num_vision_tokens + seq_len, vocab_size)``.
        """
        if inputs_embeds is not None:
            return self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        if text_ids is None:
            raise ValueError("Either text_ids or inputs_embeds must be provided.")

        text_embeds = self.language_model.embed_tokens(text_ids)

        if image is not None:
            vision_features, _ = self.vision_encoder(image)
            vision_embeds = self.projector(vision_features)
            inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

            # Extend the attention mask to cover the vision tokens (all valid).
            if attention_mask is not None:
                batch = text_ids.shape[0]
                num_vision = vision_embeds.shape[1]
                vision_mask = torch.ones(batch, num_vision, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        else:
            inputs_embeds = text_embeds

        return self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        image: Optional[torch.Tensor] = None,
        text_ids: torch.Tensor = None,  # type: ignore[assignment]
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate text conditioned on an image.

        Args:
            image: Optional conditioning image.
            text_ids: Prompt token ids ``(batch, seq_len)``.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling threshold.
            eos_token_id: Optional EOS token id.

        Returns:
            Generated token ids.
        """
        self.eval()
        # Precompute vision embeddings once.
        if image is not None:
            vision_features, _ = self.vision_encoder(image)
            vision_embeds = self.projector(vision_features)
        else:
            vision_embeds = None

        generated = text_ids
        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.language_model.num_layers
        use_cache = True

        for step in range(max_tokens):
            if step == 0:
                text_embeds = self.language_model.embed_tokens(generated)
                if vision_embeds is not None:
                    inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
                else:
                    inputs_embeds = text_embeds
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
