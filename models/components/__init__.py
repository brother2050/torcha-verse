"""Building-block components for TorchaVerse models.

This sub-package contains reusable, architecture-agnostic neural
network components such as normalisation layers, activation functions,
positional encodings, and parameter-efficient fine-tuning adapters.
"""

from __future__ import annotations

from .lora import LoRALinear, apply_lora, mark_only_lora_as_trainable, merge_lora
from .rmsnorm import RMSNorm
from .rope import RotaryPositionEmbedding, apply_rotary_pos_emb, rotate_half
from .swiglu import SwiGLU

__all__ = [
    "RMSNorm",
    "SwiGLU",
    "RotaryPositionEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    "LoRALinear",
    "apply_lora",
    "merge_lora",
    "mark_only_lora_as_trainable",
]
