"""Training utilities for the TorchaVerse framework.

This package provides dataset abstractions, supervised and RLHF trainers,
and synthetic data generation:

* :mod:`dataset` -- :class:`BaseDataset`, :class:`TextDataset`,
  :class:`ChatDataset`, :class:`ImageTextDataset`, :class:`StreamingDataset`.
* :mod:`sft_trainer` -- :class:`SFTTrainer` with LoRA/QLoRA, mixed
  precision, gradient accumulation, and checkpointing.
* :mod:`rlhf_trainer` -- :class:`RLHFTrainer` supporting PPO, DPO, and GRPO.
* :mod:`synthetic_data` -- :class:`SyntheticDataGenerator` for
  programmatic data creation.
"""

from __future__ import annotations

from .dataset import (
    BaseDataset,
    ChatDataset,
    ImageTextDataset,
    StreamingDataset,
    TextDataset,
    collate_fn,
)
from .rlhf_trainer import RLHFConfig, RLHFTrainer, ValueHead
from .sft_trainer import SFTConfig, SFTTrainer
from .synthetic_data import SyntheticDataConfig, SyntheticDataGenerator

__all__ = [
    # dataset
    "BaseDataset",
    "TextDataset",
    "ChatDataset",
    "ImageTextDataset",
    "StreamingDataset",
    "collate_fn",
    # sft_trainer
    "SFTTrainer",
    "SFTConfig",
    # rlhf_trainer
    "RLHFTrainer",
    "RLHFConfig",
    "ValueHead",
    # synthetic_data
    "SyntheticDataGenerator",
    "SyntheticDataConfig",
]
