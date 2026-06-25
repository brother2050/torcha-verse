"""Smoke tests for :mod:`training.sft_trainer`.

Validates the public API of :class:`SFTConfig` and :class:`SFTTrainer`
without running a full training loop (that would require GPU + a real
dataset).  The trainer's single-step internals (``_compute_loss``,
``_get_lr``) are exercised against a tiny model.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from training.sft_trainer import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# SFTConfig
# ---------------------------------------------------------------------------
class TestSFTConfig:
    def test_defaults(self) -> None:
        cfg = SFTConfig()
        assert cfg.epochs == 3
        assert cfg.batch_size == 4
        assert cfg.learning_rate > 0.0
        assert cfg.mixed_precision in ("fp16", "bf16", "fp32", "no")

    def test_from_dict_ignores_unknown_keys(self) -> None:
        cfg = SFTConfig.from_dict(
            {"epochs": 1, "batch_size": 2, "unknown": "ignored"}
        )
        assert cfg.epochs == 1
        assert cfg.batch_size == 2

    def test_epochs_clamped_to_min_one(self) -> None:
        cfg = SFTConfig(epochs=0)
        assert cfg.epochs == 1


# ---------------------------------------------------------------------------
# SFTTrainer
# ---------------------------------------------------------------------------
def _tiny_lm() -> nn.Module:
    """A tiny ``(B, T) -> (B, T, V)`` causal-LM-style module.

    The SFT trainer's ``_compute_loss`` expects ``logits`` of shape
    ``(batch, seq_len, vocab_size)`` and applies a one-step shift
    internally, so the linear layer must accept the right shape.
    Using ``nn.Embedding`` keeps the forward signature integer-safe.
    """

    class _LM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(16, 8)  # vocab 16 -> 8-dim
            self.head = nn.Linear(8, 16)  # 8 -> vocab 16

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = self.embed(input_ids)  # (B, T, 8)
            return self.head(x)  # (B, T, 16)

    return _LM()


def _tiny_dataset(num_samples: int = 4) -> TensorDataset:
    inputs = torch.randint(0, 16, (num_samples, 4))  # token ids
    labels = torch.randint(0, 16, (num_samples, 4))
    return TensorDataset(inputs, labels)


class TestSFTTrainer:
    def test_constructs_with_defaults(self, tmp_path) -> None:
        cfg = SFTConfig(epochs=1, batch_size=2, output_dir=str(tmp_path))
        trainer = SFTTrainer(
            model=_tiny_lm(),
            train_dataset=_tiny_dataset(),
            config=cfg,
        )
        assert trainer.device is not None
        assert trainer.config.epochs == 1

    def test_compute_loss_returns_scalar(self, tmp_path) -> None:
        cfg = SFTConfig(epochs=1, batch_size=2, output_dir=str(tmp_path))
        trainer = SFTTrainer(
            model=_tiny_lm(),
            train_dataset=_tiny_dataset(),
            config=cfg,
        )
        model = trainer.model
        # ``_compute_loss`` takes (logits, labels) directly, not a
        # batch dict.  Build a vocab-sized label tensor to match.
        # The model's forward casts int inputs to float internally,
        # so we can use ``torch.randint`` for both.
        input_ids = torch.randint(0, 16, (1, 4))
        labels = torch.randint(0, 16, (1, 4))
        logits = model(input_ids)
        loss = trainer._compute_loss(logits, labels)
        assert loss.ndim == 0  # scalar

    def test_get_lr_from_config(self, tmp_path) -> None:
        cfg = SFTConfig(learning_rate=1e-3, output_dir=str(tmp_path))
        trainer = SFTTrainer(
            model=_tiny_lm(),
            train_dataset=_tiny_dataset(),
            config=cfg,
        )
        # The SFT trainer builds an AdamW optimizer and a LambdaLR
        # scheduler.  The scheduler's initial LR is ``peak * lr_lambda(0)``
        # (which is 0 during warmup), so ``optimizer.param_groups[0]['lr']``
        # is 0 at this point.  The configured peak is held on
        # ``trainer.config.learning_rate`` and is the value tests
        # should assert against.
        assert trainer.config.learning_rate == pytest.approx(1e-3)
        # Sanity-check that the optimizer's ``initial_lr`` field (used
        # by PyTorch's LR schedulers as the peak) was set correctly.
        assert trainer.optimizer.param_groups[0]["initial_lr"] == pytest.approx(1e-3)
