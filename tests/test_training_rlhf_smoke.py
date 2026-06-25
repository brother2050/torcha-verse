"""Smoke tests for :mod:`training.rlhf_trainer`.

Validates :class:`RLHFConfig` enum-membership validation, the
``ValueHead`` shape, and the private :meth:`RLHFTrainer._sequence_log_prob`
and :meth:`_infer_hidden_size` against a tiny model.  DPO / PPO / GRPO
loops are integration tests and are not exercised here.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from training.rlhf_trainer import RLHFConfig, RLHFTrainer, ValueHead


# ---------------------------------------------------------------------------
# RLHFConfig
# ---------------------------------------------------------------------------
class TestRLHFConfig:
    def test_default_method_is_dpo(self) -> None:
        cfg = RLHFConfig()
        assert cfg.method == "dpo"
        assert cfg.beta > 0.0
        assert cfg.epochs == 1

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown RLHF method"):
            RLHFConfig(method="bogus")

    def test_method_is_case_insensitive(self) -> None:
        cfg = RLHFConfig(method="PPO")
        assert cfg.method == "ppo"


# ---------------------------------------------------------------------------
# ValueHead
# ---------------------------------------------------------------------------
class TestValueHead:
    def test_value_head_returns_per_token_scalar(self) -> None:
        vh = ValueHead(hidden_size=8)
        x = torch.randn(2, 4, 8)  # (batch, seq, hidden)
        y = vh(x)
        # ValueHead outputs shape (batch, seq, 1) per the docstring.
        assert y.shape == (2, 4, 1)


# ---------------------------------------------------------------------------
# RLHFTrainer (private helpers only)
# ---------------------------------------------------------------------------
def _tiny_lm() -> nn.Module:
    """A tiny ``(B, T) -> (B, T, V)`` causal-LM-style module.

    Uses ``nn.Embedding`` so integer token ids flow through unchanged
    (the trainer's gather / log-softmax path expects long tensors and
    would otherwise hit PyTorch's dtype-mismatch checks on a
    plain ``nn.Linear``).
    """

    class _LM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(16, 8)
            self.head = nn.Linear(8, 16)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return self.head(self.embed(input_ids))

    return _LM()


class TestRLHFTrainer:
    def test_constructs_with_config(self) -> None:
        cfg = RLHFConfig(method="dpo", beta=0.2)
        trainer = RLHFTrainer(model=_tiny_lm(), config=cfg)
        assert trainer.config.method == "dpo"
        assert trainer.config.beta == pytest.approx(0.2)

    def test_sequence_log_prob_returns_batch_tensor(self) -> None:
        cfg = RLHFConfig(method="dpo")
        trainer = RLHFTrainer(model=_tiny_lm(), config=cfg)
        model = trainer.model
        # ``_sequence_log_prob`` expects integer token ids (the gather
        # inside ``_compute_log_probs`` indexes a vocab log-softmax).
        input_ids = torch.randint(0, 16, (2, 4))
        # The mock model accepts ``attention_mask`` so the call shape
        # matches the trainer's internal invocation.
        log_prob = trainer._sequence_log_prob(model, input_ids)
        assert log_prob.shape == (2,)
        assert torch.isfinite(log_prob).all()

    def test_infer_hidden_size_returns_default(self) -> None:
        cfg = RLHFConfig(method="dpo")
        trainer = RLHFTrainer(model=_tiny_lm(), config=cfg)
        # When the model has no attribute that triggers a non-default
        # inference, ``_infer_hidden_size`` falls back to the
        # ``hidden_size`` argument on the trainer (default 768).
        size = trainer._infer_hidden_size(trainer.model)
        # We only assert the result is a positive integer; the actual
        # value depends on the model's first nn.Linear.
        assert isinstance(size, int) and size > 0
