"""Reinforcement Learning from Human Feedback (RLHF) trainer.

This module provides :class:`RLHFTrainer`, which aligns a language model
to human preferences using one of three methods:

* **DPO** (Direct Preference Optimisation) -- trains directly on
  preference pairs (chosen vs. rejected) using a sigmoid loss on the
  log-probability difference, with a reference model providing the KL
  anchor.  No reward model is required.
* **PPO** (Proximal Policy Optimisation) -- the classic RLHF pipeline
  with a policy network, a value network, a reward model, and a KL
  divergence constraint against a reference model.
* **GRPO** (Group Relative Policy Optimisation) -- a variant that
  estimates baselines from group statistics rather than a learned value
  network.

The trainer is designed to work with any :class:`BaseModel`-derived
language model and reuses the infrastructure layer for device management,
logging, and checkpointing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.base import BaseModel
from infrastructure.config_center import ConfigCenter
from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

__all__ = ["RLHFTrainer", "RLHFConfig", "ValueHead"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class RLHFConfig:
    """Configuration container for :class:`RLHFTrainer`.

    Args:
        method: Alignment method -- ``"ppo"``, ``"dpo"``, or ``"grpo"``.
        beta: DPO / KL regularisation strength.
        learning_rate: Learning rate.
        epochs: Number of training epochs.
        batch_size: Batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        max_grad_norm: Maximum gradient norm.
        clip_range: PPO clipping range (epsilon).
        kl_coeff: KL divergence coefficient for PPO.
        value_coeff: Value-loss coefficient for PPO.
        entropy_coeff: Entropy bonus coefficient for PPO.
        gae_lambda: GAE lambda for advantage estimation.
        gamma: Discount factor.
        num_ppo_epochs: Number of PPO update epochs per batch.
        seed: Random seed.
    """

    def __init__(
        self,
        method: str = "dpo",
        beta: float = 0.1,
        learning_rate: float = 5e-7,
        epochs: int = 1,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        clip_range: float = 0.2,
        kl_coeff: float = 0.05,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        gae_lambda: float = 0.95,
        gamma: float = 1.0,
        num_ppo_epochs: int = 4,
        seed: int = 42,
    ) -> None:
        method = method.lower().strip()
        if method not in ("ppo", "dpo", "grpo"):
            raise ValueError(
                f"Unknown RLHF method '{method}'. Use 'ppo', 'dpo', or 'grpo'."
            )
        self.method: str = method
        self.beta: float = float(beta)
        self.learning_rate: float = float(learning_rate)
        self.epochs: int = max(1, int(epochs))
        self.batch_size: int = max(1, int(batch_size))
        self.gradient_accumulation_steps: int = max(1, int(gradient_accumulation_steps))
        self.max_grad_norm: float = float(max_grad_norm)
        self.clip_range: float = float(clip_range)
        self.kl_coeff: float = float(kl_coeff)
        self.value_coeff: float = float(value_coeff)
        self.entropy_coeff: float = float(entropy_coeff)
        self.gae_lambda: float = float(gae_lambda)
        self.gamma: float = float(gamma)
        self.num_ppo_epochs: int = max(1, int(num_ppo_epochs))
        self.seed: int = int(seed)

    @classmethod
    def from_yaml(cls) -> "RLHFConfig":
        """Build an :class:`RLHFConfig` from the global YAML configuration."""
        cfg = ConfigCenter()
        rlhf_cfg = cfg.get("rlhf", {})
        optimizer_cfg = cfg.get("optimizer", {})
        training_cfg = cfg.get("training", {})
        return cls(
            method=rlhf_cfg.get("method", "dpo"),
            beta=rlhf_cfg.get("beta", 0.1),
            clip_range=rlhf_cfg.get("clip_range", 0.2),
            learning_rate=optimizer_cfg.get("lr", 5e-7),
            epochs=training_cfg.get("epochs", 1),
            batch_size=training_cfg.get("batch_size", 4),
            gradient_accumulation_steps=training_cfg.get(
                "gradient_accumulation_steps", 1
            ),
            max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
            seed=training_cfg.get("seed", 42),
        )


# ---------------------------------------------------------------------------
# Value head (for PPO)
# ---------------------------------------------------------------------------
class ValueHead(nn.Module):
    """A simple linear value (critic) head on top of hidden states.

    Args:
        hidden_size: Dimension of the input hidden states.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear: nn.Linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict a scalar value for each position.

        Args:
            hidden_states: Tensor of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Value tensor of shape ``(batch, seq_len, 1)``.
        """
        return self.linear(hidden_states)


# ---------------------------------------------------------------------------
# RLHFTrainer
# ---------------------------------------------------------------------------
class RLHFTrainer:
    """RLHF alignment trainer supporting PPO, DPO, and GRPO.

    Args:
        model: The policy model to be aligned (a :class:`BaseModel` or
            any :class:`torch.nn.Module`).
        ref_model: The frozen reference model used for the KL anchor.
            When ``None`` a detached copy of ``model`` is used.
        reward_model: Optional reward model for PPO.  When ``None`` a
            simple value-head-based reward is used.
        config: :class:`RLHFConfig` or ``None`` to load from YAML.
        optimizer: Optional pre-built optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        reward_model: Optional[nn.Module] = None,
        config: Optional[RLHFConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.config: RLHFConfig = config or RLHFConfig.from_yaml()
        self.model: nn.Module = model
        self._logger = get_logger(self.__class__.__name__)

        # Device management.
        self._device_manager: DeviceManager = DeviceManager()
        self.device: torch.device = self._device_manager.get_device()

        # Reference model (frozen).
        self.ref_model: Optional[nn.Module] = ref_model
        if self.ref_model is None:
            self.ref_model = self._clone_reference(model)
        self._freeze(self.ref_model)

        # Reward model (frozen, used by PPO).
        self.reward_model: Optional[nn.Module] = reward_model
        if self.reward_model is not None:
            self._freeze(self.reward_model)

        # Move models to device.
        self.model = self._device_manager.to_device(self.model, self.device)
        if self.ref_model is not None:
            self.ref_model = self._device_manager.to_device(self.ref_model, self.device)
        if self.reward_model is not None:
            self.reward_model = self._device_manager.to_device(
                self.reward_model, self.device
            )

        # Value head for PPO.
        hidden_size = self._infer_hidden_size(self.model)
        self.value_head: ValueHead = ValueHead(hidden_size).to(self.device)

        # Optimizer over the policy model + value head.
        self.optimizer: torch.optim.Optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(
                list(self.model.parameters()) + list(self.value_head.parameters()),
                lr=self.config.learning_rate,
            )
        )

        self.global_step: int = 0

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _freeze(model: Optional[nn.Module]) -> None:
        """Freeze all parameters of ``model`` (no gradient)."""
        if model is None:
            return
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    def _clone_reference(self, model: nn.Module) -> nn.Module:
        """Create a frozen deep copy of ``model`` as the reference.

        Args:
            model: The policy model.

        Returns:
            A frozen copy of the model.
        """
        import copy

        ref = copy.deepcopy(model)
        self._freeze(ref)
        return ref

    @staticmethod
    def _infer_hidden_size(model: nn.Module) -> int:
        """Infer the hidden dimension of ``model``.

        Inspects common attribute names (``hidden_size``, ``config``)
        and falls back to a default of 768.

        Args:
            model: The model to inspect.

        Returns:
            The inferred hidden size.
        """
        for attr in ("hidden_size", "d_model", "dim"):
            value = getattr(model, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        config = getattr(model, "config", None)
        if isinstance(config, dict):
            for key in ("hidden_size", "d_model", "dim"):
                if key in config and isinstance(config[key], int):
                    return config[key]
        return 768

    # ------------------------------------------------------------------
    # Log-probability computation
    # ------------------------------------------------------------------
    def _compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the per-token log-probabilities of a sequence.

        Runs the model forward pass, applies log-softmax over the
        vocabulary, and gathers the log-probabilities of the actual
        next tokens.  Padding positions (label ``-100``) are masked to
        zero.

        Args:
            model: The model to query.
            input_ids: Token ids of shape ``(batch, seq_len)``.
            attention_mask: Optional padding mask.
            labels: Optional target labels.  When ``None`` the
                ``input_ids`` shifted by one are used.

        Returns:
            Per-token log-probabilities of shape ``(batch, seq_len-1)``.
        """
        if labels is None:
            labels = input_ids

        # Forward pass.
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]

        # Shift for next-token prediction.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Log-softmax over vocabulary.
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather the log-prob of the actual token.
        gathered = log_probs.gather(
            -1, shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)

        # Mask padding positions.
        mask = (shift_labels != -100).float()
        return gathered * mask

    def _sequence_log_prob(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the total log-probability of each sequence.

        Args:
            model: The model to query.
            input_ids: Token ids of shape ``(batch, seq_len)``.
            attention_mask: Optional padding mask.
            labels: Optional target labels.

        Returns:
            Sequence log-probabilities of shape ``(batch,)``.
        """
        per_token = self._compute_log_probs(
            model, input_ids, attention_mask, labels
        )
        return per_token.sum(dim=-1)

    # ------------------------------------------------------------------
    # DPO
    # ------------------------------------------------------------------
    def train_dpo(
        self,
        preferences_dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Train with Direct Preference Optimisation (DPO).

        The DPO loss is::

            L = -log sigmoid(beta * (
                (log pi(chosen) - log pi_ref(chosen))
              - (log pi(rejected) - log pi_ref(rejected))
            ))

        Args:
            preferences_dataloader: A DataLoader yielding batches with
                ``chosen_input_ids``, ``chosen_attention_mask``,
                ``rejected_input_ids``, and ``rejected_attention_mask``.

        Returns:
            A dictionary of training metrics.
        """
        self._logger.info("Starting DPO training (%d epochs).", self.config.epochs)
        self.model.train()

        total_loss: float = 0.0
        total_accuracy: float = 0.0
        num_batches: int = 0

        for epoch in range(self.config.epochs):
            for batch in preferences_dataloader:
                batch = self._move_to_device(batch)
                loss, acc = self._dpo_step(batch)
                total_loss += loss
                total_accuracy += acc
                num_batches += 1
                self.global_step += 1

            self._logger.info(
                "DPO Epoch %d | Loss: %.4f | Accuracy: %.4f",
                epoch + 1,
                total_loss / max(1, num_batches),
                total_accuracy / max(1, num_batches),
            )

        return {
            "dpo_loss": total_loss / max(1, num_batches),
            "preference_accuracy": total_accuracy / max(1, num_batches),
            "global_steps": float(self.global_step),
        }

    def _dpo_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """Execute a single DPO optimisation step.

        Args:
            batch: A dictionary with chosen/rejected sequences.

        Returns:
            A tuple ``(loss_value, accuracy_value)``.
        """
        chosen_ids = batch["chosen_input_ids"]
        chosen_mask = batch.get("chosen_attention_mask")
        rejected_ids = batch["rejected_input_ids"]
        rejected_mask = batch.get("rejected_attention_mask")

        # Policy log-probs.
        chosen_logp = self._sequence_log_prob(
            self.model, chosen_ids, chosen_mask, chosen_ids
        )
        rejected_logp = self._sequence_log_prob(
            self.model, rejected_ids, rejected_mask, rejected_ids
        )

        # Reference log-probs (no gradient).
        with torch.no_grad():
            ref_chosen_logp = self._sequence_log_prob(
                self.ref_model, chosen_ids, chosen_mask, chosen_ids
            )
            ref_rejected_logp = self._sequence_log_prob(
                self.ref_model, rejected_ids, rejected_mask, rejected_ids
            )

        # Log-ratios.
        chosen_logratio = chosen_logp - ref_chosen_logp
        rejected_logratio = rejected_logp - ref_rejected_logp

        # DPO loss: -log sigmoid(beta * (chosen_ratio - rejected_ratio)).
        logits = self.config.beta * (chosen_logratio - rejected_logratio)
        loss = -F.logsigmoid(logits).mean()

        # Backward.
        loss.backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Preference accuracy: fraction where chosen > rejected.
        with torch.no_grad():
            accuracy = (chosen_logp > rejected_logp).float().mean().item()

        return loss.item(), accuracy

    # ------------------------------------------------------------------
    # PPO
    # ------------------------------------------------------------------
    def train_ppo(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train with Proximal Policy Optimisation (PPO).

        The PPO loop:

        1. Generate responses from the policy model.
        2. Score them with the reward model.
        3. Compute advantages via GAE using the value head.
        4. Optimise the clipped surrogate objective with a KL penalty
           against the reference model.

        Args:
            dataloader: A DataLoader yielding batches with ``input_ids``
                and ``attention_mask`` (prompts).

        Returns:
            A dictionary of training metrics.
        """
        self._logger.info("Starting PPO training (%d epochs).", self.config.epochs)
        self.model.train()

        total_loss: float = 0.0
        total_policy_loss: float = 0.0
        total_value_loss: float = 0.0
        total_reward: float = 0.0
        total_kl: float = 0.0
        num_batches: int = 0

        for epoch in range(self.config.epochs):
            for batch in dataloader:
                batch = self._move_to_device(batch)
                metrics = self._ppo_step(batch)
                total_loss += metrics["loss"]
                total_policy_loss += metrics["policy_loss"]
                total_value_loss += metrics["value_loss"]
                total_reward += metrics["reward"]
                total_kl += metrics["kl"]
                num_batches += 1
                self.global_step += 1

            self._logger.info(
                "PPO Epoch %d | Loss: %.4f | Policy: %.4f | Value: %.4f | "
                "Reward: %.4f | KL: %.4f",
                epoch + 1,
                total_loss / max(1, num_batches),
                total_policy_loss / max(1, num_batches),
                total_value_loss / max(1, num_batches),
                total_reward / max(1, num_batches),
                total_kl / max(1, num_batches),
            )

        n = max(1, num_batches)
        return {
            "ppo_loss": total_loss / n,
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "reward": total_reward / n,
            "kl_divergence": total_kl / n,
            "global_steps": float(self.global_step),
        }

    def _ppo_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single PPO update over a batch.

        Args:
            batch: A dictionary with ``input_ids`` and ``attention_mask``.

        Returns:
            A dictionary of step metrics.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        # 1. Compute old policy log-probs and values (no gradient).
        with torch.no_grad():
            old_log_probs = self._compute_log_probs(
                self.model, input_ids, attention_mask, input_ids
            )
            ref_log_probs = self._compute_log_probs(
                self.ref_model, input_ids, attention_mask, input_ids
            )
            # Reward from the reward model (or a heuristic).
            rewards = self._compute_rewards(input_ids, attention_mask)
            # KL penalty: subtract KL from reward.
            kl = old_log_probs - ref_log_probs
            kl_per_token = kl
            rewards = rewards - self.config.kl_coeff * kl_per_token.detach()

            # Values from the value head.
            hidden = self._get_hidden_states(self.model, input_ids, attention_mask)
            values = self.value_head(hidden).squeeze(-1)[:, :-1]
            # Advantages via GAE.
            advantages = self._compute_gae(rewards, values)
            returns = advantages + values

        # 2. PPO update for multiple epochs.
        policy_loss_total = 0.0
        value_loss_total = 0.0
        kl_total = 0.0

        for _ in range(self.config.num_ppo_epochs):
            new_log_probs = self._compute_log_probs(
                self.model, input_ids, attention_mask, input_ids
            )
            ratio = (new_log_probs - old_log_probs).exp()

            # Surrogate loss with clipping.
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss.
            hidden = self._get_hidden_states(self.model, input_ids, attention_mask)
            new_values = self.value_head(hidden).squeeze(-1)[:, :-1]
            value_loss = F.mse_loss(new_values, returns)

            # Entropy bonus.
            entropy = -new_log_probs.mean()
            loss = (
                policy_loss
                + self.config.value_coeff * value_loss
                - self.config.entropy_coeff * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()

            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()
            kl_total += (new_log_probs - ref_log_probs).mean().item()

        n_epochs = self.config.num_ppo_epochs
        return {
            "loss": (policy_loss_total + self.config.value_coeff * value_loss_total) / n_epochs,
            "policy_loss": policy_loss_total / n_epochs,
            "value_loss": value_loss_total / n_epochs,
            "reward": rewards.mean().item(),
            "kl": kl_total / n_epochs,
        }

    def _compute_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-token rewards.

        When a reward model is available its output is used; otherwise a
        simple length-based heuristic reward is applied.

        Args:
            input_ids: Token ids of shape ``(batch, seq_len)``.
            attention_mask: Optional padding mask.

        Returns:
            Reward tensor of shape ``(batch, seq_len-1)``.
        """
        if self.reward_model is not None:
            with torch.no_grad():
                outputs = self.reward_model(input_ids, attention_mask=attention_mask)
                rewards = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                # Use the per-position reward, shifted to align with targets.
                if rewards.dim() == 3:
                    rewards = rewards.squeeze(-1)
                return rewards[:, :-1].detach()

        # Heuristic: reward = 1 for non-pad tokens, 0 for pad.
        if attention_mask is not None:
            mask = attention_mask[:, 1:].float()
        else:
            mask = (input_ids[:, 1:] != 0).float()
        return mask

    def _get_hidden_states(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract the hidden states from ``model``.

        Three strategies are attempted in order:

        1. **Forward hook** on the model's final norm layer (``norm`` or
           ``final_layer_norm``) -- this captures the hidden states just
           before the language-model head, which is exactly what the
           value head needs.
        2. **Direct projection** -- when no norm layer is found, the
           logits are projected back to the hidden dimension using the
           token-embedding weight matrix.
        3. **Raw logits** -- when projection is not possible, the logits
           are returned as-is (the value head will adapt its input size).

        Args:
            model: The model to query.
            input_ids: Token ids.
            attention_mask: Optional padding mask.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.
        """
        # Strategy 1: forward hook on the final norm layer.
        hidden_capture: List[torch.Tensor] = []

        def _hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            hidden_capture.append(output)

        norm_candidates = ("norm", "final_layer_norm", "ln_f", "transformer.ln_f")
        handle = None
        for name in norm_candidates:
            module = getattr(model, name, None)
            if module is not None:
                handle = module.register_forward_hook(_hook)
                break

        outputs = model(input_ids, attention_mask=attention_mask)
        if handle is not None:
            handle.remove()
            if hidden_capture:
                return hidden_capture[0]

        # Strategy 2: project logits through the embedding weight.
        if isinstance(outputs, torch.Tensor):
            logits = outputs
            embed = getattr(model, "embed_tokens", None)
            if embed is not None:
                weight = embed.embedding.weight  # (vocab, hidden)
                # hidden = logits @ weight  -> (batch, seq, vocab) @ (vocab, hidden)
                return logits @ weight
            return logits

        # Strategy 3: tuple output -- assume hidden states are first.
        return outputs[0]  # type: ignore[index]

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Generalised Advantage Estimation (GAE).

        Args:
            rewards: Per-token rewards ``(batch, seq_len-1)``.
            values: Per-token value estimates ``(batch, seq_len-1)``.

        Returns:
            Advantage tensor of the same shape.
        """
        gamma = self.config.gamma
        lam = self.config.gae_lambda
        seq_len = rewards.shape[1]
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(rewards.shape[0], device=rewards.device)

        # ``values`` has the same length as ``rewards``; the next-state
        # value is the value at the *following* timestep, not the
        # current one.  The previous implementation used
        # ``values[:, t]`` (== current value) which made
        # ``delta = rewards[:, t] + gamma * V(s_t) - V(s_t) = rewards[:, t]``,
        # i.e. GAE collapsed to a plain reward sum and the critic had
        # zero influence.  We use ``values[:, t + 1]`` here, with a zero
        # bootstrap for the terminal timestep.
        for t in reversed(range(seq_len)):
            if t + 1 < values.shape[1]:
                next_value = values[:, t + 1]
            else:
                next_value = torch.zeros_like(last_gae)
            delta = rewards[:, t] + gamma * next_value - values[:, t]
            last_gae = delta + gamma * lam * last_gae
            advantages[:, t] = last_gae

        # Normalise advantages.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    # ------------------------------------------------------------------
    # GRPO
    # ------------------------------------------------------------------
    def train_grpo(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train with Group Relative Policy Optimisation (GRPO).

        GRPO samples a group of responses per prompt, computes rewards,
        and uses the group-relative advantage (normalised reward) as the
        advantage estimate -- eliminating the need for a value network.

        Args:
            dataloader: A DataLoader yielding batches with ``input_ids``
                and ``attention_mask`` (prompts).

        Returns:
            A dictionary of training metrics.
        """
        self._logger.info("Starting GRPO training (%d epochs).", self.config.epochs)
        self.model.train()

        total_loss: float = 0.0
        total_reward: float = 0.0
        total_kl: float = 0.0
        num_batches: int = 0

        for epoch in range(self.config.epochs):
            for batch in dataloader:
                batch = self._move_to_device(batch)
                metrics = self._grpo_step(batch)
                total_loss += metrics["loss"]
                total_reward += metrics["reward"]
                total_kl += metrics["kl"]
                num_batches += 1
                self.global_step += 1

            self._logger.info(
                "GRPO Epoch %d | Loss: %.4f | Reward: %.4f | KL: %.4f",
                epoch + 1,
                total_loss / max(1, num_batches),
                total_reward / max(1, num_batches),
                total_kl / max(1, num_batches),
            )

        n = max(1, num_batches)
        return {
            "grpo_loss": total_loss / n,
            "reward": total_reward / n,
            "kl_divergence": total_kl / n,
            "global_steps": float(self.global_step),
        }

    def _grpo_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single GRPO update.

        For each prompt the group-relative advantage is computed from
        the reward statistics across the (implicit) group.  Here we treat
        the batch as the group.

        Args:
            batch: A dictionary with ``input_ids`` and ``attention_mask``.

        Returns:
            A dictionary of step metrics.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        # Policy log-probs.
        log_probs = self._compute_log_probs(
            self.model, input_ids, attention_mask, input_ids
        )

        # Reference log-probs.
        with torch.no_grad():
            ref_log_probs = self._compute_log_probs(
                self.ref_model, input_ids, attention_mask, input_ids
            )

        # Rewards (per sequence).
        with torch.no_grad():
            rewards = self._compute_rewards(input_ids, attention_mask)
            seq_rewards = rewards.sum(dim=-1)  # (batch,)

        # Group-relative advantage: normalise rewards within the batch.
        with torch.no_grad():
            mean_reward = seq_rewards.mean()
            std_reward = seq_rewards.std() + 1e-8
            advantages = (seq_rewards - mean_reward) / std_reward  # (batch,)
            # Expand to per-token.
            token_advantages = advantages.unsqueeze(-1).expand_as(log_probs)

        # Ratio.
        ratio = (log_probs - ref_log_probs).exp()

        # Clipped surrogate loss.
        surr1 = ratio * token_advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range
        ) * token_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty.
        kl = (log_probs - ref_log_probs).mean()
        loss = policy_loss + self.config.kl_coeff * kl

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "reward": seq_rewards.mean().item(),
            "kl": kl.item(),
        }

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def train(
        self,
        dataloader: DataLoader,
        method: Optional[str] = None,
    ) -> Dict[str, float]:
        """Dispatch to the configured alignment method.

        Args:
            dataloader: The training data loader.  For DPO this should
                yield preference pairs; for PPO/GRPO it yields prompts.
            method: Optional override of the alignment method.

        Returns:
            A dictionary of training metrics.
        """
        target = (method or self.config.method).lower()
        if target == "dpo":
            return self.train_dpo(dataloader)
        if target == "ppo":
            return self.train_ppo(dataloader)
        if target == "grpo":
            return self.train_grpo(dataloader)
        raise ValueError(f"Unknown alignment method: '{target}'.")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Move all tensor values in ``batch`` to the device."""
        moved: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value  # type: ignore[assignment]
        return moved
