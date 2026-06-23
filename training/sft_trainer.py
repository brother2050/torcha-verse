"""Supervised Fine-Tuning (SFT) trainer for TorchaVerse.

This module provides :class:`SFTTrainer`, a high-level training loop
that fine-tunes a language model on a supervised dataset.  It supports:

* **LoRA / QLoRA** -- low-rank adaptation via the :mod:`models.components.lora`
  module, toggled through configuration.
* **Gradient accumulation** -- simulate large batch sizes by accumulating
  gradients over multiple micro-batches.
* **Mixed precision** -- automatic mixed precision (AMP) via
  :func:`torch.amp.autocast` and :class:`torch.amp.GradScaler`.
* **Gradient clipping** -- global L2-norm clipping to stabilise training.
* **Checkpointing** -- periodic saving and resumption via
  :class:`CheckpointManager`.
* **Logging** -- loss, learning rate, and epoch progress.

The training loop follows the standard pattern::

    forward -> loss -> backward -> optimizer step -> scheduler step
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.model_registry import BaseModel
from infrastructure.checkpoint_manager import CheckpointManager
from infrastructure.config_manager import ConfigManager
from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

__all__ = ["SFTTrainer", "SFTConfig"]

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class SFTConfig:
    """Configuration container for :class:`SFTTrainer`.

    All hyper-parameters have sensible defaults that mirror the
    ``training_config.yaml`` shipped with the framework.

    Args:
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Number of micro-batches to
            accumulate before an optimizer step.
        learning_rate: Peak learning rate.
        weight_decay: L2 weight decay.
        max_grad_norm: Maximum gradient norm for clipping (``0`` disables).
        mixed_precision: ``"no"``, ``"fp16"``, or ``"bf16"``.
        warmup_steps: Linear warmup steps.
        lr_scheduler_type: ``"cosine"``, ``"linear"``, ``"constant"``,
            or ``"cosine_with_restarts"``.
        seed: Random seed for reproducibility.
        log_steps: Log every ``log_steps`` optimizer steps.
        save_steps: Save a checkpoint every ``save_steps`` steps.
        save_total_limit: Maximum checkpoints to keep.
        output_dir: Directory for checkpoints and logs.
        lora: LoRA configuration dictionary (or ``None`` to disable).
        qlora: When ``True`` enable 4-bit quantised LoRA.
        num_workers: DataLoader workers.
        pin_memory: Whether to pin memory in the DataLoader.
    """

    def __init__(
        self,
        epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        mixed_precision: str = "bf16",
        warmup_steps: int = 100,
        lr_scheduler_type: str = "cosine",
        seed: int = 42,
        log_steps: int = 10,
        save_steps: int = 500,
        save_total_limit: int = 3,
        output_dir: PathLike = "outputs",
        lora: Optional[Dict[str, Any]] = None,
        qlora: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        self.epochs: int = max(1, int(epochs))
        self.batch_size: int = max(1, int(batch_size))
        self.gradient_accumulation_steps: int = max(1, int(gradient_accumulation_steps))
        self.learning_rate: float = float(learning_rate)
        self.weight_decay: float = float(weight_decay)
        self.max_grad_norm: float = float(max_grad_norm)
        self.mixed_precision: str = mixed_precision
        self.warmup_steps: int = max(0, int(warmup_steps))
        self.lr_scheduler_type: str = lr_scheduler_type
        self.seed: int = int(seed)
        self.log_steps: int = max(1, int(log_steps))
        self.save_steps: int = max(1, int(save_steps))
        self.save_total_limit: int = max(1, int(save_total_limit))
        self.output_dir: Path = Path(output_dir)
        self.lora: Optional[Dict[str, Any]] = lora
        self.qlora: bool = bool(qlora)
        self.num_workers: int = max(0, int(num_workers))
        self.pin_memory: bool = bool(pin_memory)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SFTConfig":
        """Build an :class:`SFTConfig` from a flat dictionary.

        Recognised keys mirror the constructor arguments.  Unknown keys
        are ignored.
        """
        known = {
            "epochs", "batch_size", "gradient_accumulation_steps",
            "learning_rate", "weight_decay", "max_grad_norm",
            "mixed_precision", "warmup_steps", "lr_scheduler_type",
            "seed", "log_steps", "save_steps", "save_total_limit",
            "output_dir", "lora", "qlora", "num_workers", "pin_memory",
        }
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls) -> "SFTConfig":
        """Build an :class:`SFTConfig` from the global YAML configuration."""
        cfg = ConfigManager()
        optimizer_cfg = cfg.get("optimizer", {})
        scheduler_cfg = cfg.get("lr_scheduler", {})
        training_cfg = cfg.get("training", {})
        checkpoint_cfg = cfg.get("checkpoint", {})
        lora_cfg = cfg.get("lora", {})

        lora_dict: Optional[Dict[str, Any]] = None
        if lora_cfg.get("enabled", False):
            lora_dict = {
                "r": lora_cfg.get("r", 16),
                "alpha": lora_cfg.get("alpha", 32),
                "dropout": lora_cfg.get("dropout", 0.05),
                "target_modules": lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            }

        return cls(
            epochs=training_cfg.get("epochs", 3),
            batch_size=training_cfg.get("batch_size", 4),
            gradient_accumulation_steps=training_cfg.get(
                "gradient_accumulation_steps", 1
            ),
            learning_rate=optimizer_cfg.get("lr", 2e-5),
            weight_decay=optimizer_cfg.get("weight_decay", 0.01),
            max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
            mixed_precision=training_cfg.get("mixed_precision", "bf16"),
            warmup_steps=scheduler_cfg.get("warmup_steps", 100),
            lr_scheduler_type=scheduler_cfg.get("type", "cosine"),
            seed=training_cfg.get("seed", 42),
            log_steps=cfg.get("logging.log_steps", 10),
            save_steps=checkpoint_cfg.get("save_steps", 500),
            save_total_limit=checkpoint_cfg.get("save_total_limit", 3),
            output_dir=checkpoint_cfg.get("save_dir", "outputs"),
            lora=lora_dict,
            qlora=lora_cfg.get("qlora", False),
            num_workers=training_cfg.get("num_workers", 0),
            pin_memory=training_cfg.get("pin_memory", True),
        )


# ---------------------------------------------------------------------------
# SFTTrainer
# ---------------------------------------------------------------------------
class SFTTrainer:
    """Supervised fine-tuning trainer.

    The trainer wraps a model, datasets, optimizer, and scheduler and
    executes a standard supervised training loop with support for LoRA,
    mixed precision, gradient accumulation, and checkpointing.

    Args:
        model: The model to fine-tune (a :class:`BaseModel` or any
            :class:`torch.nn.Module`).
        train_dataset: Training dataset (any ``torch.utils.data.Dataset``).
        eval_dataset: Optional evaluation dataset.
        config: :class:`SFTConfig` or ``None`` to load from YAML.
        optimizer: Optional pre-built optimizer.  When ``None`` an
            AdamW optimizer is created from the config.
        lr_scheduler: Optional pre-built scheduler.
        collate_fn: Optional custom collation function.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        config: Optional[SFTConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        collate_fn: Optional[Any] = None,
    ) -> None:
        self.config: SFTConfig = config or SFTConfig.from_yaml()
        self.model: nn.Module = model
        self.train_dataset: torch.utils.data.Dataset = train_dataset
        self.eval_dataset: Optional[torch.utils.data.Dataset] = eval_dataset
        self.collate_fn: Any = collate_fn
        self._logger = get_logger(self.__class__.__name__)

        # Device management.
        self._device_manager: DeviceManager = DeviceManager()
        self.device: torch.device = self._device_manager.get_device()

        # Reproducibility.
        self._set_seed(self.config.seed)

        # Apply LoRA / QLoRA if configured.
        self._lora_applied: bool = False
        if self.config.lora is not None:
            self._apply_lora(self.config.lora)

        # Move model to device.
        self.model = self._device_manager.to_device(self.model, self.device)

        # Optimizer.
        self.optimizer: torch.optim.Optimizer = (
            optimizer
            if optimizer is not None
            else self._build_optimizer()
        )

        # Learning-rate scheduler.
        self.lr_scheduler: Optional[Any] = lr_scheduler
        if self.lr_scheduler is None and self.config.warmup_steps >= 0:
            self.lr_scheduler = self._build_scheduler()

        # Mixed-precision setup.
        self._amp_enabled: bool = self.config.mixed_precision in ("fp16", "bf16")
        self._amp_dtype: Optional[torch.dtype] = None
        self._scaler: Optional[torch.amp.GradScaler] = None
        if self._amp_enabled:
            self._amp_dtype = (
                torch.float16 if self.config.mixed_precision == "fp16"
                else torch.bfloat16
            )
            # GradScaler is only needed for fp16.
            if self.config.mixed_precision == "fp16":
                self._scaler = torch.amp.GradScaler("cuda")

        # Checkpoint manager.
        self.checkpoint_manager: CheckpointManager = CheckpointManager(
            save_dir=self.config.output_dir,
            save_total_limit=self.config.save_total_limit,
        )

        # Training state.
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.best_eval_loss: float = float("inf")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _set_seed(seed: int) -> None:
        """Seed all RNGs for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _apply_lora(self, lora_cfg: Dict[str, Any]) -> None:
        """Inject LoRA adapters into the model.

        Args:
            lora_cfg: LoRA configuration with keys ``r``, ``alpha``,
                ``dropout``, and ``target_modules``.
        """
        try:
            from models.components.lora import (
                apply_lora,
                mark_only_lora_as_trainable,
            )
        except ImportError:
            self._logger.warning(
                "LoRA components are not available; skipping LoRA injection."
            )
            return

        target_modules = lora_cfg.get(
            "target_modules", ["q_proj", "v_proj"]
        )
        apply_lora(
            self.model,
            target_modules=target_modules,
            r=lora_cfg.get("r", 16),
            alpha=lora_cfg.get("alpha", 32),
            dropout=lora_cfg.get("dropout", 0.05),
        )
        mark_only_lora_as_trainable(self.model)
        self._lora_applied = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self._logger.info(
            "LoRA applied: %d / %d parameters trainable (%.2f%%).",
            trainable, total, 100.0 * trainable / max(total, 1),
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build an AdamW optimizer over the trainable parameters."""
        # Separate decay / no-decay parameter groups.
        no_decay = ("bias", "LayerNorm.weight", "norm.weight", "layernorm")
        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        return optimizer

    def _build_scheduler(self) -> Any:
        """Build a learning-rate scheduler with linear warmup."""
        # Estimate total steps from the dataset length.
        try:
            num_examples = len(self.train_dataset)  # type: ignore[arg-type]
        except TypeError:
            num_examples = 1000
        steps_per_epoch = max(
            1,
            math.ceil(num_examples / (self.config.batch_size * self.config.gradient_accumulation_steps)),
        )
        total_steps = steps_per_epoch * self.config.epochs
        warmup = self.config.warmup_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            progress = float(current_step - warmup) / float(
                max(1, total_steps - warmup)
            )
            if self.config.lr_scheduler_type == "linear":
                return max(0.0, 1.0 - progress)
            if self.config.lr_scheduler_type == "constant":
                return 1.0
            if self.config.lr_scheduler_type == "cosine_with_restarts":
                # Single restart at the midpoint.
                if progress < 0.5:
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
                return 0.5 * (1.0 + math.cos(math.pi * (progress - 0.5) * 2))
            # Default: cosine.
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _build_dataloader(
        self, dataset: torch.utils.data.Dataset, shuffle: bool = True
    ) -> DataLoader:
        """Build a DataLoader for ``dataset``.

        Args:
            dataset: The dataset to wrap.
            shuffle: Whether to shuffle the data.

        Returns:
            A configured :class:`DataLoader`.
        """
        collate = self.collate_fn
        if collate is None and hasattr(dataset, "collate_fn"):
            collate = dataset.collate_fn  # type: ignore[attr-defined]
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and self.device.type == "cuda",
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the cross-entropy loss, ignoring padding tokens.

        The standard causal-LM shift is applied: the model predicts token
        ``t+1`` from token ``t``, so logits and labels are shifted by one
        position.  Positions labelled ``-100`` (padding) are ignored.

        Args:
            logits: Model logits of shape ``(batch, seq_len, vocab_size)``.
            labels: Target token ids of shape ``(batch, seq_len)``.

        Returns:
            The scalar loss tensor.
        """
        # Shift: predict next token from current.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, epochs: Optional[int] = None) -> Dict[str, float]:
        """Execute the supervised training loop.

        Args:
            epochs: Override the number of epochs from the config.

        Returns:
            A dictionary of training metrics (``train_loss``,
            ``eval_loss``, ``learning_rate``, ``epochs_run``).
        """
        num_epochs = epochs or self.config.epochs
        self.model.train()

        train_dataloader = self._build_dataloader(self.train_dataset, shuffle=True)

        total_loss: float = 0.0
        log_loss: float = 0.0
        log_count: int = 0

        self._logger.info(
            "Starting SFT training for %d epoch(s) on device %s.",
            num_epochs, self.device,
        )

        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            self.current_epoch = epoch
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                batch = self._move_to_device(batch)
                loss = self._training_step(batch)

                total_loss += loss
                log_loss += loss
                log_count += 1

                # Log periodically.
                if self.global_step > 0 and self.global_step % self.config.log_steps == 0:
                    avg_loss = log_loss / max(1, log_count)
                    lr = self._get_lr()
                    self._logger.info(
                        "Epoch %d | Step %d | Loss: %.4f | LR: %.2e",
                        epoch + 1, self.global_step, avg_loss, lr,
                    )
                    log_loss = 0.0
                    log_count = 0

                # Save checkpoint periodically.
                if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()

            # End-of-epoch evaluation.
            eval_metrics: Dict[str, float] = {}
            if self.eval_dataset is not None:
                eval_metrics = self.evaluate()
                eval_loss = eval_metrics.get("eval_loss", float("inf"))
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint(tag="best")

        avg_train_loss = total_loss / max(1, self.global_step)
        final_metrics: Dict[str, float] = {
            "train_loss": avg_train_loss,
            "learning_rate": self._get_lr(),
            "epochs_run": float(num_epochs),
            "global_steps": float(self.global_step),
        }
        if self.eval_dataset is not None:
            final_metrics["eval_loss"] = self.best_eval_loss

        self._logger.info("Training complete. Metrics: %s", final_metrics)
        return final_metrics

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single (micro-batch) training step.

        Handles forward, loss, backward, gradient accumulation, gradient
        clipping, optimizer step, and scheduler step.

        Args:
            batch: A dictionary of batched tensors on the device.

        Returns:
            The loss value (as a Python float) for this micro-batch.
        """
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels", input_ids)

        # Forward pass with optional autocast.
        if self._amp_enabled and self._amp_dtype is not None:
            with torch.amp.autocast("cuda", dtype=self._amp_dtype):
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = self._compute_loss(logits, labels)
        else:
            logits = self.model(input_ids, attention_mask=attention_mask)
            loss = self._compute_loss(logits, labels)

        # Scale loss for gradient accumulation.
        scaled_loss = loss / self.config.gradient_accumulation_steps

        # Backward pass.
        if self._scaler is not None:
            self._scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Optimizer step every ``gradient_accumulation_steps`` micro-batches.
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self._optimizer_step()

        self.global_step += 1
        return loss.item()

    def _optimizer_step(self) -> None:
        """Perform gradient clipping, optimizer step, and scheduler step."""
        if self._scaler is not None:
            # Unscale before clipping.
            self._scaler.unscale_(self.optimizer)

        # Gradient clipping.
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the evaluation dataset.

        Returns:
            A dictionary with ``eval_loss`` and ``eval_accuracy``.
        """
        if self.eval_dataset is None:
            return {}

        self.model.eval()
        dataloader = self._build_dataloader(self.eval_dataset, shuffle=False)

        total_loss: float = 0.0
        total_tokens: int = 0
        correct: int = 0

        for batch in dataloader:
            batch = self._move_to_device(batch)
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels", input_ids)

            if self._amp_enabled and self._amp_dtype is not None:
                with torch.amp.autocast("cuda", dtype=self._amp_dtype):
                    logits = self.model(input_ids, attention_mask=attention_mask)
                    loss = self._compute_loss(logits, labels)
            else:
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = self._compute_loss(logits, labels)

            total_loss += loss.item()

            # Accuracy on non-ignored positions.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            preds = shift_logits.argmax(dim=-1)
            mask = shift_labels != -100
            correct += (preds[mask] == shift_labels[mask]).sum().item()
            total_tokens += mask.sum().item()

        avg_loss = total_loss / max(1, len(dataloader))
        accuracy = correct / max(1, total_tokens)

        self._logger.info(
            "Evaluation: loss=%.4f, accuracy=%.4f", avg_loss, accuracy
        )

        self.model.train()
        return {"eval_loss": avg_loss, "eval_accuracy": accuracy}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self, tag: Optional[str] = None) -> None:
        """Save a training checkpoint.

        Args:
            tag: Optional tag appended to the checkpoint directory name.
        """
        step = self.global_step
        metadata = {
            "epoch": self.current_epoch,
            "global_step": step,
            "best_eval_loss": self.best_eval_loss,
            "tag": tag or "",
        }
        try:
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                step=step,
                metadata=metadata,
            )
            self._logger.info("Saved checkpoint at step %d.", step)
        except Exception as exc:
            self._logger.error("Failed to save checkpoint: %s", exc)

    def save_model(self, path: PathLike) -> Path:
        """Save the model weights (and merge LoRA when applicable).

        Args:
            path: Target file or directory path.

        Returns:
            The path to the saved weights.
        """
        # Merge LoRA weights for a clean export.
        if self._lora_applied:
            try:
                from models.components.lora import merge_lora

                merge_lora(self.model)
                self._logger.info("Merged LoRA weights before saving.")
            except ImportError:
                pass

        return self.checkpoint_manager.save_weights_only(self.model, path)

    def load_checkpoint(self, path: PathLike) -> Dict[str, Any]:
        """Load a checkpoint and restore training state.

        Args:
            path: Path to the checkpoint directory.

        Returns:
            The checkpoint metadata dictionary.
        """
        metadata = self.checkpoint_manager.load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            map_location=self.device,
        )
        self.global_step = int(metadata.get("step", 0))
        self.current_epoch = int(metadata.get("epoch", 0))
        self.best_eval_loss = float(metadata.get("best_eval_loss", float("inf")))
        self._logger.info(
            "Restored checkpoint from %s at step %d.", path, self.global_step
        )
        return metadata

    def resume(self) -> Tuple[int, Dict[str, Any]]:
        """Resume training from the latest checkpoint.

        Returns:
            A tuple ``(step, metadata)`` from the checkpoint.
        """
        step, metadata = self.checkpoint_manager.resume(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            map_location=self.device,
        )
        self.global_step = step
        self.current_epoch = int(metadata.get("epoch", 0))
        self.best_eval_loss = float(metadata.get("best_eval_loss", float("inf")))
        if step > 0:
            self._logger.info("Resumed training from step %d.", step)
        return step, metadata

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Move all tensor values in ``batch`` to the training device.

        Args:
            batch: A dictionary of tensors (and possibly other values).

        Returns:
            A new dictionary with tensors on the device.
        """
        moved: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value  # type: ignore[assignment]
        return moved

    def _get_lr(self) -> float:
        """Return the current learning rate."""
        if self.lr_scheduler is not None:
            try:
                return self.lr_scheduler.get_last_lr()[0]
            except (AttributeError, IndexError):
                pass
        return self.optimizer.param_groups[0]["lr"]
