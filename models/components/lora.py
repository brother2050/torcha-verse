"""LoRA (Low-Rank Adaptation) adapters.

LoRA freezes the pre-trained model weights and injects trainable
rank-decomposition matrices into each layer of the Transformer
architecture, greatly reducing the number of trainable parameters for
downstream tasks.

This module provides:

* :class:`LoRALinear` -- a drop-in replacement for ``nn.Linear`` that
  wraps an existing linear layer and adds a low-rank update.
* :func:`apply_lora` -- inject LoRA adapters into a model.
* :func:`merge_lora` -- merge the LoRA weights back into the base
  weights (zero-overhead inference).

Reference:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

__all__ = ["LoRALinear", "apply_lora", "merge_lora", "mark_only_lora_as_trainable"]


class LoRALinear(nn.Module):
    """Linear layer with a LoRA low-rank adaptation update.

    The forward pass is::

        y = base_layer(x) + scaling * lora_A(lora_B(x))

    where ``lora_B`` is the down-projection ``(in -> r)``, ``lora_A`` is
    the up-projection ``(r -> out)``, and ``scaling = alpha / r``.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        r: LoRA rank.
        alpha: LoRA alpha (scaling numerator).
        dropout: Dropout applied to the input of the LoRA branch.
        bias: Whether the base layer has a bias term.
        base_layer: An optional pre-existing ``nn.Linear`` to wrap.  When
            ``None`` a new ``nn.Linear`` is created.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
        bias: bool = False,
        base_layer: Optional[nn.Linear] = None,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"r must be a positive integer, got {r}.")

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.r: int = r
        self.alpha: int = alpha
        self.scaling: float = float(alpha) / float(r)
        self.dropout_p: float = dropout

        # Base (frozen) linear layer.
        if base_layer is not None:
            self.base_layer: nn.Linear = base_layer
        else:
            self.base_layer = nn.Linear(in_features, out_features, bias=bias)

        # LoRA down-projection (initialised to zero so the update starts at 0).
        self.lora_B: nn.Linear = nn.Linear(in_features, r, bias=False)
        # LoRA up-projection.
        self.lora_A: nn.Linear = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))  # type: ignore[name-defined]
        nn.init.zeros_(self.lora_A.weight)

        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Freeze the base layer by default.
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

        self._enabled: bool = True

    # ------------------------------------------------------------------
    def enable(self) -> None:
        """Enable the LoRA branch (the low-rank update is applied)."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the LoRA branch (only the base layer is used)."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Whether the LoRA branch is active."""
        return self._enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the base output plus the (optional) LoRA update.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        result = self.base_layer(x)
        if self._enabled:
            lora_out = self.lora_A(self.lora_B(self.dropout(x)))
            result = result + self.scaling * lora_out
        return result

    # ------------------------------------------------------------------
    def merge(self) -> None:
        """Merge the LoRA weights into the base layer in-place.

        After merging the forward pass becomes equivalent to a plain
        ``nn.Linear`` with the merged weights.  The LoRA branch is then
        disabled.
        """
        with torch.no_grad():
            # W' = W + scaling * A @ B
            delta = self.scaling * (self.lora_A.weight @ self.lora_B.weight)
            self.base_layer.weight.add_(delta)
        self.disable()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"enabled={self._enabled}"
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def _get_submodule(model: nn.Module, target: str) -> nn.Module:
    """Return the submodule at the dotted ``target`` path."""
    parent: nn.Module = model
    atoms = target.split(".")
    for atom in atoms[:-1]:
        parent = getattr(parent, atom)
    return parent


def apply_lora(
    model: nn.Module,
    target_modules: Union[str, List[str]],
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
) -> nn.Module:
    """Inject LoRA adapters into ``model``.

    Every ``nn.Linear`` whose qualified name ends with one of the
    ``target_modules`` suffixes is wrapped in a :class:`LoRALinear`.

    Args:
        model: The model to modify (in-place).
        target_modules: A module-name suffix (or list of suffixes) to
            target, e.g. ``"q_proj"`` or ``["q_proj", "v_proj"]``.
        r: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout.

    Returns:
        The modified ``model`` (the same object, modified in-place).
    """
    if isinstance(target_modules, str):
        target_modules = [target_modules]

    # Collect (parent, name, module) tuples first to avoid mutating during iteration.
    targets: List[tuple] = []
    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child_module, nn.Linear) and any(
                full_name.endswith(t) or child_name == t for t in target_modules
            ):
                targets.append((parent_module, child_name, child_module))

    for parent, child_name, child_module in targets:
        lora_layer = LoRALinear(
            in_features=child_module.in_features,
            out_features=child_module.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=child_module.bias is not None,
            base_layer=child_module,
        )
        setattr(parent, child_name, lora_layer)

    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRA weights in ``model`` into their base layers.

    After merging each :class:`LoRALinear` behaves as a plain
    ``nn.Linear`` (the low-rank branch is disabled).

    Args:
        model: The model containing LoRA layers.

    Returns:
        The modified ``model``.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    return model


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Freeze all parameters except the LoRA adapters.

    Args:
        model: The model to configure.
    """
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
