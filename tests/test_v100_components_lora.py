"""v1.00 -- Tests for the LoRA adapter in :mod:`models.components.lora`.

This is the **adapter** used by the MoE / RoPE / etc. components and is
distinct from the LoRA injector in :mod:`models.lora` (which is tested
in ``test_v085_lora_offload.py``).  The surface exercised here:

* :class:`LoRALinear` -- enable / disable, merge, and freezable base layer.
* :func:`apply_lora` -- selective wrapping of ``nn.Linear`` modules.
* :func:`mark_only_lora_as_trainable` -- freezing non-LoRA parameters.

All tests are CPU-only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.components.lora import (
    LoRALinear,
    apply_lora,
    mark_only_lora_as_trainable,
)


# ===========================================================================
# 1. LoRALinear -- enable / disable
# ===========================================================================
class TestLoRALinearEnableDisable:
    """The ``enable`` / ``disable`` toggle on :class:`LoRALinear`."""

    def test_lora_linear_enable_disable(self) -> None:
        """``enable()`` / ``disable()`` flip the ``enabled`` property and
        the forward pass still produces the right shape in both states.
        """
        torch.manual_seed(0)
        wrapped = LoRALinear(in_features=16, out_features=8, r=4)
        # The LoRA branch is enabled by default (initialised at zero
        # so the output equals the base linear in either case, but the
        # property must report the state correctly).
        wrapped.disable()
        assert wrapped.enabled is False
        x = torch.randn(2, 16)
        out = wrapped(x)
        assert out.shape == (2, 8)

        wrapped.enable()
        assert wrapped.enabled is True
        out = wrapped(x)
        assert out.shape == (2, 8)

        wrapped.disable()
        assert wrapped.enabled is False


# ===========================================================================
# 2. LoRALinear -- merge() restores the base linear
# ===========================================================================
class TestLoRALinearMerge:
    """``LoRALinear.merge()`` makes the forward pass equivalent to the
    plain base ``nn.Linear`` (zero-overhead inference).
    """

    def test_lora_linear_merge_makes_equivalent_to_base(self) -> None:
        """After ``enable()`` + ``merge()`` + ``disable()`` the wrapped layer
        must be numerically equivalent to the bare base linear.
        """
        torch.manual_seed(0)
        base = nn.Linear(16, 8, bias=False)
        # ``in_features`` / ``out_features`` are required positional args
        # even when wrapping a pre-built ``base_layer``.
        wrapped = LoRALinear(in_features=16, out_features=8, base_layer=base, r=4)
        wrapped.enable()
        # Merge the LoRA delta into the base weight.
        wrapped.merge()
        # After merge the LoRA branch is disabled automatically; the
        # forward is just the base linear.
        wrapped.disable()
        x = torch.randn(3, 16)
        y_wrapped = wrapped(x)
        y_base = base(x)
        assert torch.allclose(y_wrapped, y_base, atol=1e-5)


# ===========================================================================
# 3. apply_lora -- selective wrapping by suffix
# ===========================================================================
class TestApplyLora:
    """``apply_lora`` injects LoRA adapters into selected ``nn.Linear``s."""

    def test_apply_lora_injects_by_suffix(self) -> None:
        """Only the modules whose qualified name matches the suffix are
        wrapped.
        """
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 5),
            nn.ReLU(),
        )
        # Sanity: starts as plain linears.
        assert isinstance(model[0], nn.Linear)
        assert isinstance(model[1], nn.Linear)

        # Target only "0" (the first Linear); "1" and "ReLU" must remain.
        apply_lora(model, target_modules=["0"], r=2)

        assert isinstance(model[0], LoRALinear)
        assert not isinstance(model[1], LoRALinear)
        assert isinstance(model[1], nn.Linear)
        # ReLU is a plain activation, unchanged.
        assert isinstance(model[2], nn.ReLU)


# ===========================================================================
# 4. mark_only_lora_as_trainable
# ===========================================================================
class TestMarkOnlyLoraTrainable:
    """``mark_only_lora_as_trainable`` freezes everything except ``lora_*``."""

    def test_mark_only_lora_as_trainable(self) -> None:
        """After the call, the base linear is frozen and the LoRA branch
        remains trainable.
        """
        torch.manual_seed(0)
        wrapped = LoRALinear(in_features=8, out_features=4, r=2)

        # Sanity before: base is frozen, LoRA is trainable.
        assert wrapped.base_layer.weight.requires_grad is False
        assert wrapped.lora_B.weight.requires_grad is True
        assert wrapped.lora_A.weight.requires_grad is True

        # Toggle a few requires_grad manually to make the function do work.
        for p in wrapped.parameters():
            p.requires_grad = True

        mark_only_lora_as_trainable(wrapped)

        # Base layer is frozen.
        assert wrapped.base_layer.weight.requires_grad is False
        if wrapped.base_layer.bias is not None:
            assert wrapped.base_layer.bias.requires_grad is False
        # LoRA branch is trainable.
        assert wrapped.lora_B.weight.requires_grad is True
        assert wrapped.lora_A.weight.requires_grad is True
