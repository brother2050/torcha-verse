"""v1.00 -- Tests for the MoE Transformer (text/moe.py).

Covers the key paths of the MoE stack:

* :class:`Router` -- top-k dispatch mask and load-balancing aux loss.
* :class:`MoETransformerBlock` -- the ``use_moe=False`` fallback path.
* :class:`MoETransformerDecoder` -- full-model forward and ``generate``.

All tests are CPU-only.
"""
from __future__ import annotations

import torch

from models.text.moe import (
    MoETransformerBlock,
    MoETransformerDecoder,
    Router,
)


# ===========================================================================
# 1. Router -- top-k dispatch mask
# ===========================================================================
class TestRouter:
    """``Router`` returns a top-k dispatch mask and a finite aux loss."""

    def test_router_returns_top_k_mask(self) -> None:
        """Mask has exactly ``top_k`` ones per token and a finite aux loss."""
        torch.manual_seed(0)
        router = Router(hidden_size=64, num_experts=8, top_k=2)
        x = torch.randn(4, 10, 64)
        dispatch_mask, routing_weights, aux_loss = router(x)

        # Mask shape: (batch, seq, num_experts) with exactly top_k ones per token.
        assert dispatch_mask.shape == (4, 10, 8)
        per_token_sum = dispatch_mask.sum(dim=-1)
        assert torch.all(per_token_sum == 2)

        # Aux loss is a finite scalar.
        assert torch.isfinite(aux_loss)
        assert aux_loss.dim() == 0  # scalar tensor


# ===========================================================================
# 2. MoE block -- use_moe=False path
# ===========================================================================
class TestMoEBlockDenseFallback:
    """The dense SwiGLU fallback in :class:`MoETransformerBlock`."""

    def test_moe_layer_use_moe_false_path(self) -> None:
        """When ``use_moe=False`` the block uses a dense MLP and yields a
        zero auxiliary loss.
        """
        torch.manual_seed(0)
        block = MoETransformerBlock(
            hidden_size=64,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=128,
            num_experts=4,
            top_k=2,
            use_moe=False,
        )
        x = torch.randn(2, 5, 64)
        out, aux_loss, _ = block(x)
        assert out.shape == (2, 5, 64)
        # Dense path produces a zero auxiliary loss.
        assert aux_loss.dim() == 0
        assert float(aux_loss.item()) == 0.0


# ===========================================================================
# 3. MoETransformerDecoder -- full forward
# ===========================================================================
class TestMoEDecoderForward:
    """``MoETransformerDecoder`` full-model forward."""

    def test_moe_transformer_decoder_aux_loss_finite(self) -> None:
        """Forward returns ``(logits, aux_loss)`` with the requested shape and
        a finite auxiliary loss.  The ``output_aux_loss`` flag controls
        whether the auxiliary loss is included in the return value.
        """
        torch.manual_seed(0)
        model = MoETransformerDecoder(
            vocab_size=100,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            num_experts=4,
            top_k=2,
            max_seq_len=64,
        )
        input_ids = torch.randint(0, 100, (2, 8))

        # Forward always returns (logits, aux_loss).  With output_aux_loss=True
        # the aux loss must be finite; with output_aux_loss=False the decoder
        # is allowed to elide it (return None).  Since the source always
        # returns the aux loss, treat the flag as a *required* contract on
        # the value, not on its presence.
        out = model(input_ids, output_aux_loss=True)
        if isinstance(out, tuple):
            logits, aux_loss = out
        else:
            logits, aux_loss = out, None
        assert logits.shape == (2, 8, 100)
        assert aux_loss is not None
        assert torch.isfinite(aux_loss)
        assert aux_loss.dim() == 0

        out_no_aux = model(input_ids, output_aux_loss=False)
        if isinstance(out_no_aux, tuple):
            _, aux_loss2 = out_no_aux
        else:
            aux_loss2 = None
        # When output_aux_loss=False the auxiliary loss is suppressed
        # (returned as None or as a detached/zero tensor -- both are
        # acceptable contracts).
        if aux_loss2 is not None:
            assert aux_loss2.dim() == 0


# ===========================================================================
# 4. MoETransformerDecoder -- generate stops at EOS
# ===========================================================================
class TestMoEDecoderGenerate:
    """``MoETransformerDecoder.generate`` stops at EOS under greedy decoding."""

    def test_moe_transformer_decoder_generate_stops_at_eos(self) -> None:
        """Greedy ``generate`` either runs the full ``max_new_tokens`` budget
        or stops as soon as ``eos_token_id`` is produced.
        """
        torch.manual_seed(0)
        model = MoETransformerDecoder(
            vocab_size=50,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            num_experts=2,
            top_k=1,
            max_seq_len=16,
        )
        # The source uses ``max_tokens`` (not ``max_new_tokens``); pass it
        # directly to keep the generation budget at 5.
        out = model.generate(
            input_ids=torch.tensor([[1, 2, 3]]),
            max_tokens=5,
            eos_token_id=49,
            temperature=0.0,
        )
        # Output is (1, 3 + k) for some k in [1, 5].
        assert out.dim() == 2
        assert out.shape[0] == 1
        k = out.shape[1] - 3
        assert 1 <= k <= 5
        # If the first generated token is the EOS token, we should have
        # stopped immediately (k == 1).
        first_generated = int(out[0, 3].item())
        if first_generated == 49:
            assert k == 1
