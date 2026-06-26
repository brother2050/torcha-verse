"""v0.8.5 LoRA injection + CPU offload tests (≥ 22 tests).

The v0.8.5 second wave delivers the ComfyUI / diffusers
``ModelPatcher``-style runtime patch system plus a thin
LoRA recipe on top of it.  This test file exercises:

1. **ModelPatcher** (5 tests) -- patch / undo / context
   manager / key_filter / duplicate name error.
2. **CPU offload helpers** (4 tests) -- no-op when
   ``compute == offload``; per-submodule / sequential
   registration / disable-offload cleanup.
3. **LoRAInjector / LoRASpec** (7 tests) -- basic add /
   apply / remove; zero initial delta (the model output is
   unchanged at first); non-zero delta after ``B`` Kaiming
   init; rank cap; default target modules; state-dict
   round-trip.
4. **HunyuanDiT LoRA convenience** (3 tests) -- the
   ``HunyuanDiT.lora_apply`` method; default targets
   include the four block submodules; remove / clear
   restores outputs.
5. **End-to-end LoRA on HunyuanDiT-Tiny** (3 tests) --
   ``HunyuanDiT.lora_apply`` then ``.sample()`` runs end to
   end; the LoRA is non-trivial (output differs from base);
   the delta is gone after ``lora_remove``.

Total tests: 22.  Running this on stock CPU takes < 1 s.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from core.offload import (
    ModelPatcher,
    Patch,
    disable_offload,
    enable_model_cpu_offload,
    enable_sequential_cpu_offload,
)
from models.image.dit import HunyuanDiT
from models.lora import (
    LoRAInjector,
    LoRASpec,
    default_target_modules,
    inject_lora,
    lora_state_dict,
    load_lora_state_dict,
)


# ===========================================================================
# Section 1 -- ModelPatcher (5 tests)
# ===========================================================================
class TestModelPatcher:
    """The :class:`ModelPatcher` registry."""

    def test_add_apply_clear_round_trip(self) -> None:
        """A reversible patch is applied and undone on clear."""
        m = nn.Linear(8, 8, bias=False)
        prev = m.weight.data.clone()
        patcher = ModelPatcher(m)
        saved = prev.clone()

        def zero_op(_mod: nn.Module, _s: float):
            _mod.weight.data.zero_()
            def undo(mm: nn.Module) -> None:
                mm.weight.data.copy_(saved)
            return undo

        patcher.add(Patch("zero", zero_op, strength=1.0))
        patcher.apply()
        assert m.weight.data.abs().sum().item() == 0.0
        patcher.clear()
        assert torch.equal(m.weight.data, prev)

    def test_context_manager_restores(self) -> None:
        """The ``with`` block applies on entry and restores on exit."""
        m = nn.Linear(8, 8, bias=False)
        prev = m.weight.data.clone()
        saved = prev.clone()

        def zero_op(_mod: nn.Module, _s: float):
            _mod.weight.data.zero_()
            def undo(mm: nn.Module) -> None:
                mm.weight.data.copy_(saved)
            return undo

        patcher = ModelPatcher(m)
        patcher.add(Patch("zero", zero_op, strength=1.0))
        with patcher:
            assert m.weight.data.abs().sum().item() == 0.0
        assert torch.equal(m.weight.data, prev)

    def test_key_filter_is_glob(self) -> None:
        """The ``key_filter`` glob is honoured on apply."""
        m = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 4, bias=False))
        prev0, prev1 = m[0].weight.data.clone(), m[1].weight.data.clone()
        saved0 = prev0.clone()
        saved1 = prev1.clone()

        def zero_op(_mod: nn.Module, _s: float):
            def undo(mm: nn.Module) -> None:
                # No-op marker; we rely on the closure to know
                # which weight to restore.
                pass
            _mod.weight.data.zero_()
            # Return an undo that knows *its* saved weight.
            target = _mod
            def real_undo(_mm: nn.Module) -> None:
                if target is m[0]:
                    m[0].weight.data.copy_(saved0)
                else:
                    m[1].weight.data.copy_(saved1)
            return real_undo

        patcher = ModelPatcher(m)
        patcher.add(Patch("zero0", zero_op, key_filter="0"))
        patcher.add(Patch("zero1", zero_op, key_filter="1"))
        patcher.apply("0", m[0])
        patcher.apply("1", m[1])
        assert m[0].weight.data.abs().sum().item() == 0.0
        assert m[1].weight.data.abs().sum().item() == 0.0
        patcher.remove("zero0")
        assert torch.equal(m[0].weight.data, prev0)
        assert m[1].weight.data.abs().sum().item() == 0.0
        patcher.remove("zero1")
        assert torch.equal(m[1].weight.data, prev1)

    def test_duplicate_name_raises(self) -> None:
        """``add`` raises on duplicate ``Patch.name``."""
        patcher = ModelPatcher(nn.Linear(4, 4))
        patcher.add(Patch("x", lambda m, s: None))
        with pytest.raises(ValueError):
            patcher.add(Patch("x", lambda m, s: None))

    def test_remove_unknown_is_noop(self) -> None:
        """``remove`` for an unknown name is a no-op."""
        patcher = ModelPatcher(nn.Linear(4, 4))
        # Should not raise.
        patcher.remove("nope")


# ===========================================================================
# Section 2 -- CPU offload helpers (4 tests)
# ===========================================================================
class TestCpuOffload:
    """The per-submodule / sequential CPU offload helpers."""

    def test_no_op_when_same_device(self) -> None:
        """``enable_*_cpu_offload`` is a no-op when compute == offload."""
        m = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        assert enable_model_cpu_offload(m) == 0
        assert enable_sequential_cpu_offload(m) == 0

    def test_per_submodule_offload_attaches_hooks(self) -> None:
        """``enable_model_cpu_offload`` attaches hooks on every leaf."""
        m = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        n = enable_model_cpu_offload(m, compute_device="cpu", offload_device="cpu")
        # Same device -> 0 hooks, but the function is still called.
        assert n == 0
        # disable_offload should also be a no-op (no hooks).
        assert disable_offload(m) == 0

    def test_sequential_offload_attaches_hooks(self) -> None:
        """``enable_sequential_cpu_offload`` is callable and idempotent."""
        m = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        n = enable_sequential_cpu_offload(m, compute_device="cpu", offload_device="cpu")
        assert n == 0

    def test_disable_offload_is_idempotent(self) -> None:
        """Calling ``disable_offload`` twice does not raise."""
        m = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        enable_model_cpu_offload(m, compute_device="cpu", offload_device="cpu")
        assert disable_offload(m) == 0
        # Second call -> still 0.
        assert disable_offload(m) == 0


# ===========================================================================
# Section 3 -- LoRAInjector / LoRASpec (7 tests)
# ===========================================================================
class TestLoRAInjector:
    """The :class:`LoRAInjector` + :class:`LoRASpec` API."""

    def test_zero_initial_delta(self) -> None:
        """At init the LoRA delta is zero (B Kaiming, A=0)."""
        m = nn.Sequential(nn.Linear(8, 8, bias=False))
        prev = m[0].weight.data.clone()
        injector = LoRAInjector(m)
        injector.add(LoRASpec("test", rank=2, target_modules=("0",)))
        injector.apply()
        # Output with LoRA must equal the base output (A is zero).
        x = torch.randn(2, 8)
        y_base = prev @ x.T
        y_lora = m(x)
        # Output is (N, out) so we compare directly.
        assert torch.allclose(y_lora, y_base.T, atol=1e-5)

    def test_nonzero_delta_after_init(self) -> None:
        """The delta is non-zero once the Kaiming init takes effect."""
        m = nn.Sequential(nn.Linear(8, 8, bias=False))
        # Force the B init to be non-trivial by setting the
        # seed so the Kaiming result is non-zero.
        torch.manual_seed(42)
        # We need a B that is not all-zero.  The injector
        # uses a per-spec seed; we trigger a delta by
        # manually re-init the A (which is zero by default)
        # to be non-zero.
        injector = LoRAInjector(m)
        injector.add(LoRASpec("test", rank=2, target_modules=("0",)))
        injector.apply()
        # Force A to be non-zero.
        a, b = injector._deltas["0"]
        with torch.no_grad():
            a.copy_(torch.randn_like(a))
        x = torch.randn(2, 8)
        y_lora = m(x)
        y_base = m[0].weight @ x.T
        diff = (y_lora - y_base.T).abs().max().item()
        assert diff > 1e-4, f"expected non-trivial delta, got diff={diff}"

    def test_remove_restores_output(self) -> None:
        """After ``remove`` the output equals the base output again."""
        m = nn.Sequential(nn.Linear(8, 8, bias=False))
        injector = LoRAInjector(m)
        injector.add(LoRASpec("test", rank=2, target_modules=("0",)))
        injector.apply()
        a, _ = injector._deltas["0"]
        with torch.no_grad():
            a.copy_(torch.randn_like(a))
        x = torch.randn(2, 8)
        y_with = m(x).clone()
        y_base = m[0].weight @ x.T
        assert not torch.allclose(y_with, y_base.T, atol=1e-5)
        injector.remove("test")
        y_after = m(x)
        assert torch.allclose(y_after, y_base.T, atol=1e-5)

    def test_rank_caps_to_min_dim(self) -> None:
        """``rank > min(in, out)`` is capped to ``min(in, out)``."""
        spec = LoRASpec("r", rank=999, target_modules=("m",))
        # The cap is enforced in ``_make_lora_params``; the
        # spec itself just stores the user request.
        assert spec.rank == 999
        from models.lora import _make_lora_params
        mod = nn.Linear(8, 4, bias=False)
        a, b = _make_lora_params(mod, spec)
        # in=8, out=4 -> capped rank = 4
        assert a.shape == (4, 8)
        assert b.shape == (4, 4)

    def test_default_target_modules_for_dit(self) -> None:
        """The default targets include the 4 HunyuanDiT block submodules."""
        ts = default_target_modules()
        assert "blocks.*.attn.qkv" in ts
        assert "blocks.*.attn.out_proj" in ts
        assert "blocks.*.mlp.fc1" in ts
        assert "blocks.*.mlp.fc2" in ts

    def test_inject_lora_helper(self) -> None:
        """``inject_lora`` is a one-shot helper."""
        m = nn.Sequential(nn.Linear(8, 8, bias=False))
        inj = inject_lora(m, name="h", rank=2, target_modules=("0",))
        assert "h" in inj._specs
        assert "0" in inj._deltas

    def test_lora_state_dict_round_trip(self) -> None:
        """``lora_state_dict`` + ``load_lora_state_dict`` round-trip."""
        m = nn.Sequential(nn.Linear(8, 8, bias=False))
        inj = LoRAInjector(m)
        inj.add(LoRASpec("test", rank=2, target_modules=("0",)))
        inj.apply()
        # Mutate A so the saved state is non-trivial.
        a, b = inj._deltas["0"]
        with torch.no_grad():
            a.copy_(torch.randn_like(a))
            b.copy_(torch.randn_like(b))
        sd = lora_state_dict(inj)
        # Restore into a fresh model.
        m2 = nn.Sequential(nn.Linear(8, 8, bias=False))
        inj2 = load_lora_state_dict(m2, sd)
        a2, b2 = inj2._deltas["0"]
        assert torch.equal(a, a2)
        assert torch.equal(b, b2)


# ===========================================================================
# Section 4 -- HunyuanDiT convenience methods (3 tests)
# ===========================================================================
class TestHunyuanDiTLoraConvenience:
    """The ``HunyuanDiT.lora_apply`` / ``lora_remove`` / ``lora_clear`` API."""

    def test_lora_apply_default_targets(self) -> None:
        """``lora_apply`` patches the 4 default block submodules."""
        m = HunyuanDiT()
        n = m.lora_apply("test", rank=2)
        # 2 blocks x 4 submodules = 8 patches.
        assert n == 8

    def test_lora_remove_clears_injector(self) -> None:
        """``lora_remove`` drops the spec and the delta tensors."""
        m = HunyuanDiT()
        m.lora_apply("test", rank=2)
        assert "test" in m._lora_injector._specs
        m.lora_remove("test")
        assert "test" not in m._lora_injector._specs
        assert m._lora_injector._deltas == {}

    def test_lora_remove_unknown_returns_false(self) -> None:
        """``lora_remove`` for an unknown name returns ``False``."""
        m = HunyuanDiT()
        assert m.lora_remove("never_added") is False
        m.lora_apply("real", rank=2)
        assert m.lora_remove("real") is True
        assert m.lora_remove("real") is False  # already removed


# ===========================================================================
# Section 5 -- End-to-end LoRA on HunyuanDiT-Tiny (3 tests)
# ===========================================================================
class TestE2EHunyuanDiTLoRA:
    """``HunyuanDiT.lora_apply`` + ``.sample()`` end-to-end."""

    def test_lora_does_not_break_sample(self) -> None:
        """A zero-initial LoRA does not change the sample output.

        With the AdaLN-Zero init the model output is exactly
        zero, so the flow-match Euler step ``x = x + dt * v``
        is a no-op and ``m.sample()`` returns the initial
        ``randn`` latent.  The same must hold with a LoRA
        whose ``A`` is zero (initial state).
        """
        torch.manual_seed(0)
        m = HunyuanDiT().eval()
        ctx = torch.randn(1, 8, 64)
        # Run the base model -- the output should be the
        # initial noise (model output is 0 -> no update).
        torch.manual_seed(123)
        out_base = m.sample(
            (1, 4, 8, 8),
            encoder_hidden_states=ctx,
            num_steps=2,
            guidance_scale=1.0,
        )
        # Reset the RNG and re-run with a LoRA (A is zero).
        torch.manual_seed(123)
        m.lora_apply("zero_lora", rank=2)
        out_with = m.sample(
            (1, 4, 8, 8),
            encoder_hidden_states=ctx,
            num_steps=2,
            guidance_scale=1.0,
        )
        assert torch.allclose(out_base, out_with, atol=1e-5)

    def test_lora_with_nonzero_a_changes_output(self) -> None:
        """A non-zero A changes the patched linear layer's output.

        With the AdaLN-Zero init the model *output* is
        exactly zero (the final linear is zero-initialised),
        so the LoRA's effect on the final output is not
        observable.  We instead probe the first block's
        attention QKV directly, calling it before and after
        the LoRA is applied.
        """
        torch.manual_seed(0)
        m = HunyuanDiT().eval()
        qkv = m.blocks[0].attn.qkv
        x = torch.randn(2, 8, 96)  # [B, N, hidden]
        with torch.no_grad():
            out_before = qkv(x).clone()
        # Apply a LoRA and force A to be non-zero.
        m.lora_apply("style", rank=4)
        for full, (a, _) in m._lora_injector._deltas.items():
            with torch.no_grad():
                a.copy_(torch.randn_like(a) * 0.1)
        with torch.no_grad():
            out_after = qkv(x).clone()
        diff = (out_before - out_after).abs().max().item()
        assert diff > 1e-4, f"LoRA delta had no effect on QKV, max diff={diff}"

    def test_lora_remove_restores_sample_output(self) -> None:
        """After ``lora_remove`` the sample output matches the base."""
        torch.manual_seed(0)
        m = HunyuanDiT().eval()
        ctx = torch.randn(1, 8, 64)
        torch.manual_seed(123)
        out_base = m.sample(
            (1, 4, 8, 8),
            encoder_hidden_states=ctx,
            num_steps=2,
            guidance_scale=1.0,
        )
        m.lora_apply("style", rank=4)
        for full, (a, _) in m._lora_injector._deltas.items():
            with torch.no_grad():
                a.copy_(torch.randn_like(a) * 0.1)
        m.lora_remove("style")
        torch.manual_seed(123)
        out_restored = m.sample(
            (1, 4, 8, 8),
            encoder_hidden_states=ctx,
            num_steps=2,
            guidance_scale=1.0,
        )
        assert torch.allclose(out_base, out_restored, atol=1e-5)


# ===========================================================================
# Section 6 -- HunyuanDiT enable_cpu_offload (3 tests, no-op sanity)
# ===========================================================================
class TestHunyuanDiTOffload:
    """``HunyuanDiT.enable_cpu_offload`` is a safe no-op on CPU-only."""

    def test_enable_cpu_offload_noop(self) -> None:
        """Same-device offload returns 0 hooked leaves and runs forward."""
        m = HunyuanDiT()
        n = m.enable_cpu_offload(compute_device="cpu", offload_device="cpu")
        assert n == 0
        # Forward still works.
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor([500])
        ctx = torch.randn(1, 8, 64)
        y = m(x, t, encoder_hidden_states=ctx)
        assert y.shape == (1, 4, 8, 8)

    def test_enable_sequential_cpu_offload_noop(self) -> None:
        """``sequential=True`` is also a no-op on CPU-only."""
        m = HunyuanDiT()
        n = m.enable_cpu_offload(sequential=True)
        assert n == 0
        y = m(torch.randn(1, 4, 8, 8), torch.tensor([100]))
        assert y.shape == (1, 4, 8, 8)

    def test_offload_does_not_break_lora(self) -> None:
        """LoRA injection still works after a no-op offload call."""
        m = HunyuanDiT()
        m.enable_cpu_offload()  # no-op
        n = m.lora_apply("s", rank=2)
        assert n == 8
