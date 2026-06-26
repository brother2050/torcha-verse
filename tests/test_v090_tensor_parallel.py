"""v0.9.0 minimal tensor-parallel shard tests (3 tests).

Tests the v0.9.0 :func:`infrastructure.device_manager._minimal_tensor_parallel_shard`
helper.  This helper is the v0.9.0 **acceptance test target** for the
tensor-parallel story: it splits an :class:`nn.Linear`'s output
dimension into ``num_shards`` equal parts and concatenates the
per-shard outputs at forward time so the sharded computation
matches the unsharded one on CPU.

Real multi-GPU TP (NCCL all-reduce, gather, broadcast) ships in a
later milestone; for now we only need to assert the local contract
on CPU.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from infrastructure.device_manager import _minimal_tensor_parallel_shard


# ---------------------------------------------------------------------------
# 1 - even split + concatenating the shards matches the original
# ---------------------------------------------------------------------------
def test_shard_splits_last_dim_evenly() -> None:
    """Sharding an :class:`nn.Linear` with ``out_features=8`` into
    ``num_shards=2`` produces two sub-linears with ``out_features=4``
    each, and ``shard(x)`` reproduces the original ``linear(x)``.
    """
    torch.manual_seed(0)
    in_features = 6
    out_features = 8
    num_shards = 2
    expected_chunk = out_features // num_shards  # 4

    linear = nn.Linear(in_features, out_features, bias=True)
    # Sanity: the original linear really has 8 output units.
    assert linear.weight.shape == (out_features, in_features)
    assert linear.bias is not None and linear.bias.shape == (out_features,)

    sharded = _minimal_tensor_parallel_shard(linear, num_shards)
    # The sharded wrapper must hold ``num_shards`` sub-linears.
    assert hasattr(sharded, "shards"), "sharded module must expose a .shards attribute"
    assert len(sharded.shards) == num_shards

    # Each sub-linear must have ``out_features=4`` (and unchanged
    # ``in_features``).
    for sub in sharded.shards:
        assert isinstance(sub, nn.Linear)
        assert sub.weight.shape == (expected_chunk, in_features)
        if sub.bias is not None:
            assert sub.bias.shape == (expected_chunk,)

    # Concatenating the per-shard weights along the output dim must
    # match the original weight tensor exactly (the helper copies the
    # slice with ``torch.no_grad`` + ``copy_``).
    concat_w = torch.cat([sub.weight for sub in sharded.shards], dim=0)
    assert torch.equal(concat_w, linear.weight)
    if linear.bias is not None:
        concat_b = torch.cat([sub.bias for sub in sharded.shards], dim=0)
        assert torch.equal(concat_b, linear.bias)

    # Forward equivalence: a random input must produce the same output
    # through the sharded wrapper as through the original linear.
    x = torch.randn(3, in_features)
    with torch.no_grad():
        y_orig = linear(x)
        y_shard = sharded(x)
    assert y_orig.shape == y_shard.shape == (3, out_features)
    assert torch.allclose(y_orig, y_shard, atol=1e-6)


# ---------------------------------------------------------------------------
# 2 - uneven division (10 % 3 != 0) is rejected with ValueError
# ---------------------------------------------------------------------------
def test_shard_handles_uneven_division() -> None:
    """When ``out_features`` is not divisible by ``num_shards``, the
    shard helper must raise :class:`ValueError` rather than silently
    dropping or duplicating output rows.  This protects callers from
    getting an inconsistent shape downstream.
    """
    torch.manual_seed(1)
    in_features = 4
    out_features = 10  # 10 % 3 = 1 -- the leftover case
    num_shards = 3

    linear = nn.Linear(in_features, out_features, bias=True)
    with pytest.raises(ValueError, match="not divisible by num_shards"):
        _minimal_tensor_parallel_shard(linear, num_shards)


# ---------------------------------------------------------------------------
# 3 - world_size=1 is the identity (no sharding)
# ---------------------------------------------------------------------------
def test_shard_with_world_size_one_is_identity() -> None:
    """When ``num_shards=1`` the helper must return the *original*
    ``linear`` unchanged.  This is the world-size-1 fast path: a
    single-GPU process should not pay the cost of an extra
    ``nn.Module`` wrapper.
    """
    torch.manual_seed(2)
    in_features = 5
    out_features = 7

    linear = nn.Linear(in_features, out_features, bias=True)
    out = _minimal_tensor_parallel_shard(linear, num_shards=1)

    # Identity: the very same object is returned (no sharding wrapper
    # was constructed).
    assert out is linear, (
        "num_shards=1 should be a no-op identity fast path; got a "
        f"different object of type {type(out).__name__}"
    )
