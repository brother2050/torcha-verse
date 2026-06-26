"""Tests for v1.0.0 experiments/framework.py (8 tests)."""
from __future__ import annotations

import math

import pytest

from experiments.framework import (
    Experiment,
    ExperimentRunner,
    Variant,
    bucket_assign,
)


def _make_experiment(weights=None) -> Experiment:
    if weights is None:
        weights = [1.0, 1.0]
    variants = [
        Variant(name=f"v{i}", weight=w, config={"i": i})
        for i, w in enumerate(weights)
    ]
    return Experiment(name="exp_test", variants=variants, primary_metric="ctr")


# ---------------------------------------------------------------------------
# 31
# ---------------------------------------------------------------------------
def test_variant_creation():
    v = Variant(name="control", weight=2.0, config={"prompt": "old"})
    assert v.name == "control"
    assert v.weight == 2.0
    assert v.config == {"prompt": "old"}


# ---------------------------------------------------------------------------
# 32
# ---------------------------------------------------------------------------
def test_experiment_creation():
    exp = _make_experiment([1.0, 1.0])
    assert exp.name == "exp_test"
    assert exp.primary_metric == "ctr"
    assert len(exp.variants) == 2
    assert exp.variants[0].name == "v0"
    assert exp.variants[1].name == "v1"


# ---------------------------------------------------------------------------
# 33 - 10000 users, equal weight -> near-50/50 within 1%
# ---------------------------------------------------------------------------
def test_bucket_assign_equal_weight():
    exp = _make_experiment([1.0, 1.0])
    counts = {v.name: 0 for v in exp.variants}
    n_users = 10000
    for i in range(n_users):
        v = bucket_assign(f"user-{i}", exp.variants)
        counts[v.name] += 1
    expected = n_users / 2
    for name, count in counts.items():
        assert abs(count - expected) / expected < 0.01, (
            f"variant {name!r} got {count}/{n_users} "
            f"({count / n_users:.4f}); expected ~0.5"
        )


# ---------------------------------------------------------------------------
# 34 - 20000 users, weight 1:2 -> 33/67 within 2%
# ---------------------------------------------------------------------------
def test_bucket_assign_unequal_weight():
    exp = _make_experiment([1.0, 2.0])
    counts = {v.name: 0 for v in exp.variants}
    n_users = 20000
    for i in range(n_users):
        v = bucket_assign(f"user-{i}", exp.variants)
        counts[v.name] += 1
    total = sum(counts.values())
    ratios = {name: count / total for name, count in counts.items()}
    # The first variant has weight 1, second has weight 2 -> ~1/3, ~2/3.
    assert abs(ratios["v0"] - 1 / 3) < 0.02
    assert abs(ratios["v1"] - 2 / 3) < 0.02


# ---------------------------------------------------------------------------
# 35 - same user always lands in the same variant
# ---------------------------------------------------------------------------
def test_bucket_assign_stable_per_user():
    exp = _make_experiment([1.0, 1.0, 1.0])
    for uid in ("alice", "bob", "carol", "dave", "eve"):
        first = bucket_assign(uid, exp.variants)
        for _ in range(20):
            assert bucket_assign(uid, exp.variants).name == first.name


# ---------------------------------------------------------------------------
# 36
# ---------------------------------------------------------------------------
def test_runner_pick():
    exp = _make_experiment([1.0, 1.0])
    runner = ExperimentRunner(exp)
    variant = runner.pick("user-42")
    assert isinstance(variant, Variant)
    assert variant.name in {"v0", "v1"}


# ---------------------------------------------------------------------------
# 37 - record a metric, summary includes mean / count / std
# ---------------------------------------------------------------------------
def test_runner_record_and_summary():
    # Use a heavily skewed weight so all four sample users land in v0.
    exp = _make_experiment([1000.0, 1.0])
    runner = ExperimentRunner(exp)
    for i, value in enumerate([0.1, 0.2, 0.3, 0.4]):
        runner.record(f"user-{i}", "ctr", value)
    summary = runner.summary()
    metrics = summary["v0"]["ctr"]
    assert metrics["count"] == 4
    assert math.isclose(metrics["mean"], 0.25, rel_tol=1e-9)
    # population std dev of [0.1, 0.2, 0.3, 0.4] is ~0.1118
    assert metrics["std"] > 0.0
    # All three keys must be present.
    assert {"mean", "count", "std"}.issubset(set(metrics.keys()))


# ---------------------------------------------------------------------------
# 38 - 100 users over 3 variants, summary covers all variants
# ---------------------------------------------------------------------------
def test_runner_multi_variant():
    exp = _make_experiment([1.0, 1.0, 1.0])
    runner = ExperimentRunner(exp)
    for i in range(100):
        runner.record(f"user-{i}", "ctr", float(i % 5) / 5.0)
    summary = runner.summary()
    assert set(summary.keys()) == {"v0", "v1", "v2"}
    for vname in ("v0", "v1", "v2"):
        assert "ctr" in summary[vname]
        assert "count" in summary[vname]["ctr"]
        assert "mean" in summary[vname]["ctr"]
        assert "std" in summary[vname]["ctr"]
    total = sum(summary[v]["ctr"]["count"] for v in ("v0", "v1", "v2"))
    assert total == 100
