"""v1.0.0 prometheus bridge tests (6 tests).

Tests the optional :mod:`infrastructure.prometheus_bridge` module that
connects TorchaVerse's hand-rolled :class:`~infrastructure.metrics.MetricsRegistry`
to the (optional) :mod:`prometheus_client` SDK.

The module must work whether or not ``prometheus_client`` is installed:

* The snapshot helpers (:func:`snapshot_counter_values`,
  :func:`snapshot_gauge_values`, :func:`snapshot_histogram_count`)
  work without any external dependency and are tested unconditionally.
* :func:`bridge_to_prometheus_client` is exercised only when
  ``prometheus_client`` is importable; otherwise the test skips
  cleanly with :func:`pytest.importorskip`.

These tests do **not** depend on a running Prometheus server -- the
bridge is in-process only and never opens a HTTP port.
"""
from __future__ import annotations

import pytest

from infrastructure.metrics import MetricsRegistry
from infrastructure.prometheus_bridge import (
    PROMETHEUS_CLIENT_AVAILABLE,
    bridge_to_prometheus_client,
    snapshot_counter_values,
    snapshot_gauge_values,
    snapshot_histogram_count,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fresh_registry() -> MetricsRegistry:
    """Return a clean :class:`MetricsRegistry` for each test."""
    reg = MetricsRegistry()
    yield reg
    reg.clear()


# ---------------------------------------------------------------------------
# 1 - Counter increment / read-back on a fresh registry
# ---------------------------------------------------------------------------
def test_metrics_registry_basic_counters(fresh_registry: MetricsRegistry) -> None:
    """A fresh :class:`MetricsRegistry` lets you build a counter, increment
    it N times, and read the value back.  This is the contract the
    :func:`bridge_to_prometheus_client` snapshot relies on.
    """
    counter = fresh_registry.counter(
        "requests_total", "Total requests", ("route",)
    )
    n_increments = 5
    for _ in range(n_increments):
        counter.inc("/healthz")

    # The counter stores its series under a ``_LabelKey``; locate it via
    # the helper :meth:`Counter._key` so the test does not depend on the
    # private ``_series`` map's key type.
    series_value = counter._series[counter._key(("/healthz",))]
    assert series_value == pytest.approx(float(n_increments))
    # ``snapshot_counter_values`` must report the same total.
    assert snapshot_counter_values(fresh_registry) == {
        "requests_total": pytest.approx(float(n_increments)),
    }


# ---------------------------------------------------------------------------
# 2 - bridge_to_prometheus_client (skipped when prometheus_client missing)
# ---------------------------------------------------------------------------
def test_bridge_to_prometheus_client_creates_counter(
    fresh_registry: MetricsRegistry,
) -> None:
    """When ``prometheus_client`` is importable, calling
    :func:`bridge_to_prometheus_client` mirrors every counter in the
    in-process :class:`MetricsRegistry` into a fresh
    ``prometheus_client.CollectorRegistry``.

    When ``prometheus_client`` is **not** importable the test skips
    cleanly -- the bridge is documented as optional and the rest of
    TorchaVerse is expected to keep working in that case.
    """
    prometheus_client = pytest.importorskip("prometheus_client")

    counter = fresh_registry.counter(
        "bridge_test_total", "Bridge test counter", ("label",)
    )
    counter.inc("alpha")
    counter.inc("alpha", amount=2.0)
    counter.inc("beta")

    target = bridge_to_prometheus_client(fresh_registry)
    assert target is not None, (
        "bridge_to_prometheus_client returned None -- "
        "prometheus_client should be importable inside an importorskip gate"
    )

    # The bridge must have registered the counter (with the "tv_"
    # namespace prefix applied by the module) on the target registry.
    expected_name = "tv_counter_bridge_test_total"
    collected = list(target.collect())
    counter_metrics = [m for m in collected if m.name == expected_name]
    assert counter_metrics, (
        f"Expected counter {expected_name!r} in the target registry; "
        f"saw {[m.name for m in collected]!r}"
    )

    # Confirm the bridged value is the total the registry sees.
    total_value = sum(
        sample.value for sample in counter_metrics[0].samples
        if sample.name.endswith("_total") or sample.name == expected_name
    )
    # ``alpha`` saw 3 increments, ``beta`` saw 1 -> total 4.
    assert total_value == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# 3 - snapshot_counter_values returns a flat dict
# ---------------------------------------------------------------------------
def test_snapshot_counter_values(fresh_registry: MetricsRegistry) -> None:
    """Add 3 counters with different label values, increment each, and
    confirm :func:`snapshot_counter_values` returns a dict with the
    right keys and aggregated totals.
    """
    a = fresh_registry.counter("a_total", "Counter A", ("kind",))
    b = fresh_registry.counter("b_total", "Counter B", ("kind",))
    c = fresh_registry.counter("c_total", "Counter C")  # no labels

    a.inc("x")
    a.inc("x")
    a.inc("y")
    b.inc("x", amount=4.0)
    c.inc()  # unlabeled -> 1.0

    snap = snapshot_counter_values(fresh_registry)
    assert isinstance(snap, dict)
    assert set(snap.keys()) == {"a_total", "b_total", "c_total"}
    # The flat view sums every series for a given counter.
    assert snap["a_total"] == pytest.approx(3.0)  # 2 + 1
    assert snap["b_total"] == pytest.approx(4.0)
    assert snap["c_total"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4 - snapshot_gauge_values returns a flat dict
# ---------------------------------------------------------------------------
def test_snapshot_gauge_values(fresh_registry: MetricsRegistry) -> None:
    """Add 3 gauges with various values; the snapshot is a flat
    ``{name: value}`` dict (gauges with multiple label sets are
    flattened to the first observed value by contract).
    """
    fresh_registry.gauge("queue_depth", "Queue depth", ("queue",))
    fresh_registry.gauge("cpu_pct", "CPU %")
    fresh_registry.gauge("latency_ms", "Latency", ("route",))

    g_queue = fresh_registry.get("queue_depth")
    g_cpu = fresh_registry.get("cpu_pct")
    g_latency = fresh_registry.get("latency_ms")
    g_queue.set("alpha", value=10.0)  # type: ignore[union-attr]
    g_queue.set("beta", value=20.0)   # type: ignore[union-attr]
    g_cpu.set(value=42.5)             # type: ignore[union-attr]
    g_latency.set("/a", value=15.0)   # type: ignore[union-attr]
    g_latency.set("/b", value=25.0)   # type: ignore[union-attr]

    snap = snapshot_gauge_values(fresh_registry)
    assert set(snap.keys()) == {"queue_depth", "cpu_pct", "latency_ms"}
    # The contract flattens to the first observed value per gauge.
    assert snap["cpu_pct"] == pytest.approx(42.5)
    # For "queue_depth" the dict iteration order in CPython 3.7+ is
    # insertion order, so the first value is the one we set first
    # (alpha -> 10.0).  We only assert it is one of the values we set.
    assert snap["queue_depth"] in (10.0, 20.0)
    assert snap["latency_ms"] in (15.0, 25.0)


# ---------------------------------------------------------------------------
# 5 - snapshot_histogram_count returns sample count
# ---------------------------------------------------------------------------
def test_snapshot_histogram_count(fresh_registry: MetricsRegistry) -> None:
    """Build a histogram, observe N values, and confirm
    :func:`snapshot_histogram_count` returns a dict mapping the
    histogram name to its total sample count.
    """
    hist = fresh_registry.histogram(
        "latency_seconds", "Request latency", ("route",),
        buckets=(0.1, 0.5, 1.0),
    )
    hist.observe("/a", value=0.05)
    hist.observe("/a", value=0.6)
    hist.observe("/a", value=5.0)
    hist.observe("/b", value=0.1)

    snap = snapshot_histogram_count(fresh_registry)
    assert isinstance(snap, dict)
    assert snap == {"latency_seconds": 4}


# ---------------------------------------------------------------------------
# 6 - bridge gracefully ignores unknown metric kinds
# ---------------------------------------------------------------------------
def test_bridge_handles_unknown_metric_kinds(
    fresh_registry: MetricsRegistry,
) -> None:
    """The bridge's dispatch (Counter / Gauge / Histogram) must skip
    unknown metric kinds without raising.  We inject a custom metric
    class that does not match any of the known kinds and verify the
    bridge (a) does not raise and (b) still processes the well-known
    Counter sibling we registered alongside it.
    """
    # Inject an unknown-kind metric directly into the registry.
    class _UnknownMetric:
        """Stand-in for a future metric kind (e.g. Summary) that the
        bridge does not yet know how to mirror."""

        name = "future_summary_total"

        def __init__(self) -> None:
            self._help = "A metric the bridge does not know about"
            self.labelnames: tuple = ()
            self._series = {}
            self._series_buckets = {}

    fresh_registry._metrics["future_summary_total"] = _UnknownMetric()  # type: ignore[assignment]

    # Add a real Counter alongside the unknown kind.
    counter = fresh_registry.counter(
        "known_total", "Known counter", ("k",)
    )
    counter.inc("a")
    counter.inc("a", amount=2.0)

    if PROMETHEUS_CLIENT_AVAILABLE:
        prometheus_client = pytest.importorskip("prometheus_client")
        # Must not raise -- the unknown metric should be skipped.
        target = bridge_to_prometheus_client(fresh_registry)
        assert target is not None
        # The known counter must still have made it into the target.
        mirror_names = {m.name for m in target.collect()}
        assert "tv_counter_known_total" in mirror_names
        # The unknown metric must NOT be in the target.
        assert "tv_counter_future_summary_total" not in mirror_names
        assert "tv_gauge_future_summary_total" not in mirror_names
        assert "tv_histogram_future_summary_total" not in mirror_names
    else:
        # When ``prometheus_client`` is not installed the bridge is a
        # no-op and returns ``None`` -- still must not raise.
        result = bridge_to_prometheus_client(fresh_registry)
        assert result is None
