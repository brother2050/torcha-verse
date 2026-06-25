"""Tests for the v0.4.2 stdlib metrics facade.

Covers:

* Counter / Gauge / Histogram construction, label validation, type
  uniqueness and idempotent ``get`` semantics on the registry.
* Prometheus text exposition rendering, including the ``+Inf`` bucket
  and the ``_sum`` / ``_count`` synthetic series emitted by
  :class:`Histogram`.
* Float formatting (integer, float, ``+Inf`` / ``-Inf`` / ``NaN``).
"""

from __future__ import annotations

import math
import re

import pytest

from infrastructure.metrics import (
    Counter,
    Gauge,
    Histogram,
    METRICS,
    MetricsRegistry,
    render_prometheus,
)


@pytest.fixture
def fresh_registry() -> MetricsRegistry:
    """Return a clean :class:`MetricsRegistry` for each test."""
    reg = MetricsRegistry()
    yield reg
    reg.clear()


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------
def test_counter_inc_with_labels(fresh_registry: MetricsRegistry) -> None:
    counter = fresh_registry.counter(
        "api_requests_total", "Total API requests", ("route", "status")
    )
    counter.inc("/healthz", "200")
    counter.inc("/healthz", "200", amount=2)
    counter.inc("/chat", "200")
    assert counter._series[counter._key(("/healthz", "200"))] == 3.0
    assert counter._series[counter._key(("/chat", "200"))] == 1.0


def test_counter_rejects_negative_increment(
    fresh_registry: MetricsRegistry,
) -> None:
    counter = fresh_registry.counter("down", "Down counter")
    with pytest.raises(ValueError, match="cannot be decreased"):
        counter.inc(amount=-1.0)


def test_counter_label_count_mismatch(
    fresh_registry: MetricsRegistry,
) -> None:
    counter = fresh_registry.counter("labeled", "Labeled counter", ("a", "b"))
    with pytest.raises(ValueError, match="expects 2 labels"):
        counter.inc("only-one")


def test_counter_requires_nonempty_help_and_name(
    fresh_registry: MetricsRegistry,
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        fresh_registry.counter("", "x")
    with pytest.raises(ValueError, match="non-empty"):
        fresh_registry.counter("x", "")
    with pytest.raises(ValueError, match=r"\[a-zA-Z_\]"):
        fresh_registry.counter("bad name", "x")


def test_counter_duplicate_registration_returns_existing(
    fresh_registry: MetricsRegistry,
) -> None:
    first = fresh_registry.counter("dup", "Dup")
    second = fresh_registry.counter("dup", "Dup")
    assert first is second
    with pytest.raises(TypeError, match="different type"):
        fresh_registry.gauge("dup", "Dup")


def test_counter_duplicate_label_names(
    fresh_registry: MetricsRegistry,
) -> None:
    with pytest.raises(ValueError, match="duplicate label names"):
        fresh_registry.counter("dup_label", "Dup label", ("a", "a"))


# ---------------------------------------------------------------------------
# Gauge
# ---------------------------------------------------------------------------
def test_gauge_set_inc_dec(fresh_registry: MetricsRegistry) -> None:
    gauge = fresh_registry.gauge("queue_depth", "Queue depth", ("queue",))
    gauge.set("alpha", value=5.0)
    gauge.inc("alpha", amount=2.0)
    assert gauge._series[gauge._key(("alpha",))] == 7.0
    gauge.dec("alpha", amount=3.0)
    assert gauge._series[gauge._key(("alpha",))] == 4.0
    # ``inc`` with a negative amount is allowed for gauges.
    gauge.inc("alpha", amount=-4.0)
    assert gauge._series[gauge._key(("alpha",))] == 0.0


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------
def test_histogram_observe_and_buckets(fresh_registry: MetricsRegistry) -> None:
    histogram = fresh_registry.histogram(
        "latency_seconds", "Request latency", ("route",), buckets=(0.1, 0.5, 1.0)
    )
    histogram.observe("/a", value=0.05)
    histogram.observe("/a", value=0.6)
    histogram.observe("/a", value=5.0)  # falls into the +Inf bucket
    state = histogram._series_buckets[histogram._key(("/a",))]
    # [sum, count, b_0.1, b_0.5, b_1.0, b_+Inf]
    assert state[1] == 3.0
    assert state[0] == pytest.approx(5.65)
    assert state[2] == 1.0  # 0.05 <= 0.1
    assert state[3] == 1.0  # 0.05 <= 0.5
    assert state[4] == 2.0  # 0.05, 0.6 <= 1.0
    assert state[5] == 3.0  # everything <= +Inf


def test_histogram_appends_inf_bucket(fresh_registry: MetricsRegistry) -> None:
    histogram = fresh_registry.histogram(
        "h", "h", buckets=(1.0, 2.0)
    )
    assert histogram.buckets[-1] == float("inf")
    assert len(histogram.buckets) == 3


# ---------------------------------------------------------------------------
# Prometheus rendering
# ---------------------------------------------------------------------------
def test_render_counter_unlabeled(fresh_registry: MetricsRegistry) -> None:
    fresh_registry.counter("hits_total", "Hits").inc()
    text = fresh_registry.render()
    assert "# HELP hits_total Hits" in text
    assert "# TYPE hits_total counter" in text
    # ``_format_float`` canonicalises integer-valued floats to "1".
    assert re.search(r"^hits_total\s+1$", text, re.MULTILINE)


def test_render_gauge_with_labels(fresh_registry: MetricsRegistry) -> None:
    gauge = fresh_registry.gauge("mem", "Memory", ("kind",))
    gauge.set("vram", value=24.0)
    gauge.set("ram", value=64.0)
    text = fresh_registry.render()
    assert 'mem{kind="vram"} 24' in text
    assert 'mem{kind="ram"} 64' in text


def test_render_histogram_emits_buckets_sum_count(
    fresh_registry: MetricsRegistry,
) -> None:
    histogram = fresh_registry.histogram(
        "lat", "Latency", ("route",), buckets=(0.5, 1.0)
    )
    histogram.observe("/x", value=0.25)
    histogram.observe("/x", value=0.75)
    text = fresh_registry.render()
    assert 'lat_bucket{route="/x",le="0.5"} 1' in text
    assert 'lat_bucket{route="/x",le="1"} 2' in text
    assert 'lat_bucket{route="/x",le="+Inf"} 2' in text
    assert 'lat_sum{route="/x"} 1' in text
    assert 'lat_count{route="/x"} 2' in text


def test_render_handles_special_floats() -> None:
    from infrastructure.metrics import _format_float

    assert _format_float(float("inf")) == "+Inf"
    assert _format_float(float("-inf")) == "-Inf"
    assert _format_float(float("nan")) == "NaN"
    assert _format_float(5.0) == "5"
    assert _format_float(5.5) == "5.5"


def test_render_label_escaping(fresh_registry: MetricsRegistry) -> None:
    counter = fresh_registry.counter("c", "c", ("k",))
    counter.inc('val"with\\special\n')
    text = fresh_registry.render()
    # Backslash, double-quote and newline must be escaped in the
    # exposition format.
    assert 'val\\"with\\\\special\\n' in text


def test_render_empty_registry_returns_trailing_newline() -> None:
    reg = MetricsRegistry()
    assert reg.render() == ""


def test_global_METRICS_singleton() -> None:
    """The :data:`METRICS` singleton is shared across imports."""
    assert isinstance(METRICS, MetricsRegistry)
    a = METRICS.counter("a_unique_metric_xyz", "x")
    b = METRICS.counter("a_unique_metric_xyz", "x")
    assert a is b
    # Cleanup so we don't pollute other tests' state.
    METRICS.clear()


def test_registry_len_and_contains(fresh_registry: MetricsRegistry) -> None:
    fresh_registry.counter("a", "a")
    fresh_registry.gauge("b", "b")
    assert len(fresh_registry) == 2
    assert "a" in fresh_registry
    assert "missing" not in fresh_registry


def test_render_unknown_metric_type_does_not_crash(
    fresh_registry: MetricsRegistry,
) -> None:
    # The render path only handles Counter / Gauge / Histogram, but
    # unknown entries should be skipped silently rather than raising.
    text = fresh_registry.render()
    assert text == ""
