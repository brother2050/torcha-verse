"""Tests for the v0.4.3 prometheus_client swap-in helpers.

* :func:`is_prometheus_client_available` returns a bool (no exception).
* :func:`export_to_prometheus_client` raises a clear ``ImportError``
  with install instructions when the optional dep is missing.
* When ``prometheus_client`` is installed, the function mirrors
  counters / gauges into a fresh ``CollectorRegistry`` so the
  caller's process-global registry is left untouched.
"""

from __future__ import annotations

import pytest

from infrastructure.metrics import (
    METRICS,
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    export_to_prometheus_client,
    is_prometheus_client_available,
)


@pytest.fixture
def fresh_registry() -> MetricsRegistry:
    reg = MetricsRegistry()
    yield reg
    reg.clear()


# ---------------------------------------------------------------------------
# is_prometheus_client_available
# ---------------------------------------------------------------------------
def test_is_prometheus_client_available_returns_bool() -> None:
    result = is_prometheus_client_available()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# export_to_prometheus_client - ImportError path
# ---------------------------------------------------------------------------
def test_export_to_prometheus_client_raises_when_missing(
    monkeypatch, fresh_registry: MetricsRegistry
) -> None:
    """When the optional dep is not installed, the helper must raise ``ImportError``
    with a clear install hint rather than a bare ``ModuleNotFoundError``.
    """
    # Easiest reliable path: monkeypatch ``is_prometheus_client_available``
    # to return True (so the function does not bail early on the
    # "optional not installed" short-circuit) and then make the
    # actual ``import prometheus_client`` inside the helper fail by
    # pointing the cached module at a stub that raises on use.
    monkeypatch.setattr(
        "infrastructure.metrics.is_prometheus_client_available",
        lambda: True,
    )
    import sys
    import types

    fake_mod = types.ModuleType("prometheus_client")

    def _raise_on_use(*args, **kwargs):  # pragma: no cover - error path only
        raise ModuleNotFoundError(
            "No module named 'prometheus_client' (simulated)"
        )

    fake_mod.__getattr__ = _raise_on_use  # type: ignore[attr-defined]
    fake_mod.Counter = _raise_on_use  # type: ignore[attr-defined]
    fake_mod.Gauge = _raise_on_use  # type: ignore[attr-defined]
    fake_mod.Histogram = _raise_on_use  # type: ignore[attr-defined]
    fake_mod.CollectorRegistry = _raise_on_use  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prometheus_client", fake_mod)
    with pytest.raises(ImportError, match="pip install"):
        export_to_prometheus_client(fresh_registry)


# ---------------------------------------------------------------------------
# export_to_prometheus_client - happy path
# ---------------------------------------------------------------------------
def test_export_to_prometheus_client_mirrors_counters(
    fresh_registry: MetricsRegistry,
) -> None:
    # Only run the happy path if prometheus_client is actually
    # installed; otherwise the negative test above already covers
    # the missing-dep behaviour.
    if not is_prometheus_client_available():
        pytest.skip("prometheus_client is not installed in this env")
    fresh_registry.counter("reqs_total", "Total", ("route",)).inc("/a")
    fresh_registry.counter("reqs_total", "Total", ("route",)).inc("/a", amount=4)
    result = export_to_prometheus_client(fresh_registry)
    assert "reqs_total" in result["counters"]
    rendered = result["registry"]
    body = rendered.collect()
    metric_families = {family.name: family for family in body}
    assert "reqs_total" in metric_families


def test_export_to_prometheus_client_mirrors_gauges(
    fresh_registry: MetricsRegistry,
) -> None:
    if not is_prometheus_client_available():
        pytest.skip("prometheus_client is not installed in this env")
    gauge: Gauge = fresh_registry.gauge("depth", "Depth", ("queue",))
    gauge.set("alpha", value=5.0)
    gauge.set("beta", value=3.0)
    result = export_to_prometheus_client(fresh_registry)
    rendered = result["registry"].collect()
    names = {family.name for family in rendered}
    assert "depth" in names


def test_export_to_prometheus_client_mirrors_histograms(
    fresh_registry: MetricsRegistry,
) -> None:
    if not is_prometheus_client_available():
        pytest.skip("prometheus_client is not installed in this env")
    histogram: Histogram = fresh_registry.histogram(
        "lat", "Latency", ("route",), buckets=(0.1, 1.0)
    )
    histogram.observe("/x", value=0.05)
    histogram.observe("/x", value=0.5)
    result = export_to_prometheus_client(fresh_registry)
    rendered = result["registry"].collect()
    names = {family.name for family in rendered}
    assert "lat" in names


def test_export_to_prometheus_client_does_not_mutate_global(
    fresh_registry: MetricsRegistry,
) -> None:
    if not is_prometheus_client_available():
        pytest.skip("prometheus_client is not installed in this env")
    # Snapshot the metric names in the v0.4.x registry so we can
    # confirm the function leaves them alone.
    before = set(fresh_registry.names())
    export_to_prometheus_client(fresh_registry)
    after = set(fresh_registry.names())
    assert before == after
    # The ``METRICS`` global must not have been touched either.
    assert "reqs_total" not in METRICS
