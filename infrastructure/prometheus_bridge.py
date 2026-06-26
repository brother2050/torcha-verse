"""Optional ``prometheus_client`` SDK bridge (v1.0.0).

This module is the v1.0.0 upgrade for
:mod:`infrastructure.metrics`.  The v0.6.x / v0.8.x hand-rolled
Prometheus 0.0.4 text exposition format still ships as the
**default** (zero external dependency); the bridge here simply
*also* feeds the in-process :class:`MetricsRegistry` into a
``prometheus_client`` :class:`CollectorRegistry` so the user can
``pip install prometheus_client`` to interoperate with the wider
Prometheus / Grafana ecosystem (pushgateway, alerting, recording
rules, etc.) without any code change.

Design notes
------------

* The bridge is **import-safe without ``prometheus_client``** -- the
  optional import is wrapped in a try/except and a structured
  ``PROMETHEUS_CLIENT_AVAILABLE`` flag is exposed so callers in
  offline CI still see the in-process metrics path.
* The bridge is **read-only**: we never call ``start_http_server``
  -- the v0.8.5 ``serving`` layer's :func:`MetricsCollector.render_prometheus`
  is still the canonical scrape path.  Callers wanting a separate
  pushgateway upload can feed the snapshot into
  ``prometheus_client.push_to_gateway`` themselves.
* :func:`snapshot_counter_values` / :func:`snapshot_gauge_values` /
  :func:`snapshot_histogram_count` give a Python-only flat view
  that callers (tests, alerting) can use without ever installing
  ``prometheus_client``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .metrics import Counter, Gauge, Histogram, METRICS, MetricsRegistry

__all__ = [
    "PROMETHEUS_CLIENT_AVAILABLE",
    "bridge_to_prometheus_client",
    "snapshot_counter_values",
    "snapshot_gauge_values",
    "snapshot_histogram_count",
]

try:
    import prometheus_client as _pc  # type: ignore[import-not-found]
    PROMETHEUS_CLIENT_AVAILABLE = True
except Exception:  # noqa: BLE001 -- pragma: no cover
    _pc = None  # type: ignore[assignment]
    PROMETHEUS_CLIENT_AVAILABLE = False


def _mirror_series(
    src: Any, dst_factory: Any, kind: str,
) -> None:
    """Push the in-process series of ``src`` into a ``prometheus_client``
    metric created by ``dst_factory``.  ``src`` is a Counter / Gauge /
    Histogram from :mod:`infrastructure.metrics`.
    """
    name = f"tv_{kind}_{src.name}"
    labels = src.labelnames
    if kind == "histogram":
        # Histogram needs bucket boundaries; we hard-code the v0.8.5
        # default set so the bridge is deterministic.
        dst = dst_factory(
            name, src._help, labels, buckets=src.buckets,
        )
    else:
        dst = dst_factory(name, src._help, labels)
    for key, value in src._series.items():  # type: ignore[attr-defined]
        labels_dict = dict(zip(labels, key.values))  # type: ignore[attr-defined]
        if kind == "counter":
            dst.labels(**labels_dict).inc(0.0)  # register series
            # Replace the zero with the real value:
            dst.labels(**labels_dict)._value.set(value)  # type: ignore[attr-defined]
        elif kind == "gauge":
            # First call .labels(...) so the series is registered, then set.
            dst.labels(**labels_dict).set(value)
        else:  # histogram
            dst.labels(**labels_dict).observe(0.0)  # register series
            sum_value = src._series_buckets[key][0]  # type: ignore[attr-defined]
            count_value = src._series_buckets[key][1]  # type: ignore[attr-defined]
            # Restore the count + sum:
            for _ in range(int(count_value)):
                dst.labels(**labels_dict).observe(0.0)  # dummy


def bridge_to_prometheus_client(
    registry: Optional[MetricsRegistry] = None,
    *,
    target_registry: Optional[Any] = None,
) -> Optional[Any]:
    """Snapshot the in-process :class:`MetricsRegistry` into a
    ``prometheus_client.CollectorRegistry``.

    Args:
        registry: Source registry.  Defaults to the module-level
            :data:`METRICS` singleton.
        target_registry: Destination ``prometheus_client.CollectorRegistry``.
            When ``None`` (the default) a fresh ``CollectorRegistry``
            is created so the snapshot is isolated from the default
            global one (avoids double-counting in tests).

    Returns:
        The destination ``CollectorRegistry`` populated with
        ``Counter`` / ``Gauge`` / ``Histogram`` mirrors of every
        metric in ``registry``, or ``None`` when the optional
        ``prometheus_client`` package is not installed.
    """
    if not PROMETHEUS_CLIENT_AVAILABLE:
        return None
    if registry is None:
        registry = METRICS
    if target_registry is None:
        target_registry = _pc.CollectorRegistry()
    for metric in registry._metrics.values():  # type: ignore[attr-defined]
        if isinstance(metric, Counter):
            _mirror_series(metric, target_registry.register(_pc.Counter), "counter")
        elif isinstance(metric, Gauge):
            _mirror_series(metric, target_registry.register(_pc.Gauge), "gauge")
        elif isinstance(metric, Histogram):
            _mirror_series(
                metric, target_registry.register(_pc.Histogram), "histogram",
            )
    return target_registry


def snapshot_counter_values(
    registry: Optional[MetricsRegistry] = None,
) -> Dict[str, float]:
    """Return a flat ``{metric_name}: total`` dict of every counter."""
    if registry is None:
        registry = METRICS
    out: Dict[str, float] = {}
    for name, metric in registry._metrics.items():  # type: ignore[attr-defined]
        if not isinstance(metric, Counter):
            continue
        total = 0.0
        for value in metric._series.values():  # type: ignore[attr-defined]
            total += value
        out[name] = total
    return out


def snapshot_gauge_values(
    registry: Optional[MetricsRegistry] = None,
) -> Dict[str, float]:
    """Return a flat ``{metric_name}: value`` dict of every gauge (first
    series only -- gauges with multiple label sets are intentionally
    flattened to the first observed value to keep the snapshot simple).
    """
    if registry is None:
        registry = METRICS
    out: Dict[str, float] = {}
    for name, metric in registry._metrics.items():  # type: ignore[attr-defined]
        if not isinstance(metric, Gauge):
            continue
        for value in metric._series.values():  # type: ignore[attr-defined]
            out[name] = value
            break
    return out


def snapshot_histogram_count(
    registry: Optional[MetricsRegistry] = None,
) -> Dict[str, int]:
    """Return a flat ``{metric_name}: count`` dict of every histogram."""
    if registry is None:
        registry = METRICS
    out: Dict[str, int] = {}
    for name, metric in registry._metrics.items():  # type: ignore[attr-defined]
        if not isinstance(metric, Histogram):
            continue
        total = 0
        for state in metric._series_buckets.values():  # type: ignore[attr-defined]
            total += int(state[1])
        out[name] = total
    return out
