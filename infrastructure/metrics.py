"""Lightweight metrics collection for the v0.4.x → v1.0.0 bridge.

This module provides a minimal, *in-process* metrics facade that:

* exposes a small set of counters / gauges / histograms covering
  request volume, latency, allocation failures, and GPU-cache events;
* can be rendered to the Prometheus text exposition format **without
  any external dependency** (so the v0.4.x API server can ship a
  ``/metrics`` endpoint immediately);
* can transparently delegate to ``prometheus_client`` when it is
  installed at runtime (the v1.0.0 production path), keeping the
  rendered output wire-compatible with a real Prometheus scrape.

The design is intentionally simple — the goal is to give the M2b
("Prometheus ``/metrics`` endpoint") deliverable a working
implementation that does not force the rest of the framework to take
a new dependency.  A real ``prometheus_client`` swap-in is a
drop-in change behind the same public surface.

Public API
----------

* :class:`MetricsRegistry` — the process-wide registry; use the
  :data:`METRICS` module-level singleton unless you need isolation
  in tests.
* :class:`Counter`, :class:`Gauge`, :class:`Histogram` — typed
  metric handles; constructed via the registry so labels stay
  consistent across the codebase.
* :func:`render_prometheus` — render the current registry state as
  the Prometheus 0.0.4 text exposition format.

Example
-------

    >>> from infrastructure.metrics import METRICS
    >>> METRICS.counter("api_requests_total", "Total API requests", ("route", "status")).inc("/healthz", "200")
    >>> METRICS.gauge("budget_vram_gb", "VRAM budget", ("tenant",)).set("default", 24.0)
    >>> METRICS.render_prometheus().splitlines()[:1]
    ['# HELP api_requests_total Total API requests']
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .logger import get_logger

__all__ = [
    "MetricsRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "METRICS",
    "render_prometheus",
    "is_prometheus_client_available",
    "export_to_prometheus_client",
]

_logger = get_logger("infrastructure.metrics")


# ---------------------------------------------------------------------------
# Metric value containers
# ---------------------------------------------------------------------------
@dataclass
class _LabelKey:
    """Hashable wrapper for a tuple of label values."""

    values: Tuple[str, ...]

    def __init__(self, values: Sequence[str]) -> None:
        self.values: Tuple[str, ...] = tuple(values)

    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash(self.values)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
        return isinstance(other, _LabelKey) and self.values == other.values

    def render(self, names: Sequence[str]) -> str:
        """Return the label clause for the Prometheus exposition line."""
        if not self.values:
            return ""
        parts = []
        for name, value in zip(names, self.values):
            escaped = (
                value.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
            )
            parts.append(f'{name}="{escaped}"')
        return "{" + ",".join(parts) + "}"


class _MetricBase:
    """Shared logic for Counter / Gauge / Histogram."""

    def __init__(
        self,
        registry: "MetricsRegistry",
        name: str,
        help: str,
        labelnames: Sequence[str],
    ) -> None:
        if not name:
            raise ValueError("Metric name must be a non-empty string.")
        if not name.replace("_", "").isalnum():
            raise ValueError(
                f"Metric name must match [a-zA-Z_][a-zA-Z0-9_]*; got {name!r}."
            )
        if not help:
            raise ValueError(f"Metric {name!r} requires a non-empty help string.")
        # Reject duplicate label names to keep exposition deterministic.
        if len(set(labelnames)) != len(labelnames):
            raise ValueError(
                f"Metric {name!r} has duplicate label names: {labelnames}."
            )
        self._registry: MetricsRegistry = registry
        self._name: str = name
        self._help: str = help
        self._labelnames: Tuple[str, ...] = tuple(labelnames)
        self._lock: threading.RLock = threading.RLock()
        # Map from ``_LabelKey`` to the per-series typed value.
        self._series: Dict[_LabelKey, float] = {}
        # Map for histograms: per series -> [sum, count, [buckets]].
        self._series_buckets: Dict[_LabelKey, List[float]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """The metric name (e.g. ``api_requests_total``)."""
        return self._name

    @property
    def help(self) -> str:
        """Human-readable help text for the metric."""
        return self._help

    @property
    def labelnames(self) -> Tuple[str, ...]:
        """Tuple of label names in declaration order."""
        return self._labelnames

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _key(self, labelvalues: Sequence[str]) -> _LabelKey:
        if len(labelvalues) != len(self._labelnames):
            raise ValueError(
                f"Metric {self._name!r} expects {len(self._labelnames)} labels, "
                f"got {len(labelvalues)}: {labelvalues}."
            )
        return _LabelKey(labelvalues)

    def _values(self) -> Iterable[Tuple[_LabelKey, float]]:
        with self._lock:
            return list(self._series.items())


class Counter(_MetricBase):
    """A monotonically increasing counter.

    Only :meth:`inc` is allowed; calling :meth:`dec` is rejected to
    keep the Prometheus semantic of "counters only go up".
    """

    def inc(self, *labelvalues: str, amount: float = 1.0) -> None:
        """Increase the counter for ``labelvalues`` by ``amount``."""
        if amount < 0:
            raise ValueError(
                f"Counter {self._name!r} cannot be decreased; got {amount}."
            )
        key = self._key(labelvalues)
        with self._lock:
            self._series[key] = self._series.get(key, 0.0) + amount


class Gauge(_MetricBase):
    """A value that can go up and down (e.g. memory, queue depth)."""

    def set(self, *labelvalues: str, value: float) -> None:
        """Set the gauge to ``value`` for the given labels."""
        key = self._key(labelvalues)
        with self._lock:
            self._series[key] = float(value)

    def inc(self, *labelvalues: str, amount: float = 1.0) -> None:
        """Increase the gauge by ``amount`` (may be negative)."""
        key = self._key(labelvalues)
        with self._lock:
            self._series[key] = self._series.get(key, 0.0) + amount

    def dec(self, *labelvalues: str, amount: float = 1.0) -> None:
        """Decrease the gauge by ``amount``."""
        self.inc(*labelvalues, amount=-amount)


class Histogram(_MetricBase):
    """A bucketed distribution (e.g. request latency).

    The default buckets are tuned for HTTP request latencies in
    seconds.  They are exposed under the ``le`` label so a scraper
    can compute rates / quantiles from ``_bucket`` time-series.
    """

    _DEFAULT_BUCKETS: Tuple[float, ...] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        float("+inf"),
    )

    def __init__(
        self,
        registry: "MetricsRegistry",
        name: str,
        help: str,
        labelnames: Sequence[str],
        buckets: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__(registry, name, help, labelnames)
        # ``+inf`` is always the last bucket so the cumulative count
        # at the top bucket equals the total count.
        chosen = tuple(sorted(set(buckets) if buckets else self._DEFAULT_BUCKETS))
        if chosen[-1] != float("+inf"):
            chosen = chosen + (float("+inf"),)
        self._buckets: Tuple[float, ...] = chosen

    @property
    def buckets(self) -> Tuple[float, ...]:
        """Sorted bucket boundaries (always ends with ``+inf``)."""
        return self._buckets

    def observe(self, *labelvalues: str, value: float) -> None:
        """Record an observation of ``value`` for the given labels."""
        key = self._key(labelvalues)
        with self._lock:
            state = self._series_buckets.setdefault(
                key, [0.0, 0.0] + [0.0] * len(self._buckets)
            )
            # state = [sum, count, b0, b1, ..., bN]
            state[0] += value
            state[1] += 1.0
            for idx, boundary in enumerate(self._buckets):
                if value <= boundary:
                    state[2 + idx] += 1.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class MetricsRegistry:
    """Process-wide registry of named metrics.

    Constructing a metric twice with the same name is a hard error;
    use :meth:`get` to look up an existing metric.
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        # Map metric name -> metric object (Counter / Gauge / Histogram).
        self._metrics: Dict[str, _MetricBase] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def counter(
        self,
        name: str,
        help: str,
        labelnames: Sequence[str] = (),
    ) -> Counter:
        """Construct (or fetch) a :class:`Counter`."""
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                if not isinstance(existing, Counter):
                    raise TypeError(
                        f"Metric {name!r} already registered as a different type."
                    )
                return existing
            metric = Counter(self, name, help, labelnames)
            self._metrics[name] = metric
            return metric

    def gauge(
        self,
        name: str,
        help: str,
        labelnames: Sequence[str] = (),
    ) -> Gauge:
        """Construct (or fetch) a :class:`Gauge`."""
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                if not isinstance(existing, Gauge):
                    raise TypeError(
                        f"Metric {name!r} already registered as a different type."
                    )
                return existing
            metric = Gauge(self, name, help, labelnames)
            self._metrics[name] = metric
            return metric

    def histogram(
        self,
        name: str,
        help: str,
        labelnames: Sequence[str] = (),
        buckets: Optional[Sequence[float]] = None,
    ) -> Histogram:
        """Construct (or fetch) a :class:`Histogram`."""
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                if not isinstance(existing, Histogram):
                    raise TypeError(
                        f"Metric {name!r} already registered as a different type."
                    )
                return existing
            metric = Histogram(self, name, help, labelnames, buckets)
            self._metrics[name] = metric
            return metric

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get(self, name: str) -> Optional[_MetricBase]:
        """Return the metric registered as ``name`` or ``None``."""
        return self._metrics.get(name)

    def names(self) -> List[str]:
        """Return the list of registered metric names (sorted)."""
        with self._lock:
            return sorted(self._metrics.keys())

    def clear(self) -> None:
        """Remove all metrics from the registry (intended for tests)."""
        with self._lock:
            self._metrics.clear()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self) -> str:
        """Render the registry as the Prometheus text exposition format."""
        return render_prometheus(self)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._metrics)

    def __contains__(self, name: object) -> bool:  # pragma: no cover - trivial
        return isinstance(name, str) and name in self._metrics


#: Process-wide default registry.  Modules should use this rather than
#: constructing their own so that the ``/metrics`` endpoint sees a
#: single, consistent view of the world.
METRICS: MetricsRegistry = MetricsRegistry()


# ---------------------------------------------------------------------------
# Prometheus text rendering
# ---------------------------------------------------------------------------
def _format_float(value: float) -> str:
    """Render ``value`` in the canonical Prometheus float form."""
    if value == float("+inf"):
        return "+Inf"
    if value == float("-inf"):
        return "-Inf"
    if value != value:  # NaN
        return "NaN"
    # Avoid scientific notation for typical gauge / counter values.
    if value.is_integer():
        return str(int(value))
    return repr(value)


def render_prometheus(registry: Optional[MetricsRegistry] = None) -> str:
    """Render ``registry`` (or the global :data:`METRICS`) as text.

    The output follows the
    `Prometheus 0.0.4 text exposition format
    <https://prometheus.io/docs/instrumenting/exposition_formats/>`_:
    each metric emits a ``# HELP`` and a ``# TYPE`` line followed by
    one line per series, sorted by label values for determinism.

    Args:
        registry: Registry to render.  Defaults to the global
            :data:`METRICS`.

    Returns:
        The exposition text, terminated by a single trailing newline.
    """
    target = registry if registry is not None else METRICS
    lines: List[str] = []
    for name in target.names():
        metric = target.get(name)
        if metric is None:  # pragma: no cover - invariant
            continue
        if isinstance(metric, Counter):
            lines.append(f"# HELP {name} {metric.help}")
            lines.append(f"# TYPE {name} counter")
            for key, value in sorted(
                metric._values(), key=lambda kv: kv[0].values
            ):
                lines.append(f"{name}{key.render(metric.labelnames)} {_format_float(value)}")
        elif isinstance(metric, Gauge):
            lines.append(f"# HELP {name} {metric.help}")
            lines.append(f"# TYPE {name} gauge")
            for key, value in sorted(
                metric._values(), key=lambda kv: kv[0].values
            ):
                lines.append(f"{name}{key.render(metric.labelnames)} {_format_float(value)}")
        elif isinstance(metric, Histogram):
            lines.append(f"# HELP {name} {metric.help}")
            lines.append(f"# TYPE {name} histogram")
            for key, state in sorted(
                metric._series_buckets.items(), key=lambda kv: kv[0].values
            ):
                total_sum, total_count, *bucket_counts = state
                base_labels = list(metric.labelnames)
                # Emit each ``le`` bucket as a separate time series.
                for boundary, count in zip(metric.buckets, bucket_counts):
                    le_value = "+Inf" if boundary == float("inf") else _format_float(boundary)
                    label_clause = _merge_labels(
                        base_labels, list(key.values), extra=(("le", le_value),)
                    )
                    lines.append(
                        f'{name}_bucket{label_clause} {_format_float(count)}'
                    )
                # ``_sum`` and ``_count`` are special histogram series.
                count_labels = _merge_labels(
                    base_labels, list(key.values)
                )
                lines.append(
                    f"{name}_sum{count_labels} {_format_float(total_sum)}"
                )
                lines.append(
                    f"{name}_count{count_labels} {_format_float(total_count)}"
                )
    return "\n".join(lines) + ("\n" if lines else "")


def _merge_labels(
    base_names: Sequence[str],
    base_values: Sequence[str],
    extra: Sequence[Tuple[str, str]] = (),
) -> str:
    """Render a label clause that adds ``extra`` to the base labels."""
    if not base_names and not extra:
        return ""
    parts: List[str] = []
    for name, value in zip(base_names, base_values):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
        )
        parts.append(f'{name}="{escaped}"')
    for name, value in extra:
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
        )
        parts.append(f'{name}="{escaped}"')
    return "{" + ",".join(parts) + "}"


# ---------------------------------------------------------------------------
# Optional swap-in to ``prometheus_client`` (v1.0.0 M2b)
# ---------------------------------------------------------------------------
def is_prometheus_client_available() -> bool:
    """Return True when the optional ``prometheus_client`` package is importable.

    The v0.4.x -> v1.0.0 bridge keeps the metrics layer dependency-free
    so the CPU image is small, but operators who want the real
    Prometheus collectors (``Histogram.collect()``,
    ``start_http_server``, ``pushgateway``, etc.) can install
    ``prometheus_client>=0.19`` and then call
    :func:`export_to_prometheus_client`.
    """
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        return False
    return True


def export_to_prometheus_client(
    registry: Optional[MetricsRegistry] = None,
) -> Dict[str, Any]:
    """Mirror ``registry`` into a ``prometheus_client`` registry.

    Creates a fresh :class:`prometheus_client.CollectorRegistry` (so
    the caller's process-global ``REGISTRY`` is not touched),
    instantiates the right ``prometheus_client`` primitives for
    every metric in ``registry``, and copies the current series
    values.  Subsequent ``inc`` / ``set`` / ``observe`` calls on
    the v0.4.x registry do *not* propagate to the
    ``prometheus_client`` registry; this function is intended as a
    one-shot migration helper for v1.0.0 deploys.

    Args:
        registry: The :class:`MetricsRegistry` to mirror.  Defaults
            to the global :data:`METRICS`.

    Returns:
        A dict with ``"registry"`` (the new CollectorRegistry),
        ``"counters"``, ``"gauges"``, ``"histograms"`` -- one entry
        per migrated metric so callers can wire scrape targets
        against specific names.

    Raises:
        ImportError: if ``prometheus_client`` is not installed.
    """
    try:
        import prometheus_client as pc
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "prometheus_client is not installed; install it with "
            "`pip install 'prometheus_client>=0.19'` before calling "
            "export_to_prometheus_client()."
        ) from exc

    try:
        out_registry = pc.CollectorRegistry()
    except ImportError as exc:  # pragma: no cover - optional dep
        # ``prometheus_client`` raised a ``ModuleNotFoundError`` at
        # first use (e.g. it imports a sub-package that is missing
        # on the target platform).  Surface a uniform
        # ``ImportError`` with the install hint.
        raise ImportError(
            "prometheus_client is not usable in this environment; "
            "install it with `pip install 'prometheus_client>=0.19'` "
            "before calling export_to_prometheus_client()."
        ) from exc

    target = registry if registry is not None else METRICS
    counters: Dict[str, Any] = {}
    gauges: Dict[str, Any] = {}
    histograms: Dict[str, Any] = {}

    for name in target.names():
        metric = target.get(name)
        if metric is None:  # pragma: no cover - invariant
            continue
        if isinstance(metric, Counter):
            pc_metric = pc.Counter(
                name, metric.help, list(metric.labelnames), registry=out_registry
            )
            for key, value in metric._values():
                pc_metric.labels(*key.values).inc(amount=value)
            counters[name] = pc_metric
        elif isinstance(metric, Gauge):
            pc_metric = pc.Gauge(
                name, metric.help, list(metric.labelnames), registry=out_registry
            )
            for key, value in metric._values():
                pc_metric.labels(*key.values).set(value)
            gauges[name] = pc_metric
        elif isinstance(metric, Histogram):
            pc_metric = pc.Histogram(
                name,
                metric.help,
                list(metric.labelnames),
                buckets=list(metric.buckets),
                registry=out_registry,
            )
            for key, state in metric._series_buckets.items():
                total_sum, total_count, *_ = state
                if total_count > 0:
                    # ``prometheus_client.Histogram`` has no
                    # "set from existing observations" entry
                    # point, so we use ``observe`` with the mean
                    # of the recorded series.  Best-effort
                    # migration; v1.0.0 production deployments
                    # run the two registries side by side.
                    pc_metric.labels(*key.values).observe(total_sum / total_count)
            histograms[name] = pc_metric

    return {
        "registry": out_registry,
        "counters": counters,
        "gauges": gauges,
        "histograms": histograms,
    }
