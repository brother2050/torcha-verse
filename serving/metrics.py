"""Prometheus-style metrics collection for the serving API.

This module hosts :class:`MetricsCollector`, which tracks request counts
and latency per endpoint in a thread-safe manner and renders the metrics
in the Prometheus text exposition format for scraping by ``GET /metrics``.

It was extracted from the original monolithic ``api_server.py``.
"""

from __future__ import annotations

import time
from typing import Dict, List

__all__ = ["MetricsCollector"]


class MetricsCollector:
    """Collect and expose Prometheus-format metrics.

    Tracks request counts and latency per endpoint in a thread-safe
    manner.  The metrics are rendered in the Prometheus text exposition
    format for scraping by ``GET /metrics``.
    """

    def __init__(self) -> None:
        self._request_counts: Dict[str, int] = {}
        self._error_counts: Dict[str, int] = {}
        self._latency_sum: Dict[str, float] = {}
        self._latency_count: Dict[str, int] = {}
        self._engine_loads: Dict[str, int] = {}
        self._start_time: float = time.time()

    def record_request(self, endpoint: str, latency: float, error: bool = False) -> None:
        """Record a completed request.

        Args:
            endpoint: The endpoint name.
            latency: Request latency in seconds.
            error: Whether the request resulted in an error.
        """
        self._request_counts[endpoint] = self._request_counts.get(endpoint, 0) + 1
        self._latency_sum[endpoint] = self._latency_sum.get(endpoint, 0.0) + latency
        self._latency_count[endpoint] = self._latency_count.get(endpoint, 0) + 1
        if error:
            self._error_counts[endpoint] = self._error_counts.get(endpoint, 0) + 1

    def record_engine_load(self, engine_type: str) -> None:
        """Record an engine load event."""
        self._engine_loads[engine_type] = self._engine_loads.get(engine_type, 0) + 1

    def render(self) -> str:
        """Render metrics in Prometheus text exposition format.

        Returns:
            A string suitable for a ``text/plain`` metrics response.
        """
        lines: List[str] = []
        uptime = time.time() - self._start_time

        # Uptime gauge.
        lines.append("# HELP torcha_uptime_seconds Server uptime in seconds.")
        lines.append("# TYPE torcha_uptime_seconds gauge")
        lines.append(f"torcha_uptime_seconds {uptime:.2f}")
        lines.append("")

        # Request counter.
        lines.append("# HELP torcha_requests_total Total number of requests.")
        lines.append("# TYPE torcha_requests_total counter")
        for ep, count in sorted(self._request_counts.items()):
            lines.append(f'torcha_requests_total{{endpoint="{ep}"}} {count}')
        lines.append("")

        # Error counter.
        lines.append("# HELP torcha_errors_total Total number of errors.")
        lines.append("# TYPE torcha_errors_total counter")
        for ep, count in sorted(self._error_counts.items()):
            lines.append(f'torcha_errors_total{{endpoint="{ep}"}} {count}')
        lines.append("")

        # Latency summary.
        lines.append("# HELP torcha_request_latency_seconds_avg Average request latency.")
        lines.append("# TYPE torcha_request_latency_seconds_avg gauge")
        for ep in sorted(self._latency_sum.keys()):
            total = self._latency_sum[ep]
            count = self._latency_count.get(ep, 1)
            avg = total / count if count else 0.0
            lines.append(
                f'torcha_request_latency_seconds_avg{{endpoint="{ep}"}} {avg:.6f}'
            )
        lines.append("")

        # Engine loads.
        lines.append("# HELP torcha_engine_loads_total Total engine load events.")
        lines.append("# TYPE torcha_engine_loads_total counter")
        for et, count in sorted(self._engine_loads.items()):
            lines.append(f'torcha_engine_loads_total{{engine="{et}"}} {count}')

        return "\n".join(lines) + "\n"
