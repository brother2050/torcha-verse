"""A/B testing framework for TorchaVerse experiments (v1.0.0).

Purpose
-------
:mod:`experiments.framework` is a small, dependency-free A/B testing
framework that supports the workflow TorchaVerse needs for online
evaluation: declare an :class:`Experiment` with one or more
:class:`Variant` arms, assign incoming user IDs to a deterministic
variant via :func:`bucket_assign`, record metrics through
:meth:`ExperimentRunner.record`, and roll everything up into a per-arm
summary via :meth:`ExperimentRunner.summary`.

The bucket-assignment strategy follows the well-known Optimizely /
growth-book pattern: a stable hash of the user ID is reduced modulo
the total weight, then walked through the cumulative weight ranges to
pick the arm.  This is intentionally simple - it has no feature flags,
no remote config, no online learning - but it is enough for "ship a
prompt change to 5% of users and watch the metric" use cases that come
up routinely in generative-model evaluation.

References
----------
* Optimizely's "How bucketing works" docs - the canonical reference
  for hash-modulo assignment with per-variant weights.
* growth-book's ``hash`` helper - same algorithm, different language.
"""
from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Union

__all__ = [
    "Experiment",
    "Variant",
    "ExperimentRunner",
    "bucket_assign",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Variant:
    """A single arm of an :class:`Experiment`.

    Attributes:
        name: Stable string identifier; used as the key in
            :meth:`ExperimentRunner.summary`.
        weight: Relative weight in the bucket assignment.  Weights
            are normalised internally, so ``[1, 1]`` and
            ``[0.5, 0.5]`` produce identical 50/50 splits.
        config: Free-form per-variant configuration blob.  This is
            typically the place to stash sampling parameters,
            prompt templates or any other arm-specific knob.
    """

    name: str
    weight: float = 1.0
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Variant.name must be a non-empty string")
        if self.weight < 0:
            raise ValueError(
                f"Variant.weight must be non-negative, got {self.weight!r}",
            )


@dataclass
class Experiment:
    """A bundle of variants that share a primary metric.

    Attributes:
        name: Stable string identifier for the experiment.
        variants: The list of arms.  Must contain at least one
            variant with positive weight.
        primary_metric: Name of the metric that the experiment is
            optimised for.  Surfaced in :meth:`ExperimentRunner.summary`
            and in the logs.
        start_time: Optional ISO-8601 start timestamp.  Set by
            :class:`ExperimentRunner` on first use when the caller
            leaves it as ``None``.
        end_time: Optional ISO-8601 end timestamp.  Set by
            :class:`ExperimentRunner` on :meth:`stop`.
    """

    name: str
    variants: List[Variant]
    primary_metric: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Experiment.name must be a non-empty string")
        if not self.variants:
            raise ValueError("Experiment.variants must be non-empty")
        total = sum(v.weight for v in self.variants)
        if total <= 0:
            raise ValueError(
                "Experiment.variants must have positive total weight",
            )


# ---------------------------------------------------------------------------
# Bucket assignment
# ---------------------------------------------------------------------------
def _stable_int_hash(user_id: Union[str, int]) -> int:
    """Return a stable 32-bit hash for ``user_id``.

    We use ``md5`` (not for security - just for spread) and take the
    first 4 bytes as an unsigned int.  ``md5`` is in the stdlib and
    gives a much better distribution than Python's built-in
    :func:`hash`, which is salted per process.
    """
    if isinstance(user_id, int):
        # ``int`` is its own stable hash modulo the platform int width
        # - good enough for the assignment we do here.
        return user_id & 0xFFFFFFFF
    key = str(user_id).encode("utf-8")
    digest = hashlib.md5(key).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def bucket_assign(
    user_id: Union[str, int],
    variants: List[Variant],
) -> Variant:
    """Deterministically assign ``user_id`` to one of ``variants``.

    The assignment uses a 32-bit stable hash of ``user_id`` reduced
    modulo the total weight, then walked through the cumulative
    weight ranges of ``variants``.  This mirrors Optimizely's
    bucketing strategy: the same user always lands in the same
    variant (for a fixed experiment configuration) and the empirical
    distribution of assignments converges to the configured weights
    as the user population grows.

    Args:
        user_id: Anything hashable that uniquely identifies a user.
        variants: The list of arms to choose from.  Must contain at
            least one variant with positive weight.

    Returns:
        The chosen :class:`Variant`.

    Raises:
        ValueError: If ``variants`` is empty or has zero total weight.
    """
    if not variants:
        raise ValueError("bucket_assign requires at least one variant")
    total = sum(v.weight for v in variants)
    if total <= 0:
        raise ValueError("bucket_assign requires positive total weight")
    # Reduce the 32-bit hash to a ``[0, total)`` bucket.
    bucket = _stable_int_hash(user_id) % total
    # Walk the cumulative weight ranges.
    cumulative = 0.0
    for variant in variants:
        cumulative += variant.weight
        if bucket < cumulative:
            return variant
    # Numerical edge case: ``bucket`` may equal ``total`` after the
    # modulo due to float round-off.  Fall through to the last
    # variant with positive weight, which is what Optimizely does.
    for variant in reversed(variants):
        if variant.weight > 0:
            return variant
    # Unreachable - the ``total > 0`` guard above ensures at least
    # one variant has positive weight.
    raise RuntimeError("bucket_assign: no positive-weight variant found")  # pragma: no cover


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class ExperimentRunner:
    """Track per-user variant assignments and metric outcomes.

    The runner is the in-memory companion to an :class:`Experiment`:
    :meth:`pick` resolves a user to a variant via :func:`bucket_assign`
    and :meth:`record` captures a metric value for the (user, variant)
    pair.  :meth:`summary` then rolls every metric up to a per-arm
    ``{mean, count, std}`` dict ready for logging.

    A single :class:`ExperimentRunner` instance is meant to live for
    the duration of a single experiment run; it is not thread-safe and
    it does not persist anything to disk.  For production use, wire
    :meth:`record` up to your metrics backend of choice.
    """

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment
        # ``results`` is a nested ``{variant_name: {metric: [values]}}``
        # mapping.  We store raw values rather than running statistics
        # so the caller can compute medians / percentiles on demand.
        self.results: Dict[str, Dict[str, List[float]]] = {
            v.name: {} for v in experiment.variants
        }
        # Track the per-user assignment so a second ``pick`` for the
        # same user returns the same variant - this is the
        # ``user_id``-keyed memoisation that makes the runner
        # deterministic across calls.
        self._assignments: Dict[Hashable, str] = {}
        # Mark the experiment start time lazily on first ``pick``.
        if self.experiment.start_time is None:
            self.experiment.start_time = _now_iso()

    # ------------------------------------------------------------------
    # pick
    # ------------------------------------------------------------------
    def pick(self, user_id: Union[str, int]) -> Variant:
        """Return the variant assigned to ``user_id``.

        Repeated calls for the same ``user_id`` return the same
        variant.  Unknown user IDs trigger a fresh assignment via
        :func:`bucket_assign`.
        """
        cached = self._assignments.get(user_id)
        if cached is not None:
            for variant in self.experiment.variants:
                if variant.name == cached:
                    return variant
        variant = bucket_assign(user_id, list(self.experiment.variants))
        self._assignments[user_id] = variant.name
        return variant

    # ------------------------------------------------------------------
    # record
    # ------------------------------------------------------------------
    def record(
        self,
        user_id: Union[str, int],
        metric_name: str,
        value: float,
    ) -> None:
        """Append ``value`` to ``metric_name`` for the user's variant.

        The user is auto-assigned to a variant on first sight, so the
        caller does not have to call :meth:`pick` first.
        """
        if not metric_name:
            raise ValueError("metric_name must be a non-empty string")
        variant = self.pick(user_id)
        per_metric = self.results.setdefault(variant.name, {})
        per_metric.setdefault(metric_name, []).append(float(value))

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Roll every recorded metric up to a per-arm summary.

        Returns a nested dict of the form::

            {
              "<variant_name>": {
                "<metric_name>": {"mean": float, "count": int, "std": float}
              }
            }

        Variants with no recorded samples are reported with
        ``count = 0`` and ``mean = std = 0.0`` so the output shape is
        stable for log consumers.
        """
        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for variant_name, per_metric in self.results.items():
            summary[variant_name] = {}
            for metric_name, values in per_metric.items():
                if not values:
                    summary[variant_name][metric_name] = {
                        "mean": 0.0, "count": 0, "std": 0.0,
                    }
                else:
                    summary[variant_name][metric_name] = {
                        "mean": float(statistics.fmean(values)),
                        "count": len(values),
                        "std": float(statistics.pstdev(values))
                        if len(values) > 1 else 0.0,
                    }
        return summary

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Mark the experiment end-time.  Safe to call multiple times."""
        self.experiment.end_time = _now_iso()


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import random

    experiment = Experiment(
        name="prompt_v2_smoke",
        variants=[
            Variant("control", weight=1.0, config={"prompt": "old"}),
            Variant("treatment", weight=1.0, config={"prompt": "new"}),
        ],
        primary_metric="ctr",
    )
    runner = ExperimentRunner(experiment)
    rng = random.Random(42)
    for i in range(100):
        user_id = f"user-{i}"
        runner.record(user_id, "ctr", rng.random())
    # Bucket distribution over a larger population so the empirical
    # proportion stabilises.
    distribution: Dict[str, int] = {v.name: 0 for v in experiment.variants}
    for i in range(1000):
        distribution[runner.pick(f"user-{i}").name] += 1
    print("[experiments] smoke OK")
    print("[experiments] bucket distribution:", distribution)
    print("[experiments] summary:", runner.summary())
    runner.stop()
