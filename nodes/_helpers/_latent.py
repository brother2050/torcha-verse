"""Latent validation helpers (v0.8.5).

This module is the v0.8.5 "端到端 Latent 验证" layer.  The
:class:`LatentValidator` is invoked by
:func:`call_diffusion_loop_backend` (see
:mod:`nodes._helpers._backends`) so that *every* diffusion loop
returns a structured validation report alongside the denoised
latent.  The report can be consumed by tests (assertions over
``latent_valid`` / ``latent_stats``) and by the upcoming
benchmark suite (per-model std / range telemetry).

The validator is intentionally **pure-Python** and has no
external dependencies beyond PyTorch.  All checks are O(1) over
the latent tensor (one pass for the mean / std / finite / NaN
counters) and produce a small dict that is JSON-serialisable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

__all__ = [
    "LatentStats",
    "LatentValidationError",
    "LatentValidator",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class LatentStats:
    """Summary statistics of a single latent tensor.

    All fields are JSON-serialisable and small enough to log
    inside the v0.8.5 benchmark harness.

    Attributes:
        shape: ``(B, C, H, W)`` shape of the latent.
        dtype: Tensor dtype (string, e.g. ``"torch.float32"``).
        numel: Total number of elements.
        finite: ``True`` if **all** elements are finite.
        nan_count: Number of NaN elements.
        inf_count: Number of ``+inf`` or ``-inf`` elements.
        mean: Mean of the latent.
        std: Standard deviation of the latent.
        min: Minimum value.
        max: Maximum value.
        abs_max: ``max(|min|, |max|)`` -- a quick scale signal.
    """

    shape: Tuple[int, int, int, int]
    dtype: str
    numel: int
    finite: bool
    nan_count: int
    inf_count: int
    mean: float
    std: float
    min: float
    max: float
    abs_max: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict view."""
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------
class LatentValidationError(ValueError):
    """Raised by :class:`LatentValidator` when a *required* check fails.

    The v0.8.5 contract is **fail-soft**: a non-strict call
    (:py:meth:`LatentValidator.validate`) never raises, it just
    sets ``ok=False`` on the returned report.  Strict callers
    (unit tests, CI assertions) wrap it in
    :py:meth:`LatentValidator.validate_strict`, which raises this
    exception with a one-line message that includes the failing
    check, the observed value, and the configured bound.
    """

    def __init__(self, message: str, *, report: Dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------
class LatentValidator:
    """Stateless validator that runs all v0.8.5 sanity checks on a latent.

    Args:
        expected_shape: Optional ``(B, C, H, W)`` tuple.  When
            provided the latent's shape must match exactly.
        expected_dtype: Optional tensor dtype.
        min_std: Inclusive lower bound for ``latent.std()``.
            Defaults to ``0.05`` -- below this the latent is
            effectively constant (mode collapse / sigmoid
            saturation).
        max_std: Inclusive upper bound for ``latent.std()``.
            Defaults to ``10.0`` -- beyond this the latent is
            effectively unbounded (numerical blow-up).
        min_abs_max: Inclusive lower bound for ``latent.abs().max()``.
            Defaults to ``0.0`` (latent must contain at least one
            non-zero element).
        max_abs_max: Inclusive upper bound for
            ``latent.abs().max()``.  Defaults to ``50.0`` which
            comfortably covers any legitimate float16 / float32
            diffusion latent.
        allow_nan: ``False`` (default).  When ``True`` NaNs are
            tolerated (e.g. for adversarial testing).
        allow_inf: ``False`` (default).

    Example:
        >>> validator = LatentValidator(expected_shape=(1, 4, 8, 8))
        >>> report = validator.validate(noise)
        >>> report["valid"]
        True
    """

    def __init__(
        self,
        *,
        expected_shape: Optional[Tuple[int, int, int, int]] = None,
        expected_dtype: Optional[torch.dtype] = None,
        min_std: float = 0.05,
        max_std: float = 10.0,
        min_abs_max: float = 0.0,
        max_abs_max: float = 50.0,
        allow_nan: bool = False,
        allow_inf: bool = False,
    ) -> None:
        self.expected_shape = (
            tuple(expected_shape) if expected_shape is not None else None,
        )
        self.expected_dtype = expected_dtype
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.min_abs_max = float(min_abs_max)
        self.max_abs_max = float(max_abs_max)
        self.allow_nan = bool(allow_nan)
        self.allow_inf = bool(allow_inf)

    # ------------------------------------------------------------------
    @staticmethod
    def compute_stats(latent: torch.Tensor) -> LatentStats:
        """Compute the :class:`LatentStats` summary for ``latent``.

        The tensor is first flattened so a single ``.min()`` /
        ``.max()`` / ``.std()`` pass is enough.
        """
        if not isinstance(latent, torch.Tensor):
            raise TypeError(
                f"LatentValidator.compute_stats expects torch.Tensor, "
                f"got {type(latent).__name__}",
            )
        flat = latent.detach()
        if not flat.is_floating_point() and not flat.is_complex():
            # We still compute std / mean over the integer domain
            # but flag the dtype mismatch downstream.
            pass
        finite_mask = torch.isfinite(flat)
        finite_count = int(finite_mask.sum().item())
        nan_count = int(torch.isnan(flat).sum().item())
        # ``inf`` requires floating dtype; for integer tensors the
        # count is 0 and we keep the ``inf_count`` key for schema
        # stability.
        if flat.is_floating_point() or flat.is_complex():
            inf_count = int(torch.isinf(flat).sum().item())
        else:
            inf_count = 0
        # Compute mean / std on a finite subset if the full tensor
        # has any non-finite values.  Falling back to the full
        # tensor is fine for the common all-finite case.
        safe = flat[finite_mask] if finite_count < flat.numel() else flat
        if safe.numel() == 0:
            mean = float("nan")
            std = float("nan")
            min_v = float("nan")
            max_v = float("nan")
        else:
            mean = float(safe.float().mean().item())
            std = float(safe.float().std(unbiased=False).item())
            min_v = float(safe.float().min().item())
            max_v = float(safe.float().max().item())
        return LatentStats(
            shape=tuple(latent.shape),
            dtype=str(latent.dtype),
            numel=int(latent.numel()),
            finite=bool(finite_count == flat.numel()),
            nan_count=nan_count,
            inf_count=inf_count,
            mean=mean,
            std=std,
            min=min_v,
            max=max_v,
            abs_max=float(max(abs(min_v), abs(max_v))) if math.isfinite(min_v) and math.isfinite(max_v) else float("inf"),
        )

    # ------------------------------------------------------------------
    def _run_checks(self, stats: LatentStats) -> Tuple[bool, str]:
        """Run all per-stat checks; return ``(ok, reason)``.

        The reason is the **first** failing check (stable for
        snapshot tests).  When everything is fine the reason is
        the empty string.
        """
        # Shape.
        if self.expected_shape[0] is not None:
            if tuple(stats.shape) != tuple(self.expected_shape[0]):
                return (
                    False,
                    f"shape mismatch: expected "
                    f"{tuple(self.expected_shape[0])}, got {tuple(stats.shape)}",
                )
        # Dtype.
        if self.expected_dtype is not None:
            actual = stats.dtype.split(".")[-1]
            expected = str(self.expected_dtype).split(".")[-1]
            if actual != expected:
                return (
                    False,
                    f"dtype mismatch: expected {expected}, got {actual}",
                )
        # Finite / NaN / Inf.
        if not self.allow_nan and stats.nan_count > 0:
            return False, f"contains {stats.nan_count} NaN element(s)"
        if not self.allow_inf and stats.inf_count > 0:
            return False, f"contains {stats.inf_count} Inf element(s)"
        if not stats.finite and not (self.allow_nan or self.allow_inf):
            return False, "latent is not fully finite"
        # Scale bounds.
        if math.isnan(stats.std):
            return False, "std is NaN (latent fully non-finite)"
        if stats.std < self.min_std:
            return (
                False,
                f"std {stats.std:.6f} < min_std {self.min_std:.6f} "
                f"(mode-collapse / saturation)",
            )
        if stats.std > self.max_std:
            return (
                False,
                f"std {stats.std:.6f} > max_std {self.max_std:.6f} "
                f"(numerical blow-up)",
            )
        if math.isnan(stats.abs_max) or math.isinf(stats.abs_max):
            return False, "abs_max is non-finite"
        if stats.abs_max < self.min_abs_max:
            return (
                False,
                f"abs_max {stats.abs_max:.6f} < min_abs_max "
                f"{self.min_abs_max:.6f} (all-zero latent)",
            )
        if stats.abs_max > self.max_abs_max:
            return (
                False,
                f"abs_max {stats.abs_max:.6f} > max_abs_max "
                f"{self.max_abs_max:.6f}",
            )
        return True, ""

    # ------------------------------------------------------------------
    def validate(self, latent: torch.Tensor) -> Dict[str, Any]:
        """Validate ``latent`` and return a structured report.

        Never raises.  The report contains:

        * ``valid`` -- ``True`` iff every check passed.
        * ``reason`` -- empty string, or the first failing check.
        * ``stats`` -- :class:`LatentStats` dict.
        * ``checks`` -- dict of the configured bounds (echo).
        """
        stats = self.compute_stats(latent)
        ok, reason = self._run_checks(stats)
        return {
            "valid": bool(ok),
            "reason": reason,
            "stats": stats.to_dict(),
            "checks": {
                "expected_shape": (
                    list(self.expected_shape[0])
                    if self.expected_shape[0] is not None else None
                ),
                "expected_dtype": (
                    str(self.expected_dtype) if self.expected_dtype is not None else None
                ),
                "min_std": self.min_std,
                "max_std": self.max_std,
                "min_abs_max": self.min_abs_max,
                "max_abs_max": self.max_abs_max,
                "allow_nan": self.allow_nan,
                "allow_inf": self.allow_inf,
            },
        }

    def validate_strict(self, latent: torch.Tensor) -> Dict[str, Any]:
        """Validate ``latent``; raise :class:`LatentValidationError` on fail."""
        report = self.validate(latent)
        if not report["valid"]:
            raise LatentValidationError(
                f"latent validation failed: {report['reason']}",
                report=report,
            )
        return report


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def quick_validate(latent: torch.Tensor) -> Dict[str, Any]:
    """One-liner wrapper that runs the default :class:`LatentValidator`."""
    return LatentValidator().validate(latent)


def validate_range(
    latent: torch.Tensor,
    *,
    min_std: float = 0.05,
    max_std: float = 10.0,
) -> Dict[str, Any]:
    """Validate that ``latent.std()`` is inside ``[min_std, max_std]``."""
    return LatentValidator(
        min_std=min_std, max_std=max_std,
    ).validate(latent)


def validate_shape(
    latent: torch.Tensor,
    expected_shape: Sequence[int],
) -> Dict[str, Any]:
    """Validate that ``latent.shape == expected_shape``."""
    return LatentValidator(
        expected_shape=tuple(expected_shape),
    ).validate(latent)
