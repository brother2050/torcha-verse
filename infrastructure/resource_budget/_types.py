"""Data types and exceptions for the v0.6.x resource-budget sub-package.

This module contains the *plain data* and *exception* classes that
underpin :class:`~infrastructure.resource_budget.BudgetTracker`:

* :class:`BudgetExceededError` -- raised when an allocation request
  cannot be satisfied within the declared budget.
* :class:`ResourceBudget` -- the immutable upper-bound declaration
  (vram / ram / disk / slots / offload).
* :class:`FeasibilityEstimate` -- the structured return type of
  :meth:`BudgetTracker.estimate_feasibility`.
* :class:`AllocationHandle` -- the context-manager handle returned
  by :meth:`BudgetTracker.allocate`.

The thread-safe :class:`~infrastructure.resource_budget.BudgetTracker`
lives in :mod:`infrastructure.resource_budget._tracker`.  The
:func:`@threadsafe <infrastructure.resource_budget._lock.threadsafe>`
decorator that the tracker uses lives in
:mod:`infrastructure.resource_budget._lock`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

__all__ = [
    "BudgetExceededError",
    "ResourceBudget",
    "FeasibilityEstimate",
    "AllocationHandle",
    "VALID_OFFLOAD_TARGETS",
    "EPSILON",
]

#: Allowed values for :attr:`ResourceBudget.offload_to`.
VALID_OFFLOAD_TARGETS: Tuple[str, ...] = ("cpu", "disk", "none")

#: Numerical tolerance (in GB) used when comparing floating-point budgets
#: so that round-off does not produce spurious :class:`BudgetExceededError`
#: when a request is exactly at the limit.
EPSILON: float = 1e-6


class BudgetExceededError(RuntimeError):
    """Raised when an allocation request exceeds the available budget.

    The exception message describes which resource was exhausted, the
    requested amount, and the amount that was actually available.

    Args:
        vram_gb: VRAM that the caller asked for.  ``0.0`` when the
            request was a slot-only allocation.
        ram_gb: RAM that the caller asked for.
        disk_gb: Disk that the caller asked for.
        model_slot: Whether the request asked for a model slot.
        request_slot: Whether the request asked for a request slot.
        budget: Optional :class:`ResourceBudget` describing the
            static limits.  Used to enrich the message.
        used: Optional dict of currently-used resources.  Used to
            enrich the message.
        message: Optional human-readable message; when omitted, the
            constructor builds a canonical description.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        vram_gb: float = 0.0,
        ram_gb: float = 0.0,
        disk_gb: float = 0.0,
        model_slot: bool = False,
        request_slot: bool = False,
        budget: Optional[ResourceBudget] = None,
        used: Optional[Dict[str, float]] = None,
    ) -> None:
        if message is None:
            parts: list = []
            if vram_gb:
                parts.append(f"vram_gb={vram_gb}")
            if ram_gb:
                parts.append(f"ram_gb={ram_gb}")
            if disk_gb:
                parts.append(f"disk_gb={disk_gb}")
            if model_slot:
                parts.append("model_slot")
            if request_slot:
                parts.append("request_slot")
            if budget is not None:
                parts.append(
                    f"budget=ResourceBudget(vram={budget.vram_gb}, "
                    f"ram={budget.ram_gb}, disk={budget.disk_gb}, "
                    f"models={budget.max_concurrent_models}, "
                    f"reqs={budget.max_concurrent_requests})"
                )
            if used:
                parts.append(f"used={used}")
            message = (
                "Budget exceeded for allocation: " + ", ".join(parts)
                if parts
                else "Budget exceeded"
            )
        super().__init__(message)
        self.vram_gb = float(vram_gb)
        self.ram_gb = float(ram_gb)
        self.disk_gb = float(disk_gb)
        self.model_slot = bool(model_slot)
        self.request_slot = bool(request_slot)
        self.budget = budget
        self.used = used


# ---------------------------------------------------------------------------
# ResourceBudget dataclass
# ---------------------------------------------------------------------------
@dataclass
class ResourceBudget:
    """Immutable description of the hard resource limits for a run.

    All ``*_gb`` fields are expressed in gigabytes (decimal, base 10 is not
    used -- 1 GB == 1024**3 bytes is implied by the surrounding memory
    accounting which works in bytes and converts with ``/ (1024**3)``).

    Attributes:
        vram_gb: Total GPU memory budget across all visible devices.
        ram_gb: Total host (CPU) memory budget.
        disk_gb: Total disk budget for caches, checkpoints and downloads.
        max_concurrent_models: Maximum number of models resident at once.
        max_concurrent_requests: Maximum number of in-flight inference
            requests the scheduler will admit.
        kv_cache_gb: Portion of VRAM reserved for KV cache accounting.
        activations_gb: Portion of VRAM reserved for activation accounting.
        offload_to: Where to spill weights when VRAM is exhausted.  One of
            ``"cpu"``, ``"disk"`` or ``"none"`` (raise instead of offload).

    Example:
        >>> ResourceBudget(vram_gb=24, ram_gb=64, disk_gb=200,
        ...                max_concurrent_models=2, offload_to="cpu")
        ResourceBudget(vram_gb=24.0, ram_gb=64.0, disk_gb=200.0, \
max_concurrent_models=2, max_concurrent_requests=1, kv_cache_gb=0.0, \
activations_gb=0.0, offload_to='cpu')
    """

    vram_gb: float = 0.0
    ram_gb: float = 0.0
    disk_gb: float = 0.0
    max_concurrent_models: int = 1
    max_concurrent_requests: int = 1
    kv_cache_gb: float = 0.0
    activations_gb: float = 0.0
    offload_to: str = "none"

    def __post_init__(self) -> None:
        """Validate the budget fields after dataclass initialisation."""
        if self.vram_gb < 0:
            raise ValueError(f"vram_gb must be >= 0, got {self.vram_gb}.")
        if self.ram_gb < 0:
            raise ValueError(f"ram_gb must be >= 0, got {self.ram_gb}.")
        if self.disk_gb < 0:
            raise ValueError(f"disk_gb must be >= 0, got {self.disk_gb}.")
        if self.max_concurrent_models < 0:
            raise ValueError(
                f"max_concurrent_models must be >= 0, got "
                f"{self.max_concurrent_models}."
            )
        if self.max_concurrent_requests < 0:
            raise ValueError(
                f"max_concurrent_requests must be >= 0, got "
                f"{self.max_concurrent_requests}."
            )
        if self.kv_cache_gb < 0:
            raise ValueError(f"kv_cache_gb must be >= 0, got {self.kv_cache_gb}.")
        if self.activations_gb < 0:
            raise ValueError(
                f"activations_gb must be >= 0, got {self.activations_gb}."
            )
        if self.offload_to not in VALID_OFFLOAD_TARGETS:
            raise ValueError(
                f"offload_to must be one of {VALID_OFFLOAD_TARGETS}, "
                f"got {self.offload_to!r}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation of the budget."""
        return {
            "vram_gb": self.vram_gb,
            "ram_gb": self.ram_gb,
            "disk_gb": self.disk_gb,
            "max_concurrent_models": self.max_concurrent_models,
            "max_concurrent_requests": self.max_concurrent_requests,
            "kv_cache_gb": self.kv_cache_gb,
            "activations_gb": self.activations_gb,
            "offload_to": self.offload_to,
        }


# ---------------------------------------------------------------------------
# Feasibility estimate result
# ---------------------------------------------------------------------------
@dataclass
class FeasibilityEstimate:
    """Result of :meth:`BudgetTracker.estimate`.

    Attributes:
        feasible: Whether the requested footprint fits within the budget
            (taking :attr:`ResourceBudget.offload_to` into account).
        required_vram_gb: Total VRAM the request would need.
        available_vram_gb: VRAM currently free in the tracker.
        vram_deficit_gb: VRAM that must be offloaded (``0`` if it all fits).
        offload_required: ``True`` when VRAM alone is insufficient.
        offload_target: Where the deficit would be spilled.
        reason: Human-readable explanation (empty when feasible).
    """

    feasible: bool
    required_vram_gb: float
    available_vram_gb: float
    vram_deficit_gb: float
    offload_required: bool
    offload_target: str
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation of the estimate."""
        return {
            "feasible": self.feasible,
            "required_vram_gb": self.required_vram_gb,
            "available_vram_gb": self.available_vram_gb,
            "vram_deficit_gb": self.vram_deficit_gb,
            "offload_required": self.offload_required,
            "offload_target": self.offload_target,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# AllocationHandle
# ---------------------------------------------------------------------------
class AllocationHandle:
    """Opaque handle returned by :meth:`BudgetTracker.allocate`.

    The handle is a context manager: entering a ``with`` block returns the
    handle itself, and exiting the block releases the associated budget
    automatically (even if an exception propagates).  Manual release is also
    possible via :meth:`release`; calling it more than once is a no-op.

    Attributes:
        handle_id: Unique identifier for this allocation.
        name: Human-readable name supplied at allocation time.
        vram_gb: VRAM reserved by this allocation.
        ram_gb: Host RAM reserved by this allocation.
        disk_gb: Disk reserved by this allocation.
        model_slot: Whether a model concurrency slot was taken.
        request_slot: Whether a request concurrency slot was taken.
    """

    def __init__(
        self,
        tracker: "BudgetTracker",
        handle_id: str,
        name: str,
        vram_gb: float,
        ram_gb: float,
        disk_gb: float,
        model_slot: bool,
        request_slot: bool,
    ) -> None:
        self._tracker: BudgetTracker = tracker
        self._handle_id: str = handle_id
        self._name: str = name
        self._vram_gb: float = vram_gb
        self._ram_gb: float = ram_gb
        self._disk_gb: float = disk_gb
        self._model_slot: bool = model_slot
        self._request_slot: bool = request_slot
        self._released: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def handle_id(self) -> str:
        """Unique identifier for this allocation."""
        return self._handle_id

    @property
    def name(self) -> str:
        """Human-readable name supplied at allocation time."""
        return self._name

    @property
    def vram_gb(self) -> float:
        """VRAM reserved by this allocation (GB)."""
        return self._vram_gb

    @property
    def ram_gb(self) -> float:
        """Host RAM reserved by this allocation (GB)."""
        return self._ram_gb

    @property
    def disk_gb(self) -> float:
        """Disk reserved by this allocation (GB)."""
        return self._disk_gb

    @property
    def model_slot(self) -> bool:
        """Whether a model concurrency slot was taken."""
        return self._model_slot

    @property
    def request_slot(self) -> bool:
        """Whether a request concurrency slot was taken."""
        return self._request_slot

    @property
    def released(self) -> bool:
        """``True`` once the budget has been returned to the tracker."""
        return self._released

    # ------------------------------------------------------------------
    # Release / context manager
    # ------------------------------------------------------------------
    def release(self) -> None:
        """Return the reserved budget to the tracker.

        Safe to call multiple times; only the first call has an effect.
        """
        if self._released:
            return
        self._tracker.release(self)
        self._released = True

    def __enter__(self) -> "AllocationHandle":
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        self.release()
        return False  # do not suppress exceptions

    def __repr__(self) -> str:
        return (
            f"AllocationHandle(id={self._handle_id[:8]}, name={self._name!r}, "
            f"vram_gb={self._vram_gb}, ram_gb={self._ram_gb}, "
            f"disk_gb={self._disk_gb}, released={self._released})"
        )
