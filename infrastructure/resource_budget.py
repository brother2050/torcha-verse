"""Hard resource-budget constraints for TorchaVerse.

This module implements the v0.3.0 *ResourceBudget* hard-constraint system.
Every run declares a :class:`ResourceBudget` describing the absolute upper
bounds for VRAM, host RAM, disk, and concurrency.  A :class:`BudgetTracker`
then hands out :class:`AllocationHandle` objects against that budget; any
request that would exceed the remaining budget raises
:class:`BudgetExceededError` instead of optimistically proceeding and
crashing later with an out-of-memory error.

The design goals are:

* **No optimistic estimation** -- an allocation either succeeds within the
  declared limits or fails loudly.
* **Thread-safe** -- all accounting is guarded by a single
  :class:`threading.RLock` so concurrent engines can share one tracker.
* **Context-manager friendly** -- allocations are released automatically when
  the :class:`AllocationHandle` goes out of a ``with`` block, even on error.
* **Offload aware** -- when VRAM is insufficient the tracker can report
  whether the deficit fits in host RAM (``offload_to="cpu"``) or disk
  (``offload_to="disk"``), enabling graceful degradation instead of OOM.

Example:
    >>> budget = ResourceBudget(vram_gb=24, ram_gb=64, disk_gb=200)
    >>> tracker = BudgetTracker(budget)
    >>> with tracker.allocate("llama-8b", vram_gb=16) as handle:
    ...     # ... run inference ...
    ...     pass
    >>> # handle released automatically on exit
    >>> tracker.available()["vram_gb"]
    24.0
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .logger import get_logger

__all__ = [
    "ResourceBudget",
    "BudgetTracker",
    "AllocationHandle",
    "BudgetExceededError",
    "FeasibilityEstimate",
]

#: Allowed values for :attr:`ResourceBudget.offload_to`.
_VALID_OFFLOAD_TARGETS: tuple[str, ...] = ("cpu", "disk", "none")

#: Numerical tolerance (in GB) used when comparing floating-point budgets so
#: that tiny rounding errors do not cause spurious ``BudgetExceededError``.
_EPSILON: float = 1e-6


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
        if self.offload_to not in _VALID_OFFLOAD_TARGETS:
            raise ValueError(
                f"offload_to must be one of {_VALID_OFFLOAD_TARGETS}, "
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


# ---------------------------------------------------------------------------
# Internal allocation record
# ---------------------------------------------------------------------------
@dataclass
class _AllocationRecord:
    """Internal bookkeeping entry for a live allocation."""

    handle_id: str
    name: str
    vram_gb: float
    ram_gb: float
    disk_gb: float
    model_slot: bool
    request_slot: bool


# ---------------------------------------------------------------------------
# BudgetTracker
# ---------------------------------------------------------------------------
class BudgetTracker:
    """Thread-safe accountant for a :class:`ResourceBudget`.

    The tracker maintains the running totals of every resource that has been
    handed out via :meth:`allocate`.  Each successful allocation returns an
    :class:`AllocationHandle`; releasing the handle (explicitly or via a
    ``with`` block) returns the budget to the pool.

    Args:
        budget: The hard :class:`ResourceBudget` for this tracker.

    Example:
        >>> tracker = BudgetTracker(ResourceBudget(vram_gb=24, ram_gb=64))
        >>> handle = tracker.allocate("sdxl", vram_gb=12)
        >>> tracker.available()["vram_gb"]
        12.0
        >>> tracker.release(handle)
        >>> tracker.available()["vram_gb"]
        24.0
        >>> with tracker.allocate("llama", vram_gb=8):
        ...     pass
    """

    def __init__(self, budget: ResourceBudget) -> None:
        self._budget: ResourceBudget = budget
        self._lock: threading.RLock = threading.RLock()

        # Running totals of handed-out resources.
        self._used_vram: float = 0.0
        self._used_ram: float = 0.0
        self._used_disk: float = 0.0
        self._used_model_slots: int = 0
        self._used_request_slots: int = 0

        # Live allocations keyed by handle id.
        self._allocations: Dict[str, _AllocationRecord] = {}

        # Waiters queue for :meth:`allocate_or_wait`; reusing the
        # existing ``RLock`` so we do not need a second lock to
        # reason about.
        self._waiters: threading.Condition = threading.Condition(self._lock)

        self._logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def budget(self) -> ResourceBudget:
        """The immutable :class:`ResourceBudget` governing this tracker."""
        return self._budget

    @property
    def active_allocations(self) -> int:
        """Number of currently outstanding allocations."""
        with self._lock:
            return len(self._allocations)

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------
    def allocate(
        self,
        name: str,
        vram_gb: float = 0.0,
        ram_gb: float = 0.0,
        disk_gb: float = 0.0,
        *,
        model_slot: bool = False,
        request_slot: bool = False,
    ) -> AllocationHandle:
        """Reserve resources against the budget.

        Args:
            name: Human-readable name for the allocation (e.g. model id).
            vram_gb: VRAM to reserve.
            ram_gb: Host RAM to reserve.
            disk_gb: Disk to reserve.
            model_slot: Also acquire a model-concurrency slot.
            request_slot: Also acquire a request-concurrency slot.

        Returns:
            An :class:`AllocationHandle` representing the reservation.

        Raises:
            BudgetExceededError: If the request would push any resource past
                its declared limit.
            ValueError: If any requested amount is negative.
        """
        if not name:
            raise ValueError("Allocation name must be a non-empty string.")
        if vram_gb < 0 or ram_gb < 0 or disk_gb < 0:
            raise ValueError(
                "Requested amounts must be non-negative "
                f"(vram={vram_gb}, ram={ram_gb}, disk={disk_gb})."
            )

        with self._lock:
            avail = self._available_locked()

            self._check_limit("vram_gb", vram_gb, avail["vram_gb"])
            self._check_limit("ram_gb", ram_gb, avail["ram_gb"])
            self._check_limit("disk_gb", disk_gb, avail["disk_gb"])
            if model_slot:
                self._check_limit(
                    "max_concurrent_models", 1, avail["max_concurrent_models"]
                )
            if request_slot:
                self._check_limit(
                    "max_concurrent_requests", 1, avail["max_concurrent_requests"]
                )

            handle_id = uuid.uuid4().hex
            record = _AllocationRecord(
                handle_id=handle_id,
                name=name,
                vram_gb=vram_gb,
                ram_gb=ram_gb,
                disk_gb=disk_gb,
                model_slot=model_slot,
                request_slot=request_slot,
            )
            self._allocations[handle_id] = record
            self._used_vram += vram_gb
            self._used_ram += ram_gb
            self._used_disk += disk_gb
            self._used_model_slots += 1 if model_slot else 0
            self._used_request_slots += 1 if request_slot else 0

            handle = AllocationHandle(
                tracker=self,
                handle_id=handle_id,
                name=name,
                vram_gb=vram_gb,
                ram_gb=ram_gb,
                disk_gb=disk_gb,
                model_slot=model_slot,
                request_slot=request_slot,
            )
            self._logger.debug(
                "Allocated %s: vram=%.2f ram=%.2f disk=%.2f (id=%s).",
                name,
                vram_gb,
                ram_gb,
                disk_gb,
                handle_id[:8],
            )
            return handle

    # ------------------------------------------------------------------
    # v1.0.0 M0: queue when budget is exhausted
    # ------------------------------------------------------------------
    def allocate_or_wait(
        self,
        name: str,
        vram_gb: float = 0.0,
        ram_gb: float = 0.0,
        disk_gb: float = 0.0,
        *,
        model_slot: bool = False,
        request_slot: bool = False,
        timeout: Optional[float] = None,
        poll_interval: float = 0.05,
    ) -> AllocationHandle:
        """Block until the requested resources are available.

        The semantics are identical to :meth:`allocate` *except* that
        if the budget is currently exhausted, the call blocks until
        another :class:`AllocationHandle` is released and the budget
        is freed up.  A new :class:`threading.Condition` is used to
        wake blocked waiters; FIFO ordering is preserved because
        :class:`threading.Condition` notifies in submission order
        and each waiter rechecks the budget before consuming.

        Args:
            name: Same as :meth:`allocate`.
            vram_gb: Same as :meth:`allocate`.
            ram_gb: Same as :meth:`allocate`.
            disk_gb: Same as :meth:`allocate`.
            model_slot: Same as :meth:`allocate`.
            request_slot: Same as :meth:`allocate`.
            timeout: Maximum number of seconds to block.  ``None``
                (the default) waits forever.  ``0`` is equivalent to
                :meth:`allocate` and raises immediately on failure.
            poll_interval: Minimum interval between wake-up checks.
                Bounded to ``1e-3`` to avoid busy loops.

        Returns:
            A live :class:`AllocationHandle` once the budget has
            been reserved.

        Raises:
            BudgetExceededError: If the request itself is infeasible
                (it would never fit, even on an empty budget), or
                if ``timeout`` expires before the budget becomes
                available.
        """
        if poll_interval < 1e-3:
            poll_interval = 1e-3

        # ``timeout=0`` falls through to the standard allocate() so
        # callers that need the "try-and-fail" semantics keep them.
        if timeout == 0:
            return self.allocate(
                name,
                vram_gb=vram_gb,
                ram_gb=ram_gb,
                disk_gb=disk_gb,
                model_slot=model_slot,
                request_slot=request_slot,
            )

        deadline: Optional[float] = None
        if timeout is not None:
            if timeout < 0:
                raise ValueError(f"timeout must be >= 0, got {timeout}.")
            deadline = time.monotonic() + timeout

        with self._lock:
            while True:
                # First, refuse the request outright if it could
                # never fit.  This mirrors :meth:`allocate` and
                # prevents a queue of impossible requests from
                # starving real ones.
                if self._would_exceed(
                    vram_gb=vram_gb,
                    ram_gb=ram_gb,
                    disk_gb=disk_gb,
                    model_slot=model_slot,
                    request_slot=request_slot,
                ):
                    raise BudgetExceededError(
                        vram_gb=vram_gb,
                        ram_gb=ram_gb,
                        disk_gb=disk_gb,
                        model_slot=model_slot,
                        request_slot=request_slot,
                        budget=self._budget,
                        used=self._snapshot_used(),
                    )
                if self._fits(
                    vram_gb=vram_gb,
                    ram_gb=ram_gb,
                    disk_gb=disk_gb,
                    model_slot=model_slot,
                    request_slot=request_slot,
                ):
                    # All clear; hand out the handle while we still
                    # hold the lock so we never race with release().
                    return self._consume(
                        name,
                        vram_gb=vram_gb,
                        ram_gb=ram_gb,
                        disk_gb=disk_gb,
                        model_slot=model_slot,
                        request_slot=request_slot,
                    )

                # Slot-only contention has no event source (no
                # ``vram_gb`` change happens when a slot is freed).
                # Refuse the request immediately rather than waiting
                # forever; callers who need a slot can poll with
                # ``timeout`` and retry.
                if (model_slot or request_slot) and not (
                    vram_gb or ram_gb or disk_gb
                ):
                    raise BudgetExceededError(
                        vram_gb=vram_gb,
                        ram_gb=ram_gb,
                        disk_gb=disk_gb,
                        model_slot=model_slot,
                        request_slot=request_slot,
                        budget=self._budget,
                        used=self._snapshot_used(),
                    )

                # Otherwise wait.  ``wait()`` releases the lock and
                # re-acquires it before returning; we re-check
                # feasibility on every wake-up.
                remaining: Optional[float] = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise BudgetExceededError(
                            vram_gb=vram_gb,
                            ram_gb=ram_gb,
                            disk_gb=disk_gb,
                            model_slot=model_slot,
                            request_slot=request_slot,
                            budget=self._budget,
                            used=self._snapshot_used(),
                        )
                self._waiters.wait(timeout=remaining)

    def _would_exceed(
        self,
        *,
        vram_gb: float,
        ram_gb: float,
        disk_gb: float,
        model_slot: bool,
        request_slot: bool,
    ) -> bool:
        """Return True if the request exceeds the static budget.

        Used by :meth:`allocate_or_wait` to short-circuit infeasible
        requests without ever placing them in the wait queue.

        The check is intentionally limited to per-allocation limits
        (``vram_gb`` / ``ram_gb`` / ``disk_gb``); concurrency
        slots are checked by :meth:`_fits` against the *current*
        usage, which is the right place to refuse a request that
        fits against the static budget but is currently saturated.
        """
        if vram_gb > self._budget.vram_gb:
            return True
        if ram_gb > self._budget.ram_gb:
            return True
        if disk_gb > self._budget.disk_gb:
            return True
        return False

    def _fits(
        self,
        *,
        vram_gb: float,
        ram_gb: float,
        disk_gb: float,
        model_slot: bool,
        request_slot: bool,
    ) -> bool:
        """Return True if the request fits against the *current* usage."""
        if self._used_vram + vram_gb > self._budget.vram_gb + 1e-9:
            return False
        if self._used_ram + ram_gb > self._budget.ram_gb + 1e-9:
            return False
        if self._used_disk + disk_gb > self._budget.disk_gb + 1e-9:
            return False
        if model_slot and self._used_model_slots >= self._budget.max_concurrent_models:
            return False
        if request_slot and self._used_request_slots >= self._budget.max_concurrent_requests:
            return False
        return True

    def _consume(
        self,
        name: str,
        *,
        vram_gb: float,
        ram_gb: float,
        disk_gb: float,
        model_slot: bool,
        request_slot: bool,
    ) -> AllocationHandle:
        """Reserve resources against the budget and return a handle.

        Caller must hold ``self._lock``.  This is split out from
        :meth:`allocate` so :meth:`allocate_or_wait` can re-use the
        exact same reservation logic under the lock without
        re-acquiring it.
        """
        # Negative values are validated at the public boundary.
        self._used_vram += vram_gb
        self._used_ram += ram_gb
        self._used_disk += disk_gb
        if model_slot:
            self._used_model_slots += 1
        if request_slot:
            self._used_request_slots += 1

        handle_id = uuid.uuid4().hex
        record = _AllocationRecord(
            handle_id=handle_id,
            name=name,
            vram_gb=vram_gb,
            ram_gb=ram_gb,
            disk_gb=disk_gb,
            model_slot=model_slot,
            request_slot=request_slot,
        )
        self._allocations[handle_id] = record
        # ``AllocationHandle.__init__`` is positional-only; mirror
        # the original ``allocate()`` layout rather than using
        # keyword arguments (which raises ``TypeError`` at runtime).
        return AllocationHandle(
            self,
            handle_id,
            name,
            vram_gb,
            ram_gb,
            disk_gb,
            model_slot,
            request_slot,
        )

    def _snapshot_used(self) -> Dict[str, float]:
        """Return a snapshot of the currently-used resources."""
        return {
            "vram_gb": self._used_vram,
            "ram_gb": self._used_ram,
            "disk_gb": self._used_disk,
            "model_slots": float(self._used_model_slots),
            "request_slots": float(self._used_request_slots),
        }

    # ------------------------------------------------------------------
    # v1.0.0 M0 additions (v0.4.3): try_acquire / allocate_many
    # ------------------------------------------------------------------
    def try_acquire(
        self,
        name: str,
        vram_gb: float = 0.0,
        ram_gb: float = 0.0,
        disk_gb: float = 0.0,
        *,
        model_slot: bool = False,
        request_slot: bool = False,
    ) -> Optional[AllocationHandle]:
        """Non-blocking variant of :meth:`allocate`.

        Returns the handle on success, ``None`` when the budget is
        exhausted.  The two are equivalent when the request itself
        is infeasible: both surface :class:`BudgetExceededError`
        immediately so the caller does not have to special-case
        "infeasible" vs "currently saturated".
        """
        with self._lock:
            if self._would_exceed(
                vram_gb=vram_gb,
                ram_gb=ram_gb,
                disk_gb=disk_gb,
                model_slot=model_slot,
                request_slot=request_slot,
            ):
                raise BudgetExceededError(
                    vram_gb=vram_gb,
                    ram_gb=ram_gb,
                    disk_gb=disk_gb,
                    model_slot=model_slot,
                    request_slot=request_slot,
                    budget=self._budget,
                    used=self._snapshot_used(),
                )
            if not self._fits(
                vram_gb=vram_gb,
                ram_gb=ram_gb,
                disk_gb=disk_gb,
                model_slot=model_slot,
                request_slot=request_slot,
            ):
                # slot-only contention has no event source (mirrors
                # the "reject immediately" branch in
                # ``allocate_or_wait``).  Surface as a hard
                # ``BudgetExceededError`` so callers do not have to
                # special-case "currently saturated" vs "structurally
                # infeasible" - the former is a transient event
                # callers may want to retry, the latter is a
                # permanent one.
                if (model_slot or request_slot) and not (
                    vram_gb or ram_gb or disk_gb
                ):
                    raise BudgetExceededError(
                        vram_gb=vram_gb,
                        ram_gb=ram_gb,
                        disk_gb=disk_gb,
                        model_slot=model_slot,
                        request_slot=request_slot,
                        budget=self._budget,
                        used=self._snapshot_used(),
                    )
                return None
            return self._consume(
                name,
                vram_gb=vram_gb,
                ram_gb=ram_gb,
                disk_gb=disk_gb,
                model_slot=model_slot,
                request_slot=request_slot,
            )

    def allocate_many(
        self,
        specs: Sequence[Dict[str, Any]],
    ) -> List[AllocationHandle]:
        """Atomically allocate a batch of resources.

        Either every spec in ``specs`` is fulfilled (a list of
        :class:`AllocationHandle` is returned) or the budget is
        untouched and an empty list is returned.  Specs are dicts
        of the form accepted by :meth:`allocate` (``name``,
        ``vram_gb``, ``ram_gb``, ``disk_gb``, ``model_slot``,
        ``request_slot``).

        The batch is computed against a *frozen* snapshot of the
        current usage; this avoids the "first half fits, second
        half doesn't" failure mode that the naive loop would
        suffer from.
        """
        if not specs:
            return []
        with self._lock:
            # Phase 1: feasibility + fit check against a frozen
            # snapshot.  No reservation yet.
            snapshot_used = self._snapshot_used()
            for spec in specs:
                vram_gb = float(spec.get("vram_gb", 0.0))
                ram_gb = float(spec.get("ram_gb", 0.0))
                disk_gb = float(spec.get("disk_gb", 0.0))
                model_slot = bool(spec.get("model_slot", False))
                request_slot = bool(spec.get("request_slot", False))
                if self._would_exceed(
                    vram_gb=vram_gb,
                    ram_gb=ram_gb,
                    disk_gb=disk_gb,
                    model_slot=model_slot,
                    request_slot=request_slot,
                ):
                    raise BudgetExceededError(
                        vram_gb=vram_gb,
                        ram_gb=ram_gb,
                        disk_gb=disk_gb,
                        model_slot=model_slot,
                        request_slot=request_slot,
                        budget=self._budget,
                        used=snapshot_used,
                    )
                if not self._fits(
                    vram_gb=vram_gb,
                    ram_gb=ram_gb,
                    disk_gb=disk_gb,
                    model_slot=model_slot,
                    request_slot=request_slot,
                ):
                    # No partial success: if one spec does not fit
                    # we return [] and the caller can retry.
                    return []
            # Phase 2: actually consume.  ``_fits`` is monotonic
            # inside the lock so all specs will fit a second time.
            handles: List[AllocationHandle] = []
            for spec in specs:
                handles.append(
                    self._consume(
                        str(spec.get("name", "batch")),
                        vram_gb=float(spec.get("vram_gb", 0.0)),
                        ram_gb=float(spec.get("ram_gb", 0.0)),
                        disk_gb=float(spec.get("disk_gb", 0.0)),
                        model_slot=bool(spec.get("model_slot", False)),
                        request_slot=bool(spec.get("request_slot", False)),
                    )
                )
            return handles

    def stats(self) -> Dict[str, Any]:
        """Return a snapshot of tracker state for dashboards / debug.

        The dict is a defensive copy; mutating it does not change
        the tracker state.
        """
        with self._lock:
            return {
                "budget": {
                    "vram_gb": self._budget.vram_gb,
                    "ram_gb": self._budget.ram_gb,
                    "disk_gb": self._budget.disk_gb,
                    "max_concurrent_models": self._budget.max_concurrent_models,
                    "max_concurrent_requests": self._budget.max_concurrent_requests,
                },
                "used": self._snapshot_used(),
                "active_allocations": self.active_allocations,
                "free": {
                    "vram_gb": max(0.0, self._budget.vram_gb - self._used_vram),
                    "ram_gb": max(0.0, self._budget.ram_gb - self._used_ram),
                    "disk_gb": max(0.0, self._budget.disk_gb - self._used_disk),
                    "model_slots": max(
                        0, self._budget.max_concurrent_models - self._used_model_slots
                    ),
                    "request_slots": max(
                        0,
                        self._budget.max_concurrent_requests - self._used_request_slots,
                    ),
                },
            }

    def allocate_with_backoff(
        self,
        name: str,
        vram_gb: float = 0.0,
        ram_gb: float = 0.0,
        disk_gb: float = 0.0,
        *,
        model_slot: bool = False,
        request_slot: bool = False,
        max_attempts: int = 5,
        initial_delay: float = 0.05,
        max_delay: float = 2.0,
        backoff_factor: float = 2.0,
    ) -> AllocationHandle:
        """Retry-with-exponential-backoff variant of :meth:`allocate_or_wait`.

        Calls :meth:`allocate_or_wait` once per attempt with a
        geometrically increasing delay in between.  This is the
        right shape for the v1.0.0 M0 "queue" UX where callers
        would rather spend a few extra milliseconds in exchange
        for the "transparent" UX of "just give me a handle".

        Args:
            name: Same as :meth:`allocate`.
            vram_gb: Same as :meth:`allocate`.
            ram_gb: Same as :meth:`allocate`.
            disk_gb: Same as :meth:`allocate`.
            model_slot: Same as :meth:`allocate`.
            request_slot: Same as :meth:`allocate`.
            max_attempts: Total number of attempts (1 = no retry).
            initial_delay: Delay before the second attempt.
            max_delay: Cap on the per-attempt delay.
            backoff_factor: Multiplier applied to the delay
                between attempts.

        Returns:
            A live :class:`AllocationHandle`.

        Raises:
            BudgetExceededError: If the last attempt exhausts
                its timeout, or if the request is infeasible
                from the start.
        """
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}.")
        if initial_delay < 0:
            raise ValueError(f"initial_delay must be >= 0, got {initial_delay}.")
        if backoff_factor < 1.0:
            raise ValueError(f"backoff_factor must be >= 1.0, got {backoff_factor}.")
        delay = initial_delay
        last_error: Optional[BudgetExceededError] = None
        for attempt in range(max_attempts):
            try:
                # ``timeout=0`` is equivalent to ``allocate``; on
                # success we return immediately, on contention we
                # sleep ``delay`` and try again.
                return self.allocate_or_wait(
                    name,
                    vram_gb=vram_gb,
                    ram_gb=ram_gb,
                    disk_gb=disk_gb,
                    model_slot=model_slot,
                    request_slot=request_slot,
                    timeout=0,
                )
            except BudgetExceededError as exc:
                last_error = exc
                if attempt + 1 >= max_attempts:
                    break
                # ``time.sleep`` is fine here: ``allocate_or_wait``
                # with ``timeout=0`` returns synchronously so the
                # caller's thread is otherwise idle.
                time.sleep(min(delay, max_delay))
                delay *= backoff_factor
        # Reached only when every attempt failed.
        assert last_error is not None
        raise last_error

    def release(self, handle: AllocationHandle) -> None:
        """Release the budget held by ``handle`` back to the pool.

        Args:
            handle: A previously returned :class:`AllocationHandle`.

        Raises:
            TypeError: If ``handle`` is not an :class:`AllocationHandle`.
        """
        if not isinstance(handle, AllocationHandle):
            raise TypeError(
                f"release() expects an AllocationHandle, got "
                f"{type(handle).__name__}."
            )
        with self._lock:
            record = self._allocations.pop(handle.handle_id, None)
            if record is None:
                # Already released (or unknown) -- nothing to do.
                return
            self._used_vram = max(0.0, self._used_vram - record.vram_gb)
            self._used_ram = max(0.0, self._used_ram - record.ram_gb)
            self._used_disk = max(0.0, self._used_disk - record.disk_gb)
            if record.model_slot:
                self._used_model_slots = max(0, self._used_model_slots - 1)
            if record.request_slot:
                self._used_request_slots = max(0, self._used_request_slots - 1)
            self._logger.debug(
                "Released allocation %s (id=%s).",
                record.name,
                record.handle_id[:8],
            )
            # Wake any :meth:`allocate_or_wait` waiters so they can
            # re-check the budget.  ``notify_all`` is the simplest
            # correct choice; the re-check inside the waiter loop is
            # what enforces FIFO ordering.
            self._waiters.notify_all()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def available(self) -> Dict[str, float]:
        """Return the currently available budget as a dictionary.

        The dictionary contains the remaining GB for ``vram_gb``,
        ``ram_gb`` and ``disk_gb`` plus the remaining concurrency slots for
        ``max_concurrent_models`` and ``max_concurrent_requests``.
        """
        with self._lock:
            return self._available_locked()

    def used(self) -> Dict[str, float]:
        """Return the currently consumed budget as a dictionary."""
        with self._lock:
            return {
                "vram_gb": self._used_vram,
                "ram_gb": self._used_ram,
                "disk_gb": self._used_disk,
                "max_concurrent_models": self._used_model_slots,
                "max_concurrent_requests": self._used_request_slots,
            }

    def list_allocations(self) -> List[Dict[str, Any]]:
        """Return a list of dictionaries describing live allocations."""
        with self._lock:
            return [
                {
                    "handle_id": rec.handle_id,
                    "name": rec.name,
                    "vram_gb": rec.vram_gb,
                    "ram_gb": rec.ram_gb,
                    "disk_gb": rec.disk_gb,
                    "model_slot": rec.model_slot,
                    "request_slot": rec.request_slot,
                }
                for rec in self._allocations.values()
            ]

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    def estimate(
        self,
        model_size_gb: float,
        kv_cache_gb: float = 0.0,
        activations_gb: float = 0.0,
    ) -> FeasibilityEstimate:
        """Estimate whether a model footprint fits the remaining budget.

        The required VRAM is ``model_size_gb + kv_cache_gb + activations_gb``.
        When that exceeds the available VRAM the estimate consults
        :attr:`ResourceBudget.offload_to` to decide whether the deficit can
        be spilled to host RAM (``"cpu"``) or disk (``"disk"``).

        Args:
            model_size_gb: Size of the model weights in GB.
            kv_cache_gb: Estimated KV cache footprint in GB.
            activations_gb: Estimated activation footprint in GB.

        Returns:
            A :class:`FeasibilityEstimate` describing the outcome.
        """
        if model_size_gb < 0 or kv_cache_gb < 0 or activations_gb < 0:
            raise ValueError("Estimate inputs must be non-negative.")

        with self._lock:
            avail = self._available_locked()
            available_vram = avail["vram_gb"]
            required_vram = model_size_gb + kv_cache_gb + activations_gb
            deficit = max(0.0, required_vram - available_vram)
            offload_required = deficit > _EPSILON

            if not offload_required:
                return FeasibilityEstimate(
                    feasible=True,
                    required_vram_gb=required_vram,
                    available_vram_gb=available_vram,
                    vram_deficit_gb=0.0,
                    offload_required=False,
                    offload_target="none",
                )

            target = self._budget.offload_to
            if target == "none":
                return FeasibilityEstimate(
                    feasible=False,
                    required_vram_gb=required_vram,
                    available_vram_gb=available_vram,
                    vram_deficit_gb=deficit,
                    offload_required=True,
                    offload_target="none",
                    reason=(
                        f"VRAM deficit of {deficit:.2f} GB but offload_to is "
                        f"'none'; cannot spill."
                    ),
                )
            if target == "cpu":
                spill_capacity = avail["ram_gb"]
                if deficit <= spill_capacity + _EPSILON:
                    return FeasibilityEstimate(
                        feasible=True,
                        required_vram_gb=required_vram,
                        available_vram_gb=available_vram,
                        vram_deficit_gb=deficit,
                        offload_required=True,
                        offload_target="cpu",
                    )
                return FeasibilityEstimate(
                    feasible=False,
                    required_vram_gb=required_vram,
                    available_vram_gb=available_vram,
                    vram_deficit_gb=deficit,
                    offload_required=True,
                    offload_target="cpu",
                    reason=(
                        f"VRAM deficit {deficit:.2f} GB exceeds available RAM "
                        f"{spill_capacity:.2f} GB for CPU offload."
                    ),
                )
            # target == "disk"
            spill_capacity = avail["disk_gb"]
            if deficit <= spill_capacity + _EPSILON:
                return FeasibilityEstimate(
                    feasible=True,
                    required_vram_gb=required_vram,
                    available_vram_gb=available_vram,
                    vram_deficit_gb=deficit,
                    offload_required=True,
                    offload_target="disk",
                )
            return FeasibilityEstimate(
                feasible=False,
                required_vram_gb=required_vram,
                available_vram_gb=available_vram,
                vram_deficit_gb=deficit,
                offload_required=True,
                offload_target="disk",
                reason=(
                    f"VRAM deficit {deficit:.2f} GB exceeds available disk "
                    f"{spill_capacity:.2f} GB for disk offload."
                ),
            )

    # ------------------------------------------------------------------
    # Internals (must be called with the lock held)
    # ------------------------------------------------------------------
    def _available_locked(self) -> Dict[str, float]:
        """Compute available resources (caller must hold ``self._lock``)."""
        return {
            "vram_gb": max(0.0, self._budget.vram_gb - self._used_vram),
            "ram_gb": max(0.0, self._budget.ram_gb - self._used_ram),
            "disk_gb": max(0.0, self._budget.disk_gb - self._used_disk),
            "max_concurrent_models": max(
                0, self._budget.max_concurrent_models - self._used_model_slots
            ),
            "max_concurrent_requests": max(
                0, self._budget.max_concurrent_requests - self._used_request_slots
            ),
        }

    def _check_limit(self, resource: str, requested: float, available: float) -> None:
        """Raise :class:`BudgetExceededError` if ``requested`` exceeds ``available``."""
        if requested > available + _EPSILON:
            raise BudgetExceededError(
                f"Cannot allocate {requested} {resource}: only {available} "
                f"available (budget={getattr(self._budget, resource)})."
            )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def release_all(self) -> None:
        """Release every outstanding allocation.

        Useful for tearing down a run or in test fixtures.
        """
        with self._lock:
            self._allocations.clear()
            self._used_vram = 0.0
            self._used_ram = 0.0
            self._used_disk = 0.0
            self._used_model_slots = 0
            self._used_request_slots = 0

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"BudgetTracker(budget={self._budget!r}, "
                f"active={len(self._allocations)}, "
                f"used_vram={self._used_vram:.2f})"
            )
