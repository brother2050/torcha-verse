"""Resource-budget subsystem (v0.6.x refactored from a single 1279-line file).

The :mod:`infrastructure.resource_budget` sub-package is the
framework's hard-constraint budget system.  Every run declares a
:class:`ResourceBudget` describing the absolute upper bounds for
VRAM, host RAM, disk, and concurrency.  A :class:`BudgetTracker`
then hands out :class:`AllocationHandle` objects against that
budget; any request that would exceed the remaining budget raises
:class:`BudgetExceededError` instead of optimistically proceeding
and crashing later with an out-of-memory error.

The v0.6.x refactor splits the previous single-file
``infrastructure/resource_budget.py`` (1279 lines) into four
focused modules:

* :mod:`infrastructure.resource_budget._types` -- pure data
  classes (``ResourceBudget`` / ``AllocationHandle`` /
  ``FeasibilityEstimate``) and the ``BudgetExceededError``
  exception.
* :mod:`infrastructure.resource_budget._lock` -- the
  :func:`threadsafe` decorator that the tracker uses for
  re-entrant locking.
* :mod:`infrastructure.resource_budget._feasibility` -- the
  pure function :func:`feasibility_for` that decides whether a
  model footprint fits the budget (with offload).
* :mod:`infrastructure.resource_budget._tracker` -- the
  thread-safe :class:`BudgetTracker` and its private
  :class:`_AllocationRecord` bookkeeping row.

The public API is unchanged -- ``from infrastructure.resource_budget
import ResourceBudget, BudgetTracker, AllocationHandle,
BudgetExceededError, FeasibilityEstimate`` keeps working.  The
single-file import ``from infrastructure.resource_budget import
VALID_OFFLOAD_TARGETS`` is *also* preserved via the shim module
described below.

Sub-package layout::

    infrastructure/resource_budget/
        __init__.py        -- this file (public facade)
        _types.py          -- data classes + exception
        _lock.py           -- @threadsafe decorator
        _feasibility.py    -- feasibility_for() pure function
        _tracker.py        -- BudgetTracker + _AllocationRecord

Backward compatibility
----------------------

The legacy import path
``from infrastructure.resource_budget import (
    ResourceBudget, BudgetTracker, AllocationHandle,
    BudgetExceededError, FeasibilityEstimate,
    _VALID_OFFLOAD_TARGETS, _EPSILON,
)``
is preserved by the shim at
``/workspace/torcha-verse/infrastructure/resource_budget.py``
which re-exports everything from the sub-package.
"""

from __future__ import annotations

from ._lock import threadsafe
from ._types import (
    EPSILON,
    VALID_OFFLOAD_TARGETS,
    AllocationHandle,
    BudgetExceededError,
    FeasibilityEstimate,
    ResourceBudget,
)
from ._feasibility import FeasibilityInputs, feasibility_for
from ._tracker import BudgetTracker, _AllocationRecord

__all__ = [
    # Data classes
    "ResourceBudget",
    "AllocationHandle",
    "FeasibilityEstimate",
    # Exception
    "BudgetExceededError",
    # Tracker
    "BudgetTracker",
    # Helpers
    "threadsafe",
    "feasibility_for",
    "FeasibilityInputs",
    "VALID_OFFLOAD_TARGETS",
    "EPSILON",
    # Internal (exposed for testing)
    "_AllocationRecord",
]
