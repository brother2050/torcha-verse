"""Feasibility-probe helper for :class:`BudgetTracker`.

The :meth:`BudgetTracker.estimate` method is a *pure function* of
``(model_size_gb, kv_cache_gb, activations_gb)`` plus the snapshot
of available resources at the moment of the call.  Because the
function does not mutate tracker state (it only reads the locked
snapshot), we factor it out as :func:`feasibility_for` and have the
tracker call it inside its lock-held section.

This split keeps :class:`BudgetTracker` focused on CRUD-style
allocate / release while :func:`feasibility_for` remains a small
``~100``-line function that is easy to unit-test in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ._types import FeasibilityEstimate, EPSILON

__all__ = ["feasibility_for", "FeasibilityInputs"]


@dataclass(frozen=True)
class FeasibilityInputs:
    """Frozen snapshot of the inputs to :func:`feasibility_for`.

    The dataclass is immutable so callers can share a snapshot
    across threads without a separate lock.
    """

    model_size_gb: float
    kv_cache_gb: float
    activations_gb: float
    available_vram_gb: float
    available_ram_gb: float
    available_disk_gb: float
    offload_to: str  # one of "cpu" / "disk" / "none"


def feasibility_for(inputs: FeasibilityInputs) -> FeasibilityEstimate:
    """Return a :class:`FeasibilityEstimate` for the given snapshot.

    The algorithm is:

    1. ``required_vram = model + kv + activations``
    2. ``deficit = max(0, required_vram - available_vram)``
    3. If ``deficit == 0`` -> feasible, no offload.
    4. Else, consult ``offload_to``:

       * ``"none"`` -> not feasible.
       * ``"cpu"`` -> feasible if ``deficit <= available_ram`` else not.
       * ``"disk"`` -> feasible if ``deficit <= available_disk`` else not.

    Args:
        inputs: The frozen snapshot of budget / offload policy.

    Returns:
        The structured :class:`FeasibilityEstimate`.
    """
    if (
        inputs.model_size_gb < 0
        or inputs.kv_cache_gb < 0
        or inputs.activations_gb < 0
    ):
        raise ValueError("Estimate inputs must be non-negative.")

    required_vram = (
        inputs.model_size_gb + inputs.kv_cache_gb + inputs.activations_gb
    )
    deficit = max(0.0, required_vram - inputs.available_vram_gb)
    offload_required = deficit > EPSILON

    if not offload_required:
        return FeasibilityEstimate(
            feasible=True,
            required_vram_gb=required_vram,
            available_vram_gb=inputs.available_vram_gb,
            vram_deficit_gb=0.0,
            offload_required=False,
            offload_target="none",
        )

    target = inputs.offload_to
    if target == "none":
        return FeasibilityEstimate(
            feasible=False,
            required_vram_gb=required_vram,
            available_vram_gb=inputs.available_vram_gb,
            vram_deficit_gb=deficit,
            offload_required=True,
            offload_target="none",
            reason=(
                f"VRAM deficit of {deficit:.2f} GB but offload_to is "
                f"'none'; cannot spill."
            ),
        )

    if target == "cpu":
        spill = inputs.available_ram_gb
        if deficit <= spill + EPSILON:
            return FeasibilityEstimate(
                feasible=True,
                required_vram_gb=required_vram,
                available_vram_gb=inputs.available_vram_gb,
                vram_deficit_gb=deficit,
                offload_required=True,
                offload_target="cpu",
            )
        return FeasibilityEstimate(
            feasible=False,
            required_vram_gb=required_vram,
            available_vram_gb=inputs.available_vram_gb,
            vram_deficit_gb=deficit,
            offload_required=True,
            offload_target="cpu",
            reason=(
                f"VRAM deficit {deficit:.2f} GB exceeds available RAM "
                f"{spill:.2f} GB for CPU offload."
            ),
        )

    # target == "disk"
    spill = inputs.available_disk_gb
    if deficit <= spill + EPSILON:
        return FeasibilityEstimate(
            feasible=True,
            required_vram_gb=required_vram,
            available_vram_gb=inputs.available_vram_gb,
            vram_deficit_gb=deficit,
            offload_required=True,
            offload_target="disk",
        )
    return FeasibilityEstimate(
        feasible=False,
        required_vram_gb=required_vram,
        available_vram_gb=inputs.available_vram_gb,
        vram_deficit_gb=deficit,
        offload_required=True,
        offload_target="disk",
        reason=(
            f"VRAM deficit {deficit:.2f} GB exceeds available disk "
            f"{spill:.2f} GB for disk offload."
        ),
    )
