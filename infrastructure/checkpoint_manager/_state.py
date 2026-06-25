"""RNG state capture / restore for reproducible resume.

The :class:`CheckpointManager` saves the random-number-generator
state of the active stacks (Python / numpy / torch / CUDA) so a
training run can resume deterministically.  The capture /
restore pair is factored out into this module because the
operations are pure and do not depend on the manager's instance
state.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

import torch

from ..logger import get_logger

__all__ = ["capture_rng_states", "restore_rng_states"]


_logger = get_logger("infrastructure.checkpoint_manager._state")


def capture_rng_states() -> Dict[str, Any]:
    """Capture RNG states for reproducible resume.

    Returns:
        A dict with ``python`` (always), ``numpy`` (if numpy is
        installed), ``torch`` (always) and ``torch_cuda`` (only
        when CUDA is available).
    """
    states: Dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np

        states["numpy"] = np.random.get_state()
    except Exception as exc:  # pragma: no cover - numpy optional
        _logger.debug("Skipping numpy RNG state capture: %s", exc)
    states["torch"] = torch.get_rng_state()
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: Optional[Dict[str, Any]]) -> None:
    """Restore previously captured RNG states.

    A ``None`` or empty ``states`` is a no-op so callers do not
    have to check.
    """
    if not states:
        return
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        try:
            import numpy as np

            np.random.set_state(states["numpy"])
        except Exception as exc:  # pragma: no cover
            _logger.debug("Skipping numpy RNG state restore: %s", exc)
    if "torch" in states:
        torch.set_rng_state(states["torch"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])
