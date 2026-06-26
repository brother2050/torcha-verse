"""Backwards-compatible shim for the v0.6.x CI-gate runner.

The actual implementation lives in :mod:`scripts.check.ci_gates`.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.check.ci_gates import *  # noqa: F401,F403
from scripts.check.ci_gates import (  # noqa: F401
    GATE_REGISTRY,
    GateResult,
    main,
)


if __name__ == "__main__":
    sys.exit(main())
