"""Backwards-compatible shim for the v0.6.x degrade-logging scanner.

The actual implementation lives in :mod:`scripts.check.degrade_logging`.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.check.degrade_logging import *  # noqa: F401,F403
from scripts.check.degrade_logging import main  # noqa: F401


if __name__ == "__main__":
    sys.exit(main())
