"""Backwards-compatible shim for the v0.6.x placeholder-registry scanner.

The actual implementation lives in :mod:`scripts.check.placeholders`.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.check.placeholders import *  # noqa: F401,F403
from scripts.check.placeholders import main  # noqa: F401


if __name__ == "__main__":
    sys.exit(main())
