"""Node-level logger + module-level constants shared across the base.

The :data:`_NODE_KIND` constant is the :class:`ModuleBus` namespace
under which every node is registered; the :data:`_logger` is a
stdlib-only logger (no torch) used by the registration helpers
and the safe-execute wrapper.
"""

from __future__ import annotations

import logging

from infrastructure.logger import get_logger

__all__ = ["_NODE_KIND", "_logger"]


#: ModuleBus ``kind`` namespace under which every node is registered.
_NODE_KIND: str = "node"

#: Module-level logger for the node system (stdlib only -- no torch).
_logger: logging.Logger = get_logger("nodes")
