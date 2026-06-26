"""Experiments package (v1.0.0).

Re-exports the A/B testing framework primitives from
:mod:`experiments.framework` so callers can do::

    from experiments import Experiment, Variant, ExperimentRunner

without having to know the underlying submodule layout.
"""
from __future__ import annotations

from .framework import *  # noqa: F401,F403
from .framework import (  # noqa: F401  - explicit re-export for type checkers
    Experiment,
    ExperimentRunner,
    Variant,
    bucket_assign,
)

__all__ = [
    "Experiment",
    "Variant",
    "ExperimentRunner",
    "bucket_assign",
]
