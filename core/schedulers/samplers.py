"""Concrete samplers (v0.8.0).

> **v0.9.0 refactor**: this module is now a thin re-export
> shim.  The concrete samplers live in their own files
> (:mod:`core.schedulers.euler`, :mod:`core.schedulers.euler_ancestral`,
> :mod:`core.schedulers.dpmpp_2m`, :mod:`core.schedulers.dpm_solver`,
> :mod:`core.schedulers.flow_match_euler`) so each
> integration method can be edited / benchmarked / replaced
> independently.  Importing from this module is unchanged
> (``from core.schedulers.samplers import EulerSampler`` still
> works), so the v0.8.0 / v0.8.5 call-sites are unaffected.

Each sampler implements the diffusers-style interface:

* ``set_timesteps(num_steps, shift=1.0)``
* ``step(model_output, t, sample, ...)``

The signature mirrors the existing
:class:`core.diffusion_scheduler.Sampler` base class so the two
implementations can coexist.  The samplers here are deliberately
self-contained (no external diffusers dependency) and are
registered via :data:`core.schedulers.SAMPLER_REGISTRY`.
"""
from __future__ import annotations

from ._base import _BaseSampler
from .dpm_solver import DPMSolverSampler
from .dpmpp_2m import DPMpp2MSampler
from .euler import EulerSampler
from .euler_ancestral import EulerAncestralSampler
from .flow_match_euler import FlowMatchEulerSampler, FlowMatchHeunSampler

__all__ = [
    "_BaseSampler",
    "EulerSampler",
    "EulerAncestralSampler",
    "DPMpp2MSampler",
    "DPMSolverSampler",
    "FlowMatchEulerSampler",
    "FlowMatchHeunSampler",
]
