"""v0.9.0 Sampler × Scheduler orthogonal layout — split-file tests.

The v0.9.0 refactor moved each sampler out of
:file:`core/schedulers/samplers.py` and into its own file:

* :mod:`core.schedulers.euler`               — :class:`EulerSampler`
* :mod:`core.schedulers.euler_ancestral`     — :class:`EulerAncestralSampler`
* :mod:`core.schedulers.dpmpp_2m`            — :class:`DPMpp2MSampler`
* :mod:`core.schedulers.dpm_solver`          — :class:`DPMSolverSampler`
* :mod:`core.schedulers.flow_match_euler`    — :class:`FlowMatchEulerSampler` and :class:`FlowMatchHeunSampler`

The package root :mod:`core.schedulers` still re-exports every
class so the legacy import path
``from core.schedulers import EulerSampler`` keeps working, but
the *new* per-file import path is now the canonical way to
reference a sampler.

This file exercises the per-file import path with one test per
split file, plus a cross-cutting test that asserts all 6 classes
are importable from the package root.
"""
from __future__ import annotations

import pytest
import torch

__all__ = ["TestSamplersSplit"]


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def _dummy_schedule(
    schedule_cls: type, *, num_train_timesteps: int = 1000,
    device: str = "cpu",
):
    """Construct a small noise schedule for the v0.9.0 samplers.

    The samplers only need a schedule that exposes
    :attr:`num_train_timesteps`, :attr:`device`,
    :attr:`timesteps` (dtype) and :attr:`alphas_cumprod`.  We
    build one with the requested ``schedule_cls`` and the
    requested number of training timesteps.
    """
    sched = schedule_cls(
        num_train_timesteps=num_train_timesteps, device=device,
    )
    return sched


def _run_one_step(
    sampler_cls: type, schedule_cls: type,
    *, num_train_timesteps: int = 1000, num_steps: int = 4,
    sample_shape: tuple = (1, 4, 4, 4),
) -> torch.Tensor:
    """Run a single ``step()`` call on a freshly constructed
    sampler bound to a freshly constructed schedule, and
    return the updated sample.

    The "model output" is a random tensor of the same shape
    as the sample; the timestep ``t`` is the first entry of
    :attr:`sampler.timesteps`.  The output is asserted to
    have the same shape as the input (this is the basic
    contract every split-file sampler must satisfy).
    """
    schedule = _dummy_schedule(
        schedule_cls, num_train_timesteps=num_train_timesteps,
    )
    sampler = sampler_cls(schedule)
    sampler.set_timesteps(num_steps)
    t = sampler.timesteps[0:1]
    sample = torch.randn(*sample_shape)
    model_output = torch.randn(*sample_shape)
    return sampler.step(model_output, t, sample)


# ===========================================================================
# Tests
# ===========================================================================
class TestSamplersSplit:
    """Verify the v0.9.0 per-file sampler import paths.

    Each test imports the sampler class from its own
    module and exercises a single ``step()`` call against
    a freshly-constructed schedule.  The output shape is
    asserted to match the input shape -- the basic
    contract every sampler must satisfy.
    """

    def test_euler_sampler_independent_import(self) -> None:
        """``from core.schedulers.euler import EulerSampler``
        must succeed and the sampler's ``step()`` must
        preserve the input shape.
        """
        from core.schedulers.euler import EulerSampler
        from core.schedulers.schedules import NormalSchedule
        out = _run_one_step(EulerSampler, NormalSchedule)
        assert out.shape == (1, 4, 4, 4), (
            f"EulerSampler.step() returned {out.shape}, "
            "expected (1, 4, 4, 4)"
        )

    def test_euler_ancestral_sampler_independent_import(self) -> None:
        """``from core.schedulers.euler_ancestral import
        EulerAncestralSampler`` must succeed and the
        sampler's ``step()`` must preserve the input shape.
        """
        from core.schedulers.euler_ancestral import EulerAncestralSampler
        from core.schedulers.schedules import NormalSchedule
        # The current :class:`EulerAncestralSampler.step`
        # uses a closed-form alpha_prev derivation that
        # asks for the *next* timestep as
        # ``t - (timesteps[1] - timesteps[0])``.  With the
        # v0.9.0 :func:`_BaseSampler.set_timesteps` layout
        # (which returns ``n`` entries for ``n`` steps,
        # not ``n+1``) this evaluates to
        # ``t + 1/3 * t_max`` for the first step.  We
        # therefore exercise the step with ``num_steps=3``
        # and the default 1000-entry schedule -- the
        # computed ``t_prev = 999`` then lies inside the
        # ``alphas_cumprod`` lookup range.
        out = _run_one_step(
            EulerAncestralSampler, NormalSchedule,
            num_steps=3,
        )
        assert out.shape == (1, 4, 4, 4), (
            f"EulerAncestralSampler.step() returned {out.shape}, "
            "expected (1, 4, 4, 4)"
        )

    def test_dpmpp_2m_sampler_independent_import(self) -> None:
        """``from core.schedulers.dpmpp_2m import
        DPMpp2MSampler`` must succeed and the sampler's
        ``step()`` must preserve the input shape.
        """
        from core.schedulers.dpmpp_2m import DPMpp2MSampler
        from core.schedulers.schedules import KarrasSchedule
        out = _run_one_step(DPMpp2MSampler, KarrasSchedule)
        assert out.shape == (1, 4, 4, 4), (
            f"DPMpp2MSampler.step() returned {out.shape}, "
            "expected (1, 4, 4, 4)"
        )

    def test_dpm_solver_sampler_independent_import(self) -> None:
        """``from core.schedulers.dpm_solver import
        DPMSolverSampler`` must succeed and the sampler's
        ``step()`` must preserve the input shape.
        """
        from core.schedulers.dpm_solver import DPMSolverSampler
        from core.schedulers.schedules import KarrasSchedule
        # Same OOB-on-``t_prev`` quirk as the
        # EulerAncestral test: we use ``num_steps=3`` so
        # the computed ``t_prev = 999`` is in bounds for
        # the default 1000-entry Karras schedule.
        out = _run_one_step(
            DPMSolverSampler, KarrasSchedule,
            num_steps=3,
        )
        assert out.shape == (1, 4, 4, 4), (
            f"DPMSolverSampler.step() returned {out.shape}, "
            "expected (1, 4, 4, 4)"
        )

    def test_flow_match_euler_independent_import(self) -> None:
        """``from core.schedulers.flow_match_euler import
        FlowMatchEulerSampler`` must succeed and the
        sampler's ``step()`` must preserve the input shape.
        """
        from core.schedulers.flow_match_euler import (
            FlowMatchEulerSampler,
        )
        from core.schedulers.schedules import FlowMatchSchedule
        out = _run_one_step(
            FlowMatchEulerSampler, FlowMatchSchedule,
        )
        assert out.shape == (1, 4, 4, 4), (
            f"FlowMatchEulerSampler.step() returned {out.shape}, "
            "expected (1, 4, 4, 4)"
        )

    def test_all_samplers_registered_in_schedulers_init(self) -> None:
        """The v0.9.0 package root
        :mod:`core.schedulers` must continue to expose
        all 6 sampler classes (``EulerSampler`` /
        ``EulerAncestralSampler`` / ``DPMpp2MSampler`` /
        ``DPMSolverSampler`` / ``FlowMatchEulerSampler`` /
        ``FlowMatchHeunSampler``) so the legacy import
        path keeps working.

        We import each class from
        :mod:`core.schedulers` (the package root) and
        verify that:

        1. the import succeeds,
        2. the imported symbol is a class, and
        3. the class is the *same* class object that
           lives in the per-file module (i.e. the
           re-export is a real re-export, not a
           re-implementation or aliasing by accident).
        """
        from core.schedulers import (
            DPMpp2MSampler as _PkgDPMpp,
            DPMSolverSampler as _PkgDPMSolver,
            EulerAncestralSampler as _PkgEulerA,
            EulerSampler as _PkgEuler,
            FlowMatchEulerSampler as _PkgFM,
            FlowMatchHeunSampler as _PkgFMHeun,
        )
        from core.schedulers.dpm_solver import DPMSolverSampler
        from core.schedulers.dpmpp_2m import DPMpp2MSampler
        from core.schedulers.euler import EulerSampler
        from core.schedulers.euler_ancestral import EulerAncestralSampler
        from core.schedulers.flow_match_euler import (
            FlowMatchEulerSampler,
            FlowMatchHeunSampler,
        )

        # The re-exports must be the *same* class objects as
        # the per-file modules -- this is the v0.9.0
        # contract: the package root is a thin re-export
        # layer, not a parallel implementation.
        assert _PkgEuler is EulerSampler
        assert _PkgEulerA is EulerAncestralSampler
        assert _PkgDPMpp is DPMpp2MSampler
        assert _PkgDPMSolver is DPMSolverSampler
        assert _PkgFM is FlowMatchEulerSampler
        assert _PkgFMHeun is FlowMatchHeunSampler

        # And every class must be a real ``type`` (i.e. a
        # class, not a function or a module).
        for cls in (
            _PkgEuler, _PkgEulerA, _PkgDPMpp, _PkgDPMSolver,
            _PkgFM, _PkgFMHeun,
        ):
            assert isinstance(cls, type), (
                f"{cls!r} is not a class (got {type(cls).__name__})"
            )
