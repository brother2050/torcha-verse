"""TorchaVerse evaluation framework (v0.4.0).

This package provides minimum-viable image / cross-modal evaluation
metrics that can be used to score the output of the L4 generation
nodes.  All metrics are implemented in pure PyTorch (plus standard
library) so they run in any environment that has ``torch`` installed
-- no ``scipy``, ``torchmetrics`` or other third-party metric
packages are required.

Modules
-------

* :mod:`evaluation.metrics` -- PSNR / SSIM / LPIPS (LPIPS as a
  structural placeholder; see the module docstring for the migration
  path to a real LPIPS network).
* :mod:`evaluation.fid` -- Frechet Inception Distance with a
  placeholder Inception-style backbone (same migration path).
* :mod:`evaluation.prompt_recall` -- image-prompt cosine similarity
  (a.k.a. CLIP-score) with a placeholder dual encoder.
* :mod:`evaluation.runner` -- :class:`EvaluationRunner` facade and
  :func:`load_image_dir` helper for CI / CLI use.

Public API
----------

The high-level entry points are re-exported here so callers can write
``from evaluation import image_fid, prompt_recall, psnr, ssim, lpips,
EvaluationRunner``.

Examples
--------

Quick smoke-test of the public API::

    >>> from evaluation import image_fid, prompt_recall, psnr, ssim
    >>> import torch
    >>> a = torch.rand(3, 32, 32)
    >>> b = torch.rand(3, 32, 32)
    >>> psnr(a, b) > 0
    True
    >>> 0.0 <= ssim(a, b) <= 1.0
    True
    >>> result = prompt_recall([a, b], ["a cat", "a dog"])
    >>> 0.0 <= result.mean <= 1.0
    True
    >>> fid = image_fid([a, b], [a, b])
    >>> fid >= 0.0
    True
"""

from __future__ import annotations

from .metrics import LpipPlaceholder, lpips, psnr, ssim
from .fid import FidCalculator, InceptionPlaceholder, compute_statistics, frechet_distance, image_fid
from .prompt_recall import (
    DualEncoderPlaceholder,
    PromptRecallCalculator,
    PromptRecallResult,
    prompt_recall,
    score,
)
from .runner import EvaluationReport, EvaluationRunner, load_image_dir

__all__ = [
    # metrics
    "psnr",
    "ssim",
    "lpips",
    "LpipPlaceholder",
    # fid
    "image_fid",
    "compute_statistics",
    "frechet_distance",
    "FidCalculator",
    "InceptionPlaceholder",
    # prompt_recall
    "prompt_recall",
    "score",
    "PromptRecallResult",
    "PromptRecallCalculator",
    "DualEncoderPlaceholder",
    # runner
    "EvaluationRunner",
    "EvaluationReport",
    "load_image_dir",
]


__version__ = "0.4.0"
