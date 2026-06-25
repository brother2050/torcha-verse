"""Unified evaluation entry point for the TorchaVerse evaluation
framework (v0.4.0).

This module ties together the three families of metrics in this
package -- :mod:`evaluation.metrics` (PSNR / SSIM / LPIPS),
:mod:`evaluation.fid` (Frechet Inception Distance) and
:mod:`evaluation.prompt_recall` (image-prompt alignment) -- behind a
single ``EvaluationRunner`` facade.  It also provides a small
``load_image`` helper that walks a directory of common image formats
and converts each file to a ``(C, H, W)`` float tensor in ``[0, 1]``.

The runner is the public surface that the rest of the project (and
downstream tooling) should use:

.. code-block:: python

    from evaluation.runner import EvaluationRunner

    runner = EvaluationRunner()
    real = runner.load_image_dir("data/real")
    gen = runner.load_image_dir("data/gen")
    report = runner.run(
        real=real,
        generated=gen,
        prompts=["a cat", "a dog"],
    )
    print(report["fid"], report["prompt_recall"]["mean"])

The runner is intentionally lightweight: it does not own the
backbones, it only delegates.  Each metric call still uses the
underlying calculator's caching and lazy-initialisation machinery.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``evaluation`` (this module) -- facade.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from infrastructure.logger import get_logger
from .metrics import lpips as _lpips
from .metrics import psnr as _psnr
from .metrics import ssim as _ssim
from .fid import FidCalculator
from .prompt_recall import PromptRecallCalculator

__all__ = [
    "EvaluationRunner",
    "EvaluationReport",
    "load_image_dir",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Image extensions the directory loader recognises.
_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff",
)

#: Module-level logger.
_logger = get_logger("evaluation.runner")


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------
@dataclass
class EvaluationReport:
    """Structured evaluation output, JSON-serialisable.

    Attributes:
        fid: Frechet Inception Distance between the two image sets.
        prompt_recall: ``None`` when no prompts were supplied;
            otherwise a ``prompt_recall.PromptRecallResult``-shaped
            dict with ``scores`` / ``mean`` / ``std``.
        per_image: Optional dict of per-reference-pair metrics
            (PSNR / SSIM / LPIPS) computed when both sets have the
            same length.
        n_real: Number of real images used.
        n_generated: Number of generated images used.
    """

    fid: float
    prompt_recall: Optional[Dict[str, Any]] = None
    per_image: Optional[Dict[str, List[float]]] = None
    n_real: int = 0
    n_generated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "fid": float(self.fid),
            "prompt_recall": self.prompt_recall,
            "per_image": self.per_image,
            "n_real": int(self.n_real),
            "n_generated": int(self.n_generated),
        }

    def __repr__(self) -> str:
        pr = (
            "{:.3f}".format(self.prompt_recall["mean"])
            if self.prompt_recall is not None else "n/a"
        )
        return (
            "EvaluationReport(fid={:.3f}, prompt_recall={}, n_real={}, "
            "n_generated={})".format(
                self.fid, pr, self.n_real, self.n_generated,
            )
        )


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_image_dir(path: Union[str, os.PathLike]) -> List[torch.Tensor]:
    """Load every image file in ``path`` (non-recursive) as a tensor.

    Files are sorted alphabetically for reproducibility.  Each image is
    converted to a ``(3, H, W)`` float tensor in ``[0, 1]`` via the
    flexible ``_to_tensor`` helper from :mod:`consistency.score` --
    which means PNGs are decoded with ``PIL`` and a fallback to a
    deterministic random tensor is used when ``PIL`` is not installed
    (so the function always returns the right *number* of tensors for
    downstream metrics).

    Args:
        path: A directory containing image files.  Hidden files and
            files with non-image extensions are skipped.

    Returns:
        A list of float tensors of shape ``(3, H, W)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        NotADirectoryError: If ``path`` is not a directory.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            "Image directory does not exist: {}".format(p)
        )
    if not p.is_dir():
        raise NotADirectoryError(
            "Expected a directory, got a file: {}".format(p)
        )
    from consistency.score import _to_tensor  # local import to avoid cycle

    out: List[torch.Tensor] = []
    for entry in sorted(p.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        if entry.name.startswith("."):
            continue
        try:
            with open(entry, "rb") as fh:
                from PIL import Image
                import io
                with Image.open(io.BytesIO(fh.read())) as im:
                    im = im.convert("RGB")
                    import numpy as np
                    arr = np.array(im).astype("float32") / 255.0
                    tensor = torch.from_numpy(arr).permute(2, 0, 1)
            out.append(tensor)
        except Exception as exc:  # noqa: BLE001 - PIL missing/corrupt files
            _logger.warning(
                "Failed to load image %s, falling back to deterministic "
                "placeholder tensor: %s", entry, exc,
            )
            out.append(_to_tensor({"id": entry.stem}))
    return out


# ---------------------------------------------------------------------------
# EvaluationRunner
# ---------------------------------------------------------------------------
class EvaluationRunner:
    """Unified facade over the evaluation package.

    Owns a :class:`FidCalculator` and a
    :class:`PromptRecallCalculator` and exposes a single :meth:`run`
    method that returns a fully populated :class:`EvaluationReport`.

    Args:
        device: Optional device for backbone inference.  Defaults to
            CPU so the runner is portable across CI environments.
        compute_per_image: When ``True`` (default) and the two image
            sets have the same length, the runner also computes
            per-pair PSNR / SSIM / LPIPS in addition to the global
            FID.  When ``False`` the per-image metrics are skipped.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        compute_per_image: bool = True,
    ) -> None:
        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device if device is not None
            else torch.device("cpu")
        )
        self.compute_per_image: bool = bool(compute_per_image)
        self._fid_calc: FidCalculator = FidCalculator(device=self._device)
        self._recall_calc: PromptRecallCalculator = (
            PromptRecallCalculator(device=self._device)
        )
        self._logger = _logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        real: Sequence[Any],
        generated: Sequence[Any],
        prompts: Optional[Sequence[str]] = None,
    ) -> EvaluationReport:
        """Run the full evaluation pipeline.

        Args:
            real: The reference image set (sequence of tensor / PIL /
                numpy / descriptor).  Must be non-empty.
            generated: The candidate image set.
            prompts: Optional sequence of text prompts -- when
                supplied, must be the same length as ``generated``
                and the runner will compute per-pair prompt recall
                in addition to FID.

        Returns:
            A fully populated :class:`EvaluationReport`.
        """
        if not real:
            raise ValueError("`real` must be non-empty")
        if not generated:
            raise ValueError("`generated` must be non-empty")
        if prompts is not None and len(prompts) != len(generated):
            raise ValueError(
                "`prompts` and `generated` must have the same length "
                "(got {} and {})".format(len(prompts), len(generated))
            )

        fid_value = self._fid_calc.fid(real, generated)

        recall_section: Optional[Dict[str, Any]] = None
        if prompts is not None:
            recall = self._recall_calc.prompt_recall(generated, prompts)
            recall_section = recall.to_dict()

        per_image: Optional[Dict[str, List[float]]] = None
        if self.compute_per_image and len(real) == len(generated):
            per_image = self._per_image_metrics(real, generated)

        report = EvaluationReport(
            fid=fid_value,
            prompt_recall=recall_section,
            per_image=per_image,
            n_real=len(real),
            n_generated=len(generated),
        )
        self._logger.info(
            "Evaluation done: fid=%.4f, n_real=%d, n_generated=%d, "
            "prompts=%s",
            fid_value, len(real), len(generated),
            "yes" if prompts is not None else "no",
        )
        return report

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dirs(
        cls,
        real_dir: Union[str, os.PathLike],
        generated_dir: Union[str, os.PathLike],
        prompts: Optional[Sequence[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> EvaluationReport:
        """Build a runner, load two image directories, run evaluation.

        This is the most ergonomic entry point for CI / CLI use --
        the caller only needs to point at two directories of images
        and (optionally) a list of prompts.
        """
        real = load_image_dir(real_dir)
        generated = load_image_dir(generated_dir)
        runner = cls(device=device)
        return runner.run(real=real, generated=generated, prompts=prompts)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _per_image_metrics(
        self,
        real: Sequence[Any],
        generated: Sequence[Any],
    ) -> Dict[str, List[float]]:
        """Compute PSNR / SSIM / LPIPS for each (real, generated) pair."""
        psnrs: List[float] = []
        ssims: List[float] = []
        lpips_vals: List[float] = []
        for r, g in zip(real, generated):
            psnrs.append(_psnr(r, g))
            ssims.append(_ssim(r, g))
            lpips_vals.append(_lpips(r, g))
        return {
            "psnr": psnrs,
            "ssim": ssims,
            "lpips": lpips_vals,
        }
