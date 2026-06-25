"""Paper adapters -- concrete :class:`PaperAdapter` implementations.

This package ships production-ready paper adapters that wire the
v0.5.x line of TorchaVerse to two foundational image-diffusion
papers:

* :class:`StableDiffusion3Adapter` -- Stable Diffusion 3 (SD3).
* :class:`HunyuanDiTAdapter` -- Tencent HunyuanDiT (bilingual
  English / Chinese text-to-image).

Each adapter is a real, import-safe implementation: it builds a
project-internal ``MMDiTDenoiser`` (a minimal-but-faithful clone of
the MM-DiT block that appears in both papers) using the framework's
own ``LocalTorchTextProvider`` for text encoding, then runs a
rectified-flow sampling loop to produce the final image.

The adapters are deliberately **dependency-free** so they are
importable in any environment, and they always use the
project-internal ``MMDiTDenoiser`` (a tiny model) by default.
Plugging the official Stability AI / Tencent weights is a v0.6.x
follow-up -- the architectural plumbing lives behind
:meth:`_build_denoiser` and can be swapped with a different backbone
without touching the adapter contract.
"""

from __future__ import annotations

from .hunyuan_dit import HunyuanDiTAdapter
from .stable_diffusion_3 import StableDiffusion3Adapter

__all__ = [
    "StableDiffusion3Adapter",
    "HunyuanDiTAdapter",
    "MMDiTDenoiser",
    "rectified_flow_sample",
]
