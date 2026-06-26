"""Concrete :class:`PaperAdapter` implementations (R-18 -- lazy).

The two bundled adapters — :class:`StableDiffusion3Adapter` (SD3) and
:class:`HunyuanDiTAdapter` (Tencent bilingual DiT) — depend on
``torch`` + ``torch.nn``, totalling roughly 1,000 lines of model code.
Importing :mod:`papers` is supposed to be cheap (so it can run in any
environment, including CPU-only sandboxes and offline YAML tooling),
so this sub-package follows the same lazy-export pattern as
:mod:`papers` itself: ``import papers.adapters`` does **not** import
either adapter module, and each class is loaded on first attribute
access (or on the first :meth:`AdapterRegistry.get` call for a
registered name).

Public surface (preserved from v0.5.x):

* :data:`StableDiffusion3Adapter` -- resolved lazily to the
  :class:`StableDiffusion3Adapter` class in
  :mod:`papers.adapters.stable_diffusion_3`.
* :data:`HunyuanDiTAdapter` -- resolved lazily to the
  :class:`HunyuanDiTAdapter` class in
  :mod:`papers.adapters.hunyuan_dit`.

The :class:`PaperAdapter` base class itself is defined in
:mod:`papers.adapter` and is still importable directly.
"""

from __future__ import annotations

import importlib
from typing import Any

# Re-export the base class eagerly so ``from papers.adapters import
# PaperAdapter`` keeps working.  The base class has no ``torch``
# dependency, so the eager import is cheap.
from papers.adapter import PaperAdapter

_LAZY_MODULE_FOR_NAME: dict[str, str] = {
    "StableDiffusion3Adapter": "papers.adapters.stable_diffusion_3",
    "HunyuanDiTAdapter": "papers.adapters.hunyuan_dit",
}

__all__ = [
    "PaperAdapter",
    "StableDiffusion3Adapter",
    "HunyuanDiTAdapter",
]


def __getattr__(name: str) -> Any:  # PEP 562
    """Lazy import of the ``torch``-backed adapter classes.

    Triggered on first attribute access; subsequent lookups are
    served from the module's ``__dict__`` (cached in
    ``globals()``).
    """
    if name in _LAZY_MODULE_FOR_NAME:
        module = importlib.import_module(_LAZY_MODULE_FOR_NAME[name])
        cls = getattr(module, name)
        globals()[name] = cls
        return cls
    raise AttributeError(
        "module 'papers.adapters' has no attribute {!r}".format(name)
    )


def __dir__() -> list[str]:
    """Advertise lazy exports to ``dir()`` and IDE auto-completion."""
    return sorted(set(__all__) | set(globals().keys()))
