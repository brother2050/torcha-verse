"""Paper integration system for TorchaVerse.

This package links research papers to their integration points in the
framework.  Each paper is described by a declarative
:class:`PaperSpec` (bibliographic metadata, integration kind, model
artifacts, reproducibility config, reference implementations and
compatibility constraints) held in the process-wide
:class:`PaperRegistry` singleton.  Concrete implementations plug in
through :class:`PaperAdapter` subclasses registered with
:class:`AdapterRegistry`, and the :mod:`papers.cli` module exposes
high-level list / info / install / reproduce / benchmark operations.

Layering: ``papers`` depends only on :mod:`papers.spec`,
:mod:`papers.registry`, :mod:`papers.adapter` and :mod:`yaml` (PyYAML,
already a framework dependency).  It does **not** import ``torch`` or
any L1/L2/L3 module, so it is importable in any environment.

Importing this package eagerly loads the bundled ``*.yaml`` paper specs
shipped alongside it (consistent with the ``nodes`` package, which
eagerly registers every node on import).  As a result a freshly
constructed :class:`PaperRegistry` immediately sees the full bundled
catalogue::

    from papers import PaperRegistry
    print(len(PaperRegistry().list()))   # the bundled papers
"""

from __future__ import annotations

from . import cli
from .adapter import (
    AdapterNotFoundError,
    AdapterRegistry,
    PaperAdapter,
    default_registry,
)
from .registry import PaperNotFoundError, PaperRegistry
from .spec import ModelRef, PaperSpec

# Eagerly load the bundled paper YAML specs so the catalogue is
# available immediately after import.  This mirrors the ``nodes``
# package, which eagerly registers every node on import.  Failures are
# logged but never raised so a missing/malformed file cannot break the
# import of the package itself.
try:
    PaperRegistry().load_bundled()
except Exception:  # noqa: BLE001 - import must never fail
    import logging

    logging.getLogger("papers").warning(
        "Failed to load bundled paper specs; registry will be empty "
        "until load_from_dir() is called.",
        exc_info=True,
    )

__all__ = [
    # Specs
    "PaperSpec",
    "ModelRef",
    # Registry
    "PaperRegistry",
    "PaperNotFoundError",
    # Adapters
    "PaperAdapter",
    "AdapterRegistry",
    "AdapterNotFoundError",
    # Concrete paper adapters
    "StableDiffusion3Adapter",
    "HunyuanDiTAdapter",
    # CLI
    "cli",
]


def _register_default_adapters() -> None:
    """Register the v0.5.x bundled paper adapters with the default registry.

    Failures are swallowed (and logged) so a missing optional
    dependency can never break the import of :mod:`papers`.
    """
    import logging

    log = logging.getLogger("papers")
    try:
        from .adapters import HunyuanDiTAdapter, StableDiffusion3Adapter

        default_registry.register("stable-diffusion-3", StableDiffusion3Adapter)
        default_registry.register("sd3", StableDiffusion3Adapter)
        default_registry.register("hunyuan-dit", HunyuanDiTAdapter)
        default_registry.register("hunyuan_dit", HunyuanDiTAdapter)
    except Exception:  # noqa: BLE001 - import must never fail
        log.warning(
            "Failed to register bundled paper adapters; "
            "AdapterRegistry will be empty until the adapters "
            "are imported explicitly.",
            exc_info=True,
        )


_register_default_adapters()
