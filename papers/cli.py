"""Paper CLI commands for the TorchaVerse paper integration system.

This module exposes the high-level operations a user performs against a
registered paper:

* :func:`paper_list` -- enumerate registered papers.
* :func:`paper_info` -- show full details for a paper.
* :func:`paper_install` -- compute the install plan (models to download
  + Python dependencies to install) for a paper.
* :func:`paper_reproduce` -- run a reproducibility verification against a
  paper's declared configuration.
* :func:`paper_benchmark` -- run a performance benchmark placeholder.

The functions return plain dictionaries / lists so they can be driven
both from a REPL and from a ``click``-based CLI (see :mod:`serving.cli`).
They are deliberately side-effect free in the sense that they never
actually download weights or execute third-party code -- they compute
*plans* and *reports* that the caller may act upon.  This keeps the
paper system safe to invoke in any environment.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .registry import PaperNotFoundError, PaperRegistry
from .spec import PaperSpec

__all__ = [
    "paper_list",
    "paper_info",
    "paper_install",
    "paper_reproduce",
    "paper_benchmark",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _registry() -> PaperRegistry:
    """Return the process-wide :class:`PaperRegistry` singleton."""
    return PaperRegistry()


def _require(name: str) -> PaperSpec:
    """Return the spec for ``name`` or raise :class:`PaperNotFoundError`."""
    return _registry().get(name)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def paper_list() -> List[PaperSpec]:
    """Return every registered :class:`PaperSpec`, sorted by name.

    Returns:
        A list of :class:`PaperSpec`.
    """
    return _registry().list()


def paper_info(name: str) -> Dict[str, Any]:
    """Return a detailed info dictionary for a paper.

    Args:
        name: The paper name.

    Returns:
        A dictionary with the paper's bibliographic, integration, model,
        reproducibility, reference-implementation and compatibility
        fields.

    Raises:
        PaperNotFoundError: If no paper is registered for ``name``.
    """
    spec = _require(name)
    return {
        "name": spec.name,
        "title": spec.title,
        "authors": list(spec.authors),
        "arxiv_id": spec.arxiv_id,
        "github": spec.github,
        "license": spec.license,
        "published": spec.published,
        "integration_type": spec.integration_type,
        "node_type": spec.node_type,
        "method": spec.method,
        "category": spec.category,
        "models": [
            {
                "name": m.name,
                "source": m.source,
                "repo": m.repo,
                "size_gb": m.size_gb,
                "vram_gb": m.vram_gb,
                "dependencies": list(m.dependencies),
            }
            for m in spec.models
        ],
        "seed": spec.seed,
        "deterministic": spec.deterministic,
        "config": dict(spec.config),
        "reference_impl": dict(spec.reference_impl),
        "min_torcha_verse": spec.min_torcha_verse,
        "gpu_required": spec.gpu_required,
        "min_vram_gb": spec.min_vram_gb,
    }


def paper_install(name: str) -> Dict[str, Any]:
    """Compute the install plan for a paper.

    The plan enumerates the model artifacts to download (with their
    source / repo / size) and the deduplicated set of Python
    dependencies to install.  No download is actually performed -- the
    returned plan is a report the caller may act upon.

    Args:
        name: The paper name.

    Returns:
        A dictionary with keys ``paper``, ``models`` and
        ``dependencies``.

    Raises:
        PaperNotFoundError: If no paper is registered for ``name``.
    """
    spec = _require(name)

    models: List[Dict[str, Any]] = []
    deps: List[str] = []
    seen_deps: set[str] = set()
    total_size_gb = 0.0
    total_vram_gb = 0.0

    for m in spec.models:
        models.append(
            {
                "name": m.name,
                "source": m.source,
                "repo": m.repo,
                "size_gb": m.size_gb,
                "vram_gb": m.vram_gb,
                "dependencies": list(m.dependencies),
            }
        )
        total_size_gb += m.size_gb
        total_vram_gb = max(total_vram_gb, m.vram_gb)
        for dep in m.dependencies:
            if dep not in seen_deps:
                seen_deps.add(dep)
                deps.append(dep)

    return {
        "paper": spec.name,
        "status": "planned",
        "models": models,
        "total_size_gb": round(total_size_gb, 4),
        "peak_vram_gb": total_vram_gb,
        "dependencies": deps,
        "gpu_required": spec.gpu_required,
        "min_vram_gb": spec.min_vram_gb,
    }


def paper_reproduce(name: str) -> Dict[str, Any]:
    """Run a reproducibility verification for a paper.

    The verification checks that the paper declares a deterministic
    configuration (a fixed seed and ``deterministic=True``) and reports
    the declared seed and config.  The ``status`` field is ``"ok"`` when
    the configuration is reproducible, ``"non-deterministic"`` otherwise.

    Args:
        name: The paper name.

    Returns:
        A dictionary with the reproducibility report.

    Raises:
        PaperNotFoundError: If no paper is registered for ``name``.
    """
    spec = _require(name)
    reproducible = bool(spec.deterministic) and spec.seed is not None
    return {
        "paper": spec.name,
        "status": "ok" if reproducible else "non-deterministic",
        "seed": spec.seed,
        "deterministic": spec.deterministic,
        "config": dict(spec.config),
        "reference_impl": dict(spec.reference_impl),
    }


def paper_benchmark(name: str) -> Dict[str, Any]:
    """Run a performance benchmark placeholder for a paper.

    The benchmark reports the resource envelope declared by the paper
    (GPU requirement, minimum VRAM) together with a placeholder
    throughput figure.  Real benchmarking is delegated to the
    :mod:`evaluation` layer once a concrete adapter is wired in.

    Args:
        name: The paper name.

    Returns:
        A dictionary with the benchmark report.

    Raises:
        PaperNotFoundError: If no paper is registered for ``name``.
    """
    spec = _require(name)
    return {
        "paper": spec.name,
        "method": spec.method,
        "node_type": spec.node_type,
        "category": spec.category,
        "gpu_required": spec.gpu_required,
        "min_vram_gb": spec.min_vram_gb,
        "peak_vram_gb": max(
            (m.vram_gb for m in spec.models), default=0.0
        ),
        "model_count": len(spec.models),
        # Placeholder throughput -- replaced by real measurements once
        # an adapter is registered with the AdapterRegistry.
        "throughput": "n/a (no adapter registered)",
        "status": "placeholder",
    }
