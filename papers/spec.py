"""Paper specification dataclasses for the TorchaVerse paper integration system.

This module defines the declarative records that describe an *integrated
paper*: its bibliographic metadata, how it plugs into the framework
(node / model / pipeline / tool), the model artifacts it requires, its
reproducibility configuration, links to community reference
implementations, and its compatibility constraints.

The records are plain :mod:`dataclasses` with **no third-party
dependencies** so they can be imported in any environment.  YAML
round-tripping is handled by :meth:`PaperSpec.from_dict` /
:meth:`PaperSpec.to_dict`, which understand the nested ``paper`` /
``integration`` / ``models`` / ``reproducibility`` / ``reference_impl`` /
``compatibility`` schema used by the bundled ``*.yaml`` files.

Public surface
--------------
* :class:`ModelRef` -- a reference to a model artifact required by a
  paper implementation.
* :class:`PaperSpec` -- the full declarative specification of an
  integrated paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

__all__ = [
    "ModelRef",
    "PaperSpec",
]


# ---------------------------------------------------------------------------
# ModelRef
# ---------------------------------------------------------------------------
@dataclass
class ModelRef:
    """A reference to a model artifact required by a paper implementation.

    Attributes:
        name: Model identifier (e.g. ``"musetalk"``).
        source: Provenance of the weights -- one of ``"huggingface"``,
            ``"modelscope"`` or ``"github"``.
        repo: Repository path, e.g. ``"TMElyralab/MuseTalk"``.
        size_gb: Approximate download size in gigabytes (for resource
            planning).
        vram_gb: Approximate VRAM footprint in gigabytes when loaded.
        dependencies: Python package dependencies required to load the
            model.
    """

    name: str
    source: str = ""
    repo: str = ""
    size_gb: float = 0.0
    vram_gb: float = 0.0
    dependencies: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            "ModelRef(name={!r}, source={!r}, repo={!r}, "
            "size_gb={!r}, vram_gb={!r})".format(
                self.name, self.source, self.repo, self.size_gb, self.vram_gb
            )
        )


# ---------------------------------------------------------------------------
# PaperSpec
# ---------------------------------------------------------------------------
@dataclass
class PaperSpec:
    """Declarative specification of an integrated paper.

    A :class:`PaperSpec` is the single source of truth that links a
    research paper to its integration point in TorchaVerse.  It captures
    bibliographic metadata, the integration kind (node / model /
    pipeline / tool), the model artifacts it depends on, a
    reproducibility configuration, links to community reference
    implementations, and compatibility constraints.

    Instances should be treated as immutable once published to the
    :class:`~papers.registry.PaperRegistry`.

    Attributes:
        name: Unique short identifier (e.g. ``"musetalk"``).
        title: Full paper title.
        authors: List of author names.
        arxiv_id: arXiv identifier (e.g. ``"2501.01895"``).
        github: URL of the official GitHub repository.
        license: SPDX-style license string (e.g. ``"CC-BY-NC-4.0"``).
        published: Publication date, ``YYYY-MM`` or ``YYYY-MM-DD``.

        integration_type: How the paper plugs in -- one of
            ``"node"``, ``"model"``, ``"pipeline"`` or ``"tool"``.
        node_type: The framework node type this paper maps to (e.g.
            ``"dh_lip_sync"``).  Empty for non-node integrations.
        method: Method identifier (e.g. ``"musetalk"``).
        category: Coarse category (e.g. ``"digital_human"``,
            ``"foundation"``).

        models: Model artifacts required by the implementation.

        seed: Random seed for reproducible execution.
        deterministic: Whether the implementation is deterministic.
        config: Free-form reproducibility configuration dictionary.

        reference_impl: Mapping of collection name to a reference
            implementation path.  Recognised collections are
            ``"sutskever_30"``, ``"labml"``, ``"karpathy"`` and
            ``"lucidrains"``.

        min_torcha_verse: Minimum compatible TorchaVerse version.
        gpu_required: Whether a GPU is required.
        min_vram_gb: Minimum VRAM in gigabytes.
    """

    # --- Bibliographic metadata ------------------------------------------
    name: str
    title: str
    authors: List[str] = field(default_factory=list)
    arxiv_id: str = ""
    github: str = ""
    license: str = ""
    published: str = ""

    # --- Integration information -----------------------------------------
    integration_type: str = ""  # "node" | "model" | "pipeline" | "tool"
    node_type: str = ""  # corresponding node type
    method: str = ""  # method identifier
    category: str = ""  # category

    # --- Model information ------------------------------------------------
    models: List[ModelRef] = field(default_factory=list)

    # --- Reproducibility --------------------------------------------------
    seed: int = 42
    deterministic: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

    # --- Reference implementations ---------------------------------------
    reference_impl: Dict[str, str] = field(default_factory=dict)

    # --- Compatibility ---------------------------------------------------
    min_torcha_verse: str = "0.3.1"
    gpu_required: bool = True
    min_vram_gb: int = 4

    # ------------------------------------------------------------------
    # YAML round-tripping
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperSpec":
        """Build a :class:`PaperSpec` from a parsed YAML dictionary.

        The expected schema mirrors the bundled ``*.yaml`` files::

            paper:          {name, title, authors, arxiv_id, ...}
            integration:    {type, node_type, method, category}
            models:         [{name, source, repo, size_gb, ...}]
            reproducibility:{seed, deterministic, config}
            reference_impl: {sutskever_30, labml, karpathy, lucidrains}
            compatibility:  {min_torcha_verse, gpu_required, min_vram_gb}

        Missing sections default to their empty / default values so a
        partial YAML still loads.

        Args:
            data: The parsed YAML mapping.

        Returns:
            A populated :class:`PaperSpec`.
        """
        paper = data.get("paper") or {}
        integration = data.get("integration") or {}
        models_raw = data.get("models") or []
        repro = data.get("reproducibility") or {}
        ref_impl = data.get("reference_impl") or {}
        compat = data.get("compatibility") or {}

        models: List[ModelRef] = []
        for entry in models_raw:
            if not isinstance(entry, dict):
                continue
            models.append(
                ModelRef(
                    name=str(entry.get("name", "")),
                    source=str(entry.get("source", "")),
                    repo=str(entry.get("repo", "")),
                    size_gb=float(entry.get("size_gb", 0.0) or 0.0),
                    vram_gb=float(entry.get("vram_gb", 0.0) or 0.0),
                    dependencies=list(entry.get("dependencies") or []),
                )
            )

        return cls(
            name=str(paper.get("name", "")),
            title=str(paper.get("title", "")),
            authors=list(paper.get("authors") or []),
            arxiv_id=str(paper.get("arxiv_id", "")),
            github=str(paper.get("github", "")),
            license=str(paper.get("license", "")),
            published=str(paper.get("published", "")),
            integration_type=str(integration.get("type", "")),
            node_type=str(integration.get("node_type", "")),
            method=str(integration.get("method", "")),
            category=str(integration.get("category", "")),
            models=models,
            seed=int(repro.get("seed", 42)),
            deterministic=bool(repro.get("deterministic", True)),
            config=dict(repro.get("config") or {}),
            reference_impl={
                str(k): str(v) for k, v in (ref_impl or {}).items()
            },
            min_torcha_verse=str(compat.get("min_torcha_verse", "0.3.1")),
            gpu_required=bool(compat.get("gpu_required", True)),
            min_vram_gb=int(compat.get("min_vram_gb", 4)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this spec back to the bundled-YAML schema.

        The returned mapping round-trips through :meth:`from_dict`::

            PaperSpec.from_dict(spec.to_dict()) == spec  (structurally)

        Returns:
            A YAML-serialisable dictionary.
        """
        return {
            "paper": {
                "name": self.name,
                "title": self.title,
                "authors": list(self.authors),
                "arxiv_id": self.arxiv_id,
                "github": self.github,
                "license": self.license,
                "published": self.published,
            },
            "integration": {
                "type": self.integration_type,
                "node_type": self.node_type,
                "method": self.method,
                "category": self.category,
            },
            "models": [
                {
                    "name": m.name,
                    "source": m.source,
                    "repo": m.repo,
                    "size_gb": m.size_gb,
                    "vram_gb": m.vram_gb,
                    "dependencies": list(m.dependencies),
                }
                for m in self.models
            ],
            "reproducibility": {
                "seed": self.seed,
                "deterministic": self.deterministic,
                "config": dict(self.config),
            },
            "reference_impl": dict(self.reference_impl),
            "compatibility": {
                "min_torcha_verse": self.min_torcha_verse,
                "gpu_required": self.gpu_required,
                "min_vram_gb": self.min_vram_gb,
            },
        }

    def __repr__(self) -> str:
        return (
            "PaperSpec(name={!r}, title={!r}, method={!r}, "
            "node_type={!r}, models={})".format(
                self.name, self.title, self.method,
                self.node_type, len(self.models),
            )
        )
