"""Predefined pipeline templates for the TorchaVerse pipeline layer (L5).

This module ships a curated catalogue of ready-made pipeline *templates* --
reusable DAG definitions for common generative-AI workflows (animation
shorts, music videos, product showcases, character cards, ...).  Each
template is a plain :class:`PipelineTemplate` dataclass holding a
``dag_dict`` (the serialised form of a :class:`~pipeline.dag.DAG`) plus
default parameters and metadata.

A :class:`TemplateRegistry` indexes templates by name and category, supports
free-text search, and can load additional templates from a directory of YAML
files.  The twelve built-in templates are registered automatically when the
module is imported.

Templates are *structural blueprints*: their nodes use placeholder
``node_type`` identifiers (e.g. ``"script_generate"``, ``"image_txt2img"``)
and the edges declare the intended data flow.  Instantiating a template into
a runnable :class:`~pipeline.composer.Pipeline` is a matter of overriding the
default parameters and resolving the node executors through a
:class:`~pipeline.composer.NodeContext`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .dag import DAG, DAGEdge, DAGNode

__all__ = ["PipelineTemplate", "TemplateRegistry", "BUILTIN_TEMPLATES"]


# ---------------------------------------------------------------------------
# PipelineTemplate
# ---------------------------------------------------------------------------
@dataclass
class PipelineTemplate:
    """A reusable, serialised pipeline blueprint.

    Attributes:
        name: Unique template name (used as the registry key).
        description: Human-readable description of what the template does.
        category: Coarse category for grouping (e.g. ``"video"``,
            ``"image"``, ``"audio"``).
        dag_dict: Serialised :class:`DAG` (``{"nodes": [...], "edges": [...]}``)
            describing the node graph.
        default_params: Default parameter overrides applied when
            instantiating the template.
        tags: Free-form tags used by :meth:`TemplateRegistry.search`.
    """

    name: str
    description: str
    category: str
    dag_dict: Dict[str, Any] = field(default_factory=dict)
    default_params: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this template to a JSON-serialisable dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "dag_dict": dict(self.dag_dict),
            "default_params": dict(self.default_params),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineTemplate":
        """Reconstruct a :class:`PipelineTemplate` from a serialised dict."""
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            category=d.get("category", "general"),
            dag_dict=dict(d.get("dag_dict") or {}),
            default_params=dict(d.get("default_params") or {}),
            tags=list(d.get("tags") or []),
        )

    def to_dag(self) -> DAG:
        """Materialise this template's ``dag_dict`` into a live :class:`DAG`."""
        return DAG.from_dict(self.dag_dict)

    def __repr__(self) -> str:
        node_count = len(self.dag_dict.get("nodes", []))
        edge_count = len(self.dag_dict.get("edges", []))
        return (
            "PipelineTemplate(name={!r}, category={!r}, "
            "nodes={}, edges={})".format(
                self.name, self.category, node_count, edge_count
            )
        )


# ---------------------------------------------------------------------------
# TemplateRegistry
# ---------------------------------------------------------------------------
class TemplateRegistry:
    """Thread-safe registry of :class:`PipelineTemplate` instances.

    Templates are keyed by their (case-insensitive) ``name``.  The registry
    supports category filtering, free-text search over name / description /
    tags, and bulk loading from a directory of YAML files.

    Example:
        >>> reg = TemplateRegistry()
        >>> len(reg.list()) >= 12
        True
        >>> tmpl = reg.get("anime_short_film")
        >>> dag = tmpl.to_dag()
    """

    def __init__(self) -> None:
        self._templates: Dict[str, PipelineTemplate] = {}
        self._lock: threading.RLock = threading.RLock()
        # Register the built-in catalogue.
        for template in _BUILTIN_TEMPLATE_FACTORIES():
            self.register(template)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, template: PipelineTemplate) -> None:
        """Register (or replace) a template.

        Args:
            template: The :class:`PipelineTemplate` to register.

        Raises:
            ValueError: If ``template.name`` is empty.
        """
        if not template.name or not isinstance(template.name, str):
            raise ValueError("Template name must be a non-empty string.")
        key = template.name.strip().lower()
        with self._lock:
            self._templates[key] = template

    def unregister(self, name: str) -> Optional[PipelineTemplate]:
        """Remove and return the template named ``name`` (or ``None``)."""
        key = name.strip().lower()
        with self._lock:
            return self._templates.pop(key, None)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def get(self, name: str) -> PipelineTemplate:
        """Return the template named ``name``.

        Args:
            name: Template name (case-insensitive).

        Returns:
            The :class:`PipelineTemplate`.

        Raises:
            KeyError: If no template with that name is registered.
        """
        key = name.strip().lower()
        with self._lock:
            template = self._templates.get(key)
        if template is None:
            raise KeyError("No template named {!r}.".format(name))
        return template

    def has(self, name: str) -> bool:
        """Return ``True`` if a template named ``name`` is registered."""
        key = name.strip().lower()
        with self._lock:
            return key in self._templates

    def list(self, category: Optional[str] = None) -> List[PipelineTemplate]:
        """List registered templates, optionally filtered by category.

        Args:
            category: When given, only templates whose ``category`` matches
                (case-insensitive) are returned.

        Returns:
            A list of :class:`PipelineTemplate` sorted by name.
        """
        with self._lock:
            templates = list(self._templates.values())
        if category is not None:
            wanted = category.strip().lower()
            templates = [
                t for t in templates
                if t.category.strip().lower() == wanted
            ]
        templates.sort(key=lambda t: t.name)
        return templates

    def search(self, query: str) -> List[PipelineTemplate]:
        """Free-text search over name, description and tags.

        The query is split into whitespace-separated terms; a template
        matches if *every* term appears (case-insensitive) in its name,
        description or any of its tags.

        Args:
            query: The search query.

        Returns:
            A list of matching :class:`PipelineTemplate` sorted by name.
        """
        terms = [t.lower() for t in query.split() if t]
        if not terms:
            return self.list()
        matches: List[PipelineTemplate] = []
        with self._lock:
            templates = list(self._templates.values())
        for tmpl in templates:
            haystack = " ".join(
                [tmpl.name, tmpl.description, tmpl.category]
                + list(tmpl.tags)
            ).lower()
            if all(term in haystack for term in terms):
                matches.append(tmpl)
        matches.sort(key=lambda t: t.name)
        return matches

    # ------------------------------------------------------------------
    # Bulk loading
    # ------------------------------------------------------------------
    def load_from_dir(self, path: Union[str, Path]) -> int:
        """Load templates from every ``*.yaml`` / ``*.yml`` file in a directory.

        Each YAML file may contain either a single template mapping or a
        list of template mappings.  A mapping must include at least a
        ``name``; the remaining fields default sensibly.

        Args:
            path: Directory to scan.

        Returns:
            The number of templates loaded.
        """
        directory = Path(path).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(
                "Template directory not found: {}".format(directory)
            )
        loaded = 0
        for yaml_file in sorted(directory.glob("*.y*ml")):
            with open(yaml_file, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            if data is None:
                continue
            if isinstance(data, dict):
                items: List[Dict[str, Any]] = [data]
            elif isinstance(data, list):
                items = [d for d in data if isinstance(d, dict)]
            else:
                continue
            for item in items:
                if "name" not in item:
                    continue
                self.register(PipelineTemplate.from_dict(item))
                loaded += 1
        return loaded

    # ------------------------------------------------------------------
    def count(self) -> int:
        """Return the number of registered templates."""
        with self._lock:
            return len(self._templates)

    def __repr__(self) -> str:
        return "TemplateRegistry(templates={})".format(self.count())


# ---------------------------------------------------------------------------
# Built-in template catalogue
# ---------------------------------------------------------------------------
def _node(
    nid: str,
    node_type: str,
    inputs: Optional[Dict[str, Any]] = None,
    dependencies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Helper to build a serialised node dict for the built-in templates."""
    return {
        "id": nid,
        "node_type": node_type,
        "inputs": dict(inputs or {}),
        "dependencies": list(dependencies or []),
    }


def _edge(
    from_node: str,
    to_node: str,
    output_key: str = "output",
    input_key: str = "input",
) -> Dict[str, Any]:
    """Helper to build a serialised edge dict for the built-in templates."""
    return {
        "from_node": from_node,
        "to_node": to_node,
        "output_key": output_key,
        "input_key": input_key,
    }


def _tmpl(
    name: str,
    description: str,
    category: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    default_params: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> PipelineTemplate:
    """Assemble a :class:`PipelineTemplate` from node/edge dicts."""
    return PipelineTemplate(
        name=name,
        description=description,
        category=category,
        dag_dict={"nodes": nodes, "edges": edges},
        default_params=dict(default_params or {}),
        tags=list(tags or []),
    )


def _anime_short_film() -> PipelineTemplate:
    """3-minute anime short film: script -> cast -> storyboard -> generate -> voice -> subtitle."""
    nodes = [
        _node("script", "text_chat", {"duration_minutes": 3, "genre": "anime"}, []),
        _node("cast", "character_apply", {"style": "anime"}, ["script"]),
        _node("storyboard", "image_txt2img", {"shots_per_minute": 8}, ["script"]),
        _node("shots", "image_txt2img", {"width": 1280, "height": 720}, ["storyboard"]),
        _node("voice", "audio_tts", {"language": "ja"}, ["script", "cast"]),
        _node("subtitle", "subtitle_burn", {"font": "sans", "language": "zh"}, ["shots", "voice"]),
    ]
    edges = [
        _edge("script", "cast", "text", "prompt"),
        _edge("script", "storyboard", "text", "prompt"),
        _edge("storyboard", "shots", "image", "prompt"),
        _edge("script", "voice", "text", "text"),
        _edge("cast", "voice", "image", "voice"),
        _edge("shots", "subtitle", "image", "video"),
        _edge("voice", "subtitle", "audio", "subtitle_track"),
    ]
    return _tmpl(
        "anime_short_film",
        "3-minute anime short film: script -> cast -> storyboard -> shot "
        "generation -> voice-over -> subtitle burn-in.",
        "video",
        nodes,
        edges,
        default_params={"duration_minutes": 3, "resolution": "1280x720"},
        tags=["anime", "video", "short_film", "voiceover"],
    )


def _mv_music_video() -> PipelineTemplate:
    """Music video: lyrics -> beat map -> shot sequencer -> render."""
    nodes = [
        _node("lyrics", "text_chat", {"theme": "love", "language": "en"}, []),
        _node("beat", "audio_music", {"bpm": 120}, ["lyrics"]),
        _node("shots", "image_txt2img", {"aspect": "16:9"}, ["lyrics", "beat"]),
        _node("render", "export_video", {"fps": 30}, ["shots"]),
    ]
    edges = [
        _edge("lyrics", "beat", "text", "prompt"),
        _edge("lyrics", "shots", "text", "prompt"),
        _edge("beat", "shots", "audio", "prompt"),
        _edge("shots", "render", "image", "video"),
    ]
    return _tmpl(
        "mv_music_video",
        "Music video: lyrics -> beat map -> shot sequencer -> final render.",
        "video",
        nodes,
        edges,
        default_params={"bpm": 120, "aspect": "16:9"},
        tags=["music", "video", "mv", "lyrics"],
    )


def _product_showcase() -> PipelineTemplate:
    """E-commerce product showcase: input -> background swap -> light match -> upscale."""
    nodes = [
        _node("input", "image_txt2img", {"source": "upload"}, []),
        _node("bg", "scene_apply", {"scenes": ["studio", "lifestyle"]}, ["input"]),
        _node("light", "scene_apply", {"target": "studio_softbox"}, ["bg"]),
        _node("upscale", "image_upscale", {"scale": 4}, ["light"]),
    ]
    edges = [
        _edge("input", "bg", "image", "image"),
        _edge("bg", "light", "image", "image"),
        _edge("light", "upscale", "image", "image"),
    ]
    return _tmpl(
        "product_showcase",
        "E-commerce product image set: input -> background swap -> light "
        "match -> upscale.",
        "image",
        nodes,
        edges,
        default_params={"scale": 4, "scenes": ["studio", "lifestyle"]},
        tags=["ecommerce", "image", "product", "upscale"],
    )


def _character_card_5view() -> PipelineTemplate:
    """Character card five-view: ref -> five view -> outfit variants -> contact sheet."""
    nodes = [
        _node("ref", "image_txt2img", {"source": "upload"}, []),
        _node("views", "character_five_view", {"views": ["front", "side", "back", "3q", "top"]}, ["ref"]),
        _node("outfits", "outfit_apply", {"count": 4}, ["views"]),
        _node("sheet", "export_image", {"cols": 5, "rows": 2}, ["views", "outfits"]),
    ]
    edges = [
        _edge("ref", "views", "image", "reference_image"),
        _edge("views", "outfits", "five_views", "image"),
        _edge("views", "sheet", "five_views", "image"),
        _edge("outfits", "sheet", "image", "image"),
    ]
    return _tmpl(
        "character_card_5view",
        "Character card five-view: reference -> five-view -> outfit "
        "variants -> contact sheet.",
        "image",
        nodes,
        edges,
        default_params={"views": ["front", "side", "back", "3q", "top"]},
        tags=["character", "image", "concept", "sheet"],
    )


def _story_illustration() -> PipelineTemplate:
    """Novel illustration: chapter text -> scene extract -> illustration grid."""
    nodes = [
        _node("chapter", "text_chat", {"source": "novel"}, []),
        _node("scenes", "text_chat", {"max_scenes": 8}, ["chapter"]),
        _node("grid", "export_image", {"style": "watercolor", "cols": 3}, ["scenes"]),
    ]
    edges = [
        _edge("chapter", "scenes", "text", "prompt"),
        _edge("scenes", "grid", "text", "image"),
    ]
    return _tmpl(
        "story_illustration",
        "Novel illustration: chapter text -> scene extraction -> "
        "illustration grid.",
        "image",
        nodes,
        edges,
        default_params={"max_scenes": 8, "style": "watercolor"},
        tags=["illustration", "image", "novel", "scene"],
    )


def _douyin_vertical_clip() -> PipelineTemplate:
    """Vertical short video: topic -> script -> 9:16 shot -> subtitle burn."""
    nodes = [
        _node("topic", "text_chat", {"platform": "douyin"}, []),
        _node("script", "text_chat", {"duration_seconds": 30}, ["topic"]),
        _node("shot", "image_txt2img", {"width": 1080, "height": 1920}, ["script"]),
        _node("subtitle", "subtitle_burn", {"font": "bold", "language": "zh"}, ["shot"]),
    ]
    edges = [
        _edge("topic", "script", "text", "prompt"),
        _edge("script", "shot", "text", "prompt"),
        _edge("shot", "subtitle", "image", "video"),
    ]
    return _tmpl(
        "douyin_vertical_clip",
        "Vertical short video (9:16): topic -> script -> vertical shot -> "
        "subtitle burn-in.",
        "video",
        nodes,
        edges,
        default_params={"resolution": "1080x1920", "duration_seconds": 30},
        tags=["douyin", "video", "vertical", "short"],
    )


def _kids_storybook() -> PipelineTemplate:
    """Children's storybook: story -> illustration loop -> PDF export."""
    nodes = [
        _node("story", "text_chat", {"audience": "children"}, []),
        _node("pages", "image_txt2img", {"pages": 12, "style": "cartoon"}, ["story"]),
        _node("pdf", "export_image", {"format": "pdf"}, ["pages"]),
    ]
    edges = [
        _edge("story", "pages", "text", "prompt"),
        _edge("pages", "pdf", "image", "image"),
    ]
    return _tmpl(
        "kids_storybook",
        "Children's storybook: story text -> illustration loop -> PDF export.",
        "document",
        nodes,
        edges,
        default_params={"style": "cartoon", "page_size": "A4"},
        tags=["kids", "storybook", "pdf", "illustration"],
    )


def _concept_art_pack() -> PipelineTemplate:
    """Concept art pack: theme -> multi angle -> mood board."""
    nodes = [
        _node("theme", "text_chat", {"domain": "sci-fi"}, []),
        _node("angles", "character_five_view", {"angles": ["front", "side", "top", "detail"]}, ["theme"]),
        _node("mood", "image_txt2img", {"palette": "auto", "cols": 4}, ["angles"]),
    ]
    edges = [
        _edge("theme", "angles", "text", "reference_image"),
        _edge("angles", "mood", "five_views", "prompt"),
    ]
    return _tmpl(
        "concept_art_pack",
        "Concept art pack: theme -> multi-angle renders -> mood board.",
        "image",
        nodes,
        edges,
        default_params={"angles": ["front", "side", "top", "detail"]},
        tags=["concept", "image", "mood_board", "design"],
    )


def _tutorial_overlay() -> PipelineTemplate:
    """Tutorial screen recording packaging: recording -> caption -> highlight -> export."""
    nodes = [
        _node("recording", "export_video", {"fps": 30, "resolution": "1080p"}, []),
        _node("caption", "subtitle_generate", {"language": "en"}, ["recording"]),
        _node("highlight", "subtitle_burn", {"style": "highlight"}, ["recording", "caption"]),
        _node("export", "export_video", {"format": "mp4"}, ["caption", "highlight"]),
    ]
    edges = [
        _edge("recording", "caption", "path", "media_path"),
        _edge("recording", "highlight", "path", "video"),
        _edge("caption", "highlight", "subtitle_track", "subtitle_track"),
        _edge("caption", "export", "subtitle_track", "video"),
        _edge("highlight", "export", "video", "video"),
    ]
    return _tmpl(
        "tutorial_overlay",
        "Tutorial screen-recording packaging: recording -> caption -> "
        "highlight overlay -> export.",
        "video",
        nodes,
        edges,
        default_params={"fps": 30, "format": "mp4"},
        tags=["tutorial", "video", "screen_recording", "caption"],
    )


def _image_variation_grid() -> PipelineTemplate:
    """Image variation matrix: input -> style matrix -> grid compose."""
    nodes = [
        _node("input", "image_txt2img", {"source": "upload"}, []),
        _node("matrix", "export_image", {"styles": ["anime", "realistic", "oil", "pixel"]}, ["input"]),
        _node("grid", "export_image", {"cols": 4, "rows": 4}, ["matrix"]),
    ]
    edges = [
        _edge("input", "matrix", "image", "image"),
        _edge("matrix", "grid", "path", "image"),
    ]
    return _tmpl(
        "image_variation_grid",
        "Image variation matrix: input -> style matrix -> grid composition.",
        "image",
        nodes,
        edges,
        default_params={"styles": ["anime", "realistic", "oil", "pixel"]},
        tags=["variation", "image", "grid", "style"],
    )


def _restoration_pipeline() -> PipelineTemplate:
    """Old film restoration: input -> denoise -> colorize -> interpolate -> upscale."""
    nodes = [
        _node("input", "export_video", {"source": "archive"}, []),
        _node("denoise", "image_upscale", {"strength": 0.6}, ["input"]),
        _node("colorize", "image_upscale", {"model": "deoldify"}, ["denoise"]),
        _node("interp", "video_interpolate", {"target_fps": 60}, ["colorize"]),
        _node("upscale", "image_upscale", {"scale": 4}, ["interp"]),
    ]
    edges = [
        _edge("input", "denoise", "path", "image"),
        _edge("denoise", "colorize", "image", "image"),
        _edge("colorize", "interp", "image", "video"),
        _edge("interp", "upscale", "video", "image"),
    ]
    return _tmpl(
        "restoration_pipeline",
        "Old film restoration: input -> denoise -> colorize -> frame "
        "interpolation -> upscale.",
        "video",
        nodes,
        edges,
        default_params={"target_fps": 60, "scale": 4},
        tags=["restoration", "video", "colorize", "upscale"],
    )


def _audiobook_with_bgm() -> PipelineTemplate:
    """Audiobook with BGM: text -> TTS -> BGM mix -> chapter markers."""
    nodes = [
        _node("text", "text_chat", {"source": "epub"}, []),
        _node("tts", "audio_tts", {"voice": "warm_male", "language": "zh"}, ["text"]),
        _node("bgm", "audio_music", {"mood": "calm", "ducking": 0.7}, ["tts"]),
        _node("markers", "subtitle_generate", {"format": "m4b"}, ["text", "bgm"]),
    ]
    edges = [
        _edge("text", "tts", "text", "text"),
        _edge("tts", "bgm", "audio", "prompt"),
        _edge("text", "markers", "text", "text"),
        _edge("bgm", "markers", "audio", "media_path"),
    ]
    return _tmpl(
        "audiobook_with_bgm",
        "Audiobook with background music: text -> TTS -> BGM mix -> "
        "chapter markers.",
        "audio",
        nodes,
        edges,
        default_params={"voice": "warm_male", "format": "m4b"},
        tags=["audiobook", "audio", "tts", "bgm"],
    )


def _BUILTIN_TEMPLATE_FACTORIES() -> List[PipelineTemplate]:
    """Return the list of all built-in :class:`PipelineTemplate` instances."""
    return [
        _anime_short_film(),
        _mv_music_video(),
        _product_showcase(),
        _character_card_5view(),
        _story_illustration(),
        _douyin_vertical_clip(),
        _kids_storybook(),
        _concept_art_pack(),
        _tutorial_overlay(),
        _image_variation_grid(),
        _restoration_pipeline(),
        _audiobook_with_bgm(),
    ]


#: The immutable list of built-in templates shipped with the framework.
BUILTIN_TEMPLATES: List[PipelineTemplate] = _BUILTIN_TEMPLATE_FACTORIES()
