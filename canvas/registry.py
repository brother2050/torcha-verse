"""Community template registry for the TorchaVerse v0.3.0 architecture (L5).

This module provides a lightweight, in-memory registry for *community*
canvas templates -- user-submitted canvas states that can be searched,
downloaded, rated and discovered by popularity or recency.

Unlike the built-in :class:`~pipeline.templates.TemplateRegistry` (which
stores serialised DAG blueprints), the community registry stores full
:class:`~canvas.canvas.CanvasState` objects, so a downloaded template is
immediately usable as a :class:`~canvas.canvas.Canvas`.

The registry layer is *torch-free* and depends only on the canvas core
(:mod:`canvas.canvas`) and the Python standard library.

Public surface
--------------

* :class:`CommunityTemplate` -- a dataclass describing a community
  template (name, author, description, canvas state, tags, downloads,
  rating).
* :class:`CommunityRegistry` -- a thread-safe registry supporting submit,
  search, download, rate, list-popular and list-recent operations.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .canvas import Canvas, CanvasState

__all__ = ["CommunityTemplate", "CommunityRegistry"]

# ---------------------------------------------------------------------------
# Module-level logger.
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger("canvas.registry")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Minimum allowed rating value.
_MIN_RATING: float = 0.0
#: Maximum allowed rating value.
_MAX_RATING: float = 5.0
#: Default download count for a newly submitted template.
_DEFAULT_DOWNLOADS: int = 0
#: Default rating for a newly submitted template.
_DEFAULT_RATING: float = 0.0


# ---------------------------------------------------------------------------
# CommunityTemplate
# ---------------------------------------------------------------------------
@dataclass
class CommunityTemplate:
    """A user-submitted canvas template in the community registry.

    A community template bundles a :class:`CanvasState` with metadata
    (author, description, tags) and engagement metrics (downloads,
    rating).  Templates are keyed by their (case-insensitive) ``name``.

    Attributes:
        name: Unique template name (used as the registry key).
        author: Name of the submitter.
        description: Human-readable description.
        canvas_state: The :class:`CanvasState` captured by this template.
        tags: Free-form tags used for search and discovery.
        downloads: Number of times this template has been downloaded.
        rating: Average rating (``0.0`` -- ``5.0``).
    """

    name: str
    author: str
    description: str
    canvas_state: CanvasState
    tags: List[str] = field(default_factory=list)
    downloads: int = _DEFAULT_DOWNLOADS
    rating: float = _DEFAULT_RATING
    submitted_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this template to a JSON-serialisable dictionary."""
        return {
            "name": self.name,
            "author": self.author,
            "description": self.description,
            "canvas_state": self.canvas_state.to_dict(),
            "tags": list(self.tags),
            "downloads": self.downloads,
            "rating": self.rating,
            "submitted_at": self.submitted_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CommunityTemplate":
        """Reconstruct a :class:`CommunityTemplate` from a serialised dict.

        Args:
            d: Dictionary produced by :meth:`to_dict`.

        Returns:
            A new :class:`CommunityTemplate` instance.
        """
        return cls(
            name=d["name"],
            author=d.get("author", ""),
            description=d.get("description", ""),
            canvas_state=CanvasState.from_dict(d.get("canvas_state") or {}),
            tags=list(d.get("tags") or []),
            downloads=int(d.get("downloads", _DEFAULT_DOWNLOADS)),
            rating=float(d.get("rating", _DEFAULT_RATING)),
            submitted_at=float(d.get("submitted_at", time.time())),
        )

    def __repr__(self) -> str:
        return (
            "CommunityTemplate(name={!r}, author={!r}, "
            "downloads={}, rating={})".format(
                self.name, self.author, self.downloads, self.rating
            )
        )


# ---------------------------------------------------------------------------
# CommunityRegistry
# ---------------------------------------------------------------------------
class CommunityRegistry:
    """Thread-safe registry of community canvas templates.

    Templates are keyed by their (case-insensitive) ``name``.  The
    registry supports free-text search (over name, description and tags),
    tag-based filtering, popularity and recency sorting, downloading (which
    increments the download counter) and rating.

    All operations are guarded by a re-entrant lock for thread safety.

    Example::

        cr = CommunityRegistry()
        template_id = cr.submit(CommunityTemplate(
            name="my_cool_canvas",
            author="alice",
            description="A cool canvas",
            canvas_state=my_state,
            tags=["video", "anime"],
        ))
        results = cr.search("anime", tags=["video"])
        canvas = cr.download("my_cool_canvas")
    """

    def __init__(self) -> None:
        self._templates: Dict[str, CommunityTemplate] = {}
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------
    def submit(self, template: CommunityTemplate) -> str:
        """Submit a community template to the registry.

        If a template with the same name already exists it is replaced.

        Args:
            template: The :class:`CommunityTemplate` to submit.

        Returns:
            The name under which the template was registered.

        Raises:
            ValueError: If ``template.name`` is empty.
        """
        if not template.name or not isinstance(template.name, str):
            raise ValueError("Template name must be a non-empty string.")
        key = template.name.strip().lower()
        with self._lock:
            self._templates[key] = template
            _logger.debug("Submitted community template %r.", template.name)
        return template.name

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        tags: Optional[List[str]] = None,
    ) -> List[CommunityTemplate]:
        """Search community templates by free-text query and optional tags.

        The query is split into whitespace-separated terms; a template
        matches if *every* term appears (case-insensitive) in its name,
        description or tags.  When ``tags`` is provided, the template must
        also contain *all* of the specified tags.

        Args:
            query: The free-text search query.
            tags: Optional list of tags that must all be present.

        Returns:
            A list of matching :class:`CommunityTemplate` sorted by name.
        """
        terms = [t.lower() for t in query.split() if t]
        wanted_tags = (
            {t.strip().lower() for t in tags} if tags else None
        )
        matches: List[CommunityTemplate] = []
        with self._lock:
            templates = list(self._templates.values())
        for tmpl in templates:
            haystack = " ".join(
                [tmpl.name, tmpl.description, tmpl.author]
                + list(tmpl.tags)
            ).lower()
            if terms and not all(term in haystack for term in terms):
                continue
            if wanted_tags is not None:
                tmpl_tags = {t.strip().lower() for t in tmpl.tags}
                if not wanted_tags.issubset(tmpl_tags):
                    continue
            matches.append(tmpl)
        matches.sort(key=lambda t: t.name)
        return matches

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    def download(self, name: str) -> Canvas:
        """Download a community template as a new :class:`Canvas`.

        The download counter is incremented.  The returned canvas is an
        independent copy; modifying it does not affect the stored
        template.

        Args:
            name: The template name (case-insensitive).

        Returns:
            A new :class:`Canvas` initialised from the template's state.

        Raises:
            KeyError: If no template with that name is registered.
        """
        key = name.strip().lower()
        with self._lock:
            template = self._templates.get(key)
            if template is None:
                raise KeyError(
                    "No community template named {!r}.".format(name)
                )
            template.downloads += 1
            state = CanvasState.from_dict(template.canvas_state.to_dict())
        canvas = Canvas(template.name, state)
        _logger.debug("Downloaded community template %r.", name)
        return canvas

    # ------------------------------------------------------------------
    # Rate
    # ------------------------------------------------------------------
    def rate(self, name: str, rating: float) -> None:
        """Rate a community template.

        The rating is clamped to ``[0, 5]`` and stored directly on the
        template.  In a full deployment this would track individual votes
        and compute a running average; this simplified implementation
        sets the rating to the provided value.

        Args:
            name: The template name (case-insensitive).
            rating: The rating value, clamped to ``[0, 5]``.

        Raises:
            KeyError: If no template with that name is registered.
        """
        clamped = max(_MIN_RATING, min(_MAX_RATING, float(rating)))
        key = name.strip().lower()
        with self._lock:
            template = self._templates.get(key)
            if template is None:
                raise KeyError(
                    "No community template named {!r}.".format(name)
                )
            template.rating = clamped
        _logger.debug(
            "Rated community template %r -> %.2f.", name, clamped
        )

    # ------------------------------------------------------------------
    # List popular / recent
    # ------------------------------------------------------------------
    def list_popular(self, limit: int = 10) -> List[CommunityTemplate]:
        """Return the most-downloaded community templates.

        Args:
            limit: Maximum number of templates to return.

        Returns:
            A list of :class:`CommunityTemplate` sorted by download count
            (descending).
        """
        with self._lock:
            templates = list(self._templates.values())
        templates.sort(key=lambda t: t.downloads, reverse=True)
        return templates[:limit]

    def list_recent(self, limit: int = 10) -> List[CommunityTemplate]:
        """Return the most-recently-submitted community templates.

        Args:
            limit: Maximum number of templates to return.

        Returns:
            A list of :class:`CommunityTemplate` sorted by submission
            timestamp (descending).
        """
        with self._lock:
            templates = list(self._templates.values())
        templates.sort(key=lambda t: t.submitted_at, reverse=True)
        return templates[:limit]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def count(self) -> int:
        """Return the number of registered community templates."""
        with self._lock:
            return len(self._templates)

    def __repr__(self) -> str:
        with self._lock:
            return "CommunityRegistry(templates={})".format(
                len(self._templates)
            )
