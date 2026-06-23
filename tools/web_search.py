"""Web search tool with pluggable backends.

This module provides :class:`WebSearchTool`, a :class:`BaseTool`
implementation that queries the web for information.  It supports multiple
search backends and gracefully degrades when none are available.

Supported backends (in priority order):

1. **Google Custom Search API** -- requires ``GOOGLE_API_KEY`` and
   ``GOOGLE_CSE_ID`` environment variables.
2. **DuckDuckGo** -- uses the optional ``duckduckgo_search`` package.

When no backend is available the tool returns a descriptive message
instead of raising, so agents can fall back to their internal knowledge.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from core.tool_registry import BaseTool
from infrastructure.logger import get_logger

__all__ = ["WebSearchTool", "SearchResult"]

# ---------------------------------------------------------------------------
# Backend identifiers
# ---------------------------------------------------------------------------
_BACKEND_GOOGLE: str = "google"
_BACKEND_DUCKDUCKGO: str = "duckduckgo"
_BACKEND_NONE: str = "none"


class SearchResult(dict):
    """A single web search result.

    A thin :class:`dict` subclass providing attribute access for the
    common fields ``title``, ``url``, and ``snippet``.
    """

    @property
    def title(self) -> str:  # type: ignore[override]
        """The result title."""
        return self.get("title", "")

    @property
    def url(self) -> str:  # type: ignore[override]
        """The result URL."""
        return self.get("url", "")

    @property
    def snippet(self) -> str:  # type: ignore[override]
        """A short snippet/excerpt of the result."""
        return self.get("snippet", "")


class WebSearchTool(BaseTool):
    """Search the web for information.

    The tool auto-detects the best available backend at construction
    time.  Backends are tried in priority order: Google Custom Search
    (when API credentials are present), then DuckDuckGo (when the
    ``duckduckgo_search`` package is installed).

    Example::

        >>> tool = WebSearchTool()
        >>> results = tool.search("PyTorch tutorial", num_results=3)
        >>> len(results) <= 3
        True
    """

    name: str = "web_search"
    description: str = "Search the web for information"
    parameter_schema: Dict[str, Any] = {
        "query": {
            "type": "string",
            "description": "The search query",
            "required": True,
        },
        "num_results": {
            "type": "integer",
            "description": "Maximum number of results to return",
            "default": 5,
            "required": False,
        },
    }

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        timeout: int = 10,
    ) -> None:
        """Initialise the search tool.

        Args:
            google_api_key: Google API key.  When ``None`` the
                ``GOOGLE_API_KEY`` environment variable is consulted.
            google_cse_id: Google Custom Search Engine ID.  When ``None``
                the ``GOOGLE_CSE_ID`` environment variable is consulted.
            timeout: Request timeout in seconds.
        """
        self._google_api_key: Optional[str] = google_api_key or os.environ.get(
            "GOOGLE_API_KEY"
        )
        self._google_cse_id: Optional[str] = google_cse_id or os.environ.get(
            "GOOGLE_CSE_ID"
        )
        self.timeout: int = max(1, int(timeout))
        self._logger = get_logger(self.__class__.__name__)
        self._backend: str = self._detect_backend()
        self._logger.info("WebSearchTool using backend: %s", self._backend)

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------
    def _detect_backend(self) -> str:
        """Determine the best available search backend.

        Returns:
            One of ``"google"``, ``"duckduckgo"``, or ``"none"``.
        """
        if self._google_api_key and self._google_cse_id:
            return _BACKEND_GOOGLE
        if self._has_duckduckgo():
            return _BACKEND_DUCKDUCKGO
        return _BACKEND_NONE

    @staticmethod
    def _has_duckduckgo() -> bool:
        """Return ``True`` if the ``duckduckgo_search`` package is importable."""
        try:
            import duckduckgo_search  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def backend(self) -> str:
        """The active search backend identifier."""
        return self._backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(self, **params: Any) -> List[Dict[str, Any]]:
        """Execute a web search.

        Args:
            **params: Keyword arguments matching :attr:`parameter_schema`.
                Must include ``query``; ``num_results`` defaults to 5.

        Returns:
            A list of result dictionaries with ``title``, ``url``, and
            ``snippet`` keys.
        """
        query = params.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Parameter 'query' must be a non-empty string.")

        num_results = params.get("num_results", 5)
        if not isinstance(num_results, int) or num_results <= 0:
            num_results = 5

        return self.search(query, num_results=num_results)

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web for ``query``.

        Args:
            query: The search query string.
            num_results: Maximum number of results to return.

        Returns:
            A list of result dictionaries.  Each dictionary has the
            keys ``title``, ``url``, and ``snippet``.  When no backend
            is available a single informational result is returned.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        num_results = max(1, int(num_results))

        if self._backend == _BACKEND_GOOGLE:
            return self._search_google(query, num_results)
        if self._backend == _BACKEND_DUCKDUCKGO:
            return self._search_duckduckgo(query, num_results)

        # No backend available -- return an informational message.
        self._logger.warning(
            "No search backend available. Install 'duckduckgo_search' or "
            "set GOOGLE_API_KEY / GOOGLE_CSE_ID to enable web search."
        )
        return [
            {
                "title": "Web search unavailable",
                "url": "",
                "snippet": (
                    "No search API is configured. Install the "
                    "'duckduckgo_search' package or provide Google "
                    "Custom Search credentials (GOOGLE_API_KEY and "
                    "GOOGLE_CSE_ID) to enable web search."
                ),
            }
        ]

    # ------------------------------------------------------------------
    # Google Custom Search backend
    # ------------------------------------------------------------------
    def _search_google(
        self, query: str, num_results: int
    ) -> List[Dict[str, Any]]:
        """Query the Google Custom Search JSON API.

        Args:
            query: The search query.
            num_results: Maximum results (Google allows 1-10 per request).

        Returns:
            A list of result dictionaries.
        """
        # Google CSE allows at most 10 results per request.
        num = min(max(1, num_results), 10)
        params = urllib.parse.urlencode(
            {
                "key": self._google_api_key,
                "cx": self._google_cse_id,
                "q": query,
                "num": num,
            }
        )
        url = f"https://www.googleapis.com/customsearch/v1?{params}"

        try:
            data = self._http_get_json(url)
        except Exception as exc:
            self._logger.error("Google search failed: %s", exc)
            return [
                {
                    "title": "Search error",
                    "url": "",
                    "snippet": f"Google Custom Search request failed: {exc}",
                }
            ]

        items = data.get("items", [])
        results: List[Dict[str, Any]] = []
        for item in items[:num_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
            )
        return results

    # ------------------------------------------------------------------
    # DuckDuckGo backend
    # ------------------------------------------------------------------
    def _search_duckduckgo(
        self, query: str, num_results: int
    ) -> List[Dict[str, Any]]:
        """Query DuckDuckGo via the ``duckduckgo_search`` package.

        Args:
            query: The search query.
            num_results: Maximum results.

        Returns:
            A list of result dictionaries.
        """
        try:
            # The package was renamed across versions; try both APIs.
            try:
                from duckduckgo_search import DDGS  # type: ignore
            except ImportError:  # pragma: no cover
                from duckduckgo_search import ddgs as _ddgs_mod  # type: ignore

                DDGS = _ddgs_mod  # type: ignore[assignment]
        except ImportError:
            return [
                {
                    "title": "DuckDuckGo unavailable",
                    "url": "",
                    "snippet": (
                        "The 'duckduckgo_search' package is not installed."
                    ),
                }
            ]

        results: List[Dict[str, Any]] = []
        try:
            with DDGS() as ddgs:  # type: ignore[call-arg]
                for r in ddgs.text(query, max_results=num_results):
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href") or r.get("link") or r.get("url", ""),
                            "snippet": r.get("body") or r.get("snippet", ""),
                        }
                    )
        except Exception as exc:
            self._logger.error("DuckDuckGo search failed: %s", exc)
            return [
                {
                    "title": "Search error",
                    "url": "",
                    "snippet": f"DuckDuckGo search failed: {exc}",
                }
            ]
        return results

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------
    def _http_get_json(self, url: str) -> Dict[str, Any]:
        """Perform a GET request and parse the JSON response.

        Args:
            url: The request URL.

        Returns:
            The parsed JSON dictionary.

        Raises:
            Exception: If the request fails or the response is not valid
                JSON.
        """
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "TorchaVerse/1.0 WebSearchTool",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            body = response.read().decode("utf-8")
        return json.loads(body)
