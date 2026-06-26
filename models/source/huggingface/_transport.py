"""The :class:`HttpTransport` abstract base.

The :class:`HttpTransport` is the pluggable HTTP surface the
:class:`HuggingFaceSource` (and the sibling :class:`CivitaiSource`)
talks to.  Two methods are required:

* :meth:`get_json` -- GET a URL, decode the body as JSON, return
  the value (or ``None`` on an empty body).
* :meth:`get_bytes` -- GET a URL, return ``(body_bytes,
  response_headers_dict)``.

The two methods are deliberately small so the test suite can
easily fake them: a single ``MockTransport`` recording every
call and returning canned responses is enough to test the
:class:`HuggingFaceSource` end-to-end without touching the
network.

The default implementations live in :mod:`._urllib_transport`
(``urllib.request``), :mod:`._openai_transport`
(OpenAI-compatible Bearer-auth) and :mod:`._ollama_transport`
(local Ollama daemon).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

__all__ = ["HttpTransport"]


class HttpTransport:
    """Pluggable HTTP transport interface.

    The fetcher talks to HuggingFace through this interface so
    that tests can swap in a fake transport without
    monkey-patching ``urllib``.  The default implementation
    :class:`~.UrllibTransport` uses the standard library.
    """

    def get_json(
        self, url: str, *, headers: Optional[Dict[str, str]] = None
    ) -> Any:
        """Issue a GET, decode the response as JSON, return the value.

        Args:
            url: The URL to fetch.
            headers: Optional extra headers to send (merged
                with the transport's defaults).

        Returns:
            The decoded JSON value, or ``None`` when the body
            is empty.
        """
        # Abstract method -- every concrete transport in
        # :mod:`._urllib_transport`, :mod:`._openai_transport`
        # and :mod:`._ollama_transport` implements this.  See
        # placeholder #73 in docs/placeholder_registry.md.
        raise NotImplementedError

    def get_bytes(
        self, url: str, *, headers: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, Dict[str, str]]:
        """Issue a GET, return ``(body, response_headers)``.

        Args:
            url: The URL to fetch.
            headers: Optional extra headers to send (merged
                with the transport's defaults).

        Returns:
            A ``(body, response_headers)`` tuple.  The headers
            dict is normalised to lower-case keys (matching
            ``urllib`` / ``requests`` conventions).
        """
        # Abstract method -- see placeholder #74 in
        # docs/placeholder_registry.md.
        raise NotImplementedError
