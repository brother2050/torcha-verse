"""The :class:`OllamaTransport`.

Ollama is a popular local model runtime that exposes a JSON
HTTP API on the loopback address by default (default URL
``http://127.0.0.1:11434``).  The API is largely
self-describing and is **not** OpenAI-compatible in the v0.4.x
line, so this transport is shipped as a separate
:class:`HttpTransport` implementation rather than an
:class:`OpenAICompatTransport` variant.

This transport is a thin :class:`UrllibTransport` that injects
the ``Content-Type: application/json`` header on JSON requests,
tolerates the absence of auth headers, and accepts the
``OLLAMA_HOST`` / ``OLLAMA_API_KEY`` environment variables for
non-default deployments.

Args:
    host: Base URL of the Ollama daemon (no trailing slash).
        When ``None`` the constructor reads ``$OLLAMA_HOST`` and
        falls back to ``"http://127.0.0.1:11434"``.
    api_key: Optional bearer token.  When ``None`` the
        constructor falls back to ``$OLLAMA_API_KEY``.
    user_agent: Override the User-Agent header.
    timeout: Request timeout in seconds.  Ollama blob pulls can
        be very slow, so the default is 5x the
        :data:`DEFAULT_TIMEOUT`.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, Optional, Tuple

from ._constants import DEFAULT_TIMEOUT, DEFAULT_USER_AGENT
from ._transport import HttpTransport

__all__ = ["OllamaTransport"]


class OllamaTransport(HttpTransport):
    """HTTP transport that talks to a local or remote Ollama daemon.

    The transport is intentionally similar to
    :class:`OpenAICompatTransport` but injects ``Content-Type:
    application/json`` on JSON requests (Ollama's POST endpoints
    expect it) and tolerates the absence of auth headers.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: Optional[float] = None,
    ) -> None:
        if not host:
            host = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
        if not api_key:
            api_key = os.environ.get("OLLAMA_API_KEY")
        self._host: str = str(host).rstrip("/")
        self._api_key: str = str(api_key or "")
        self._user_agent: str = str(user_agent)
        # Ollama blob downloads can be slow (multi-GB models on
        # spinning disks); bump the default to 5x DEFAULT_TIMEOUT.
        self._timeout: float = float(
            timeout if timeout is not None else DEFAULT_TIMEOUT * 5
        )

    def _request(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        *,
        json: bool = True,
    ) -> Tuple[bytes, Dict[str, str]]:
        hdrs: Dict[str, str] = {"User-Agent": self._user_agent}
        if json:
            hdrs["Accept"] = "application/json"
            hdrs["Content-Type"] = "application/json"
        if self._api_key:
            hdrs["Authorization"] = "Bearer {}".format(self._api_key)
        if headers:
            for k, v in headers.items():
                if v is not None:
                    hdrs[k] = str(v)
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            raw = resp.read()
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
        return raw, resp_headers

    def get_json(
        self, url: str, *, headers: Optional[Dict[str, str]] = None
    ) -> Any:
        raw, _ = self._request(url, headers=headers, json=True)
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))

    def get_bytes(
        self, url: str, *, headers: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, Dict[str, str]]:
        raw, resp_headers = self._request(url, headers=headers, json=False)
        return raw, resp_headers
