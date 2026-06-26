"""The :class:`OpenAICompatTransport`.

The OpenAI REST API is a de-facto industry standard -- the
same shape is served by a long list of providers (Together,
Anyscale, Fireworks, OpenRouter, vLLM with
``--served-model-name``, local llama-cpp-python servers, etc.).
This transport is a thin :class:`UrllibTransport` that injects
an ``Authorization: Bearer <api_key>`` header by default and
accepts the same payload as the OpenAI REST API.

The transport works for *any* endpoint that speaks
OpenAI-compatible JSON (e.g. model metadata at
``/v1/models/{id}``, file download at
``/v1/files/{id}/content``).  It is registered as an
:class:`HttpTransport` so it can be plugged into
:class:`HuggingFaceSource` / :class:`CivitaiSource` whenever the
caller wants the adapter to talk to an OpenAI-compatible
mirror instead of the canonical HF / Civitai endpoints.

Args:
    api_key: Bearer token.  When ``None`` the constructor falls
        back to ``$OPENAI_API_KEY`` (and then
        ``$OPENAI_COMPAT_API_KEY``) so the typical
        ``OpenAICompatTransport()`` no-arg form Just Works in
        CI.
    base_url: The provider's base URL (no trailing slash).
        Defaults to ``"https://api.openai.com"``.
    user_agent: Override the User-Agent header.
    timeout: Request timeout in seconds.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, Optional, Tuple

from ._constants import DEFAULT_TIMEOUT, DEFAULT_USER_AGENT
from ._transport import HttpTransport

__all__ = ["OpenAICompatTransport"]


class OpenAICompatTransport(HttpTransport):
    """HTTP transport that talks to OpenAI-compatible model providers.

    The transport is stateless from the auth point of view: it
    injects ``Bearer`` + ``Accept`` on every call.  Callers may
    still pass extra headers (e.g. ``OpenAI-Organization``) via
    the per-call ``headers`` argument.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com",
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
                "OPENAI_COMPAT_API_KEY"
            )
        self._api_key: str = str(api_key or "")
        self._base_url: str = str(base_url).rstrip("/")
        self._user_agent: str = str(user_agent)
        self._timeout: float = float(timeout)

    def _request(
        self, url: str, headers: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, Dict[str, str]]:
        hdrs: Dict[str, str] = {
            "User-Agent": self._user_agent,
            "Accept": "application/json",
        }
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
        raw, _ = self._request(url, headers=headers)
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))

    def get_bytes(
        self, url: str, *, headers: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, Dict[str, str]]:
        raw, resp_headers = self._request(url, headers=headers)
        return raw, resp_headers
