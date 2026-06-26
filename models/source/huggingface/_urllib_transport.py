"""The default :class:`UrllibTransport` (stdlib-only HTTP).

The :class:`UrllibTransport` is the default :class:`HttpTransport`
backed by :mod:`urllib.request`.  It has no third-party
dependencies and works in any Python 3.9+ environment, which is
important for TorchaVerse's "single-system, zero-optional-deps
default" policy.

Args:
    user_agent: Override the User-Agent header.  Defaults to
        :data:`DEFAULT_USER_AGENT` from the public facade.
    timeout: Request timeout in seconds.  Defaults to
        :data:`DEFAULT_TIMEOUT` (30s).
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, Optional, Tuple

from ._constants import (
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
)
from ._transport import HttpTransport

__all__ = ["UrllibTransport"]


class UrllibTransport(HttpTransport):
    """Default :class:`HttpTransport` backed by ``urllib.request``.

    No third-party dependencies; works in any Python 3.9+
    environment.
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._user_agent = str(user_agent)
        self._timeout = float(timeout)

    def _request(
        self, url: str, headers: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, Dict[str, str]]:
        hdrs = {"User-Agent": self._user_agent}
        if headers:
            hdrs.update(headers)
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            raw = resp.read()
            # urllib's headers are case-insensitive; casefold for safety.
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
        return raw, resp_headers

    def get_json(
        self, url: str, *, headers: Optional[Dict[str, str]] = None
    ) -> Any:
        hdrs = {"Accept": "application/json"}
        if headers:
            hdrs.update(headers)
        raw, _ = self._request(url, headers=hdrs)
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))

    def get_bytes(
        self, url: str, *, headers: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, Dict[str, str]]:
        raw, resp_headers = self._request(url, headers=headers)
        return raw, resp_headers
