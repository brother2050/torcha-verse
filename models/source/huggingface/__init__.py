"""HuggingFace Hub source adapter for the TorchaVerse model fetcher.

The :mod:`models.source.huggingface` sub-package is the
HuggingFace side of the v0.6.x model fetcher.  It exposes a
thin :class:`HuggingFaceSource` class that wraps the upstream
``https://huggingface.co`` API:

* :meth:`HuggingFaceSource.resolve_license` -- reads the repo
  metadata and returns the SPDX license id (or ``""`` when the
  source does not declare one).
* :meth:`HuggingFaceSource.list_files` -- enumerates the files
  associated with a revision.
* :meth:`HuggingFaceSource.download_files` -- downloads a list
  of files and returns ``[{name, data, sha256}, ...]``.

The adapter is designed to be *testable without a network*.
The HTTP transport is provided by an injectable
:class:`HttpTransport` object -- the default implementation is
:class:`UrllibTransport` (``urllib.request`` from the standard
library), but tests can swap in a fake that records calls or
returns canned responses.  This is why the module works in any
environment that has ``torch`` (and therefore any environment
in which the rest of TorchaVerse runs), without forcing a hard
dependency on the optional ``huggingface_hub`` package.

The v0.6.x refactor splits the previous single-file
``models/source/huggingface.py`` (983 lines) into seven
focused modules:

* :mod:`models.source.huggingface._constants` -- module-level
  constants (default API base, user-agent, timeout, URL
  templates).
* :mod:`models.source.huggingface._types` --
  :class:`FileDownload` / :class:`DownloadProgress` data
  classes.
* :mod:`models.source.huggingface._transport` -- the
  :class:`HttpTransport` abstract base.
* :mod:`models.source.huggingface._urllib_transport` -- the
  default :class:`UrllibTransport`.
* :mod:`models.source.huggingface._openai_transport` -- the
  :class:`OpenAICompatTransport` (Bearer-auth).
* :mod:`models.source.huggingface._ollama_transport` -- the
  :class:`OllamaTransport` (local daemon).
* :mod:`models.source.huggingface._download` -- the
  :func:`download_one_with_fallback` /
  :func:`download_default_artifacts` helpers (extracted from
  the largest method of the original
  :class:`HuggingFaceSource`).
* :mod:`models.source.huggingface._source` -- the
  :class:`HuggingFaceSource` core.

The public API is unchanged --
``from models.source.huggingface import HuggingFaceSource,
HttpTransport, UrllibTransport, OpenAICompatTransport,
OllamaTransport, FileDownload, DownloadProgress`` keeps
working.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``models.source`` (this module) -- HuggingFace adapter.
"""

from __future__ import annotations

from ._constants import (
    DEFAULT_API_BASE,
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
)
from ._download import (
    DEFAULT_ARTIFACT_CANDIDATES,
    download_default_artifacts,
    download_one_with_fallback,
)
from ._ollama_transport import OllamaTransport
from ._openai_transport import OpenAICompatTransport
from ._source import HuggingFaceSource
from ._transport import HttpTransport
from ._types import DownloadProgress, FileDownload
from ._urllib_transport import UrllibTransport

__all__ = [
    "HuggingFaceSource",
    "HttpTransport",
    "UrllibTransport",
    "OpenAICompatTransport",
    "OllamaTransport",
    "FileDownload",
    "DownloadProgress",
    # Constants exported for the sibling Civitai adapter so it
    # can construct a default transport with the same
    # user-agent / timeout.
    "DEFAULT_USER_AGENT",
    "DEFAULT_TIMEOUT",
    "DEFAULT_API_BASE",
    "DEFAULT_ARTIFACT_CANDIDATES",
    "download_one_with_fallback",
    "download_default_artifacts",
]
