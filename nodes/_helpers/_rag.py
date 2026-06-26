"""RAG node helpers: defaults and document normalisation (v0.6.x).

These helpers used to live in :mod:`nodes._helpers` (the v0.4.x
``_helpers.py`` of 710 lines) and are kept in their own sub-module
because they depend on :mod:`infrastructure.rag` -- a heavier
import than the rest of :mod:`nodes._helpers`.  Splitting them
out keeps the import cost of the lighter helpers bounded.
"""

from __future__ import annotations

import re
from typing import Any, List as _List, Mapping as _Mapping

__all__ = [
    "_RAG_INDEX_NAME_PATTERN",
    "_RAG_DEFAULT_TOP_K",
    "_RAG_DEFAULT_CHUNK_SIZE",
    "_RAG_DEFAULT_CHUNK_OVERLAP",
    "_RAG_DEFAULT_BACKEND",
    "_normalise_rag_documents",
]


#: Allowed pattern for an RAG index name -- alphanumeric, dash,
#: underscore and forward-slash (so tenant-style names like
#: ``tenant.alpha/docs`` are accepted).  Length is capped at 128
#: characters to keep index names friendly in logs, file systems
#: and the serving layer's URL paths.
_RAG_INDEX_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-/]{1,128}$")

#: Default top-k for RAG queries when the node input does not
#: provide one.
_RAG_DEFAULT_TOP_K: int = 5

#: Default character-window size for the :class:`TextChunker`
#: used by :class:`RAGIngestor` -- small enough to keep the
#: byte tokenizer within its O(n) sweet spot, large enough to
#: give meaningful context.
_RAG_DEFAULT_CHUNK_SIZE: int = 512

#: Default overlap between consecutive RAG chunks.
_RAG_DEFAULT_CHUNK_OVERLAP: int = 64

#: Default vector-store backend.  ``"auto"`` selects FAISS when
#: it is importable and the in-memory NumPy backend otherwise.
_RAG_DEFAULT_BACKEND: str = "auto"


def _normalise_rag_documents(value: Any) -> _List:
    """Convert a free-form ``documents`` payload into a list of
    :class:`infrastructure.rag.RAGDocument` instances.

    Accepted shapes:

    * ``list[RAGDocument]`` -- returned unchanged.
    * ``list[dict]`` with ``"doc_id"`` / ``"text"`` /
      ``"metadata"`` keys.
    * ``dict[doc_id -> text]`` -- turned into one
      :class:`RAGDocument` per entry.
    * ``None`` / empty -- returns ``[]``.

    Raises:
        TypeError: when an entry has the wrong shape.
        ValueError: when an entry is missing ``text`` or has a
            non-string ``text``.
    """
    from infrastructure.rag import RAGDocument  # local import: keep _helpers dependency-light

    if value is None:
        return []
    if isinstance(value, RAGDocument):
        return [value]
    if isinstance(value, _Mapping):
        return [
            RAGDocument(
                doc_id=str(k), text=str(v) if v is not None else ""
            )
            for k, v in value.items()
        ]
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"documents must be a list, tuple, dict or None; "
            f"got {type(value).__name__}."
        )
    documents: _List = []
    for i, entry in enumerate(value):
        if isinstance(entry, RAGDocument):
            documents.append(entry)
            continue
        if not isinstance(entry, _Mapping):
            raise TypeError(
                f"documents[{i}] must be a dict or RAGDocument; "
                f"got {type(entry).__name__}."
            )
        text = entry.get("text")
        if not isinstance(text, str):
            raise ValueError(
                f"documents[{i}].text must be a string; "
                f"got {type(text).__name__}."
            )
        doc_id = str(entry.get("doc_id") or "")
        meta = entry.get("metadata") or {}
        if not isinstance(meta, _Mapping):
            raise ValueError(
                f"documents[{i}].metadata must be a dict; "
                f"got {type(meta).__name__}."
            )
        documents.append(
            RAGDocument(
                doc_id=doc_id,
                text=text,
                metadata=dict(meta),
            )
        )
    return documents
