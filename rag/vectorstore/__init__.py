"""Vector stores for the TorchaVerse RAG subsystem.

This package provides storage backends for dense embeddings with
similarity-search functionality.
"""

from __future__ import annotations

from .vector_store import (
    BaseVectorStore,
    FaissVectorStore,
    InMemoryVectorStore,
    SearchResult,
)

__all__ = [
    "BaseVectorStore",
    "FaissVectorStore",
    "InMemoryVectorStore",
    "SearchResult",
]
