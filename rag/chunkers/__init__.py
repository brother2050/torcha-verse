"""Text chunkers for the TorchaVerse RAG subsystem.

This package provides several chunking strategies for splitting
documents into smaller, retrievable pieces.
"""

from __future__ import annotations

from .text_chunker import (
    BaseChunker,
    Chunk,
    FixedLengthChunker,
    RecursiveChunker,
    SemanticChunker,
)

__all__ = [
    "BaseChunker",
    "Chunk",
    "FixedLengthChunker",
    "RecursiveChunker",
    "SemanticChunker",
]
