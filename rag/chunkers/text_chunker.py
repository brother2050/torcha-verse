"""Text chunkers for the TorchaVerse RAG subsystem.

This module provides several strategies for splitting documents into
smaller, retrievable chunks:

* :class:`FixedLengthChunker` -- splits at fixed character boundaries with
  optional overlap.
* :class:`SemanticChunker` -- splits at sentence boundaries, grouping
  sentences up to a maximum size.
* :class:`RecursiveChunker` -- recursively splits using a hierarchy of
  separators (paragraph -> sentence -> word) to respect natural
  boundaries.

All chunkers inherit from :class:`BaseChunker` and implement the
``chunk(text) -> List[Chunk]`` contract.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from infrastructure.logger import get_logger

__all__ = [
    "Chunk",
    "BaseChunker",
    "FixedLengthChunker",
    "SemanticChunker",
    "RecursiveChunker",
]


# ---------------------------------------------------------------------------
# Chunk data class
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """A text chunk extracted from a document.

    Attributes:
        text: The chunk text content.
        metadata: Metadata dictionary.  Conventionally contains
            ``index`` (sequential chunk number), ``start`` (character
            offset in the source text), and ``end`` (exclusive offset).
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def index(self) -> int:
        """The sequential chunk index."""
        return self.metadata.get("index", 0)

    @property
    def start(self) -> int:
        """The start character offset in the source text."""
        return self.metadata.get("start", 0)

    @property
    def end(self) -> int:
        """The end character offset (exclusive) in the source text."""
        return self.metadata.get("end", 0)

    def __repr__(self) -> str:
        preview = self.text[:50].replace("\n", " ")
        return f"Chunk(index={self.index}, chars={len(self.text)}, text={preview!r}...)"


# ---------------------------------------------------------------------------
# BaseChunker
# ---------------------------------------------------------------------------
class BaseChunker(abc.ABC):
    """Abstract base class for all text chunkers.

    Args:
        chunk_size: Target maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive
            chunks.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}.")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than chunk_size "
                f"({chunk_size})."
            )
        self.chunk_size: int = chunk_size
        self.overlap: int = overlap
        self._logger = get_logger(self.__class__.__name__)

    @abc.abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """Split ``text`` into chunks.

        Args:
            text: The text to split.

        Returns:
            A list of :class:`Chunk` objects.
        """
        ...

    # ------------------------------------------------------------------
    def _make_chunk(
        self,
        text: str,
        index: int,
        start: int,
        end: int,
        **extra: Any,
    ) -> Chunk:
        """Create a :class:`Chunk` with standard metadata.

        Args:
            text: The chunk text.
            index: Sequential chunk index.
            start: Start character offset.
            end: End character offset (exclusive).
            **extra: Additional metadata fields.

        Returns:
            A :class:`Chunk` instance.
        """
        metadata: Dict[str, Any] = {
            "index": index,
            "start": start,
            "end": end,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
        }
        metadata.update(extra)
        return Chunk(text=text, metadata=metadata)


# ---------------------------------------------------------------------------
# FixedLengthChunker
# ---------------------------------------------------------------------------
class FixedLengthChunker(BaseChunker):
    """Split text into fixed-length chunks with optional overlap.

    This is the simplest chunker: it cuts the text at every
    ``chunk_size`` characters, optionally overlapping consecutive chunks
    by ``overlap`` characters.

    Args:
        chunk_size: Maximum characters per chunk.
        overlap: Overlap characters between consecutive chunks.
    """

    def chunk(self, text: str) -> List[Chunk]:
        """Split ``text`` into fixed-length overlapping chunks.

        Args:
            text: The text to split.

        Returns:
            A list of :class:`Chunk` objects.
        """
        if not text or not text.strip():
            return []

        chunks: List[Chunk] = []
        step = max(1, self.chunk_size - self.overlap)
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(self._make_chunk(chunk_text, index, start, end))
            if end >= len(text):
                break
            start += step
            index += 1

        self._logger.debug("Split text (%d chars) into %d fixed chunks.", len(text), len(chunks))
        return chunks


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------
class SemanticChunker(BaseChunker):
    """Split text at sentence boundaries.

    Groups consecutive sentences into chunks that stay under
    ``chunk_size`` characters.  Sentence boundaries are detected using a
    regular expression that matches common sentence terminators
    (``.``, ``!``, ``?``) followed by whitespace.

    Args:
        chunk_size: Maximum characters per chunk.
        overlap: Target overlap in characters (used to carry trailing
            sentences into the next chunk).
        min_chunk_size: Minimum characters before a chunk is flushed.
    """

    # Matches a sentence: non-terminator characters followed by .!?
    _SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+[\s]*", re.MULTILINE)

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        min_chunk_size: int = 100,
    ) -> None:
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self.min_chunk_size: int = min_chunk_size

    # ------------------------------------------------------------------
    def chunk(self, text: str) -> List[Chunk]:
        """Split ``text`` into semantically coherent chunks.

        Args:
            text: The text to split.

        Returns:
            A list of :class:`Chunk` objects.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return [self._make_chunk(text.strip(), 0, 0, len(text))]

        chunks: List[Chunk] = []
        current = ""
        current_start = 0
        index = 0

        for sent_text, sent_start in sentences:
            if current and len(current) + len(sent_text) > self.chunk_size:
                # Flush current chunk.
                chunks.append(
                    self._make_chunk(
                        current.strip(), index, current_start, current_start + len(current)
                    )
                )
                index += 1
                # Start new chunk, optionally with overlap.
                if self.overlap > 0 and len(current) > self.overlap:
                    overlap_text = current[-self.overlap:]
                    current = overlap_text + sent_text
                    current_start = sent_start - self.overlap
                else:
                    current = sent_text
                    current_start = sent_start
            else:
                if not current:
                    current_start = sent_start
                current += sent_text

        if current.strip():
            chunks.append(
                self._make_chunk(
                    current.strip(), index, current_start, current_start + len(current)
                )
            )

        self._logger.debug("Split text (%d chars) into %d semantic chunks.", len(text), len(chunks))
        return chunks

    # ------------------------------------------------------------------
    def _split_sentences(self, text: str) -> List[tuple]:
        """Split text into (sentence, start_offset) pairs.

        Args:
            text: The text to split.

        Returns:
            A list of ``(sentence_text, start_position)`` tuples.
        """
        sentences: List[tuple] = []
        for match in self._SENTENCE_RE.finditer(text):
            sentences.append((match.group(0), match.start()))

        # Capture any trailing text that doesn't end with a terminator.
        if sentences:
            last_end = sentences[-1][1] + len(sentences[-1][0])
            remainder = text[last_end:].strip()
            if remainder:
                sentences.append((text[last_end:], last_end))
        elif text.strip():
            sentences.append((text, 0))

        return sentences


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------
class RecursiveChunker(BaseChunker):
    """Recursively split text using a hierarchy of separators.

    Tries to split on the most natural boundary first (paragraphs),
    falling back to finer separators (sentences, words) when a piece is
    still too large.  This preserves semantic structure better than
    fixed-length splitting.

    Args:
        chunk_size: Maximum characters per chunk.
        overlap: Overlap characters between consecutive chunks.
        separators: Ordered list of separators to try, from coarsest
            to finest.  An empty string ``""`` signals fixed-length
            fallback.
    """

    DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self.separators: List[str] = list(separators) if separators else list(self.DEFAULT_SEPARATORS)

    # ------------------------------------------------------------------
    def chunk(self, text: str) -> List[Chunk]:
        """Split ``text`` recursively.

        Args:
            text: The text to split.

        Returns:
            A list of :class:`Chunk` objects with positional metadata.
        """
        if not text or not text.strip():
            return []

        raw_chunks = self._split_text(text)
        if not raw_chunks:
            return []

        # Assign positional metadata by locating each chunk in the source.
        chunks: List[Chunk] = []
        search_pos = 0
        for i, chunk_text in enumerate(raw_chunks):
            stripped = chunk_text.strip()
            if not stripped:
                continue
            # Find the chunk in the original text for accurate offsets.
            start = text.find(stripped[:40], search_pos)
            if start == -1:
                start = search_pos
            end = min(start + len(stripped), len(text))
            search_pos = end
            chunks.append(self._make_chunk(stripped, i, start, end))

        self._logger.debug("Split text (%d chars) into %d recursive chunks.", len(text), len(chunks))
        return chunks

    # ------------------------------------------------------------------
    def _split_text(self, text: str) -> List[str]:
        """Recursively split ``text`` into pieces under ``chunk_size``.

        Args:
            text: The text to split.

        Returns:
            A list of text pieces.
        """
        if len(text) <= self.chunk_size:
            return [text]

        for i, sep in enumerate(self.separators):
            if sep == "":
                # Last resort: fixed-length split.
                return self._fixed_split(text)

            parts = text.split(sep)
            if len(parts) <= 1:
                continue

            # Merge parts greedily up to chunk_size.
            merged = self._merge_parts(parts, sep)

            # Recurse on any piece that is still too large.
            final: List[str] = []
            for piece in merged:
                if len(piece) > self.chunk_size:
                    final.extend(self._split_text(piece))
                else:
                    final.append(piece)
            return final

        return self._fixed_split(text)

    # ------------------------------------------------------------------
    def _merge_parts(self, parts: List[str], sep: str) -> List[str]:
        """Greedily merge ``parts`` joined by ``sep`` up to ``chunk_size``.

        Args:
            parts: Text pieces split by ``sep``.
            sep: The separator used to join parts.

        Returns:
            A list of merged text pieces.
        """
        merged: List[str] = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                # If the single part exceeds chunk_size it will be
                # recursed on later.
                current = part
        if current:
            merged.append(current)
        return merged

    # ------------------------------------------------------------------
    def _fixed_split(self, text: str) -> List[str]:
        """Split ``text`` into fixed-length overlapping pieces.

        Args:
            text: The text to split.

        Returns:
            A list of text pieces.
        """
        step = max(1, self.chunk_size - self.overlap)
        pieces: List[str] = []
        start = 0
        while start < len(text):
            pieces.append(text[start : start + self.chunk_size])
            if start + self.chunk_size >= len(text):
                break
            start += step
        return pieces
