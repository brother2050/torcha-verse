"""Document loaders for the TorchaVerse RAG subsystem.

This module provides a family of document loaders that convert files in
various formats -- plain text, PDF, HTML, and Markdown -- into a uniform
list of :class:`Document` objects.  A factory
(:class:`DocumentLoaderFactory`) automatically selects the appropriate
loader based on the file extension.

All loaders inherit from :class:`BaseDocumentLoader` and implement the
``load(path) -> List[Document]`` contract.
"""

from __future__ import annotations

import abc
import os
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Type, Union

from infrastructure.logger import get_logger

__all__ = [
    "Document",
    "BaseDocumentLoader",
    "TextFileLoader",
    "PDFLoader",
    "HTMLLoader",
    "MarkdownLoader",
    "DocumentLoaderFactory",
]


# ---------------------------------------------------------------------------
# Document data class
# ---------------------------------------------------------------------------
@dataclass
class Document:
    """A loaded document.

    Attributes:
        content: The full text content of the document.
        metadata: Metadata dictionary.  Conventionally contains at least
            ``source`` (file path), ``page`` (page number, 0 for
            single-page documents), and ``format`` (e.g. ``"text"``,
            ``"pdf"``, ``"html"``, ``"markdown"``).
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        """The source file path (empty string when unknown)."""
        return self.metadata.get("source", "")

    @property
    def page(self) -> int:
        """The page number (0-indexed for single-page documents)."""
        return self.metadata.get("page", 0)

    @property
    def format(self) -> str:
        """The document format identifier."""
        return self.metadata.get("format", "")

    def __repr__(self) -> str:
        return (
            f"Document(format={self.format!r}, source={self.source!r}, "
            f"chars={len(self.content)})"
        )


# ---------------------------------------------------------------------------
# BaseDocumentLoader
# ---------------------------------------------------------------------------
class BaseDocumentLoader(abc.ABC):
    """Abstract base class for all document loaders.

    Subclasses implement :meth:`load` to read a file and return a list of
    :class:`Document` objects.

    Args:
        encoding: The text encoding used when reading files.
    """

    #: File extensions supported by this loader (without the leading dot).
    SUPPORTED_EXTENSIONS: tuple = ()

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding: str = encoding
        self._logger = get_logger(self.__class__.__name__)

    @abc.abstractmethod
    def load(self, path: Union[str, os.PathLike]) -> List[Document]:
        """Load documents from ``path``.

        Args:
            path: Path to the file to load.

        Returns:
            A list of :class:`Document` objects (one per page for
            multi-page formats such as PDF).
        """
        ...

    # ------------------------------------------------------------------
    def _read_file(self, path: str) -> str:
        """Read a text file and return its contents.

        Args:
            path: File path.

        Returns:
            The file contents as a string.

        Raises:
            IOError: If the file cannot be read.
        """
        with open(path, "r", encoding=self.encoding) as handle:
            return handle.read()


# ---------------------------------------------------------------------------
# TextFileLoader
# ---------------------------------------------------------------------------
class TextFileLoader(BaseDocumentLoader):
    """Loader for plain-text files (``.txt``, ``.text``)."""

    SUPPORTED_EXTENSIONS = (".txt", ".text")

    def load(self, path: Union[str, os.PathLike]) -> List[Document]:
        """Load a plain-text file.

        Args:
            path: Path to the ``.txt`` file.

        Returns:
            A single-element list containing the loaded :class:`Document`.
        """
        path = str(path)
        try:
            content = self._read_file(path)
        except (IOError, UnicodeDecodeError) as exc:
            self._logger.error("Failed to read '%s': %s", path, exc)
            return []

        return [
            Document(
                content=content,
                metadata={
                    "source": path,
                    "page": 0,
                    "format": "text",
                    "filename": os.path.basename(path),
                },
            )
        ]


# ---------------------------------------------------------------------------
# PDFLoader
# ---------------------------------------------------------------------------
class PDFLoader(BaseDocumentLoader):
    """Loader for PDF files.

    Uses :mod:`PyPDF2` when available, falling back to
    :mod:`pdfplumber`.  If neither library is installed an
    :class:`ImportError` is raised with installation instructions.
    """

    SUPPORTED_EXTENSIONS = (".pdf",)

    def load(self, path: Union[str, os.PathLike]) -> List[Document]:
        """Load a PDF file, returning one :class:`Document` per page.

        Args:
            path: Path to the ``.pdf`` file.

        Returns:
            A list of :class:`Document` objects, one per page.

        Raises:
            ImportError: If neither PyPDF2 nor pdfplumber is installed.
        """
        path = str(path)

        # Try PyPDF2 first.
        try:
            import PyPDF2  # type: ignore[import-untyped]

            return self._load_with_pypdf2(path, PyPDF2)
        except ImportError as exc:
            self._logger.debug("PyPDF2 unavailable, falling back to pdfplumber: %s", exc)

        # Fall back to pdfplumber.
        try:
            import pdfplumber  # type: ignore[import-untyped]

            return self._load_with_pdfplumber(path, pdfplumber)
        except ImportError as exc:
            self._logger.debug("pdfplumber unavailable, skipping PDF %s: %s", path, exc)

        raise ImportError(
            "PDF loading requires PyPDF2 or pdfplumber. "
            "Install one of them with:\n"
            "    pip install PyPDF2\n"
            "or\n"
            "    pip install pdfplumber"
        )

    # ------------------------------------------------------------------
    def _load_with_pypdf2(self, path: str, PyPDF2: Any) -> List[Document]:
        """Load a PDF using PyPDF2."""
        documents: List[Document] = []
        with open(path, "rb") as handle:
            reader = PyPDF2.PdfReader(handle)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                documents.append(
                    Document(
                        content=text,
                        metadata={
                            "source": path,
                            "page": i,
                            "format": "pdf",
                            "filename": os.path.basename(path),
                        },
                    )
                )
        self._logger.info("Loaded %d pages from '%s' (PyPDF2).", len(documents), path)
        return documents

    def _load_with_pdfplumber(self, path: str, pdfplumber: Any) -> List[Document]:
        """Load a PDF using pdfplumber."""
        documents: List[Document] = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                documents.append(
                    Document(
                        content=text,
                        metadata={
                            "source": path,
                            "page": i,
                            "format": "pdf",
                            "filename": os.path.basename(path),
                        },
                    )
                )
        self._logger.info(
            "Loaded %d pages from '%s' (pdfplumber).", len(documents), path
        )
        return documents


# ---------------------------------------------------------------------------
# HTMLLoader
# ---------------------------------------------------------------------------
class _HTMLTextExtractor(HTMLParser):
    """A lightweight HTML parser that extracts visible text.

    Skips ``<script>`` and ``<style>`` content and inserts line breaks
    at block-level element boundaries.
    """

    _BLOCK_TAGS = frozenset(
        {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}
    )
    _SKIP_TAGS = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: List[Any]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        """Return the extracted text with collapsed whitespace."""
        raw = "".join(self._parts)
        # Collapse runs of blank lines.
        text = re.sub(r"\n{3,}", "\n\n", raw)
        return text.strip()


class HTMLLoader(BaseDocumentLoader):
    """Loader for HTML files using the standard-library :mod:`html.parser`.

    Extracts visible text content while discarding ``<script>`` and
    ``<style>`` blocks.
    """

    SUPPORTED_EXTENSIONS = (".html", ".htm")

    def load(self, path: Union[str, os.PathLike]) -> List[Document]:
        """Load an HTML file and extract its visible text.

        Args:
            path: Path to the ``.html`` file.

        Returns:
            A single-element list containing the loaded :class:`Document`.
        """
        path = str(path)
        try:
            html = self._read_file(path)
        except (IOError, UnicodeDecodeError) as exc:
            self._logger.error("Failed to read '%s': %s", path, exc)
            return []

        extractor = _HTMLTextExtractor()
        extractor.feed(html)
        text = extractor.get_text()

        return [
            Document(
                content=text,
                metadata={
                    "source": path,
                    "page": 0,
                    "format": "html",
                    "filename": os.path.basename(path),
                },
            )
        ]


# ---------------------------------------------------------------------------
# MarkdownLoader
# ---------------------------------------------------------------------------
class MarkdownLoader(BaseDocumentLoader):
    """Loader for Markdown files (``.md``, ``.markdown``).

    Reads the raw Markdown and strips common formatting syntax (code
    fences, inline code, links, images, emphasis, headers, list markers,
    blockquotes) to produce clean plain text.
    """

    SUPPORTED_EXTENSIONS = (".md", ".markdown")

    # Regex patterns for stripping Markdown syntax.
    _IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
    _LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
    _BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)
    _ITALIC_RE = re.compile(r"\*(.+?)\*|_(.+?)_", re.DOTALL)
    _INLINE_CODE_RE = re.compile(r"`([^`]+)`")
    _HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
    _LIST_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
    _NUM_LIST_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
    _BLOCKQUOTE_RE = re.compile(r"^\s*>\s?", re.MULTILINE)
    _HR_RE = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)

    def load(self, path: Union[str, os.PathLike]) -> List[Document]:
        """Load a Markdown file and strip formatting syntax.

        Args:
            path: Path to the ``.md`` file.

        Returns:
            A single-element list containing the loaded :class:`Document`.
        """
        path = str(path)
        try:
            content = self._read_file(path)
        except (IOError, UnicodeDecodeError) as exc:
            self._logger.error("Failed to read '%s': %s", path, exc)
            return []

        text = self._strip_markdown(content)

        return [
            Document(
                content=text,
                metadata={
                    "source": path,
                    "page": 0,
                    "format": "markdown",
                    "filename": os.path.basename(path),
                },
            )
        ]

    def _strip_markdown(self, text: str) -> str:
        """Remove Markdown formatting, returning plain text.

        Args:
            text: Raw Markdown text.

        Returns:
            Cleaned plain text.
        """
        # Remove fenced code blocks, keeping their inner content.
        def _strip_fence(match: re.Match) -> str:
            block = match.group(0)
            lines = block.split("\n")
            # Drop the opening ```lang and closing ``` lines.
            if len(lines) >= 2:
                return "\n".join(lines[1:-1])
            return block

        text = re.sub(r"```[^\n]*\n.*?```", _strip_fence, text, flags=re.DOTALL)

        # Strip remaining inline syntax.
        text = self._IMAGE_RE.sub(r"\1", text)
        text = self._LINK_RE.sub(r"\1", text)
        text = self._INLINE_CODE_RE.sub(r"\1", text)
        text = self._BOLD_RE.sub(lambda m: m.group(1) or m.group(2), text)
        text = self._ITALIC_RE.sub(lambda m: m.group(1) or m.group(2), text)
        text = self._HEADER_RE.sub("", text)
        text = self._LIST_RE.sub("", text)
        text = self._NUM_LIST_RE.sub("", text)
        text = self._BLOCKQUOTE_RE.sub("", text)
        text = self._HR_RE.sub("", text)

        # Collapse excessive blank lines.
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ---------------------------------------------------------------------------
# DocumentLoaderFactory
# ---------------------------------------------------------------------------
class DocumentLoaderFactory:
    """Factory that selects a loader based on file extension.

    Maintains a registry mapping file extensions to loader classes.
    Custom loaders can be registered via :meth:`register_loader`.
    """

    _loaders: Dict[str, Type[BaseDocumentLoader]] = {
        ".txt": TextFileLoader,
        ".text": TextFileLoader,
        ".pdf": PDFLoader,
        ".html": HTMLLoader,
        ".htm": HTMLLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
    }

    # ------------------------------------------------------------------
    @classmethod
    def register_loader(cls, extension: str, loader_cls: Type[BaseDocumentLoader]) -> None:
        """Register a loader for a file extension.

        Args:
            extension: File extension including the leading dot
                (e.g. ``".docx"``).  Case-insensitive.
            loader_cls: A :class:`BaseDocumentLoader` subclass.
        """
        if not issubclass(loader_cls, BaseDocumentLoader):
            raise TypeError("loader_cls must be a subclass of BaseDocumentLoader.")
        cls._loaders[extension.lower()] = loader_cls

    # ------------------------------------------------------------------
    @classmethod
    def create_loader(
        cls,
        path: Union[str, os.PathLike],
        encoding: str = "utf-8",
    ) -> BaseDocumentLoader:
        """Create a loader appropriate for the given file path.

        Args:
            path: File path (used only to determine the extension).
            encoding: Text encoding for the loader.

        Returns:
            A :class:`BaseDocumentLoader` instance.

        Raises:
            ValueError: If no loader is registered for the extension.
        """
        ext = os.path.splitext(str(path))[1].lower()
        loader_cls = cls._loaders.get(ext)
        if loader_cls is None:
            raise ValueError(
                f"No document loader registered for extension '{ext}'. "
                f"Supported: {', '.join(sorted(cls._loaders))}."
            )
        return loader_cls(encoding=encoding)

    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: Union[str, os.PathLike],
        encoding: str = "utf-8",
    ) -> List[Document]:
        """Load documents from ``path`` using the auto-selected loader.

        Args:
            path: File path.
            encoding: Text encoding.

        Returns:
            A list of :class:`Document` objects.
        """
        loader = cls.create_loader(path, encoding=encoding)
        return loader.load(path)

    # ------------------------------------------------------------------
    @classmethod
    def supported_extensions(cls) -> List[str]:
        """Return a sorted list of supported file extensions."""
        return sorted(cls._loaders.keys())
