"""Document loaders for the TorchaVerse RAG subsystem.

This package provides loaders for various document formats and a factory
for automatic loader selection based on file extension.
"""

from __future__ import annotations

from .document_loader import (
    BaseDocumentLoader,
    Document,
    DocumentLoaderFactory,
    HTMLLoader,
    MarkdownLoader,
    PDFLoader,
    TextFileLoader,
)

__all__ = [
    "BaseDocumentLoader",
    "Document",
    "DocumentLoaderFactory",
    "HTMLLoader",
    "MarkdownLoader",
    "PDFLoader",
    "TextFileLoader",
]
