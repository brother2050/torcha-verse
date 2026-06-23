"""Tool implementations for the TorchaVerse framework.

This package provides concrete :class:`BaseTool` subclasses that can be
registered with the :class:`ToolRegistry` and invoked by LLM-based agents:

* :class:`CalculatorTool` -- safe mathematical expression evaluation.
* :class:`PythonExecutorTool` -- sandboxed Python code execution.
* :class:`FileOpsTool` -- file read/write/management with path safety.
* :class:`WebSearchTool` -- web search with pluggable backends.
"""

from __future__ import annotations

from core.tool_registry import BaseTool, ToolRegistry
from .calculator import CalculatorTool
from .file_ops import FileOpsTool
from .python_executor import ExecutionResult, PythonExecutorTool
from .web_search import SearchResult, WebSearchTool

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "CalculatorTool",
    "PythonExecutorTool",
    "ExecutionResult",
    "FileOpsTool",
    "WebSearchTool",
    "SearchResult",
]
