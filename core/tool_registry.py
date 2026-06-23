"""Tool registry center for TorchaVerse.

This module provides :class:`ToolRegistry`, a central registry that
catalogues callable tools (functions) that can be invoked by LLM-based
agents.  Tools are registered with a name, description, and parameter
schema, enabling automatic parameter validation and LLM function-calling
support.

Key features:

* :class:`BaseTool` -- Abstract base class defining the ``execute``
  contract.
* :class:`Tool` -- Data class holding tool metadata.
* :func:`register_tool` -- Decorator for convenient registration.
* Automatic parameter validation against JSON-schema-style
  ``parameter_schema``.
* :meth:`discover_tools` -- Keyword-based tool discovery.
* :meth:`get_tool_descriptions` -- Export tool descriptions in a format
  suitable for LLM function calling.
"""

from __future__ import annotations

import abc
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

from infrastructure.logger import get_logger

__all__ = [
    "BaseTool",
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "register_tool",
    "validate_params",
]


# ---------------------------------------------------------------------------
# Tool data class
# ---------------------------------------------------------------------------
@dataclass
class Tool:
    """Metadata for a registered tool.

    Attributes:
        name: Unique tool name.
        description: Human-readable description of what the tool does.
        parameter_schema: JSON-schema-style dictionary describing the
            expected parameters.
        func: The callable invoked when the tool is executed.
    """

    name: str
    description: str
    parameter_schema: Dict[str, Any] = field(default_factory=dict)
    func: Optional[Callable[..., Any]] = None

    def __post_init__(self) -> None:
        self.name = self.name.strip()


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------
@dataclass
class ToolResult:
    """The result of a tool execution.

    Attributes:
        success: Whether the execution succeeded.
        output: The return value of the tool.
        error: Error message when ``success`` is ``False``.
        metadata: Additional execution metadata.
    """

    success: bool = True
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, output: Any, **metadata: Any) -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, output=output, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> "ToolResult":
        """Create a failed result."""
        return cls(success=False, error=error, metadata=metadata)


# ---------------------------------------------------------------------------
# BaseTool
# ---------------------------------------------------------------------------
class BaseTool(abc.ABC):
    """Abstract base class for all tools.

    Subclasses implement :meth:`execute` and optionally override
    :attr:`name`, :attr:`description`, and :attr:`parameter_schema`.

    Example:
        >>> class CalculatorTool(BaseTool):
        ...     name = "calculator"
        ...     description = "Perform arithmetic calculations"
        ...     parameter_schema = {
        ...         "expression": {"type": "string", "required": True}
        ...     }
        ...     def execute(self, expression: str) -> float:
        ...         return eval(expression)
    """

    name: str = ""
    description: str = ""
    parameter_schema: Dict[str, Any] = {}

    @abc.abstractmethod
    def execute(self, **params: Any) -> Any:
        """Execute the tool with the given parameters.

        Args:
            **params: Keyword arguments matching ``parameter_schema``.

        Returns:
            The tool's output.
        """
        ...

    def to_tool(self) -> Tool:
        """Convert this tool instance to a :class:`Tool` data object."""
        return Tool(
            name=self.name or self.__class__.__name__,
            description=self.description,
            parameter_schema=self.parameter_schema,
            func=self.execute,
        )


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------
def validate_params(
    params: Dict[str, Any],
    schema: Dict[str, Any],
) -> List[str]:
    """Validate ``params`` against a JSON-schema-style ``schema``.

    The schema is a dictionary mapping parameter names to their
    specifications::

        {
            "param_name": {
                "type": "string" | "integer" | "float" | "boolean" | "list" | "dict",
                "required": True | False,
                "default": <default value>,
                "description": "..."
            }
        }

    Args:
        params: The parameters to validate.
        schema: The parameter schema.

    Returns:
        A list of error messages (empty if valid).
    """
    errors: List[str] = []

    _type_map: Dict[str, tuple] = {
        "string": (str,),
        "integer": (int,),
        "float": (float, int),  # int is acceptable for float
        "boolean": (bool,),
        "list": (list, tuple),
        "dict": (dict,),
        "any": (object,),
    }

    for param_name, spec in schema.items():
        expected_type = spec.get("type", "any")
        required = spec.get("required", False)
        has_default = "default" in spec

        value = params.get(param_name, _MISSING)

        if value is _MISSING:
            if required and not has_default:
                errors.append(f"Missing required parameter: '{param_name}'.")
            continue

        # Type checking.
        if expected_type != "any":
            accepted_types = _type_map.get(expected_type)
            if accepted_types is None:
                errors.append(
                    f"Unknown type '{expected_type}' for parameter '{param_name}'."
                )
            elif not isinstance(value, accepted_types):
                errors.append(
                    f"Parameter '{param_name}' expected type '{expected_type}', "
                    f"got '{type(value).__name__}'."
                )

    return errors


# Sentinel for missing values.
_MISSING: Any = object()


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------
class ToolRegistry:
    """Central registry for callable tools.

    Tools can be registered in three ways:

    1. Explicitly via :meth:`register_tool`.
    2. Via the :func:`register_tool` decorator.
    3. By registering a :class:`BaseTool` subclass instance.

    Example:
        >>> registry = ToolRegistry()
        >>> @register_tool("greet", "Greet a person by name")
        ... def greet(name: str) -> str:
        ...     return f"Hello, {name}!"
        >>> result = registry.execute_tool("greet", {"name": "World"})
        >>> result.output
        'Hello, World!'
    """

    _instance: Optional["ToolRegistry"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._tools: Dict[str, Tool] = {}
        self._logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_tool(
        self,
        name: str,
        func: Optional[Callable[..., Any]] = None,
        description: str = "",
        parameter_schema: Optional[Dict[str, Any]] = None,
    ) -> Callable[..., Any]:
        """Register a tool.

        Can be used as a function call or as a decorator factory::

            registry.register_tool("my_tool", my_func, "Does something")

            @registry.register_tool("my_tool", description="Does something")
            def my_func(...): ...

        When called as a decorator factory (``func`` is ``None``), it
        returns a decorator.  When called directly (``func`` is provided),
        it registers the tool and returns ``func`` unchanged.

        Args:
            name: Unique tool name.
            func: The callable to register.  When ``None``, this method
                returns a decorator.
            description: Human-readable description.
            parameter_schema: Parameter schema for validation.

        Returns:
            The original ``func`` when called directly, or a decorator
            when ``func`` is ``None``.
        """
        if func is not None:
            self._do_register(name, func, description, parameter_schema)
            return func

        # Decorator mode.
        def _decorator(target: Callable[..., Any]) -> Callable[..., Any]:
            self._do_register(name, target, description, parameter_schema)
            return target

        return _decorator

    def _do_register(
        self,
        name: str,
        func: Callable[..., Any],
        description: str,
        parameter_schema: Optional[Dict[str, Any]],
    ) -> None:
        """Internal registration logic."""
        key = name.strip().lower()
        if not key:
            raise ValueError("Tool name must be a non-empty string.")

        # Auto-generate description and schema from the function signature
        # when not provided.
        if not description:
            description = func.__doc__ or f"Tool: {name}"

        if parameter_schema is None:
            parameter_schema = self._infer_schema(func)

        tool = Tool(
            name=key,
            description=description.strip(),
            parameter_schema=parameter_schema,
            func=func,
        )
        self._tools[key] = tool
        self._logger.debug("Registered tool '%s'.", key)

    def register_base_tool(self, tool_instance: BaseTool) -> None:
        """Register a :class:`BaseTool` subclass instance.

        Args:
            tool_instance: An instantiated :class:`BaseTool`.
        """
        tool = tool_instance.to_tool()
        self._tools[tool.name.lower()] = tool
        self._logger.debug("Registered base tool '%s'.", tool.name)

    def unregister_tool(self, name: str) -> bool:
        """Remove a tool from the registry.

        Args:
            name: Tool name.

        Returns:
            ``True`` if removed, ``False`` if not found.
        """
        key = name.strip().lower()
        if key in self._tools:
            del self._tools[key]
            self._logger.debug("Unregistered tool '%s'.", key)
            return True
        return False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover_tools(
        self,
        query: Optional[str] = None,
        limit: int = 10,
    ) -> List[Tool]:
        """Discover tools matching ``query``.

        When ``query`` is ``None`` all tools are returned.  Otherwise a
        simple keyword match is performed against the tool name and
        description.

        Args:
            query: Search query (case-insensitive).
            limit: Maximum number of results.

        Returns:
            A list of matching :class:`Tool` objects.
        """
        if query is None:
            return list(self._tools.values())[:limit]

        query_lower = query.lower()
        scored: List[tuple] = []
        for tool in self._tools.values():
            name_match = query_lower in tool.name.lower()
            desc_match = query_lower in tool.description.lower()
            if name_match or desc_match:
                score = (2 if name_match else 0) + (1 if desc_match else 0)
                scored.append((score, tool))

        scored.sort(key=lambda x: -x[0])
        return [tool for _, tool in scored[:limit]]

    def list_available(self) -> List[str]:
        """Return a sorted list of all registered tool names."""
        return sorted(self._tools.keys())

    def is_registered(self, name: str) -> bool:
        """Return ``True`` if ``name`` is a registered tool."""
        return name.strip().lower() in self._tools

    def get_tool(self, name: str) -> Optional[Tool]:
        """Return the :class:`Tool` registered under ``name``.

        Args:
            name: Tool name.

        Returns:
            The :class:`Tool` or ``None`` if not found.
        """
        return self._tools.get(name.strip().lower())

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute_tool(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute a registered tool.

        Parameters are validated against the tool's ``parameter_schema``
        before execution.  Missing optional parameters with defaults are
        filled in automatically.

        Args:
            name: Tool name.
            params: Parameters to pass to the tool.

        Returns:
            A :class:`ToolResult` containing the output or error.
        """
        tool = self.get_tool(name)
        if tool is None:
            return ToolResult.fail(
                f"Tool '{name}' is not registered. "
                f"Available: {', '.join(self.list_available()) or '(none)'}."
            )

        if tool.func is None:
            return ToolResult.fail(f"Tool '{name}' has no callable function.")

        params = params or {}

        # Fill in defaults.
        for param_name, spec in tool.parameter_schema.items():
            if param_name not in params and "default" in spec:
                params[param_name] = spec["default"]

        # Validate parameters.
        errors = validate_params(params, tool.parameter_schema)
        if errors:
            return ToolResult.fail(
                f"Parameter validation failed: {'; '.join(errors)}"
            )

        # Execute.
        try:
            output = tool.func(**params)
            return ToolResult.ok(output, tool=name)
        except Exception as exc:
            self._logger.exception("Tool '%s' execution failed: %s", name, exc)
            return ToolResult.fail(str(exc), tool=name, exception_type=type(exc).__name__)

    # ------------------------------------------------------------------
    # LLM function-calling support
    # ------------------------------------------------------------------
    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Return tool descriptions for LLM function calling.

        The output follows the OpenAI function-calling format::

            [
                {
                    "name": "tool_name",
                    "description": "What the tool does",
                    "parameters": {
                        "type": "object",
                        "properties": { ... },
                        "required": [ ... ]
                    }
                }
            ]

        Returns:
            A list of tool description dictionaries.
        """
        descriptions: List[Dict[str, Any]] = []
        for tool in self._tools.values():
            properties: Dict[str, Any] = {}
            required: List[str] = []
            for param_name, spec in tool.parameter_schema.items():
                prop: Dict[str, Any] = {"type": spec.get("type", "any")}
                if "description" in spec:
                    prop["description"] = spec["description"]
                if "default" in spec:
                    prop["default"] = spec["default"]
                properties[param_name] = prop
                if spec.get("required", False):
                    required.append(param_name)

            descriptions.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                }
            )
        return descriptions

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Alias for :meth:`get_tool_descriptions`."""
        return self.get_tool_descriptions()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_schema(func: Callable[..., Any]) -> Dict[str, Any]:
        """Infer a parameter schema from the function signature.

        Args:
            func: The function to inspect.

        Returns:
            A parameter schema dictionary.
        """
        sig = inspect.signature(func)
        schema: Dict[str, Any] = {}

        _annotation_map: Dict[type, str] = {
            str: "string",
            int: "integer",
            float: "float",
            bool: "boolean",
            list: "list",
            dict: "dict",
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            spec: Dict[str, Any] = {}
            if param.annotation != inspect.Parameter.empty:
                spec["type"] = _annotation_map.get(param.annotation, "any")

            if param.default != inspect.Parameter.empty:
                spec["required"] = False
                spec["default"] = param.default
            else:
                spec["required"] = True

            schema[param_name] = spec

        return schema

    # ------------------------------------------------------------------
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
        cls._initialized = False


# ---------------------------------------------------------------------------
# Module-level decorator
# ---------------------------------------------------------------------------
def register_tool(
    name: str,
    description: str = "",
    parameter_schema: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a function as a tool.

    Usage::

        @register_tool("calculator", "Perform arithmetic")
        def calculate(expression: str) -> float:
            return eval(expression)

    Args:
        name: Unique tool name.
        description: Human-readable description.
        parameter_schema: Optional parameter schema.  When ``None`` the
            schema is inferred from the function signature.

    Returns:
        The original function (unchanged) after registration.
    """

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        ToolRegistry().register_tool(name, func, description, parameter_schema)
        return func

    return _decorator
