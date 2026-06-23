"""Safe mathematical expression calculator tool.

This module provides :class:`CalculatorTool`, a :class:`BaseTool`
implementation that evaluates mathematical expressions without using the
unsafe :func:`eval` builtin.  Instead the expression is parsed into an
abstract syntax tree (AST) via the :mod:`ast` module and a restricted set
of AST node types is evaluated recursively.

Supported operations:

* Arithmetic: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``
* Unary: ``+x``, ``-x``
* Parentheses for grouping
* Constants: ``pi``, ``e``, ``tau``, ``inf``, ``nan``
* Functions: ``sqrt``, ``sin``, ``cos``, ``tan``, ``asin``, ``acos``,
  ``atan``, ``log`` (natural), ``log2``, ``log10``, ``exp``, ``abs``,
  ``floor``, ``ceil``, ``round``, ``min``, ``max``
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict

from core.tool_registry import BaseTool
from infrastructure.logger import get_logger

__all__ = ["CalculatorTool"]


# ---------------------------------------------------------------------------
# Allowed binary operators (AST node -> callable).
# ---------------------------------------------------------------------------
_BINARY_OPS: Dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# ---------------------------------------------------------------------------
# Allowed unary operators.
# ---------------------------------------------------------------------------
_UNARY_OPS: Dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# ---------------------------------------------------------------------------
# Allowed mathematical constants.
# ---------------------------------------------------------------------------
_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
}

# ---------------------------------------------------------------------------
# Allowed unary functions.
# ---------------------------------------------------------------------------
_UNARY_FUNCS: Dict[str, Any] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "log": math.log,  # natural logarithm
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "radians": math.radians,
    "degrees": math.degrees,
}

# ---------------------------------------------------------------------------
# Allowed variadic functions.
# ---------------------------------------------------------------------------
_VARIADIC_FUNCS: Dict[str, Any] = {
    "min": min,
    "max": max,
    "pow": pow,
}


class _SafeEvaluator(ast.NodeVisitor):
    """Evaluate a restricted subset of Python AST nodes.

    The evaluator walks the AST recursively and computes the value of
    each node.  Any node type not present in the allow-list raises a
    :class:`ValueError`, preventing arbitrary code execution.
    """

    def visit_Expression(self, node: ast.Expression) -> float:
        """Visit the top-level expression node."""
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        """Visit a numeric or string constant."""
        if isinstance(node.value, bool):
            # ``True``/``False`` are subclasses of int; reject them to
            # avoid surprising semantics in arithmetic.
            raise ValueError("Boolean constants are not allowed.")
        if not isinstance(node.value, (int, float)):
            raise ValueError(
                f"Unsupported constant type: {type(node.value).__name__}."
            )
        return node.value

    def visit_Num(self, node: ast.Num) -> float:  # pragma: no cover - legacy
        """Visit a legacy ``Num`` node (Python < 3.8)."""
        return node.n

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        """Visit a unary operation (``+x``, ``-x``)."""
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}.")
        operand = self.visit(node.operand)
        return _UNARY_OPS[op_type](operand)

    def visit_BinOp(self, node: ast.BinOp) -> float:
        """Visit a binary operation (``a + b``, ``a ** b``, ...)."""
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}.")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if op_type is ast.Div and right == 0:
            raise ZeroDivisionError("Division by zero.")
        if op_type is ast.FloorDiv and right == 0:
            raise ZeroDivisionError("Floor division by zero.")
        if op_type is ast.Mod and right == 0:
            raise ZeroDivisionError("Modulo by zero.")
        return _BINARY_OPS[op_type](left, right)

    def visit_Name(self, node: ast.Name) -> float:
        """Visit a name reference (e.g. ``pi``, ``e``)."""
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"Unknown variable: '{node.id}'.")

    def visit_Call(self, node: ast.Call) -> float:
        """Visit a function call (e.g. ``sqrt(4)``, ``max(1, 2)``)."""
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed.")
        func_name = node.func.id

        if node.keywords:
            raise ValueError("Keyword arguments are not allowed.")

        args = [self.visit(arg) for arg in node.args]

        if func_name in _UNARY_FUNCS:
            if len(args) != 1:
                raise ValueError(
                    f"Function '{func_name}' expects exactly 1 argument, "
                    f"got {len(args)}."
                )
            return _UNARY_FUNCS[func_name](args[0])

        if func_name in _VARIADIC_FUNCS:
            if not args:
                raise ValueError(
                    f"Function '{func_name}' requires at least 1 argument."
                )
            return _VARIADIC_FUNCS[func_name](*args)

        raise ValueError(f"Unknown function: '{func_name}'.")

    def generic_visit(self, node: ast.AST) -> Any:
        """Reject any AST node type not explicitly handled."""
        raise ValueError(
            f"Unsupported expression element: {type(node).__name__}."
        )


class CalculatorTool(BaseTool):
    """Evaluate mathematical expressions safely.

    The tool parses the provided ``expression`` string into an AST and
    evaluates it using a restricted visitor that only permits arithmetic
    operators, mathematical constants, and a curated set of functions.
    This avoids the security risks associated with :func:`eval`.

    Example::

        >>> tool = CalculatorTool()
        >>> tool.execute(expression="2 + 3 * 4")
        14
        >>> tool.execute(expression="sqrt(16) + log(e)")
        5.0
    """

    name: str = "calculator"
    description: str = "Evaluate mathematical expressions safely"
    parameter_schema: Dict[str, Any] = {
        "expression": {
            "type": "string",
            "description": "Mathematical expression to evaluate",
            "required": True,
        }
    }

    def __init__(self) -> None:
        self._logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calculate(self, expression: str) -> float:
        """Safely evaluate a mathematical expression.

        Args:
            expression: The mathematical expression string, e.g.
                ``"2 + 3 * sqrt(16)"``.

        Returns:
            The numeric result of the expression.

        Raises:
            ValueError: If the expression contains unsupported syntax
                or unknown names/functions.
            ZeroDivisionError: If the expression divides by zero.
        """
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("Expression must be a non-empty string.")

        # Strip whitespace to normalise the input.
        expression = expression.strip()

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression syntax: {exc.msg}") from exc

        evaluator = _SafeEvaluator()
        result = evaluator.visit(tree)
        return result

    def execute(self, **params: Any) -> float:
        """Execute the calculator tool.

        Args:
            **params: Keyword arguments matching :attr:`parameter_schema`.
                Must include ``expression``.

        Returns:
            The numeric result of the evaluated expression.
        """
        expression = params.get("expression")
        if expression is None:
            raise ValueError("Missing required parameter: 'expression'.")
        return self.calculate(expression)
