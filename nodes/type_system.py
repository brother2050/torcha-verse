"""Port type system for canvas connection validation.

This module provides a pure-Python (torch-free) type system that defines
the set of port type identifiers used across the node catalogue and a
compatibility matrix governing which output types may be wired into which
input types.

The :class:`TypeSystem` is intentionally dependency-free so that it can be
imported in any environment -- including minimal CI sandboxes -- and used
by both the L4 node layer (for spec declarations) and the L5 canvas layer
(for connection validation).

Type identifiers
---------------
Every port type is a short uppercase string (e.g. ``"IMAGE"``,
``"TEXT"``).  Two convenience wrappers are supported:

* ``"Optional[T]"`` -- an optional port of type ``T``.  Used in
  :class:`~nodes.base.NodeSpec` input declarations to mark inputs that may
  be omitted.  The :func:`unwrap_optional` helper strips the wrapper.
* ``"LIST[T]"`` -- a homogeneous list of elements of type ``T``.  A
  ``LIST[T]`` output is compatible with a ``LIST[T]`` input (element-wise)
  and, for convenience, a ``LIST[T]`` output may feed a single ``T``
  input (the inner element type is checked).
"""
from __future__ import annotations

__all__ = ["TypeSystem", "is_optional", "unwrap_optional"]


# Prefix/suffix tokens used to encode optionality and lists in type strings.
_OPT_PREFIX: str = "Optional["
_LIST_PREFIX: str = "LIST["


def is_optional(type_str: str) -> bool:
    """Return ``True`` when ``type_str`` is an ``Optional[T]`` wrapper."""
    return (
        isinstance(type_str, str)
        and type_str.startswith(_OPT_PREFIX)
        and type_str.endswith("]")
    )


def unwrap_optional(type_str: str) -> str:
    """Strip a single ``Optional[...]`` wrapper, returning the inner type.

    If ``type_str`` is not optional it is returned unchanged.
    """
    if is_optional(type_str):
        return type_str[len(_OPT_PREFIX):-1]
    return type_str


class TypeSystem:
    """Port type compatibility system.

    The class exposes:

    * A set of type-identifier constants (``IMAGE``, ``VIDEO``, ...).
    * A ``COMPATIBILITY`` mapping from an output type to the list of input
      types it may connect to (in addition to itself -- a type is always
      compatible with itself).
    * The :meth:`is_compatible` predicate used by the canvas layer to
      validate connections.
    * The :meth:`compatible_inputs` helper returning every input type
      compatible with a given output type.
    * The :meth:`all_types` helper returning every registered type.
    """

    # ------------------------------------------------------------------
    # Type identifier constants
    # ------------------------------------------------------------------
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"
    TEXT = "TEXT"
    INT = "INT"
    FLOAT = "FLOAT"
    BOOL = "BOOL"
    SEED = "SEED"
    PROMPT = "PROMPT"
    ASSET_REF = "ASSET_REF"
    CHARACTER = "CHARACTER"
    OUTFIT = "OUTFIT"
    SCENE = "SCENE"
    DEPTH = "DEPTH"
    SUBTITLE = "SUBTITLE"
    LATENT = "LATENT"
    MODEL = "MODEL"
    LORA = "LORA"
    CONTROLNET = "CONTROLNET"

    # ------------------------------------------------------------------
    # Compatibility matrix
    # ------------------------------------------------------------------
    #: Mapping of output type -> list of additional input types it may
    #: connect to.  A type is always compatible with itself (handled in
    #: :meth:`is_compatible`), so it does not need to be listed here.
    COMPATIBILITY: dict[str, list[str]] = {
        IMAGE: [LATENT],
        VIDEO: [IMAGE],
        AUDIO: [],
        TEXT: [PROMPT],
        INT: [FLOAT, SEED],
        FLOAT: [],
        BOOL: [],
        SEED: [INT],
        PROMPT: [TEXT],
        CHARACTER: [ASSET_REF],
        OUTFIT: [ASSET_REF],
        SCENE: [ASSET_REF],
        DEPTH: [ASSET_REF],
        LORA: [ASSET_REF],
        CONTROLNET: [ASSET_REF],
        # Types below are leaf types with no extra compatibility beyond
        # themselves; they are listed so that :meth:`all_types` reports
        # them.
        ASSET_REF: [],
        SUBTITLE: [],
        LATENT: [],
        MODEL: [],
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    def is_compatible(cls, from_type: str, to_type: str) -> bool:
        """Check if an output of type ``from_type`` can connect to an input of type ``to_type``.

        The check is reflexive for identical types and understands the
        ``Optional[T]`` and ``LIST[T]`` wrappers:

        * ``Optional[T]`` is unwrapped before comparison (an optional
          input accepts the same types as its inner type).
        * ``LIST[T]`` is compatible with ``LIST[U]`` when ``T`` is
          compatible with ``U``; a ``LIST[T]`` output may also feed a
          single ``T`` input (the inner element type is checked).

        Args:
            from_type: The output port's type string.
            to_type: The input port's type string.

        Returns:
            ``True`` when the connection is type-safe.
        """
        if from_type == to_type:
            return True

        # Unwrap Optional[...] wrappers on both sides.
        from_type = unwrap_optional(from_type)
        to_type = unwrap_optional(to_type)

        if from_type == to_type:
            return True

        # Handle LIST[X] types.
        if from_type.startswith(_LIST_PREFIX) and to_type.startswith(_LIST_PREFIX):
            inner_from = from_type[len(_LIST_PREFIX):-1]
            inner_to = to_type[len(_LIST_PREFIX):-1]
            return cls.is_compatible(inner_from, inner_to)
        if from_type.startswith(_LIST_PREFIX) and not to_type.startswith(_LIST_PREFIX):
            inner = from_type[len(_LIST_PREFIX):-1]
            return cls.is_compatible(inner, to_type)

        compatible = cls.COMPATIBILITY.get(from_type, [])
        return to_type in compatible

    @classmethod
    def compatible_inputs(cls, output_type: str) -> list[str]:
        """Return all input types compatible with the given output type.

        The result always includes the output type itself followed by the
        types listed in :attr:`COMPATIBILITY`.

        Args:
            output_type: The output port's type string.

        Returns:
            A list of compatible input type strings.
        """
        output_type = unwrap_optional(output_type)
        result = [output_type]
        result.extend(cls.COMPATIBILITY.get(output_type, []))
        return result

    @classmethod
    def all_types(cls) -> list[str]:
        """Return all registered type identifiers."""
        return list(cls.COMPATIBILITY.keys())
