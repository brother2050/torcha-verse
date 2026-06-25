"""Type-checked configuration schemas (``@config_schema`` + ``Field``).

The v0.6.x refactor introduces two new public helpers that make
configuration declarative and self-validating:

* :class:`Field` -- a small descriptor-style wrapper that records
  the default, type, docstring, and (optional) ``min`` / ``max`` /
  ``choices`` for a single configuration key.
* :func:`config_schema` -- a class decorator that walks the class
  body, harvests every :class:`Field` annotation, registers a
  schema with the global :class:`ConfigSchemaRegistry`, and (at
  class-creation time) seeds the singleton
  :class:`ConfigCenter` instance with the defaults so the values
  are immediately queryable via :func:`get_config`.

Example:

    >>> from infrastructure.config_center import config_schema, Field
    >>> @config_schema
    ... class Train:
    ...     '''Training loop feature flags.'''
    ...     fast_mode: bool = Field(default=True, doc="fast mode")
    ...     max_steps: int = Field(default=1000, doc="max steps", min=1)
    ...
    >>> from infrastructure.config_center import get_config
    >>> get_config("Train.fast_mode")
    True
    >>> get_config("Train.max_steps")
    1000

Validation semantics
--------------------

The :func:`set_default` helper that
:func:`config_schema` registers with the
:class:`ConfigCenter` enforces the type and the optional
``min`` / ``max`` / ``choices`` constraints.  A
:class:`ConfigSchemaError` is raised when an override violates
the schema.

The decorator is intentionally *not* a heavy metaclass: it is a
plain class decorator that scans the class's ``__annotations__``
and ``__dict__`` at decoration time.  This keeps the framework's
startup cost flat even when many schemas are registered.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Type

__all__ = [
    "Field",
    "ConfigSchema",
    "ConfigSchemaError",
    "config_schema",
    "ConfigSchemaRegistry",
    "default_registry",
]


class ConfigSchemaError(ValueError):
    """Raised when a configuration value violates its schema."""


@dataclass(frozen=True)
class Field:
    """A single field declaration inside a ``@config_schema`` class.

    Attributes:
        default: The default value; must be compatible with ``type_``.
        type_:   The Python type of the value (``bool`` / ``int`` /
            ``float`` / ``str`` / ``List[str]`` / ...).  When
            ``None`` the type is inferred from ``default``.
        doc:     Human-readable description; surfaced through
            :func:`ConfigSchemaRegistry.describe`.
        min:     Optional inclusive lower bound (numeric fields).
        max:     Optional inclusive upper bound (numeric fields).
        choices: Optional whitelist of allowed values (string /
            enum-like fields).
    """

    default: Any
    type_: Optional[Type[Any]] = None
    doc: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[Sequence[Any]] = None

    def coerce(self, value: Any) -> Any:
        """Coerce ``value`` to ``type_`` and validate constraints.

        Returns the coerced value; raises :class:`ConfigSchemaError`
        if the value is of the wrong type or violates the
        ``min`` / ``max`` / ``choices`` constraints.
        """
        target = self.type_ or type(self.default)
        if target is bool and isinstance(value, int):
            value = bool(value)
        elif target in (int, float) and isinstance(value, str):
            try:
                value = target(value)
            except ValueError as exc:
                raise ConfigSchemaError(
                    f"Cannot coerce string {value!r} to {target.__name__}"
                ) from exc
        elif target in (list, tuple, set) and isinstance(value, str):
            value = [p.strip() for p in value.split(",") if p.strip()]
            if target is tuple:
                value = tuple(value)
            elif target is set:
                value = set(value)
        if not isinstance(value, target):
            raise ConfigSchemaError(
                f"Expected {target.__name__}, got {type(value).__name__}: "
                f"{value!r}"
            )
        if self.min is not None and isinstance(value, (int, float)) and value < self.min:
            raise ConfigSchemaError(
                f"Value {value!r} is below min={self.min}"
            )
        if self.max is not None and isinstance(value, (int, float)) and value > self.max:
            raise ConfigSchemaError(
                f"Value {value!r} is above max={self.max}"
            )
        if self.choices is not None and value not in self.choices:
            raise ConfigSchemaError(
                f"Value {value!r} not in choices={list(self.choices)}"
            )
        return value


@dataclass(frozen=True)
class ConfigSchema:
    """The harvested schema for a single :func:`config_schema` class.

    Attributes:
        name:    The class name, used as the dotted prefix in the
            :class:`ConfigCenter` (e.g. ``"Train"``).
        doc:     The class docstring, surfaced through ``describe()``.
        fields:  An ordered list of :class:`Field` entries, in the
            order they were declared on the class body.
    """

    name: str
    doc: str
    fields: Tuple[Tuple[str, Field], ...]


class ConfigSchemaRegistry:
    """Process-wide registry of :class:`ConfigSchema` declarations."""

    def __init__(self) -> None:
        self._schemas: List[ConfigSchema] = []
        self._by_name: dict[str, ConfigSchema] = {}

    def register(self, schema: ConfigSchema) -> None:
        """Register ``schema`` (idempotent on ``schema.name``)."""
        if schema.name in self._by_name:
            return
        self._by_name[schema.name] = schema
        self._schemas.append(schema)

    def all(self) -> List[ConfigSchema]:
        return list(self._schemas)

    def get(self, name: str) -> Optional[ConfigSchema]:
        return self._by_name.get(name)

    def describe(self) -> List[Mapping[str, Any]]:
        """Return a serialisable list of schema metadata for tooling."""
        out: List[Mapping[str, Any]] = []
        for schema in self._schemas:
            fields_meta: List[Mapping[str, Any]] = []
            for fname, f in schema.fields:
                fields_meta.append({
                    "name": fname,
                    "default": f.default,
                    "type": (f.type_ or type(f.default)).__name__,
                    "doc": f.doc,
                    "min": f.min,
                    "max": f.max,
                    "choices": f.choices,
                })
            out.append({
                "name": schema.name,
                "doc": schema.doc,
                "fields": fields_meta,
            })
        return out


#: Module-level singleton; importable as
#: :data:`infrastructure.config_center.default_registry`.
default_registry: ConfigSchemaRegistry = ConfigSchemaRegistry()


def config_schema(cls: Type[Any]) -> Type[Any]:
    """Class decorator that registers a :class:`ConfigSchema`.

    The decorator:

    1. Walks the class body and harvests every :class:`Field` (or
       :func:`Field` call returning a :class:`Field` instance) into
       a :class:`ConfigSchema`.
    2. Registers the schema with :data:`default_registry`.
    3. Seeds the :class:`ConfigCenter` singleton with the field
       defaults via :func:`get_config` and the
       :class:`ConfigCenter.set` method (lazy import to avoid a
       circular dependency between this module and the
       :class:`ConfigCenter` core).
    4. Returns the original class unchanged (so ``@config_schema``
       is composable with any other class decorator).
    """
    schema_name = cls.__name__
    doc = (cls.__doc__ or "").strip()
    fields: List[Tuple[str, Field]] = []
    for attr_name, annotation in getattr(cls, "__annotations__", {}).items():
        raw = cls.__dict__.get(attr_name, None)
        if isinstance(raw, Field):
            fields.append((attr_name, raw))
        elif raw is not None:
            # Bare value: wrap it in a Field with no constraints.
            fields.append((attr_name, Field(default=raw, type_=annotation or type(raw), doc="")))
    schema = ConfigSchema(
        name=schema_name, doc=doc,
        fields=tuple(fields),
    )
    default_registry.register(schema)

    # Seed the ConfigCenter with the defaults.  Lazy import to break
    # the cycle (this module is loaded by ConfigCenter during boot).
    try:
        from ._center import ConfigCenter  # type: ignore[import-not-found]
        cc = ConfigCenter()
        for fname, f in fields:
            key = f"{schema_name}.{fname}"
            if not cc.has(key):
                cc.set(key, f.default)
    except Exception:
        # First-import path: ConfigCenter itself imports this module
        # during class creation.  Seeding is best-effort; the
        # ConfigCenter constructor will re-seed when it finishes
        # loading.
        pass

    cls.__torcha_config_schema__ = schema  # type: ignore[attr-defined]
    return cls
