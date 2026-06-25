"""Tiered configuration merge centre for TorchaVerse (v0.6.x).

This sub-package promotes configuration to a first-class citizen by
introducing :class:`ConfigCenter`, a singleton that merges
configuration from four ordered layers, each with a strictly
increasing precedence:

1. **System** -- built-in defaults shipped with the package under
   ``<package>/config/_defaults/``.  These are *immutable* from
   the user's perspective and are snapshotted in CI as golden files.
2. **Project** -- the ``./config/*.yaml`` files committed with the
   repository.
3. **User** -- per-user overrides living under
   ``~/.config/torcha-verse/`` (Linux/macOS) or
   ``%APPDATA%/torcha-verse/`` (Windows): UI preferences, API
   keys, local paths.
4. **Run** -- every run produces a ``config_snapshot.json`` so
   that the exact configuration used for a generation can be
   replayed later.

The v0.6.x refactor splits the previous single-file
``infrastructure/config_center.py`` (862 lines) into six focused
modules:

* :mod:`infrastructure.config_center._paths` -- env-var conventions
  and path resolution helpers.
* :mod:`infrastructure.config_center._io` -- YAML / JSON file
  reading + deep-merge.
* :mod:`infrastructure.config_center._budget` -- config-driven
  :class:`ResourceBudget` builder.
* :mod:`infrastructure.config_center._jsonable` -- recursive
  JSON-serialisation helper.
* :mod:`infrastructure.config_center._schema` -- ``@config_schema``
  + :class:`Field` declarative configuration.
* :mod:`infrastructure.config_center._center` -- the
  :class:`ConfigCenter` core class.

The public API is unchanged.  ``from infrastructure.config_center
import ConfigCenter, get_config`` keeps working; the
:func:`get_config` accessor now goes through the package's
``__init__`` which re-exports it as a thin wrapper around the
singleton.

The :func:`config_schema` decorator and the
:class:`Field` helper are new in v0.6.x.  They let feature flags
be declared as a typed dataclass-style class and registered with
the :class:`ConfigCenter` automatically::

    from infrastructure.config_center import config_schema, Field, get_config

    @config_schema
    class Train:
        '''Training loop feature flags.'''
        fast_mode: bool = Field(default=True, doc="fast mode")
        max_steps: int = Field(default=1000, doc="max steps", min=1)

    assert get_config("Train.fast_mode") is True
"""

from __future__ import annotations

from typing import Any

from ..resource_budget import ResourceBudget
from ._center import ConfigCenter
from ._schema import (
    ConfigSchema,
    ConfigSchemaError,
    ConfigSchemaRegistry,
    Field,
    config_schema,
    default_registry,
)

__all__ = [
    # Core
    "ConfigCenter",
    "get_config",
    "ResourceBudget",
    # Schema DSL (v0.6.x new)
    "Field",
    "config_schema",
    "ConfigSchema",
    "ConfigSchemaError",
    "ConfigSchemaRegistry",
    "default_registry",
]


def get_config(key: str, default: Any = None) -> Any:
    """Read a dotted configuration key from the singleton.

    Args:
        key: Dotted key (e.g. ``"Train.fast_mode"``).
        default: Fallback value.

    Returns:
        The active value or ``default``.
    """
    return ConfigCenter().get(key, default)
