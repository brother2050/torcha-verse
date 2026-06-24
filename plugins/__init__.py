"""TorchaVerse plugin system.

This package provides a declarative plugin discovery and loading
mechanism that sits on top of the programmatic
:class:`core.module_bus.ModuleBus` registry.  Plugins can be discovered
from three independent sources:

1. **Entry points** -- ``pip``-installed packages advertising a
   ``torcha_verse.plugins`` entry point.
2. **Directories** -- plugin folders (with a ``plugin.toml`` /
   ``plugin.yaml`` manifest) dropped into a plugin directory.
3. **Code** -- plugins registered programmatically via
   :meth:`PluginManager.register`.

The package is import-safe in minimal environments: importing it does
**not** pull in ``torch``.  The :mod:`nodes` package is imported lazily
the first time a plugin that contributes nodes is actually loaded.

Public surface
--------------

* :class:`PluginManager` -- discover / load / unload / enable / disable.
* :class:`PluginSpec` -- declarative plugin description.
* :class:`Plugin` -- a loaded plugin instance.
* :class:`ManifestParser` -- parse / validate ``plugin.toml`` &
  ``plugin.yaml`` manifests.
* :class:`BasePlugin`, :func:`create_plugin_scaffold` -- the plugin
  development SDK.
* Exception types: :class:`PluginError`, :class:`PluginNotFoundError`,
  :class:`PluginAlreadyLoadedError`, :class:`ManifestError`.

Example::

    from plugins import PluginManager

    mgr = PluginManager()
    for spec in mgr.discover():
        print(spec.name, spec.version, spec.source)
    plugin = mgr.load("my_plugin")
    print(plugin.node_classes)
    mgr.unload("my_plugin")
"""

from __future__ import annotations

from .manifest import ManifestError, ManifestParser
from .manager import (
    PluginAlreadyLoadedError,
    PluginError,
    PluginManager,
    PluginNotFoundError,
)
from .sdk import BasePlugin, create_plugin_scaffold
from .spec import Plugin, PluginSpec

__all__ = [
    # Core
    "PluginManager",
    "PluginSpec",
    "Plugin",
    # Manifest
    "ManifestParser",
    "ManifestError",
    # SDK
    "BasePlugin",
    "create_plugin_scaffold",
    # Exceptions
    "PluginError",
    "PluginNotFoundError",
    "PluginAlreadyLoadedError",
]

__version__ = "0.1.0"
