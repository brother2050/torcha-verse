"""Plugin specification data classes for the TorchaVerse plugin system.

This module is intentionally dependency-free (standard library only) so
that ``from plugins import PluginSpec`` works in any environment,
including minimal CI sandboxes without ``torch`` installed.

Public surface
--------------

* :class:`PluginSpec` -- declarative description of a plugin (name,
  version, author, node modules, load/unload hooks, source, ...).
* :class:`Plugin` -- a loaded plugin instance bundling its
  :class:`PluginSpec` with the node classes it contributed and its
  runtime state (loaded / enabled).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Type

__all__ = ["PluginSpec", "Plugin"]


# ---------------------------------------------------------------------------
# Valid ``source`` values for a :class:`PluginSpec`.
# ---------------------------------------------------------------------------
#: Plugin discovered through a ``pip``-installed entry point.
SOURCE_ENTRY_POINT: str = "entry_point"
#: Plugin discovered by scanning a plugin directory.
SOURCE_DIRECTORY: str = "directory"
#: Plugin registered programmatically in code.
SOURCE_CODE: str = "code"

#: All recognised plugin sources.
VALID_SOURCES: tuple = (SOURCE_ENTRY_POINT, SOURCE_DIRECTORY, SOURCE_CODE)


# ---------------------------------------------------------------------------
# PluginSpec
# ---------------------------------------------------------------------------
@dataclass
class PluginSpec:
    """Declarative description of a plugin.

    A :class:`PluginSpec` is the single source of truth for a plugin's
    identity, metadata, the node modules it contributes and the optional
    load / unload hooks.  It is produced either by parsing a manifest
    file (``plugin.toml`` / ``plugin.yaml``), by loading an entry point,
    or constructed directly in code.

    Attributes:
        name: Unique plugin name (a valid Python identifier suffix,
            e.g. ``"my_image_tools"``).
        version: Semantic version string, e.g. ``"0.1.0"``.
        author: Author name or organisation.
        description: Short human-readable description.
        license: SPDX license identifier (default ``"MIT"``).
        homepage: Optional project homepage URL.
        dependencies: Optional list of pip dependency specifiers that
            must be installed for the plugin to work, e.g.
            ``["transformers>=4.30", "accelerate"]``.
        node_modules: List of node module paths contributed by the
            plugin.  For directory plugins these are paths (relative to
            the plugin directory) to ``.py`` files or packages; for
            entry-point / code plugins these are dotted module paths.
            When empty, a directory plugin auto-scans its ``nodes/``
            sub-directory.
        on_load: Optional dotted path of a callable invoked after the
            plugin's node modules are imported, e.g.
            ``"hooks:on_load"`` or ``"my_plugin.hooks.on_load"``.
        on_unload: Optional dotted path of a callable invoked before the
            plugin's nodes are unregistered.
        source: How the plugin was discovered -- one of
            :data:`SOURCE_ENTRY_POINT`, :data:`SOURCE_DIRECTORY`,
            :data:`SOURCE_CODE`.
        path: Filesystem path of the plugin (its directory for
            directory plugins, the manifest file, or ``None`` for
            entry-point / code plugins).
    """

    name: str
    version: str
    author: str
    description: str
    license: str = "MIT"
    homepage: str = ""
    dependencies: List[str] = field(default_factory=list)
    node_modules: List[str] = field(default_factory=list)
    on_load: str = ""
    on_unload: str = ""
    source: str = ""
    path: Optional[Path] = None

    def __post_init__(self) -> None:
        """Lightweight validation of the spec fields.

        Only the identity fields (``name`` / ``version`` / ``author``)
        are checked here; deeper validation lives in
        :func:`plugins.manifest.ManifestParser.validate`.  ``path`` is
        coerced to :class:`pathlib.Path` when a string is supplied.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("PluginSpec.name must be a non-empty string.")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("PluginSpec.version must be a non-empty string.")
        if not isinstance(self.author, str) or not self.author.strip():
            raise ValueError("PluginSpec.author must be a non-empty string.")
        if not isinstance(self.description, str):
            raise ValueError("PluginSpec.description must be a string.")
        if self.source and self.source not in VALID_SOURCES:
            raise ValueError(
                "PluginSpec.source must be one of {} (got {!r}).".format(
                    VALID_SOURCES, self.source
                )
            )
        if self.path is not None and not isinstance(self.path, Path):
            self.path = Path(self.path)

    # ------------------------------------------------------------------
    @property
    def is_entry_point(self) -> bool:
        """``True`` when the plugin was discovered via an entry point."""
        return self.source == SOURCE_ENTRY_POINT

    @property
    def is_directory(self) -> bool:
        """``True`` when the plugin was discovered via directory scan."""
        return self.source == SOURCE_DIRECTORY

    @property
    def is_code(self) -> bool:
        """``True`` when the plugin was registered programmatically."""
        return self.source == SOURCE_CODE

    def __repr__(self) -> str:
        return (
            "PluginSpec(name={!r}, version={!r}, source={!r}, "
            "node_modules={})".format(
                self.name, self.version, self.source, len(self.node_modules)
            )
        )


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------
@dataclass
class Plugin:
    """A loaded (or loadable) plugin instance.

    Bundles a :class:`PluginSpec` with the concrete node classes the
    plugin contributed and its runtime state.  Instances are created and
    owned by :class:`plugins.manager.PluginManager`.

    Attributes:
        spec: The :class:`PluginSpec` describing the plugin.
        node_classes: Node classes contributed by the plugin (populated
            at load time).
        instance: Optional :class:`plugins.sdk.BasePlugin` instance when
            the plugin was built on the SDK base class.
    """

    spec: PluginSpec
    node_classes: List[Type] = field(default_factory=list)
    instance: Optional[object] = None

    #: Runtime flag -- ``True`` once :meth:`PluginManager.load` succeeds.
    _loaded: bool = False
    #: Runtime flag -- ``False`` when the plugin has been disabled.
    _enabled: bool = True
    #: Node types registered by this plugin (for clean unregistration).
    _registered_node_types: List[str] = field(default_factory=list)

    @property
    def loaded(self) -> bool:
        """``True`` when the plugin is currently loaded."""
        return self._loaded

    @property
    def enabled(self) -> bool:
        """``True`` when the plugin is enabled."""
        return self._enabled

    def __repr__(self) -> str:
        state = "loaded" if self._loaded else "unloaded"
        if not self._enabled:
            state = "disabled"
        return "Plugin({!r} v{}, {})".format(
            self.spec.name, self.spec.version, state
        )
