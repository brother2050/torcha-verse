"""Plugin development SDK for TorchaVerse.

This module provides the small, friendly surface that plugin authors
interact with:

* :class:`BasePlugin` -- an optional base class that plugin developers
  may subclass to bundle lifecycle hooks (``on_load`` / ``on_unload``)
  with their plugin.  Using it is *not* required -- a plain module with
  ``@register_node``-decorated classes is a perfectly valid plugin -- but
  it gives a convenient, documented extension point.
* :func:`create_plugin_scaffold` -- generates a ready-to-edit plugin
  directory tree (manifest + sample node) so authors can start from a
  working skeleton instead of an empty folder.

The SDK is dependency-free at import time (standard library only) so it
can be imported in any environment.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from .spec import PluginSpec, SOURCE_DIRECTORY

__all__ = ["BasePlugin", "create_plugin_scaffold"]


#: Valid Python-identifier-ish plugin name pattern.
_NAME_RE: re.Pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# BasePlugin
# ---------------------------------------------------------------------------
class BasePlugin:
    """Optional base class for TorchaVerse plugins.

    Subclassing :class:`BasePlugin` is **not** required to write a
    plugin -- a module containing ``@register_node``-decorated classes
    is sufficient.  However, deriving from this class gives plugin
    authors a single, documented place to put lifecycle logic and
    metadata, and lets the :class:`plugins.manager.PluginManager`
    discover the plugin's hooks automatically.

    Subclasses typically set the class attributes ``name`` and
    ``version`` and override :meth:`on_load` / :meth:`on_unload`::

        class MyPlugin(BasePlugin):
            name = "my_plugin"
            version = "0.1.0"

            def on_load(self) -> None:
                print("My plugin loaded")

            def on_unload(self) -> None:
                print("My plugin unloaded")

    Class attributes:
        name: Plugin name.  Defaults to the empty string; subclasses
            should override it.
        version: Plugin version.  Defaults to ``"0.0.0"``.
        author: Plugin author.  Defaults to ``"unknown"``.
        description: Short description.  Defaults to ``""``.
    """

    name: str = ""
    version: str = "0.0.0"
    author: str = "unknown"
    description: str = ""

    # ------------------------------------------------------------------
    # Lifecycle hooks (overridable).
    # ------------------------------------------------------------------
    def on_load(self) -> None:
        """Called once after the plugin's node modules are imported.

        Override to perform plugin-specific initialisation (e.g. warming
        up a model cache, registering non-node modules on the
        :class:`ModuleBus`).  The default implementation is a no-op.
        """
        return None

    def on_unload(self) -> None:
        """Called once before the plugin's nodes are unregistered.

        Override to release resources acquired in :meth:`on_load`.  The
        default implementation is a no-op.
        """
        return None

    # ------------------------------------------------------------------
    def to_spec(self, source: str = SOURCE_DIRECTORY, path: Optional[Path] = None) -> PluginSpec:
        """Build a :class:`PluginSpec` from this instance's metadata.

        Args:
            source: The discovery source to record on the spec.
            path: Optional filesystem path to attach.

        Returns:
            A :class:`PluginSpec` populated from the class attributes.
        """
        return PluginSpec(
            name=self.name or self.__class__.__name__,
            version=self.version,
            author=self.author,
            description=self.description,
            source=source,
            path=path,
        )

    def __repr__(self) -> str:
        return "{}(name={!r}, version={!r})".format(
            self.__class__.__name__, self.name, self.version
        )


# ---------------------------------------------------------------------------
# Scaffold templates
# ---------------------------------------------------------------------------
_TOML_TEMPLATE = """\
# Plugin manifest for {name}
# Docs: https://torcha-verse.example/plugins/manifest

name = "{name}"
version = "0.1.0"
author = "{author}"
description = "{description}"
license = "MIT"
homepage = ""

# pip dependencies required by this plugin, e.g. ["transformers>=4.30"]
dependencies = []

# Node modules contributed by this plugin.  Paths are relative to this
# file.  Leave empty to auto-import every `nodes/*.py` module.
node_modules = ["nodes/example_node.py"]

# Optional lifecycle hooks (dotted paths, relative to this directory):
# on_load = "hooks:on_load"
# on_unload = "hooks:on_unload"
"""

_YAML_TEMPLATE = """\
# Plugin manifest for {name}
# Docs: https://torcha-verse.example/plugins/manifest

name: {name}
version: 0.1.0
author: {author}
description: {description}
license: MIT
homepage: ""

# pip dependencies required by this plugin, e.g. ["transformers>=4.30"]
dependencies: []

# Node modules contributed by this plugin.  Paths are relative to this
# file.  Leave empty to auto-import every `nodes/*.py` module.
node_modules:
  - nodes/example_node.py

# Optional lifecycle hooks (dotted paths, relative to this directory):
# on_load: "hooks:on_load"
# on_unload: "hooks:on_unload"
"""

_INIT_TEMPLATE = '''\
"""{name} plugin package for TorchaVerse."""
'''

_NODES_INIT_TEMPLATE = '''\
"""Node modules contributed by the {name} plugin."""
'''

_NODE_TEMPLATE = '''\
"""Sample node for the {name} plugin.

This is a minimal, working node you can edit.  It is registered on the
process-wide :class:`ModuleBus` (under the ``node`` kind) the moment this
module is imported, so the :class:`PluginManager` discovers it
automatically.
"""

from __future__ import annotations

from typing import Any, Dict

from nodes.base import BaseNode, NodeContext, NodeSpec, register_node


@register_node("{node_type}")
class ExampleNode(BaseNode):
    """A trivial example node that echoes its input.

    Replace the body of :meth:`execute` with your real logic.
    """

    spec = NodeSpec(
        type="{node_type}",
        name="Example Node",
        description="An example node contributed by the {name} plugin.",
        inputs={{"prompt": "PROMPT"}},
        outputs={{"text": "TEXT"}},
        tags=["example", "{name}"],
    )

    def execute(self, ctx: NodeContext, **inputs: Any) -> Dict[str, Any]:
        prompt = str(inputs.get("prompt", ""))
        return {{"text": "[{name}] " + prompt}}
'''


# ---------------------------------------------------------------------------
# create_plugin_scaffold
# ---------------------------------------------------------------------------
def create_plugin_scaffold(
    name: str,
    output_dir: Any,
    *,
    manifest_format: str = "toml",
    author: str = "Your Name",
    description: Optional[str] = None,
) -> Path:
    """Create a plugin scaffold directory tree.

    Generates a ready-to-edit plugin skeleton::

        <output_dir>/<name>/
            plugin.toml        (or plugin.yaml)
            __init__.py
            nodes/
                __init__.py
                example_node.py

    The generated node is immediately loadable by the
    :class:`plugins.manager.PluginManager`.

    Args:
        name: Plugin name.  Must be a valid Python-identifier-ish string
            (letters, digits, underscore; must not start with a digit).
        output_dir: Directory in which to create the plugin folder.  It
            is created (with parents) if it does not exist.
        manifest_format: ``"toml"`` (default) or ``"yaml"`` -- the
            manifest file format to generate.
        author: Author string written into the manifest.
        description: Description written into the manifest.  When
            ``None`` a sensible default is used.

    Returns:
        The :class:`pathlib.Path` of the created plugin directory.

    Raises:
        ValueError: If ``name`` is not a valid identifier or
            ``manifest_format`` is unsupported.
    """
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(
            "Plugin name must be a valid Python-identifier string "
            "(letters, digits, underscore; not starting with a digit), "
            "got {!r}.".format(name)
        )
    manifest_format = (manifest_format or "").lower()
    if manifest_format == "toml":
        manifest_name = "plugin.toml"
        template = _TOML_TEMPLATE
    elif manifest_format in ("yaml", "yml"):
        manifest_name = "plugin.yaml"
        template = _YAML_TEMPLATE
    else:
        raise ValueError(
            "manifest_format must be 'toml' or 'yaml', got {!r}.".format(
                manifest_format
            )
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plugin_dir = out / name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    nodes_dir = plugin_dir / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)

    desc = description or "{} plugin for TorchaVerse".format(name)
    node_type = "{}_example".format(name)

    # Manifest.
    (plugin_dir / manifest_name).write_text(
        template.format(name=name, author=author, description=desc),
        encoding="utf-8",
    )

    # Package init files.
    (plugin_dir / "__init__.py").write_text(
        _INIT_TEMPLATE.format(name=name), encoding="utf-8"
    )
    (nodes_dir / "__init__.py").write_text(
        _NODES_INIT_TEMPLATE.format(name=name), encoding="utf-8"
    )

    # Sample node.
    (nodes_dir / "example_node.py").write_text(
        _NODE_TEMPLATE.format(name=name, node_type=node_type),
        encoding="utf-8",
    )

    return plugin_dir
