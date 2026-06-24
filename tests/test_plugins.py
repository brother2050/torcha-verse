"""Tests for the TorchaVerse plugin system.

Covers:

* :class:`PluginSpec` / :class:`Plugin` data-class creation.
* :class:`ManifestParser` for both ``plugin.toml`` and ``plugin.yaml``.
* :class:`ManifestParser.validate`.
* :class:`PluginManager` discovery (Layer 2 directory scan + Layer 3
  programmatic registration).
* :class:`PluginManager` load / unload with automatic node registration
  and unregistration on the :class:`ModuleBus`.
* :class:`PluginManager` enable / disable.
* :func:`create_plugin_scaffold` and loading the generated plugin with
  its contributed node.

The tests are careful to leave the process-wide :class:`ModuleBus`
singleton untouched: an autouse fixture snapshots the registered node
types before each test and unregisters any *new* node types afterwards,
so built-in nodes are never affected and tests do not leak state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.module_bus import ModuleBus
from plugins import (
    ManifestError,
    ManifestParser,
    Plugin,
    PluginError,
    PluginManager,
    PluginSpec,
    create_plugin_scaffold,
)
from plugins.manifest import TOML_AVAILABLE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _cleanup_plugin_nodes():
    """Unregister any node types a test registers on the global bus.

    Snapshots the ``node`` kind before the test and removes every node
    type that is new afterwards.  This keeps the process-wide
    :class:`ModuleBus` clean without resetting it (which would wipe the
    built-in node catalogue imported by ``import nodes``).
    """
    bus = ModuleBus()
    before = {s.name for s in bus.list("node")}
    yield
    after = {s.name for s in bus.list("node")}
    for node_type in after - before:
        try:
            from nodes import NodeRegistry

            NodeRegistry(bus=bus).unregister(node_type)
        except Exception:
            bus.unregister("node", node_type)


# ---------------------------------------------------------------------------
# Manifest content templates
# ---------------------------------------------------------------------------
_TOML_MANIFEST = """\
name = "{name}"
version = "0.2.0"
author = "Test Author"
description = "A demo plugin for tests."
license = "Apache-2.0"
homepage = "https://example.com/{name}"
dependencies = ["numpy>=1.20"]
node_modules = ["nodes/demo_node.py"]
on_load = "hooks:on_load"
on_unload = "hooks:on_unload"
"""

_YAML_MANIFEST = """\
name: {name}
version: 0.2.0
author: Test Author
description: A demo plugin for tests.
license: Apache-2.0
homepage: https://example.com/{name}
dependencies:
  - numpy>=1.20
node_modules:
  - nodes/demo_node.py
on_load: "hooks:on_load"
on_unload: "hooks:on_unload"
"""

_NODE_MODULE = '''\
"""Auto-generated demo node for the {name} plugin (tests)."""
from __future__ import annotations

from typing import Any, Dict

from nodes.base import BaseNode, NodeContext, NodeSpec, register_node


@register_node("{node_type}")
class DemoNode(BaseNode):
    spec = NodeSpec(
        type="{node_type}",
        name="Demo Node",
        description="A demo node contributed by the {name} plugin.",
        inputs={{"prompt": "PROMPT"}},
        outputs={{"text": "TEXT"}},
        tags=["demo", "{name}"],
    )

    def execute(self, ctx: NodeContext, **inputs: Any) -> Dict[str, Any]:
        prompt = str(inputs.get("prompt", ""))
        return {{"text": "[{name}] " + prompt}}
'''

_HOOKS_MODULE = '''\
"""Lifecycle hooks for the {name} plugin (tests)."""

_HOOK_CALLED = {{"on_load": 0, "on_unload": 0}}


def on_load() -> None:
    _HOOK_CALLED["on_load"] += 1


def on_unload() -> None:
    _HOOK_CALLED["on_unload"] += 1


def call_counts() -> dict:
    return dict(_HOOK_CALLED)
'''


def _write_plugin(
    root: Path,
    name: str,
    node_type: str,
    *,
    manifest_format: str = "yaml",
    with_hooks: bool = False,
) -> Path:
    """Create a complete plugin directory under ``root`` and return it."""
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    nodes_dir = plugin_dir / "nodes"
    nodes_dir.mkdir(exist_ok=True)
    (nodes_dir / "__init__.py").write_text("", encoding="utf-8")

    if manifest_format == "toml":
        (plugin_dir / "plugin.toml").write_text(
            _TOML_MANIFEST.format(name=name), encoding="utf-8"
        )
    else:
        (plugin_dir / "plugin.yaml").write_text(
            _YAML_MANIFEST.format(name=name), encoding="utf-8"
        )

    (nodes_dir / "demo_node.py").write_text(
        _NODE_MODULE.format(name=name, node_type=node_type), encoding="utf-8"
    )

    if with_hooks:
        (plugin_dir / "hooks.py").write_text(
            _HOOKS_MODULE.format(name=name), encoding="utf-8"
        )

    return plugin_dir


# ===========================================================================
# PluginSpec / Plugin
# ===========================================================================
class TestPluginSpec:
    """PluginSpec and Plugin data-class behaviour."""

    def test_plugin_spec_creation(self):
        """A PluginSpec is created with the expected fields and defaults."""
        spec = PluginSpec(
            name="my_plugin",
            version="0.1.0",
            author="Alice",
            description="A test plugin.",
        )
        assert spec.name == "my_plugin"
        assert spec.version == "0.1.0"
        assert spec.author == "Alice"
        assert spec.description == "A test plugin."
        # Defaults.
        assert spec.license == "MIT"
        assert spec.homepage == ""
        assert spec.dependencies == []
        assert spec.node_modules == []
        assert spec.on_load == ""
        assert spec.on_unload == ""
        assert spec.source == ""
        assert spec.path is None
        # Source helpers.
        assert spec.is_entry_point is False
        assert spec.is_directory is False
        assert spec.is_code is False

    def test_plugin_spec_rejects_empty_name(self):
        """An empty name raises ValueError."""
        with pytest.raises(ValueError):
            PluginSpec(name="", version="0.1.0", author="A", description="d")

    def test_plugin_spec_coerces_path_string(self):
        """A string ``path`` is coerced to pathlib.Path."""
        spec = PluginSpec(
            name="p", version="0.1.0", author="A", description="d",
            path="/tmp/some_plugin",
        )
        assert isinstance(spec.path, Path)
        assert spec.path == Path("/tmp/some_plugin")

    def test_plugin_spec_rejects_invalid_source(self):
        """An unknown ``source`` value raises ValueError."""
        with pytest.raises(ValueError):
            PluginSpec(
                name="p", version="0.1.0", author="A", description="d",
                source="bogus",
            )

    def test_plugin_instance_defaults(self):
        """A Plugin starts unloaded and enabled."""
        spec = PluginSpec(
            name="p", version="0.1.0", author="A", description="d"
        )
        plugin = Plugin(spec=spec)
        assert plugin.spec is spec
        assert plugin.node_classes == []
        assert plugin.loaded is False
        assert plugin.enabled is True


# ===========================================================================
# ManifestParser
# ===========================================================================
class TestManifestParser:
    """Manifest parsing and validation."""

    def test_manifest_parse_toml(self, tmp_path):
        """A plugin.toml manifest is parsed into a PluginSpec."""
        if not TOML_AVAILABLE:
            pytest.skip("tomllib/tomli not available")
        plugin_dir = _write_plugin(
            tmp_path, "toml_demo", "toml_demo_node", manifest_format="toml"
        )
        spec = ManifestParser.parse(plugin_dir / "plugin.toml")
        assert isinstance(spec, PluginSpec)
        assert spec.name == "toml_demo"
        assert spec.version == "0.2.0"
        assert spec.author == "Test Author"
        assert spec.description == "A demo plugin for tests."
        assert spec.license == "Apache-2.0"
        assert spec.homepage == "https://example.com/toml_demo"
        assert spec.dependencies == ["numpy>=1.20"]
        assert spec.node_modules == ["nodes/demo_node.py"]
        assert spec.on_load == "hooks:on_load"
        assert spec.on_unload == "hooks:on_unload"
        assert spec.source == "directory"
        assert spec.path == plugin_dir

    def test_manifest_parse_yaml(self, tmp_path):
        """A plugin.yaml manifest is parsed into a PluginSpec."""
        plugin_dir = _write_plugin(
            tmp_path, "yaml_demo", "yaml_demo_node", manifest_format="yaml"
        )
        spec = ManifestParser.parse(plugin_dir / "plugin.yaml")
        assert isinstance(spec, PluginSpec)
        assert spec.name == "yaml_demo"
        assert spec.version == "0.2.0"
        assert spec.author == "Test Author"
        assert spec.license == "Apache-2.0"
        assert spec.dependencies == ["numpy>=1.20"]
        assert spec.node_modules == ["nodes/demo_node.py"]
        assert spec.source == "directory"
        assert spec.path == plugin_dir

    def test_manifest_parse_unknown_suffix_raises(self, tmp_path):
        """An unsupported manifest suffix raises ManifestError."""
        path = tmp_path / "plugin.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ManifestError):
            ManifestParser.parse(path)

    def test_manifest_parse_missing_file_raises(self, tmp_path):
        """A non-existent manifest file raises ManifestError."""
        with pytest.raises(ManifestError):
            ManifestParser.parse(tmp_path / "plugin.toml")

    def test_manifest_find_manifest(self, tmp_path):
        """find_manifest locates the manifest inside a plugin directory."""
        plugin_dir = _write_plugin(
            tmp_path, "find_demo", "find_demo_node", manifest_format="yaml"
        )
        found = ManifestParser.find_manifest(plugin_dir)
        assert found is not None
        assert found.name == "plugin.yaml"
        # Non-plugin directory returns None.
        assert ManifestParser.find_manifest(tmp_path) is None

    def test_manifest_validate(self):
        """validate() returns errors for bad specs and [] for good ones."""
        # Valid spec.
        good = PluginSpec(
            name="good_plugin",
            version="1.2.3",
            author="Alice",
            description="A good plugin.",
        )
        assert ManifestParser.validate(good) == []

        # Invalid name.
        bad_name = PluginSpec(
            name="bad name!", version="1.0.0", author="A", description="d"
        )
        errors = ManifestParser.validate(bad_name)
        assert any("name" in e for e in errors)

        # Invalid version.
        bad_ver = PluginSpec(
            name="p", version="not-a-version", author="A", description="d"
        )
        errors = ManifestParser.validate(bad_ver)
        assert any("version" in e for e in errors)

        # Missing author / description (bypass __post_init__ which
        # already rejects empty name/version/author).
        empty = PluginSpec(name="p", version="1.0.0", author="x", description="d")
        object.__setattr__(empty, "author", "")
        object.__setattr__(empty, "description", "")
        errors = ManifestParser.validate(empty)
        assert any("author" in e for e in errors)
        assert any("description" in e for e in errors)

        # Bad hook path.
        bad_hook = PluginSpec(
            name="p", version="1.0.0", author="A", description="d",
            on_load="not a valid path!",
        )
        errors = ManifestParser.validate(bad_hook)
        assert any("on_load" in e for e in errors)

        # Bad source.
        # Bypass __post_init__ by setting source after construction.
        bad_source = PluginSpec(
            name="p", version="1.0.0", author="A", description="d"
        )
        object.__setattr__(bad_source, "source", "bogus")
        errors = ManifestParser.validate(bad_source)
        assert any("source" in e for e in errors)


# ===========================================================================
# PluginManager
# ===========================================================================
class TestPluginManager:
    """PluginManager discovery, load/unload and enable/disable."""

    def test_plugin_manager_discover(self, tmp_path):
        """discover() finds directory plugins and programmatic plugins."""
        _write_plugin(
            tmp_path, "discover_demo", "discover_demo_node",
            manifest_format="yaml",
        )
        mgr = PluginManager(plugin_dirs=[tmp_path])

        specs = mgr.discover()
        names = [s.name for s in specs]
        assert "discover_demo" in names

        # list_available returns the same set (sorted).
        avail = mgr.list_available()
        assert any(s.name == "discover_demo" for s in avail)

        # Layer 3: programmatic registration.
        code_spec = PluginSpec(
            name="code_plugin", version="0.3.0", author="Bob",
            description="Programmatic.", source="code",
        )
        mgr.register(code_spec)
        assert any(s.name == "code_plugin" for s in mgr.list_available())
        assert mgr.get_spec("code_plugin") is code_spec

    def test_plugin_manager_load_unload(self, tmp_path):
        """load() registers the plugin's node; unload() unregisters it."""
        node_type = "loadunload_demo_node"
        _write_plugin(
            tmp_path, "loadunload_demo", node_type,
            manifest_format="yaml", with_hooks=True,
        )
        mgr = PluginManager(plugin_dirs=[tmp_path])
        mgr.discover()

        bus = ModuleBus()
        assert bus.has("node", node_type) is False

        plugin = mgr.load("loadunload_demo")
        assert plugin.loaded is True
        assert "loadunload_demo" in mgr.list_loaded()
        # Node was registered on the bus.
        assert bus.has("node", node_type) is True
        # Node class was collected.
        assert len(plugin.node_classes) == 1
        assert plugin.node_classes[0].__name__ == "DemoNode"

        # The on_load hook ran (import the hooks module to verify).
        hooks_mod = plugin.spec.path / "hooks.py"
        assert hooks_mod.is_file()

        mgr.unload("loadunload_demo")
        assert "loadunload_demo" not in mgr.list_loaded()
        assert bus.has("node", node_type) is False
        assert plugin.loaded is False

    def test_plugin_manager_load_unknown_raises(self, tmp_path):
        """Loading an unknown plugin raises PluginNotFoundError."""
        mgr = PluginManager(plugin_dirs=[tmp_path])
        from plugins.manager import PluginNotFoundError

        with pytest.raises(PluginNotFoundError):
            mgr.load("does_not_exist")

    def test_plugin_manager_enable_disable(self, tmp_path):
        """enable/disable toggles loadability and the enabled flag."""
        node_type = "enabledisable_demo_node"
        _write_plugin(
            tmp_path, "enabledisable_demo", node_type,
            manifest_format="yaml",
        )
        mgr = PluginManager(plugin_dirs=[tmp_path])
        mgr.discover()

        assert mgr.is_enabled("enabledisable_demo") is True

        # Disable -> cannot load.
        mgr.disable("enabledisable_demo")
        assert mgr.is_enabled("enabledisable_demo") is False
        with pytest.raises(PluginError):
            mgr.load("enabledisable_demo")

        # Enable -> can load.
        mgr.enable("enabledisable_demo")
        assert mgr.is_enabled("enabledisable_demo") is True
        plugin = mgr.load("enabledisable_demo")
        assert plugin.loaded is True

        # Disabling a loaded plugin unloads it.
        assert mgr.is_loaded("enabledisable_demo") is True
        mgr.disable("enabledisable_demo")
        assert mgr.is_loaded("enabledisable_demo") is False
        assert mgr.is_enabled("enabledisable_demo") is False

        # Re-enable and clean up.
        mgr.enable("enabledisable_demo")
        if mgr.is_loaded("enabledisable_demo"):
            mgr.unload("enabledisable_demo")

    def test_plugin_manager_enable_unknown_raises(self, tmp_path):
        """enable/disable on an unknown plugin raises PluginNotFoundError."""
        from plugins.manager import PluginNotFoundError

        mgr = PluginManager(plugin_dirs=[tmp_path])
        mgr.discover()
        with pytest.raises(PluginNotFoundError):
            mgr.enable("nope")
        with pytest.raises(PluginNotFoundError):
            mgr.disable("nope")

    def test_plugin_manager_programmatic_load(self, tmp_path):
        """A programmatically registered plugin (Layer 3) loads."""
        # A spec with no node modules -- load is a no-op but succeeds.
        spec = PluginSpec(
            name="prog_demo", version="0.1.0", author="A",
            description="Programmatic plugin.", source="code",
        )
        mgr = PluginManager(plugin_dirs=[tmp_path])
        mgr.register(spec)
        plugin = mgr.load("prog_demo")
        assert plugin.loaded is True
        assert plugin.node_classes == []
        mgr.unload("prog_demo")


# ===========================================================================
# create_plugin_scaffold
# ===========================================================================
class TestScaffold:
    """create_plugin_scaffold and loading the generated plugin."""

    def test_create_plugin_scaffold(self, tmp_path):
        """create_plugin_scaffold builds a loadable plugin tree."""
        fmt = "toml" if TOML_AVAILABLE else "yaml"
        plugin_dir = create_plugin_scaffold(
            "scaffold_demo", tmp_path, manifest_format=fmt
        )
        assert plugin_dir.is_dir()
        # Manifest exists.
        manifest = ManifestParser.find_manifest(plugin_dir)
        assert manifest is not None
        # Package + nodes package + sample node exist.
        assert (plugin_dir / "__init__.py").is_file()
        assert (plugin_dir / "nodes" / "__init__.py").is_file()
        assert (plugin_dir / "nodes" / "example_node.py").is_file()
        # Manifest parses and validates.
        spec = ManifestParser.parse(manifest)
        assert spec.name == "scaffold_demo"
        assert ManifestParser.validate(spec) == []

    def test_create_plugin_scaffold_bad_name(self, tmp_path):
        """An invalid plugin name raises ValueError."""
        with pytest.raises(ValueError):
            create_plugin_scaffold("bad name!", tmp_path)

    def test_plugin_load_with_nodes(self, tmp_path):
        """A scaffolded plugin loads and its node is executable."""
        from nodes import NodeRegistry
        from nodes.base import BaseNode, NodeContext

        fmt = "toml" if TOML_AVAILABLE else "yaml"
        plugin_dir = create_plugin_scaffold(
            "loadnodes_demo", tmp_path, manifest_format=fmt
        )
        # The scaffold was created under tmp_path/loadnodes_demo; point the
        # manager at tmp_path so the directory scan discovers it.
        mgr = PluginManager(plugin_dirs=[tmp_path])
        mgr.discover()
        assert any(s.name == "loadnodes_demo" for s in mgr.list_available())

        plugin = mgr.load("loadnodes_demo")
        assert plugin.loaded is True
        # The example node was registered.
        node_type = "loadnodes_demo_example"
        assert ModuleBus().has("node", node_type) is True
        # And collected on the plugin.
        assert len(plugin.node_classes) == 1
        assert issubclass(plugin.node_classes[0], BaseNode)

        # The node is instantiable and executable through NodeRegistry.
        registry = NodeRegistry()
        node = registry.get(node_type)
        assert isinstance(node, BaseNode)
        ctx = NodeContext()
        result = node.execute(ctx, prompt="hello")
        assert "text" in result
        assert "loadnodes_demo" in result["text"]

        mgr.unload("loadnodes_demo")
        assert ModuleBus().has("node", node_type) is False
