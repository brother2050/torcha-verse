"""Plugin CLI commands for TorchaVerse.

This module exposes both the *core* command functions
(:func:`plugin_list`, :func:`plugin_install`, ...) -- plain Python
functions that perform the work and print ``rich``-formatted output --
and a :data:`plugin` :class:`click.Group` of thin wrappers that can be
registered on the main ``torcha`` CLI (see
:mod:`serving.cli`).

The core functions accept an optional ``manager`` argument so they can
be unit-tested without going through ``click``'s test runner, and an
optional ``console`` argument for output capture.

Usage (once registered on the main CLI)::

    torcha plugin list
    torcha plugin install ./my_plugin
    torcha plugin uninstall my_plugin
    torcha plugin enable my_plugin
    torcha plugin disable my_plugin
    torcha plugin create my_new_plugin
    torcha plugin validate ./my_plugin/plugin.toml
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import click

from .manager import (
    PluginError,
    PluginManager,
    PluginNotFoundError,
)
from .manifest import ManifestError, ManifestParser
from .sdk import create_plugin_scaffold

__all__ = [
    # Core functions
    "plugin_list",
    "plugin_install",
    "plugin_uninstall",
    "plugin_enable",
    "plugin_disable",
    "plugin_create",
    "plugin_validate",
    # Click group
    "plugin",
]


# ---------------------------------------------------------------------------
# Optional rich output (rich is a core framework dependency).
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
except ImportError:  # pragma: no cover - rich is required
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]


def _make_console():
    """Return a :class:`rich.console.Console` or ``None`` if rich is absent."""
    if Console is None:  # pragma: no cover
        return None
    return Console()


def _print(console, message: str, style: str = "") -> None:
    """Print ``message`` with optional rich ``style`` (or plain fallback)."""
    if console is None:  # pragma: no cover
        print(message)
        return
    if style:
        console.print(message, style=style)
    else:
        console.print(message)


# ---------------------------------------------------------------------------
# Manager holder
# ---------------------------------------------------------------------------
def get_plugin_manager() -> PluginManager:
    """Return a fresh :class:`PluginManager` with default plugin dirs."""
    return PluginManager()


# ---------------------------------------------------------------------------
# Core command functions
# ---------------------------------------------------------------------------
def plugin_list(
    manager: Optional[PluginManager] = None,
    console=None,
) -> None:
    """List all available plugins with their status.

    Prints a table with columns: name, version, source, status
    (loaded / disabled / available).
    """
    mgr = manager if manager is not None else get_plugin_manager()
    if console is None:
        console = _make_console()

    specs = mgr.discover()
    if not specs:
        _print(console, "No plugins available.", style="yellow")
        return

    if Table is not None and console is not None:
        table = Table(title="TorchaVerse Plugins", border_style="cyan")
        table.add_column("Name", style="white", no_wrap=True)
        table.add_column("Version", style="cyan")
        table.add_column("Source", style="dim")
        table.add_column("Status", style="green")
        for spec in specs:
            if mgr.is_loaded(spec.name):
                status = "loaded"
            elif not mgr.is_enabled(spec.name):
                status = "disabled"
            else:
                status = "available"
            table.add_row(spec.name, spec.version, spec.source or "-", status)
        console.print(table)
    else:  # pragma: no cover - plain fallback
        for spec in specs:
            status = (
                "loaded" if mgr.is_loaded(spec.name)
                else "disabled" if not mgr.is_enabled(spec.name)
                else "available"
            )
            print("{:24} {:8} {:12} {}".format(
                spec.name, spec.version, spec.source or "-", status
            ))


# ------------------------------------------------------------------
def plugin_install(
    name: str,
    manager: Optional[PluginManager] = None,
    console=None,
) -> bool:
    """Install (and load) a plugin.

    If ``name`` is a path to an existing plugin directory, the directory
    is copied into the user-level plugin directory so the plugin
    persists across sessions, then discovered and loaded.  Otherwise
    ``name`` is treated as the name of an already-available plugin and
    loaded directly.

    Returns:
        ``True`` if the plugin was loaded successfully.
    """
    mgr = manager if manager is not None else get_plugin_manager()
    if console is None:
        console = _make_console()

    plugin_name = name
    src_path = Path(name)
    if src_path.is_dir():
        # Install from a local directory into the user plugin dir.
        dest_root = mgr._plugin_dirs[0] if mgr._plugin_dirs else None
        if dest_root is not None:
            dest_root.mkdir(parents=True, exist_ok=True)
            dest = dest_root / src_path.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src_path, dest)
            _print(
                console,
                "Installed plugin files to {}".format(dest),
                style="dim",
            )
        plugin_name = src_path.name
        mgr.discover()

    try:
        plugin = mgr.load(plugin_name)
    except PluginNotFoundError:
        _print(console, "Plugin {!r} not found.".format(plugin_name), style="red")
        return False
    except PluginError as exc:
        _print(console, "Failed to load plugin {!r}: {}".format(plugin_name, exc), style="red")
        return False

    _print(
        console,
        "Plugin {!r} v{} loaded ({} node(s)).".format(
            plugin.spec.name, plugin.spec.version, len(plugin.node_classes)
        ),
        style="green",
    )
    return True


# ------------------------------------------------------------------
def plugin_uninstall(
    name: str,
    manager: Optional[PluginManager] = None,
    console=None,
) -> bool:
    """Uninstall a plugin: unload it and mark it disabled.

    Returns:
        ``True`` if the plugin was found (and unloaded / disabled).
    """
    mgr = manager if manager is not None else get_plugin_manager()
    if console is None:
        console = _make_console()

    if mgr.is_loaded(name):
        try:
            mgr.unload(name)
        except PluginError as exc:
            _print(console, "Failed to unload {!r}: {}".format(name, exc), style="red")
            return False

    try:
        mgr.disable(name)
    except PluginNotFoundError:
        _print(console, "Plugin {!r} not found.".format(name), style="red")
        return False

    _print(console, "Plugin {!r} uninstalled (unloaded + disabled).".format(name), style="green")
    return True


# ------------------------------------------------------------------
def plugin_enable(
    name: str,
    manager: Optional[PluginManager] = None,
    console=None,
) -> bool:
    """Enable a plugin so it can be loaded.

    Returns:
        ``True`` if the plugin was enabled.
    """
    mgr = manager if manager is not None else get_plugin_manager()
    if console is None:
        console = _make_console()

    try:
        mgr.enable(name)
    except PluginNotFoundError:
        _print(console, "Plugin {!r} not found.".format(name), style="red")
        return False
    _print(console, "Plugin {!r} enabled.".format(name), style="green")
    return True


# ------------------------------------------------------------------
def plugin_disable(
    name: str,
    manager: Optional[PluginManager] = None,
    console=None,
) -> bool:
    """Disable a plugin (it will be unloaded if currently loaded).

    Returns:
        ``True`` if the plugin was disabled.
    """
    mgr = manager if manager is not None else get_plugin_manager()
    if console is None:
        console = _make_console()

    try:
        mgr.disable(name)
    except PluginNotFoundError:
        _print(console, "Plugin {!r} not found.".format(name), style="red")
        return False
    _print(console, "Plugin {!r} disabled.".format(name), style="green")
    return True


# ------------------------------------------------------------------
def plugin_create(
    name: str,
    output_dir: Optional[str] = None,
    manager: Optional[PluginManager] = None,
    console=None,
) -> Optional[Path]:
    """Create a plugin scaffold directory.

    Args:
        name: Plugin name (valid Python identifier).
        output_dir: Where to create the plugin folder.  Defaults to the
            current working directory.

    Returns:
        The :class:`pathlib.Path` of the created plugin directory, or
        ``None`` on failure.
    """
    if console is None:
        console = _make_console()
    out = Path(output_dir) if output_dir else Path.cwd()

    try:
        plugin_dir = create_plugin_scaffold(name, out)
    except ValueError as exc:
        _print(console, "Cannot create plugin: {}".format(exc), style="red")
        return None

    _print(
        console,
        "Created plugin scaffold at {}".format(plugin_dir),
        style="green",
    )
    _print(
        console,
        "Edit {}/plugin.toml and nodes/example_node.py, then:\n"
        "  torcha plugin install {}".format(plugin_dir.name, plugin_dir),
        style="dim",
    )
    return plugin_dir


# ------------------------------------------------------------------
def plugin_validate(
    path: str,
    manager: Optional[PluginManager] = None,
    console=None,
) -> bool:
    """Validate a plugin manifest file (or a plugin directory).

    Args:
        path: Path to a ``plugin.toml`` / ``plugin.yaml`` file, or to a
            plugin directory containing one.

    Returns:
        ``True`` if the manifest is valid.
    """
    if console is None:
        console = _make_console()
    target = Path(path)

    if target.is_dir():
        manifest = ManifestParser.find_manifest(target)
        if manifest is None:
            _print(console, "No manifest found in {}.".format(target), style="red")
            return False
        target = manifest

    try:
        spec = ManifestParser.parse(target)
    except ManifestError as exc:
        _print(console, "Manifest error: {}".format(exc), style="red")
        return False

    errors = ManifestParser.validate(spec)
    if errors:
        _print(console, "Manifest {} is invalid:".format(target), style="red")
        for err in errors:
            _print(console, "  - {}".format(err), style="red")
        return False

    _print(
        console,
        "Manifest {} is valid: {} v{} ({}).".format(
            target, spec.name, spec.version, spec.author
        ),
        style="green",
    )
    return True


# ===========================================================================
# Click command group
# ===========================================================================
@click.group(name="plugin")
def plugin() -> None:
    """Manage TorchaVerse plugins (list / install / enable / create ...)."""


@plugin.command("list")
@click.option("--available-only", is_flag=True, help="Only show available plugins.")
def _plugin_list_cmd(available_only: bool) -> None:
    """List all available plugins and their status."""
    plugin_list()


@plugin.command("install")
@click.argument("name")
def _plugin_install_cmd(name: str) -> None:
    """Install (and load) a plugin by name or directory path."""
    plugin_install(name)


@plugin.command("uninstall")
@click.argument("name")
def _plugin_uninstall_cmd(name: str) -> None:
    """Uninstall a plugin (unload + disable)."""
    plugin_uninstall(name)


@plugin.command("enable")
@click.argument("name")
def _plugin_enable_cmd(name: str) -> None:
    """Enable a plugin so it can be loaded."""
    plugin_enable(name)


@plugin.command("disable")
@click.argument("name")
def _plugin_disable_cmd(name: str) -> None:
    """Disable a plugin (unloads it if loaded)."""
    plugin_disable(name)


@plugin.command("create")
@click.argument("name")
@click.option(
    "--output-dir",
    "-o",
    default=".",
    help="Directory in which to create the plugin folder.",
)
@click.option(
    "--manifest-format",
    type=click.Choice(["toml", "yaml"]),
    default="toml",
    help="Manifest file format.",
)
def _plugin_create_cmd(name: str, output_dir: str, manifest_format: str) -> None:
    """Create a new plugin scaffold."""
    out = Path(output_dir)
    if console_obj := _make_console():
        try:
            from .sdk import create_plugin_scaffold as _scaffold

            plugin_dir = _scaffold(name, out, manifest_format=manifest_format)
            _print(console_obj, "Created plugin scaffold at {}".format(plugin_dir), style="green")
        except ValueError as exc:
            _print(console_obj, "Cannot create plugin: {}".format(exc), style="red")
    else:  # pragma: no cover
        plugin_create(name, output_dir)


@plugin.command("validate")
@click.argument("path")
def _plugin_validate_cmd(path: str) -> None:
    """Validate a plugin manifest file or directory."""
    plugin_validate(path)
