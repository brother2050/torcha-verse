"""Plugin manager with three-layer discovery and loading.

The :class:`PluginManager` is the central orchestrator of the TorchaVerse
plugin system.  It discovers plugins from three independent sources
("layers") and unifies them behind a single API:

* **Layer 1 -- entry points.**  Plugins installed via ``pip`` advertise
  themselves through the ``torcha_verse.plugins`` entry-point group.
  Discovery uses :func:`importlib.metadata.entry_points`.

* **Layer 2 -- directory scan.**  Plugins dropped into a plugin
  directory (``~/.local/share/torcha-verse/plugins/`` by default, or a
  caller-supplied directory) are discovered by scanning for
  sub-directories that contain a ``plugin.toml`` / ``plugin.yaml``
  manifest.

* **Layer 3 -- programmatic registration.**  Plugins constructed in
  code are registered with :meth:`PluginManager.register`.  This is the
  declarative counterpart to the low-level
  :meth:`core.module_bus.ModuleBus.register` API.

Loading a plugin imports its node modules (which register their
:class:`nodes.base.BaseNode` subclasses on the :class:`ModuleBus` via
``@register_node``), records the newly-registered node types so they can
be cleanly unregistered on unload, and invokes the optional ``on_load``
hook.  Unloading reverses the process and calls ``on_unload``.

Thread safety
-------------
All mutable manager state (the available / loaded / enabled maps) is
guarded by a single :class:`threading.RLock`.  Re-entrant locking is used
so that an ``on_load`` hook may safely call back into the manager.

The manager is dependency-free at import time (it only touches
:mod:`core.module_bus`, :mod:`security.sandbox` (pure-Python AST analyser),
:mod:`plugins.spec`, :mod:`plugins.manifest` and the standard library).
:mod:`nodes` is imported lazily inside :meth:`load` / :meth:`unload`
because importing it pulls in ``torch``.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from core.module_bus import ModuleBus
from security.sandbox import ASTAnalyzer

from .manifest import ManifestError, ManifestParser
from .spec import (
    Plugin,
    PluginSpec,
    SOURCE_CODE,
    SOURCE_ENTRY_POINT,
)

__all__ = [
    "PluginManager",
    "PluginError",
    "PluginNotFoundError",
    "PluginAlreadyLoadedError",
]


# ---------------------------------------------------------------------------
# Module-level logger (stdlib only -- no torch).
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger("PluginManager")

# ---------------------------------------------------------------------------
# R0-6: 插件源码静态安全分析器(纯 Python，无 torch 依赖)。
# 在 importlib 执行插件源码前，对源码进行 AST 静态分析，拒绝包含危险调用
# (os.system / subprocess / eval / exec ...)、危险导入 (os / socket / ctypes
# ...) 或敏感文件访问的插件。分析器无状态且线程安全。
# ---------------------------------------------------------------------------
_ast_analyzer: ASTAnalyzer = ASTAnalyzer()

# ---------------------------------------------------------------------------
# 专用导入锁：保护 ``sys.path`` 的临时修改，避免并发插件加载时
# ``sys.path.insert`` / ``sys.path.remove`` 产生竞争。
# ---------------------------------------------------------------------------
_import_lock: threading.Lock = threading.Lock()

#: 插件源码文件大小上限（10 MiB），超过则拒绝加载。
MAX_PLUGIN_SIZE: int = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class PluginError(RuntimeError):
    """Base class for plugin manager errors."""


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin cannot be found."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        super().__init__("No plugin found for name {!r}.".format(name))


class PluginAlreadyLoadedError(PluginError):
    """Raised when attempting to load a plugin that is already loaded."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        super().__init__("Plugin {!r} is already loaded.".format(name))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Entry-point group advertised by ``pip``-installed plugins.
ENTRY_POINT_GROUP: str = "torcha_verse.plugins"

#: Default user-level plugin directory.
DEFAULT_USER_PLUGIN_DIR: Path = (
    Path.home() / ".local" / "share" / "torcha-verse" / "plugins"
)

#: Name of the persisted enable/disable state file.
_STATE_FILENAME: str = ".plugin_state.json"


# ---------------------------------------------------------------------------
# PluginManager
# ---------------------------------------------------------------------------
class PluginManager:
    """Discover, load, unload and enable/disable TorchaVerse plugins.

    Args:
        plugin_dirs: Directories to scan for Layer-2 plugins.  When
            ``None`` (the default) the manager scans the user-level
            directory (``~/.local/share/torcha-verse/plugins/``) and the
            project-level ``plugins/`` directory.  When a list is given
            it *replaces* the defaults, which is useful for tests and
            isolated environments.

    Example::

        from plugins import PluginManager
        mgr = PluginManager()
        for spec in mgr.discover():
            print(spec.name, spec.version)
        plugin = mgr.load("my_plugin")
        mgr.unload("my_plugin")
    """

    #: Entry-point group name (also exposed as a class attribute).
    ENTRY_POINT_GROUP: str = ENTRY_POINT_GROUP

    # ------------------------------------------------------------------
    def __init__(self, plugin_dirs: Optional[List[Path]] = None) -> None:
        if plugin_dirs is None:
            self._plugin_dirs: List[Path] = self._default_plugin_dirs()
        else:
            self._plugin_dirs = [Path(p) for p in plugin_dirs]

        # State.
        self._available: Dict[str, PluginSpec] = {}
        self._loaded: Dict[str, Plugin] = {}
        # S1-2: 标记正在加载中的插件，防止并发重复导入同一插件。
        self._loading: set = set()
        self._programmatic: Dict[str, PluginSpec] = {}
        self._enabled_state: Dict[str, bool] = {}
        self._discovered: bool = False

        self._bus: ModuleBus = ModuleBus()
        self._lock: threading.RLock = threading.RLock()
        self._logger: logging.Logger = _logger

        # Persisted enable/disable state lives next to the first plugin
        # directory so that tests passing a temp dir get full isolation.
        if self._plugin_dirs:
            self._state_file: Optional[Path] = (
                self._plugin_dirs[0] / _STATE_FILENAME
            )
        else:
            self._state_file = DEFAULT_USER_PLUGIN_DIR / _STATE_FILENAME

        self._load_state()

    # ==================================================================
    # Defaults
    # ==================================================================
    @staticmethod
    def _default_plugin_dirs() -> List[Path]:
        """Return the default Layer-2 plugin directories.

        Includes the user-level directory and the project-level
        ``plugins/`` directory (the parent of this package's parent).
        """
        dirs: List[Path] = [DEFAULT_USER_PLUGIN_DIR]
        # Project-level: <repo_root>/plugins (this package's own dir).
        project_plugins = Path(__file__).resolve().parent
        dirs.append(project_plugins)
        return dirs

    # ==================================================================
    # Discovery
    # ==================================================================
    def discover(self) -> List[PluginSpec]:
        """Discover all available plugins across the three layers.

        This refreshes the cached available-plugin map.  It is safe to
        call repeatedly; each call re-scans entry points and directories.

        Returns:
            A list of :class:`PluginSpec` (one per discovered plugin),
            sorted by name.
        """
        with self._lock:
            available: Dict[str, PluginSpec] = {}

            # Layer 1: entry points.
            for spec in self._discover_entry_points():
                available.setdefault(spec.name, spec)

            # Layer 2: directory scan.
            for spec in self._discover_directories():
                available.setdefault(spec.name, spec)

            # Layer 3: programmatic registrations.
            for name, spec in self._programmatic.items():
                available.setdefault(name, spec)

            self._available = available
            self._discovered = True
            result = sorted(available.values(), key=lambda s: s.name)
        return result

    # ------------------------------------------------------------------
    def list_available(self) -> List[PluginSpec]:
        """Return the list of available (discovered) plugin specs.

        Triggers :meth:`discover` on first use.  Does not reflect the
        loaded / enabled state -- use :meth:`list_loaded` for loaded
        plugins.

        Returns:
            A list of :class:`PluginSpec` sorted by name.
        """
        with self._lock:
            discovered = self._discovered
        if not discovered:
            self.discover()
        with self._lock:
            return sorted(self._available.values(), key=lambda s: s.name)

    # ------------------------------------------------------------------
    def _discover_entry_points(self) -> List[PluginSpec]:
        """Layer 1: discover plugins advertised via entry points."""
        specs: List[PluginSpec] = []
        try:
            eps = importlib.metadata.entry_points()
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.debug("entry_points() failed: %s", exc)
            return specs

        if hasattr(eps, "select"):
            group_eps = list(eps.select(group=ENTRY_POINT_GROUP))
        else:  # Python < 3.10 dict-style API
            group_eps = list(eps.get(ENTRY_POINT_GROUP, []))  # type: ignore[union-attr]

        for ep in group_eps:
            try:
                obj = ep.load()
                spec = self._coerce_to_spec(obj, source=SOURCE_ENTRY_POINT)
                if spec is not None:
                    if not spec.name:
                        spec.name = ep.name
                    specs.append(spec)
            except Exception as exc:
                self._logger.warning(
                    "Failed to load entry point %s: %s", ep, exc
                )
        return specs

    # ------------------------------------------------------------------
    def _discover_directories(self) -> List[PluginSpec]:
        """Layer 2: discover plugins by scanning plugin directories."""
        specs: List[PluginSpec] = []
        seen_dirs: set = set()
        for plugin_dir in self._plugin_dirs:
            if not plugin_dir or not plugin_dir.is_dir():
                continue
            real = plugin_dir.resolve()
            if real in seen_dirs:
                continue
            seen_dirs.add(real)
            try:
                children = sorted(plugin_dir.iterdir())
            except OSError as exc:
                self._logger.debug("Cannot list %s: %s", plugin_dir, exc)
                continue
            for child in children:
                if not child.is_dir():
                    continue
                manifest = ManifestParser.find_manifest(child)
                if manifest is None:
                    continue
                try:
                    spec = ManifestParser.parse(manifest)
                except ManifestError as exc:
                    self._logger.warning(
                        "Skipping plugin at %s: %s", child, exc
                    )
                    continue
                specs.append(spec)
        return specs

    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_to_spec(
        obj: Any, source: str, path: Optional[Path] = None
    ) -> Optional[PluginSpec]:
        """Coerce an entry-point object into a :class:`PluginSpec`.

        Accepts a :class:`PluginSpec`, a :class:`plugins.sdk.BasePlugin`
        instance, a :class:`BasePlugin` subclass, or a zero-argument
        callable returning any of the above.
        """
        # Lazy import to avoid a hard torch dependency at import time.
        from .sdk import BasePlugin

        if isinstance(obj, PluginSpec):
            obj.source = obj.source or source
            if path is not None and obj.path is None:
                obj.path = path
            return obj
        if isinstance(obj, BasePlugin):
            return obj.to_spec(source=source, path=path)
        if isinstance(obj, type) and issubclass(obj, BasePlugin):
            return obj().to_spec(source=source, path=path)
        if callable(obj) and not isinstance(obj, type):
            return PluginManager._coerce_to_spec(obj(), source, path)
        return None

    # ==================================================================
    # Programmatic registration (Layer 3)
    # ==================================================================
    def register(self, spec: PluginSpec) -> None:
        """Programmatically register a plugin (Layer 3).

        This is the declarative counterpart to the low-level
        :meth:`core.module_bus.ModuleBus.register` API.  The spec is
        recorded with ``source="code"`` (unless already set) and becomes
        immediately available for :meth:`load`.

        Args:
            spec: The :class:`PluginSpec` to register.
        """
        if not isinstance(spec, PluginSpec):
            raise TypeError("spec must be a PluginSpec instance.")
        if not spec.source:
            spec.source = SOURCE_CODE
        with self._lock:
            self._programmatic[spec.name] = spec
            self._available[spec.name] = spec
            self._discovered = True

    # ==================================================================
    # Load / unload
    # ==================================================================
    def load(self, name: str) -> Plugin:
        """Load (and activate) the plugin named ``name``.

        Imports the plugin's node modules -- which registers their
        :class:`nodes.base.BaseNode` subclasses on the
        :class:`ModuleBus` -- records the newly-registered node types and
        invokes the optional ``on_load`` hook.

        Args:
            name: Name of an available (discovered) plugin.

        Returns:
            The loaded :class:`Plugin`.

        Raises:
            PluginNotFoundError: If no plugin is available for ``name``.
            PluginError: If the plugin is disabled or loading fails.
            PluginAlreadyLoadedError: If the plugin is already loaded.
        """
        # ==================================================================
        # Phase 1 (锁内): 检查状态、标记加载中。
        # 仅持有锁完成快速的状态校验与 spec 解析，不执行任何 I/O。
        # ==================================================================
        with self._lock:
            if name in self._loaded:
                raise PluginAlreadyLoadedError(name)
            if name in self._loading:
                raise PluginAlreadyLoadedError(name)
            if not self._enabled_state.get(name, True):
                raise PluginError(
                    "Plugin {!r} is disabled; enable it before loading.".format(
                        name
                    )
                )
            spec = self._available.get(name)

        if spec is None:
            self.discover()
            with self._lock:
                spec = self._available.get(name)
            if spec is None:
                raise PluginNotFoundError(name)

        with self._lock:
            # 发现后再次检查(另一线程可能已加载或正在加载)。
            if name in self._loaded:
                raise PluginAlreadyLoadedError(name)
            if name in self._loading:
                raise PluginAlreadyLoadedError(name)
            # 标记为加载中，防止并发重复导入。
            self._loading.add(name)

        # ==================================================================
        # Phase 2 (锁外): 执行 import_module 与 _call_hook。
        # 慢速 I/O(模块导入、钩子调用、sys.path 修改)在锁外完成，
        # 不阻塞其他管理器操作。sys.path 的修改在 _import_node_modules
        # 内部完成并在 finally 中回滚。
        # ==================================================================
        try:
            plugin = self._load_spec(spec)
        except Exception:
            # 加载失败时清理 loading 标记，允许后续重试。
            with self._lock:
                self._loading.discard(name)
            raise

        # ==================================================================
        # Phase 3 (锁内): 更新元数据。
        # 处理并发竞争:若另一线程在此期间已加载同一插件，则回滚本次
        # 重复注册并抛出异常。
        # ==================================================================
        with self._lock:
            self._loading.discard(name)
            if name in self._loaded:
                self._unload_plugin(plugin)
                raise PluginAlreadyLoadedError(name)
            self._loaded[name] = plugin
            self._logger.info("Loaded plugin %s v%s.", spec.name, spec.version)
            return plugin

    # ------------------------------------------------------------------
    def unload(self, name: str) -> None:
        """Unload the plugin named ``name``.

        Invokes the optional ``on_unload`` hook and unregisters every
        node type the plugin contributed from the :class:`ModuleBus` and
        the node-class index.

        Args:
            name: Name of a currently loaded plugin.

        Raises:
            PluginNotFoundError: If the plugin is not loaded.
        """
        with self._lock:
            plugin = self._loaded.get(name)
            if plugin is None:
                raise PluginNotFoundError(name)
            self._unload_plugin(plugin)
            del self._loaded[name]
            self._logger.info("Unloaded plugin %s.", name)

    # ------------------------------------------------------------------
    def _load_spec(self, spec: PluginSpec) -> Plugin:
        """Import a plugin's node modules and build the :class:`Plugin`."""
        # Ensure the built-in node catalogue is registered *before* we
        # snapshot, so its node types are not attributed to this plugin.
        self._ensure_nodes_imported()

        before = {s.name for s in self._bus.list("node")}

        self._import_node_modules(spec)

        after = {s.name for s in self._bus.list("node")}
        new_types = sorted(after - before)

        node_classes: List[type] = []
        for node_type in new_types:
            try:
                factory = self._bus.get_spec("node", node_type).factory
                if isinstance(factory, type):
                    node_classes.append(factory)
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.debug(
                    "Could not retrieve node class for %s: %s", node_type, exc
                )

        plugin = Plugin(spec=spec, node_classes=node_classes)
        plugin._registered_node_types = new_types

        # Optional lifecycle hook.
        if not self._call_hook(spec, "on_load"):
            # on_load hook 失败：回滚已注册的节点类型，不标记为 loaded。
            self._unregister_nodes(plugin._registered_node_types)
            plugin._registered_node_types = []
            plugin.node_classes = []
            plugin._loaded = False
            raise PluginError(
                "on_load hook failed for plugin {!r}".format(spec.name)
            )

        plugin._loaded = True
        return plugin

    # ------------------------------------------------------------------
    def _unload_plugin(self, plugin: Plugin) -> None:
        """Reverse :meth:`_load_spec` for a loaded plugin."""
        # Optional lifecycle hook (called before unregistering nodes).
        self._call_hook(plugin.spec, "on_unload")

        unregistered = self._unregister_nodes(plugin._registered_node_types)
        if unregistered:
            self._logger.debug(
                "Unregistered %d node(s) from plugin %s.",
                unregistered, plugin.spec.name,
            )

        plugin._registered_node_types = []
        plugin.node_classes = []
        plugin._loaded = False

    # ------------------------------------------------------------------
    def _unregister_nodes(self, node_types: List[str]) -> int:
        """Unregister node types from the bus and node-class index.

        Uses :class:`nodes.NodeRegistry` when available so the
        module-level ``_NODE_CLASSES`` index is cleaned too; falls back
        to :meth:`ModuleBus.unregister` otherwise.
        """
        if not node_types:
            return 0
        count = 0
        try:
            from nodes import NodeRegistry

            registry = NodeRegistry(bus=self._bus)
            for node_type in node_types:
                if registry.unregister(node_type):
                    count += 1
            return count
        except Exception as exc:
            self._logger.debug(
                "NodeRegistry unavailable, falling back to bus.unregister: %s",
                exc,
            )
        for node_type in node_types:
            if self._bus.unregister("node", node_type):
                count += 1
        return count

    # ------------------------------------------------------------------
    def _import_node_modules(self, spec: PluginSpec) -> None:
        """Import every node module declared by ``spec``."""
        modules = list(spec.node_modules)

        # Directory plugins auto-scan their nodes/ dir when unspecified.
        if spec.is_directory and not modules and spec.path is not None:
            nodes_dir = spec.path / "nodes"
            if nodes_dir.is_dir():
                for py in sorted(nodes_dir.glob("*.py")):
                    if py.stem == "__init__":
                        continue
                    modules.append(str(py.relative_to(spec.path)))

        if not modules:
            self._logger.debug(
                "Plugin %s declares no node modules.", spec.name
            )
            return

        # 使用专用导入锁保护 sys.path 的临时修改，避免并发插件加载
        # 时 sys.path.insert / sys.path.remove 产生竞争。
        with _import_lock:
            # Make the plugin directory importable for relative hooks/imports.
            added_path: Optional[str] = None
            if spec.is_directory and spec.path is not None:
                dir_str = str(spec.path)
                if dir_str not in sys.path:
                    sys.path.insert(0, dir_str)
                    added_path = dir_str

            try:
                for module_path in modules:
                    self._import_single_module(spec, module_path)
            finally:
                if added_path is not None and added_path in sys.path:
                    sys.path.remove(added_path)

    # ------------------------------------------------------------------
    def _import_single_module(self, spec: PluginSpec, module_path: str) -> None:
        """Import a single node module for ``spec``.

        ``module_path`` is interpreted as:

        * a relative ``.py`` file path (for directory plugins), or
        * a dotted module path (for entry-point / code plugins).
        """
        module_path = module_path.strip()
        if not module_path:
            return

        # Relative .py file (directory plugins).
        if spec.is_directory and spec.path is not None:
            candidate = spec.path / module_path
            if not candidate.suffix:
                candidate = candidate.with_suffix(".py")
            if candidate.is_file():
                self._import_file(candidate, spec.name)
                return
            # ``nodes/foo.py`` style already has suffix.
            if module_path.endswith(".py"):
                candidate2 = spec.path / module_path
                if candidate2.is_file():
                    self._import_file(candidate2, spec.name)
                    return

        # Dotted module path.
        try:
            importlib.import_module(module_path)
        except ImportError as exc:
            raise PluginError(
                "Failed to import node module {!r} for plugin {!r}: {}".format(
                    module_path, spec.name, exc
                )
            ) from exc

    # ------------------------------------------------------------------
    @staticmethod
    def _import_file(path: Path, plugin_name: str) -> Any:
        """Import a ``.py`` file as an anonymous module."""
        safe_stem = re.sub(r"\W", "_", plugin_name)
        mod_name = "_torcha_plugin_{}_{}".format(safe_stem, path.stem)
        # Avoid collisions when the same file is imported twice.
        if mod_name in sys.modules:
            return sys.modules[mod_name]
        # --- 文件大小限制 -------------------------------------------------
        # 在读取源码前检查文件大小，防止恶意超大文件耗尽内存。
        try:
            file_size = path.stat().st_size
        except OSError as exc:
            raise PluginError(
                "Cannot stat plugin file {}: {}".format(path, exc)
            ) from exc
        if file_size > MAX_PLUGIN_SIZE:
            raise PluginError(
                "Plugin file too large: {} bytes (max {}): {}".format(
                    file_size, MAX_PLUGIN_SIZE, path
                )
            )
        # --- R0-6: 静态安全分析 --------------------------------------------
        # 在 importlib 执行插件源码前，读取源码并使用 ASTAnalyzer 进行
        # 静态分析。命中危险调用 / 危险导入 / 敏感文件访问时拒绝加载，
        # 防止恶意插件在 exec_module 阶段造成破坏。
        source = path.read_text(encoding="utf-8")
        result = _ast_analyzer.analyze(source)
        if not result.is_safe:
            raise PluginError(
                "Plugin {!r} source {!r} rejected by sandbox AST analysis: "
                "{}".format(
                    plugin_name, path, "; ".join(result.violations)
                )
            )
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise PluginError(
                "Cannot create import spec for {}".format(path)
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise
        return module

    # ------------------------------------------------------------------
    def _call_hook(self, spec: PluginSpec, hook_field: str) -> bool:
        """Resolve and invoke an optional ``on_load`` / ``on_unload`` hook.

        Returns:
            ``True`` if the hook was not defined, could not be resolved, or
            ran successfully.  ``False`` if the hook was found and callable
            but raised an exception during execution.
        """
        hook_path = getattr(spec, hook_field, "")
        if not hook_path:
            return True
        func = self._resolve_hook(spec, hook_path)
        if func is None:
            self._logger.warning(
                "Could not resolve %s hook %r for plugin %r.",
                hook_field, hook_path, spec.name,
            )
            return True
        if not callable(func):
            self._logger.warning(
                "%s hook %r for plugin %r is not callable.",
                hook_field, hook_path, spec.name,
            )
            return True
        try:
            func()
            return True
        except Exception as exc:
            self._logger.warning(
                "%s hook for plugin %r raised: %s", hook_field, spec.name, exc
            )
            return False

    # ------------------------------------------------------------------
    def _resolve_hook(
        self, spec: PluginSpec, hook_path: str
    ) -> Optional[Callable[..., Any]]:
        """Resolve a dotted / ``module:func`` hook path to a callable."""
        hook_path = hook_path.strip()
        if not hook_path:
            return None

        if ":" in hook_path:
            module_part, _, func_name = hook_path.partition(":")
        else:
            parts = hook_path.rsplit(".", 1)
            if len(parts) != 2:
                return None
            module_part, func_name = parts
        module_part = module_part.strip()
        func_name = func_name.strip()
        if not module_part or not func_name:
            return None

        # Directory plugins: try a relative .py file first.
        if spec.is_directory and spec.path is not None:
            rel = module_part.replace(".", os.sep)
            candidate = spec.path / (rel + ".py")
            if candidate.is_file():
                module = self._import_file(candidate, spec.name + "_hook")
                return getattr(module, func_name, None)

        # Dotted import.
        try:
            module = importlib.import_module(module_part)
        except ImportError:
            return None
        return getattr(module, func_name, None)

    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_nodes_imported() -> None:
        """Import the ``nodes`` package so built-in nodes are registered.

        Done lazily (and only when a plugin is actually loaded) because
        importing ``nodes`` pulls in ``torch``.
        """
        try:
            import nodes  # noqa: F401
        except Exception as exc:  # pragma: no cover - defensive
            _logger.debug("Could not import nodes package: %s", exc)

    # ==================================================================
    # Listing helpers
    # ==================================================================
    def list_loaded(self) -> List[str]:
        """Return the names of currently loaded plugins (sorted)."""
        with self._lock:
            return sorted(self._loaded.keys())

    # ==================================================================
    # Enable / disable
    # ==================================================================
    def enable(self, name: str) -> None:
        """Enable a plugin (allow it to be loaded).

        Args:
            name: Plugin name.

        Raises:
            PluginNotFoundError: If the plugin is not available.
        """
        self._ensure_known(name)
        with self._lock:
            self._enabled_state[name] = True
            self._save_state()
            self._logger.info("Enabled plugin %s.", name)

    # ------------------------------------------------------------------
    def disable(self, name: str) -> None:
        """Disable a plugin (prevent it from being loaded).

        If the plugin is currently loaded it is unloaded first.

        Args:
            name: Plugin name.

        Raises:
            PluginNotFoundError: If the plugin is not available.
        """
        self._ensure_known(name)
        with self._lock:
            self._enabled_state[name] = False
            plugin = self._loaded.get(name)
            if plugin is not None:
                self._unload_plugin(plugin)
                del self._loaded[name]
            self._save_state()
            self._logger.info("Disabled plugin %s.", name)

    # ==================================================================
    # Introspection helpers
    # ==================================================================
    def is_loaded(self, name: str) -> bool:
        """Return ``True`` if plugin ``name`` is currently loaded."""
        with self._lock:
            return name in self._loaded

    # ------------------------------------------------------------------
    def is_enabled(self, name: str) -> bool:
        """Return ``True`` if plugin ``name`` is enabled."""
        with self._lock:
            return self._enabled_state.get(name, True)

    # ------------------------------------------------------------------
    def get(self, name: str) -> Optional[Plugin]:
        """Return the loaded :class:`Plugin` for ``name`` or ``None``."""
        with self._lock:
            return self._loaded.get(name)

    # ------------------------------------------------------------------
    def get_spec(self, name: str) -> Optional[PluginSpec]:
        """Return the available :class:`PluginSpec` for ``name`` or ``None``."""
        with self._lock:
            if name in self._available:
                return self._available[name]
        if not self._discovered:
            self.discover()
        with self._lock:
            return self._available.get(name)

    # ==================================================================
    # Internal helpers
    # ==================================================================
    def _is_known(self, name: str) -> bool:
        """Return ``True`` if ``name`` is a known (available) plugin."""
        if name in self._available:
            return True
        if name in self._programmatic:
            return True
        return False

    # ------------------------------------------------------------------
    def _ensure_known(self, name: str) -> None:
        """Ensure ``name`` is a known plugin, discovering if needed.

        Raises:
            PluginNotFoundError: If the plugin cannot be found after
                discovery.
        """
        with self._lock:
            known = self._is_known(name)
            discovered = self._discovered
        if not known and not discovered:
            self.discover()
            known = self._is_known(name)
        if not known:
            raise PluginNotFoundError(name)

    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        """Load persisted enable/disable state from disk (best-effort)."""
        if self._state_file is None:
            return
        try:
            with open(self._state_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._enabled_state = {
                    str(k): str(v).lower() in ("true", "1", "yes")
                    for k, v in data.items()
                }
        except (OSError, ValueError, json.JSONDecodeError):
            # Missing / corrupt state file -> start fresh.
            self._enabled_state = {}

    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        """Persist enable/disable state to disk atomically (best-effort)."""
        if self._state_file is None:
            return
        tmp: Optional[Path] = None
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: write to a unique temp file, then os.replace
            # (which is atomic on POSIX and Windows).  This avoids leaving a
            # partially-written / corrupt state file if the process crashes
            # mid-write.
            tmp = self._state_file.with_name(
                "." + self._state_file.name + "." + uuid4().hex + ".tmp"
            )
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._enabled_state, fh, indent=2)
            os.replace(str(tmp), str(self._state_file))
        except OSError as exc:
            self._logger.debug("Could not persist plugin state: %s", exc)
            try:
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
            except Exception as exc:
                self._logger.debug("Failed to remove temp plugin state file: %s", exc)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        with self._lock:
            return (
                "PluginManager(available={}, loaded={}, dirs={})".format(
                    len(self._available),
                    len(self._loaded),
                    [str(d) for d in self._plugin_dirs],
                )
            )
