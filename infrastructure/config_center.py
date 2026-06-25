"""Tiered configuration merge centre for TorchaVerse (v0.3.0).

This module promotes configuration to a first-class citizen by introducing
:class:`ConfigCenter`, a singleton that merges configuration from four
ordered layers, each with a strictly increasing precedence:

1. **System** -- built-in defaults shipped with the package under
   ``<package>/config/_defaults/``.  These are *immutable* from the user's
   perspective and are snapshotted in CI as golden files.
2. **Project** -- the ``./config/*.yaml`` files committed with the
   repository.
3. **User** -- per-user overrides living under
   ``~/.config/torcha-verse/`` (Linux/macOS) or
   ``%APPDATA%/torcha-verse/`` (Windows): UI preferences, API keys, local
   paths.
4. **Run** -- every run produces a ``config_snapshot.json`` so that the
   exact configuration used for a generation can be replayed later.

:class:`ConfigCenter` 提供完整的配置访问接口（``get`` / ``set`` /
``has`` / ``merge`` / ``to_dict``，支持点号分隔的键访问），并在此基础上
添加分层加载、快照序列化以及 :class:`ResourceBudget` 访问器。
``ConfigManager`` 现已合并为本类的薄别名。

Example:
    >>> cc = ConfigCenter()
    >>> cc.get("default.dtype")
    'bf16'
    >>> snap = cc.snapshot()              # 深拷贝当前配置的 JSON
    >>> path = cc.save_run_snapshot()     # 写入 config_snapshot.json
    >>> cc.load_run_snapshot(path)        # 重放之前的运行
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

from .logger import get_logger
from .resource_budget import ResourceBudget

__all__ = ["ConfigCenter", "ResourceBudget", "get_config"]


# ---------------------------------------------------------------------------
# Helpers (合并自 config_manager，ConfigCenter 不再继承 ConfigManager)
# ---------------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归地将 ``override`` 合并到 ``base`` 中。

    嵌套字典按键逐个合并，而 ``override`` 中的非字典值替换 ``base`` 中
    对应的值。返回一个新字典；输入永远不会被修改。

    Args:
        base: 基础配置字典。
        override: 取得优先级的字典。

    Returns:
        包含合并结果的新字典。
    """
    result: Dict[str, Any] = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _resolve_config_dir(config_dir: Optional[Union[str, Path]]) -> Path:
    """解析配置目录。

    当 ``config_dir`` 为 ``None`` 时，目录相对于本文件定位
    （``<project_root>/config``）。也可通过 ``TORCHAVERSE_CONFIG_DIR``
    环境变量覆盖默认位置。

    Args:
        config_dir: 显式路径或 ``None`` 以自动检测。

    Returns:
        指向配置目录的已解析 :class:`~pathlib.Path`。
    """
    env_dir = os.environ.get("TORCHAVERSE_CONFIG_DIR")
    if config_dir is not None:
        return Path(config_dir).expanduser().resolve()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    # 本文件位于 <project_root>/infrastructure/config_center.py
    return Path(__file__).resolve().parent.parent / "config"

#: Environment variable used to override the System defaults directory.
_ENV_SYSTEM_DIR: str = "TORCHAVERSE_SYSTEM_CONFIG_DIR"

#: Environment variable used to override the User config directory.
_ENV_USER_DIR: str = "TORCHAVERSE_USER_CONFIG_DIR"

#: Environment variable used to override the Run snapshot directory.
_ENV_RUN_DIR: str = "TORCHAVERSE_RUN_DIR"

#: File name written for every run to guarantee reproducibility.
_RUN_SNAPSHOT_FILENAME: str = "config_snapshot.json"

#: Sub-directory (relative to the package config dir) holding System defaults.
_SYSTEM_DEFAULTS_SUBDIR: str = "_defaults"


# ---------------------------------------------------------------------------
# Platform path helpers
# ---------------------------------------------------------------------------
def _system_defaults_dir() -> Path:
    """Resolve the System-layer defaults directory.

    Defaults to ``<package_root>/config/_defaults/`` but can be overridden
    via the ``TORCHAVERSE_SYSTEM_CONFIG_DIR`` environment variable.
    """
    env = os.environ.get(_ENV_SYSTEM_DIR)
    if env:
        return Path(env).expanduser().resolve()
    # This file lives at <package_root>/infrastructure/config_center.py
    package_root = Path(__file__).resolve().parent.parent
    return package_root / "config" / _SYSTEM_DEFAULTS_SUBDIR


def _user_config_dir() -> Path:
    """Resolve the User-layer config directory in a platform-aware way.

    Linux/macOS: ``~/.config/torcha-verse/``
    Windows: ``%APPDATA%/torcha-verse/``
    """
    env = os.environ.get(_ENV_USER_DIR)
    if env:
        return Path(env).expanduser().resolve()
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "torcha-verse"
    return Path.home() / ".config" / "torcha-verse"


def _user_data_dir() -> Path:
    """Resolve the per-user data directory (run snapshots, audit logs, ...).

    Linux/macOS: ``~/.local/share/torcha-verse/``
    Windows: ``%LOCALAPPDATA%/torcha-verse/``
    """
    env = os.environ.get(_ENV_RUN_DIR)
    if env:
        return Path(env).expanduser().resolve()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(
            Path.home() / "AppData" / "Local"
        )
        return Path(base) / "torcha-verse"
    return Path.home() / ".local" / "share" / "torcha-verse"


def _run_snapshot_dir() -> Path:
    """Resolve the directory where run snapshots are stored."""
    return _user_data_dir() / "runs"


# ---------------------------------------------------------------------------
# ConfigCenter
# ---------------------------------------------------------------------------
class ConfigCenter:
    """单例四级配置合并中心。

    :class:`ConfigCenter` 提供分层感知的配置加载。四个层级
    （System < Project < User < Run）按顺序深度合并，使更高优先级的
    层级覆盖更低优先级的层级。

    该单例保留点号分隔的访问器（``get`` / ``set`` / ``has`` /
    ``merge`` / ``to_dict``）并添加：

    * :meth:`load_user_config` -- （重新）加载 User 层。
    * :meth:`load_run_snapshot` -- 重放之前运行的快照。
    * :meth:`snapshot` -- 深拷贝当前配置为 JSON。
    * :meth:`save_run_snapshot` -- 持久化当前配置以供重放。
    * :meth:`resource_budget` -- 从配置构建 :class:`ResourceBudget`。

    Args:
        config_dir: Project 层配置目录的覆盖。
        environment: 活动环境（``dev`` 或 ``prod``）。
        auto_load: 为 ``True``（默认）时立即加载所有层级。
        include_user: 为 ``True``（默认）时加载 User 层。
        include_run: 为 ``True``（默认）时在加载时写入运行快照。

    Example:
        >>> cc = ConfigCenter()
        >>> cc.get("default.dtype")
        'bf16'
        >>> with cc.reset_context():
        ...     cc.set("default.dtype", "fp16")
        ...     cc.get("default.dtype")
        'fp16'
    """

    _instance: Optional["ConfigCenter"] = None
    _initialized: bool = False
    # RLock（非 Lock）：__new__ 可能被并发调用，RLock 允许可重入获取。
    _singleton_lock: threading.RLock = threading.RLock()

    #: 启动时加载的默认配置文件。
    DEFAULT_CONFIG_FILES: Sequence[str] = (
        "model_config.yaml",
        "inference_config.yaml",
        "training_config.yaml",
        "prompt_templates.yaml",
    )

    #: 支持的环境标识符。
    SUPPORTED_ENVIRONMENTS: Sequence[str] = ("dev", "prod")

    def __new__(cls, *args: Any, **kwargs: Any) -> "ConfigCenter":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:  # double-check
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        config_dir: Optional[Union[str, Path]] = None,
        environment: str = "dev",
        auto_load: bool = True,
        include_user: bool = True,
        include_run: bool = True,
    ) -> None:
        # Guard against re-initialisation because ``__init__`` is invoked on
        # every ``ConfigCenter()`` call for a singleton.  The whole
        # initialisation block runs under the same ``_singleton_lock`` used
        # by ``__new__`` so that two concurrent ``ConfigCenter()`` calls
        # cannot both pass the ``_initialized`` check (TOCTOU).
        if self._initialized:
            return
        with self._singleton_lock:
            if self._initialized:
                return
            self._initialized = True

            self._config: Dict[str, Any] = {}
            self._environment: str = environment
            self._config_dir: Path = _resolve_config_dir(config_dir)
            self._loaded_files: List[Path] = []
            self._lock: threading.RLock = threading.RLock()

            self._system_dir: Path = _system_defaults_dir()
            self._user_dir: Path = _user_config_dir()
            self._run_snapshot_path: Optional[Path] = None

            self._logger = get_logger(self.__class__.__name__)

            if auto_load:
                self.load(
                    environment=environment,
                    include_user=include_user,
                    include_run=include_run,
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def environment(self) -> str:
        """当前活动的环境（``dev`` 或 ``prod``）。"""
        return self._environment

    @property
    def config_dir(self) -> Path:
        """已解析的配置目录。"""
        return self._config_dir

    @property
    def loaded_files(self) -> List[Path]:
        """已成功加载的配置文件列表。"""
        return list(self._loaded_files)

    @property
    def system_dir(self) -> Path:
        """The System-layer defaults directory."""
        return self._system_dir

    @property
    def user_dir(self) -> Path:
        """The User-layer config directory."""
        return self._user_dir

    @property
    def run_snapshot_path(self) -> Optional[Path]:
        """Path of the run snapshot written during the last :meth:`load`."""
        return self._run_snapshot_path

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(
        self,
        config_dir: Optional[Union[str, Path]] = None,
        environment: Optional[str] = None,
        include_user: bool = True,
        include_run: bool = True,
    ) -> Dict[str, Any]:
        """Load and merge all configuration layers.

        Layers are merged in increasing precedence:
        System < Project < User < Run.

        Args:
            config_dir: Optional override for the Project-layer directory.
            environment: Optional override for the active environment.
            include_user: When ``True`` the User layer is (re)loaded.
            include_run: When ``True`` a run snapshot is written.

        Returns:
            The fully merged configuration dictionary.
        """
        with self._lock:
            if config_dir is not None:
                self._config_dir = _resolve_config_dir(config_dir)
            if environment is not None:
                self._environment = environment

            self._config = {}
            self._loaded_files = []

            # 1. System layer (immutable built-in defaults).
            self._load_dir(self._system_dir, required=False, layer="system")

            # 2. Project layer (committed ./config/*.yaml + environment override).
            for filename in self.DEFAULT_CONFIG_FILES:
                self._load_file(self._config_dir / filename)
            env_file = self._config_dir / f"config.{self._environment}.yaml"
            self._load_file(env_file, required=False)

            # 3. User layer (per-user overrides).
            if include_user:
                self.load_user_config()

            # 4. Run layer (persist a snapshot for reproducibility).
            if include_run:
                try:
                    self._run_snapshot_path = self.save_run_snapshot()
                except Exception as exc:  # pragma: no cover - best effort
                    self._logger.warning(
                        "Failed to write run snapshot: %s", exc
                    )
                    self._run_snapshot_path = None

            return deepcopy(self._config)

    def load_user_config(self) -> Dict[str, Any]:
        """Load (or reload) the User-layer overrides.

        Reads every ``*.yaml`` file under :attr:`user_dir` and merges them
        on top of the current configuration.  Missing directories are
        silently ignored.

        Returns:
            The merged configuration dictionary after applying user overrides.
        """
        with self._lock:
            self._load_dir(self._user_dir, required=False, layer="user")
            return deepcopy(self._config)

    def load_run_snapshot(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Replay a previous run's configuration snapshot.

        The snapshot is merged as the highest-precedence Run layer, so every
        value it contains overrides anything loaded from the lower layers.

        Args:
            path: Path to a ``config_snapshot.json`` file.

        Returns:
            The merged configuration dictionary after applying the snapshot.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the snapshot is not a JSON object.
        """
        snapshot_path = Path(path).expanduser().resolve()
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Run snapshot not found: {snapshot_path}")

        with open(snapshot_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            raise ValueError(
                f"Run snapshot {snapshot_path} must contain a JSON object at "
                f"the top level, got {type(data).__name__}."
            )

        # Separate the payload from the snapshot metadata envelope.
        payload = data.get("config", data)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Run snapshot {snapshot_path} 'config' field must be an "
                f"object, got {type(payload).__name__}."
            )

        with self._lock:
            self._config = _deep_merge(self._config, payload)
            self._loaded_files.append(snapshot_path)
            self._run_snapshot_path = snapshot_path
            self._logger.info("Loaded run snapshot from %s.", snapshot_path)
            return deepcopy(self._config)

    def load_file(self, path: Union[str, Path], required: bool = True) -> None:
        """加载额外的 YAML 文件并合并到配置中。

        Args:
            path: YAML 文件路径。
            required: 为 ``True`` 时，缺失文件会抛出 ``FileNotFoundError``。
        """
        with self._lock:
            self._load_file(Path(path).expanduser().resolve(), required=required)

    def _load_file(self, path: Path, required: bool = True) -> None:
        """加载并合并单个 YAML 文件。"""
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Configuration file not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(
                f"Configuration file {path} must contain a YAML mapping at "
                f"the top level, got {type(data).__name__}."
            )

        with self._lock:
            self._config = _deep_merge(self._config, data)
            self._loaded_files.append(path)

    # ------------------------------------------------------------------
    # Access (点号分隔键访问)
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """使用点号分隔表示法获取配置值。

        Args:
            key: 点号分隔路径，如 ``"sampling.default.temperature"``。
            default: 键不存在时返回的值。

        Returns:
            配置值或 ``default``。
        """
        if not key:
            return default

        with self._lock:
            current: Any = self._config
            for part in key.split("."):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            # 对可变容器类型（dict/list）返回深拷贝，避免调用方
            # 误改内部配置；标量类型直接返回即可。
            if isinstance(current, (dict, list)):
                return deepcopy(current)
            return current

    def set(self, key: str, value: Any) -> None:
        """使用点号分隔表示法动态设置配置值。

        缺失的中间字典会自动创建。

        Args:
            key: 点号分隔路径，如 ``"sampling.default.temperature"``。
            value: 要赋的值。
        """
        if not key:
            raise KeyError("Configuration key must be a non-empty string.")

        with self._lock:
            parts = key.split(".")
            node: Dict[str, Any] = self._config
            for part in parts[:-1]:
                existing = node.get(part)
                if not isinstance(existing, dict):
                    existing = {}
                    node[part] = existing
                node = existing
            node[parts[-1]] = deepcopy(value)

    def has(self, key: str) -> bool:
        """当 ``key`` 存在于配置中时返回 ``True``。"""
        sentinel = object()
        return self.get(key, sentinel) is not sentinel

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------
    def merge(self, *sources: Dict[str, Any]) -> Dict[str, Any]:
        """将一个或多个字典合并到当前配置中。

        后面的源优先级高于前面的源及现有配置。合并以递归方式进行。

        Args:
            *sources: 一个或多个配置字典。

        Returns:
            合并后的配置字典。
        """
        with self._lock:
            for source in sources:
                if not isinstance(source, dict):
                    raise TypeError(
                        f"Each merge source must be a dict, got "
                        f"{type(source).__name__}."
                    )
                self._config = _deep_merge(self._config, source)
            return deepcopy(self._config)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """返回完整配置字典的深拷贝。"""
        with self._lock:
            return deepcopy(self._config)

    def switch_environment(self, environment: str) -> Dict[str, Any]:
        """切换活动环境并重新加载配置。

        Args:
            environment: 目标环境（``dev`` 或 ``prod``）。

        Returns:
            新合并的配置字典。
        """
        if environment not in self.SUPPORTED_ENVIRONMENTS:
            raise ValueError(
                f"Unsupported environment '{environment}'. Supported: "
                f"{', '.join(self.SUPPORTED_ENVIRONMENTS)}."
            )
        return self.load(environment=environment)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Return a deep copy of the current configuration as plain JSON data.

        The returned dictionary is safe to serialise with :func:`json.dumps`
        and contains no references back into the live configuration.
        """
        with self._lock:
            return _to_jsonable(deepcopy(self._config))

    def save_run_snapshot(
        self,
        path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Persist the current configuration as a run snapshot.

        The snapshot is written as ``config_snapshot.json`` inside a
        timestamped run directory under the user data directory, unless an
        explicit ``path`` is given.

        Args:
            path: Optional explicit destination file path.

        Returns:
            The resolved path of the written snapshot file.
        """
        if path is not None:
            target = Path(path).expanduser().resolve()
        else:
            run_dir = _run_snapshot_dir() / time.strftime("%Y%m%d-%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
            target = run_dir / _RUN_SNAPSHOT_FILENAME

        target.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            envelope: Dict[str, Any] = {
                "framework": "TorchaVerse",
                "version": "0.3.1",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": time.time(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "environment": self._environment,
                "config": _to_jsonable(deepcopy(self._config)),
            }
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, indent=2, ensure_ascii=False, default=str)

        self._logger.debug("Wrote run snapshot to %s.", target)
        return target

    # ------------------------------------------------------------------
    # ResourceBudget accessor
    # ------------------------------------------------------------------
    def resource_budget(self) -> ResourceBudget:
        """Build a :class:`ResourceBudget` from the current configuration.

        Reads the optional ``resource_budget`` section of the configuration
        and falls back to sensible defaults for any missing field.  This
        keeps :class:`ResourceBudget` construction free of hard-coded
        constants -- everything comes from configuration.
        """
        section = self.get("resource_budget", {}) or {}
        if not isinstance(section, dict):
            section = {}

        def _get_float(key: str, default: float) -> float:
            # Prefer the canonical ResourceBudget field name, falling back to
            # the System-layer ``default_<field>`` convention so that the
            # immutable System defaults can seed the budget.
            for candidate in (key, f"default_{key}"):
                if candidate in section:
                    try:
                        return float(section[candidate])
                    except (TypeError, ValueError):
                        pass
            return default

        def _get_int(key: str, default: int) -> int:
            for candidate in (key, f"default_{key}"):
                if candidate in section:
                    try:
                        return int(section[candidate])
                    except (TypeError, ValueError):
                        pass
            return default

        return ResourceBudget(
            vram_gb=_get_float("vram_gb", 0.0),
            ram_gb=_get_float("ram_gb", 0.0),
            disk_gb=_get_float("disk_gb", 0.0),
            max_concurrent_models=_get_int("max_concurrent_models", 1),
            max_concurrent_requests=_get_int("max_concurrent_requests", 1),
            kv_cache_gb=_get_float("kv_cache_gb", 0.0),
            activations_gb=_get_float("activations_gb", 0.0),
            offload_to=str(section.get("offload_to", "none")),
        )

    # ------------------------------------------------------------------
    # Context manager for temporary overrides (testing convenience)
    # ------------------------------------------------------------------
    class _ResetContext:
        """Context manager that restores configuration on exit."""

        def __init__(self, owner: "ConfigCenter") -> None:
            self._owner = owner
            self._backup: Dict[str, Any] = {}

        def __enter__(self) -> "ConfigCenter":
            with self._owner._lock:
                self._backup = deepcopy(self._owner._config)
            return self._owner

        def __exit__(
            self,
            exc_type: Optional[type],
            exc_val: Optional[BaseException],
            exc_tb: Any,
        ) -> bool:
            with self._owner._lock:
                self._owner._config = self._backup
            return False

    def reset_context(self) -> "ConfigCenter._ResetContext":
        """Return a context manager that restores config on exit.

        Useful for tests that mutate configuration via :meth:`set` and want
        the changes automatically reverted::

            with ConfigCenter().reset_context():
                ConfigCenter().set("default.dtype", "fp16")
        """
        return ConfigCenter._ResetContext(self)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_dir(
        self,
        directory: Path,
        required: bool,
        layer: str,
    ) -> None:
        """Load and merge every ``*.yaml`` file inside ``directory``."""
        if not directory.exists():
            if required:
                raise FileNotFoundError(
                    f"{layer} config directory not found: {directory}"
                )
            return

        for yaml_file in sorted(directory.glob("*.yaml")):
            self._load_file(yaml_file, required=True)

    # ------------------------------------------------------------------
    # Reset (testing)
    # ------------------------------------------------------------------
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance.

        Primarily useful for testing where a fresh configuration is required.
        """
        with cls._singleton_lock:
            cls._instance = None
            cls._initialized = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-serialisable primitives.

    ``Path`` objects are converted to strings and tuples become lists, so
    the result can be safely passed to :func:`json.dump`.
    """
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def get_config(key: str, default: Any = None) -> Any:
    """单例 :class:`ConfigCenter` 的便捷访问器。

    Args:
        key: 点号分隔的配置键。
        default: 回退值。

    Returns:
        配置值或 ``default``。
    """
    return ConfigCenter().get(key, default)
