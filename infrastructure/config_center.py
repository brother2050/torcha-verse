"""Tiered configuration merge centre for TorchaVerse (v0.3.0).

This module promotes configuration to a first-class citizen by introducing
:class:`ConfigCenter`, a singleton that merges configuration from four
ordered layers, each with a strictly increasing precedence:

1. **System** -- built-in defaults shipped with the package under
   ``<package>/config/_defaults/``.  These are *immutable* from the user's
   perspective and are snapshotted in CI as golden files.
2. **Project** -- the ``./config/*.yaml`` files committed with the
   repository (loaded via the inherited :class:`ConfigManager` logic).
3. **User** -- per-user overrides living under
   ``~/.config/torcha-verse/`` (Linux/macOS) or
   ``%APPDATA%/torcha-verse/`` (Windows): UI preferences, API keys, local
   paths.
4. **Run** -- every run produces a ``config_snapshot.json`` so that the
   exact configuration used for a generation can be replayed later.

:class:`ConfigCenter` inherits the full :class:`ConfigManager` interface
(``get`` / ``set`` / ``has`` / ``merge`` / ``to_dict`` with dot-notation
access) and adds tier-aware loading, snapshot serialisation and a
:class:`ResourceBudget` accessor.

Example:
    >>> cc = ConfigCenter()
    >>> cc.get("default.dtype")
    'bf16'
    >>> snap = cc.snapshot()              # deep-copy JSON of current config
    >>> path = cc.save_run_snapshot()     # write config_snapshot.json
    >>> cc.load_run_snapshot(path)        # replay a previous run
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .config_manager import ConfigManager, _deep_merge, _resolve_config_dir
from .logger import get_logger
from .resource_budget import ResourceBudget

__all__ = ["ConfigCenter", "ResourceBudget"]

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
class ConfigCenter(ConfigManager):
    """Singleton four-level configuration merge centre.

    :class:`ConfigCenter` extends :class:`ConfigManager` with tier-aware
    loading.  The four layers (System < Project < User < Run) are deep-merged
    in order so that higher-precedence layers override lower ones.

    The singleton retains the dot-notation accessors (``get`` / ``set`` /
    ``has`` / ``merge`` / ``to_dict``) inherited from :class:`ConfigManager`
    and adds:

    * :meth:`load_user_config` -- (re)load the User layer.
    * :meth:`load_run_snapshot` -- replay a previous run's snapshot.
    * :meth:`snapshot` -- deep-copy the current config as JSON.
    * :meth:`save_run_snapshot` -- persist the current config for replay.
    * :meth:`resource_budget` -- build a :class:`ResourceBudget` from config.

    Args:
        config_dir: Override for the Project-layer config directory.
        environment: Active environment (``dev`` or ``prod``).
        auto_load: When ``True`` (default) all layers are loaded immediately.
        include_user: When ``True`` (default) the User layer is loaded.
        include_run: When ``True`` (default) a run snapshot is written on
            load.

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

    def __new__(cls, *args: Any, **kwargs: Any) -> "ConfigCenter":
        if cls._instance is None:
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
        # every ``ConfigCenter()`` call for a singleton.
        if self._initialized:
            return
        self._initialized = True

        self._config: Dict[str, Any] = {}
        self._environment: str = environment
        self._config_dir: Path = _resolve_config_dir(config_dir)
        self._loaded_files: List[Path] = []

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

        self._config = _deep_merge(self._config, payload)
        self._loaded_files.append(snapshot_path)
        self._run_snapshot_path = snapshot_path
        self._logger.info("Loaded run snapshot from %s.", snapshot_path)
        return deepcopy(self._config)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Return a deep copy of the current configuration as plain JSON data.

        The returned dictionary is safe to serialise with :func:`json.dumps`
        and contains no references back into the live configuration.
        """
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

        envelope: Dict[str, Any] = {
            "framework": "TorchaVerse",
            "version": "0.3.0",
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
            self._backup = deepcopy(self._owner._config)
            return self._owner

        def __exit__(
            self,
            exc_type: Optional[type],
            exc_val: Optional[BaseException],
            exc_tb: Any,
        ) -> bool:
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
