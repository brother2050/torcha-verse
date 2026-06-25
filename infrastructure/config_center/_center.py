"""The :class:`ConfigCenter` core.

This module is the main :class:`ConfigCenter` class.  It depends on
the four sibling modules in the :mod:`infrastructure.config_center`
sub-package:

* :mod:`._paths` -- path resolution + env-var conventions.
* :mod:`._io` -- YAML / JSON file reading + deep-merge.
* :mod:`._budget` -- :class:`ResourceBudget` construction.
* :mod:`._jsonable` -- recursive JSON-serialisation helper.
* :mod:`._schema` -- :func:`config_schema` + :class:`Field`
  (re-seeded by :class:`ConfigCenter` at boot so schemas see
  their defaults).

The class itself exposes:

* :meth:`load` / :meth:`load_file` / :meth:`load_user_config` /
  :meth:`load_run_snapshot` -- layered loading API.
* :meth:`get` / :meth:`set` / :meth:`has` / :meth:`merge` /
  :meth:`to_dict` -- query / mutation API.
* :meth:`switch_environment` / :meth:`snapshot` /
  :meth:`save_run_snapshot` -- environment / persistence API.
* :meth:`resource_budget` -- configuration-driven
  :class:`ResourceBudget` builder.
* :meth:`reset` / :meth:`reset_context` -- testing hooks.

The :func:`get_config` module-level singleton accessor lives in
:mod:`infrastructure.config_center.__init__` (the public facade).
"""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from ..logger import get_logger
from ..resource_budget import ResourceBudget
from ._budget import build_resource_budget
from ._io import (
    collect_dir as _collect_dir,
    deep_merge as _deep_merge,
    read_yaml_file as _read_yaml_file,
)
from ._jsonable import to_jsonable
from ._paths import (
    resolve_config_dir as _resolve_config_dir,
    run_snapshot_dir as _run_snapshot_dir,
    system_defaults_dir as _system_defaults_dir,
    user_config_dir as _user_config_dir,
)

__all__ = ["ConfigCenter"]


class ConfigCenter:
    """Singleton four-layer configuration merge centre.

    :class:`ConfigCenter` provides layered-aware configuration
    loading.  Four layers (System < Project < User < Run) are deep-
    merged in order so higher-precedence layers override lower-
    precedence ones.

    The singleton keeps the dotted-key accessor
    (``get`` / ``set`` / ``has`` / ``merge`` / ``to_dict``) and adds:

    * :meth:`load_user_config` -- (re)load the User layer.
    * :meth:`load_run_snapshot` -- replay a previous run snapshot.
    * :meth:`snapshot` -- deep-copy the current configuration as JSON.
    * :meth:`save_run_snapshot` -- persist the current configuration
      so a generation can be replayed later.
    * :meth:`resource_budget` -- build a :class:`ResourceBudget`
      from the configuration.

    Args:
        config_dir: Override for the Project-layer directory.
        environment: Active environment (``dev`` or ``prod``).
        auto_load: When ``True`` (default) all layers are loaded
            immediately.
        include_user: When ``True`` (default) the User layer is
            loaded.
        include_run: When ``True`` (default) a run snapshot is
            written during the load.

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
    # RLock (not Lock): ``__new__`` can be invoked concurrently, RLock
    # allows the same thread to re-enter.
    _singleton_lock: threading.RLock = threading.RLock()

    #: Default configuration files loaded at startup.
    DEFAULT_CONFIG_FILES: Sequence[str] = (
        "model_config.yaml",
        "inference_config.yaml",
        "training_config.yaml",
        "prompt_templates.yaml",
    )

    #: Supported environment identifiers.
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
        # Guard against re-init because ``__init__`` runs on every
        # ``ConfigCenter()`` call (singleton).  The whole block runs
        # under the same ``_singleton_lock`` used by ``__new__`` so
        # two concurrent ``ConfigCenter()`` calls cannot both pass
        # the ``_initialized`` check (TOCTOU).
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

            # Re-seed the @config_schema defaults so that schemas
            # registered after ConfigCenter's first instantiation
            # also see their defaults.
            try:
                from ._schema import default_registry  # local import
                for schema in default_registry.all():
                    for fname, f in schema.fields:
                        key = f"{schema.name}.{fname}"
                        with self._lock:
                            if key not in self._config:
                                self._config[key] = f.default
            except Exception:  # pragma: no cover - best effort
                pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def environment(self) -> str:
        return self._environment

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    @property
    def loaded_files(self) -> List[Path]:
        return list(self._loaded_files)

    @property
    def system_dir(self) -> Path:
        return self._system_dir

    @property
    def user_dir(self) -> Path:
        return self._user_dir

    @property
    def run_snapshot_path(self) -> Optional[Path]:
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
            config_dir: Optional override for the Project-layer dir.
            environment: Optional override for the active env.
            include_user: When ``True`` the User layer is reloaded.
            include_run: When ``True`` a run snapshot is written.

        Returns:
            The fully merged configuration dictionary.
        """
        with self._lock:
            if config_dir is not None:
                self._config_dir = _resolve_config_dir(config_dir)
            if environment is not None:
                self._environment = environment
            system_dir = self._system_dir
            user_dir = self._user_dir
            config_dir_resolved = self._config_dir
            env = self._environment

        # I/O outside the lock.
        new_config: Dict[str, Any] = {}
        new_loaded_files: List[Path] = []

        # 1. System layer (immutable built-in defaults).
        for path, data in _collect_dir(system_dir, required=False, layer="system"):
            new_config = _deep_merge(new_config, data)
            new_loaded_files.append(path)

        # 2. Project layer (committed ./config/*.yaml + env override).
        for filename in self.DEFAULT_CONFIG_FILES:
            fpath = config_dir_resolved / filename
            data = _read_yaml_file(fpath, required=True)
            if data is not None:
                new_config = _deep_merge(new_config, data)
                new_loaded_files.append(fpath)
        env_file = config_dir_resolved / f"config.{env}.yaml"
        data = _read_yaml_file(env_file, required=False)
        if data is not None:
            new_config = _deep_merge(new_config, data)
            new_loaded_files.append(env_file)

        # 3. User layer.
        if include_user:
            for path, data in _collect_dir(user_dir, required=False, layer="user"):
                new_config = _deep_merge(new_config, data)
                new_loaded_files.append(path)

        # Atomic update.
        with self._lock:
            self._config = new_config
            self._loaded_files = new_loaded_files

        # 4. Run layer.
        if include_run:
            try:
                self._run_snapshot_path = self.save_run_snapshot()
            except Exception as exc:  # pragma: no cover - best effort
                self._logger.warning("Failed to write run snapshot: %s", exc)
                self._run_snapshot_path = None

        with self._lock:
            return deepcopy(self._config)

    def load_user_config(self) -> Dict[str, Any]:
        """(Re)load the User-layer overrides.

        Reads every ``*.yaml`` file under :attr:`user_dir` and merges
        them on top of the current configuration.  Missing directories
        are silently ignored.
        """
        with self._lock:
            user_dir = self._user_dir
        if not user_dir.is_dir():
            return {}
        new_overlay: Dict[str, Any] = {}
        for path, data in _collect_dir(user_dir, required=False, layer="user"):
            new_overlay = _deep_merge(new_overlay, data)
        with self._lock:
            self._config = _deep_merge(self._config, new_overlay)
            return deepcopy(self._config)

    def load_run_snapshot(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Replay a run snapshot from ``path`` as the active config.

        The loaded snapshot is the new top of the precedence stack
        (above User) so it overrides any user-level overrides.
        """
        snapshot = _read_yaml_file(Path(path), required=True)
        with self._lock:
            self._config = _deep_merge(self._config, snapshot)
            self._run_snapshot_path = Path(path).resolve()
            return deepcopy(self._config)

    def load_file(
        self,
        path: Union[str, Path],
        required: bool = True,
    ) -> None:
        """Load a single config file and merge it into the active config.

        Args:
            path: The file to load (``.yaml`` / ``.yml`` / ``.json``).
            required: When ``True`` a missing file is a hard error.
        """
        data = _read_yaml_file(Path(path), required=required)
        with self._lock:
            self._config = _deep_merge(self._config, data)
            self._loaded_files.append(Path(path).resolve())

    # ------------------------------------------------------------------
    # Query / mutation
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Return the value at dotted ``key`` or ``default``."""
        with self._lock:
            node: Any = self._config
            for part in key.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    return default
            return node

    def set(self, key: str, value: Any) -> None:
        """Set the value at dotted ``key``.

        Intermediate dicts are created automatically.
        """
        parts = key.split(".")
        with self._lock:
            node = self._config
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value

    def has(self, key: str) -> bool:
        """Return ``True`` if dotted ``key`` is set."""
        with self._lock:
            node: Any = self._config
            for part in key.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    return False
            return True

    def merge(self, *sources: Dict[str, Any]) -> Dict[str, Any]:
        """Merge every ``source`` dict into the active config.

        Each source is deep-merged onto the current config in order,
        so later sources override earlier ones.
        """
        with self._lock:
            for source in sources:
                _deep_merge(self._config, source)
            return deepcopy(self._config)

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the active configuration."""
        with self._lock:
            return deepcopy(self._config)

    def switch_environment(self, environment: str) -> Dict[str, Any]:
        """Switch to ``environment`` and reload the Project layer.

        Args:
            environment: One of ``SUPPORTED_ENVIRONMENTS``.

        Returns:
            The freshly merged configuration.
        """
        if environment not in self.SUPPORTED_ENVIRONMENTS:
            raise ValueError(
                f"Unsupported environment {environment!r}; "
                f"expected one of {self.SUPPORTED_ENVIRONMENTS}"
            )
        return self.load(environment=environment)

    # ------------------------------------------------------------------
    # Snapshot / persistence
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-safe deep copy of the active configuration."""
        with self._lock:
            return to_jsonable(self._config)

    def save_run_snapshot(
        self,
        path: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """Persist the current configuration for replay.

        Args:
            path: Optional explicit path.  Defaults to
                ``<run_dir>/config_snapshot.json``.

        Returns:
            The path the snapshot was written to, or ``None`` when
            the write was skipped (because the directory could not
            be created).
        """
        if path is None:
            target_dir = _run_snapshot_dir()
        else:
            target_dir = Path(path).parent
            target_path = Path(path)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - FS failure
            self._logger.warning("Cannot create run snapshot dir: %s", exc)
            return None
        if path is None:
            target_path = target_dir / "config_snapshot.json"
        with self._lock:
            payload = to_jsonable(self._config)
        with target_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        return target_path.resolve()

    # ------------------------------------------------------------------
    # Resource budget
    # ------------------------------------------------------------------
    def resource_budget(self) -> ResourceBudget:
        """Build a :class:`ResourceBudget` from the active configuration."""
        section = self.get("resource_budget", {}) or {}
        return build_resource_budget(section, self._logger)

    # ------------------------------------------------------------------
    # Context manager (testing convenience)
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
        """Return a context manager that restores config on exit."""
        return self._ResetContext(self)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (testing hook)."""
        with cls._singleton_lock:
            cls._instance = None
            cls._initialized = False
