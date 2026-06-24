"""Configuration manager for the TorchaVerse framework.

This module provides a singleton :class:`ConfigManager` that loads YAML
configuration files from the project ``config/`` directory, supports
environment-based overrides (``dev``/``prod``), dot-notation access for
nested keys, dynamic modification, and merging of multiple config sources.

The goal is to keep configuration and code completely separated: no magic
values should be hard-coded inside the framework logic.
"""

from __future__ import annotations

import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import yaml

__all__ = ["ConfigManager", "get_config"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base``.

    Nested dictionaries are merged key-by-key while non-dict values from
    ``override`` replace the corresponding values in ``base``.  A new
    dictionary is returned; the inputs are never mutated.

    Args:
        base: The base configuration dictionary.
        override: The dictionary whose values take precedence.

    Returns:
        A new dictionary containing the merged result.
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
    """Resolve the configuration directory.

    When ``config_dir`` is ``None`` the directory is located relative to this
    file (``<project_root>/config``).  The ``TORCHAVERSE_CONFIG_DIR``
    environment variable can also be used to override the default location.

    Args:
        config_dir: Explicit path or ``None`` to auto-detect.

    Returns:
        The resolved :class:`~pathlib.Path` to the config directory.
    """
    env_dir = os.environ.get("TORCHAVERSE_CONFIG_DIR")
    if config_dir is not None:
        return Path(config_dir).expanduser().resolve()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    # This file lives at <project_root>/infrastructure/config_manager.py
    return Path(__file__).resolve().parent.parent / "config"


# ---------------------------------------------------------------------------
# ConfigManager
# ---------------------------------------------------------------------------
class ConfigManager:
    """Singleton configuration manager.

    Loads the default YAML configuration files
    (``model_config.yaml``, ``inference_config.yaml``,
    ``training_config.yaml``, ``prompt_templates.yaml``) and overlays an
    environment-specific file (``config.dev.yaml`` or ``config.prod.yaml``)
    on top of them.

    Configuration values can be accessed with dot-separated keys, e.g.
    ``config.get("sampling.default.temperature", 0.7)``.

    Example:
        >>> cfg = ConfigManager()
        >>> cfg.get("default.dtype")
        'bf16'
        >>> cfg.set("default.dtype", "fp16")
        >>> cfg.get("default.dtype")
        'fp16'
    """

    _instance: Optional["ConfigManager"] = None
    _initialized: bool = False
    # RLock (not Lock) because ConfigCenter subclasses ConfigManager and its
    # __new__ calls super().__new__(), which re-enters this same lock via
    # ``cls._singleton_lock`` (cls is still the subclass).  A plain Lock
    # would self-deadlock; RLock permits the re-entrant acquisition.
    _singleton_lock: threading.RLock = threading.RLock()

    #: The default configuration files loaded on startup.
    DEFAULT_CONFIG_FILES: Sequence[str] = (
        "model_config.yaml",
        "inference_config.yaml",
        "training_config.yaml",
        "prompt_templates.yaml",
    )

    #: Supported environment identifiers.
    SUPPORTED_ENVIRONMENTS: Sequence[str] = ("dev", "prod")

    def __new__(cls, *args: Any, **kwargs: Any) -> "ConfigManager":
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
    ) -> None:
        # Guard against re-initialisation because ``__init__`` is invoked on
        # every ``ConfigManager()`` call for a singleton.
        if self._initialized:
            return
        self._initialized = True

        self._config: Dict[str, Any] = {}
        self._environment: str = environment
        self._config_dir: Path = _resolve_config_dir(config_dir)
        self._loaded_files: List[Path] = []
        self._lock: threading.RLock = threading.RLock()

        if auto_load:
            self.load(environment=environment)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def environment(self) -> str:
        """The currently active environment (``dev`` or ``prod``)."""
        return self._environment

    @property
    def config_dir(self) -> Path:
        """The resolved configuration directory."""
        return self._config_dir

    @property
    def loaded_files(self) -> List[Path]:
        """List of configuration files that were successfully loaded."""
        return list(self._loaded_files)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(
        self,
        config_dir: Optional[Union[str, Path]] = None,
        environment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load configuration files from disk.

        The default config files are merged first, followed by the
        environment-specific override file (``config.<env>.yaml``) if it
        exists.  Calling :meth:`load` resets any previously loaded or
        dynamically set configuration.

        Args:
            config_dir: Optional override for the config directory.
            environment: Optional override for the active environment.

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

            # 1. Load the default config files.
            for filename in self.DEFAULT_CONFIG_FILES:
                self._load_file(self._config_dir / filename)

            # 2. Overlay the environment-specific config.
            env_file = self._config_dir / f"config.{self._environment}.yaml"
            self._load_file(env_file, required=False)

            return deepcopy(self._config)

    def load_file(self, path: Union[str, Path], required: bool = True) -> None:
        """Load an additional YAML file and merge it into the configuration.

        Args:
            path: Path to the YAML file.
            required: If ``True`` a missing file raises ``FileNotFoundError``.
        """
        with self._lock:
            self._load_file(Path(path).expanduser().resolve(), required=required)

    def _load_file(self, path: Path, required: bool = True) -> None:
        """Load and merge a single YAML file."""
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
    # Access
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value using dot-separated notation.

        Args:
            key: Dot-separated path, e.g. ``"sampling.default.temperature"``.
            default: Value returned when the key is absent.

        Returns:
            The configuration value or ``default``.
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
            return current

    def set(self, key: str, value: Any) -> None:
        """Dynamically set a configuration value using dot-separated notation.

        Intermediate dictionaries are created automatically when missing.

        Args:
            key: Dot-separated path, e.g. ``"sampling.default.temperature"``.
            value: The value to assign.
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
        """Return ``True`` if ``key`` exists in the configuration."""
        sentinel = object()
        return self.get(key, sentinel) is not sentinel

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------
    def merge(self, *sources: Dict[str, Any]) -> Dict[str, Any]:
        """Merge one or more dictionaries into the current configuration.

        Later sources take precedence over earlier ones and over the existing
        configuration.  The merge is performed recursively.

        Args:
            *sources: One or more configuration dictionaries.

        Returns:
            The merged configuration dictionary.
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
        """Return a deep copy of the full configuration dictionary."""
        with self._lock:
            return deepcopy(self._config)

    def switch_environment(self, environment: str) -> Dict[str, Any]:
        """Switch the active environment and reload configuration.

        Args:
            environment: The target environment (``dev`` or ``prod``).

        Returns:
            The newly merged configuration dictionary.
        """
        if environment not in self.SUPPORTED_ENVIRONMENTS:
            raise ValueError(
                f"Unsupported environment '{environment}'. Supported: "
                f"{', '.join(self.SUPPORTED_ENVIRONMENTS)}."
            )
        return self.load(environment=environment)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance.

        Primarily useful for testing where a fresh configuration is required.
        """
        with cls._singleton_lock:
            cls._instance = None
            cls._initialized = False


def get_config(key: str, default: Any = None) -> Any:
    """Convenience accessor for the singleton :class:`ConfigManager`.

    Args:
        key: Dot-separated configuration key.
        default: Fallback value.

    Returns:
        The configuration value or ``default``.
    """
    return ConfigManager().get(key, default)
