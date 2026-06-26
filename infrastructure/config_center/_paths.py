"""Path resolution helpers for :class:`ConfigCenter`.

The v0.6.x split of :mod:`infrastructure.config_center` keeps the
path-layer-1 helpers (config-dir / system-defaults / user / run-snapshot
directories, plus the env-var conventions) in this small module so the
:class:`ConfigCenter` core can stay focused on load / query / snapshot
semantics.

The four env-var conventions are also defined here because every
function in this module reads at least one of them:

* ``TORCHAVERSE_SYSTEM_CONFIG_DIR`` -- override the system defaults dir
* ``TORCHAVERSE_USER_CONFIG_DIR`` -- override the per-user dir
* ``TORCHAVERSE_RUN_DIR`` -- the run-time working directory (e.g. from
  an experiment runner); the run snapshot is written into a
  ``config_snapshot.json`` inside it.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "ENV_SYSTEM_DIR",
    "ENV_USER_DIR",
    "ENV_RUN_DIR",
    "RUN_SNAPSHOT_FILENAME",
    "SYSTEM_DEFAULTS_SUBDIR",
    "resolve_config_dir",
    "system_defaults_dir",
    "user_config_dir",
    "user_data_dir",
    "run_snapshot_dir",
]

#: Environment variable that overrides the system defaults dir.
ENV_SYSTEM_DIR: str = "TORCHAVERSE_SYSTEM_CONFIG_DIR"
#: Environment variable that overrides the per-user config dir.
ENV_USER_DIR: str = "TORCHAVERSE_USER_CONFIG_DIR"
#: Environment variable that points at the per-run working directory.
ENV_RUN_DIR: str = "TORCHAVERSE_RUN_DIR"
#: Filename of the run-snapshot inside the per-run directory.
RUN_SNAPSHOT_FILENAME: str = "config_snapshot.json"
#: Sub-directory under the system root that holds the YAML defaults.
SYSTEM_DEFAULTS_SUBDIR: str = "_defaults"


def resolve_config_dir(
    config_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve the project config dir in order of priority.

    Resolution order:

    1. ``config_dir`` argument (caller-supplied override).
    2. ``./config/`` in the current working directory, but only
       when the directory actually contains a known config file
       (otherwise an unrelated empty ``./config/`` shadows the
       shipped defaults).
    3. The directory of the script (when running a ``python script.py``
       style entry point) -- same validation.
    4. The package root (when imported as a library), resolved by
       walking up the ``__file__`` of this module until a ``config``
       directory is found that contains the default config files.

    Returns:
        A :class:`Path` that exists or can be created.
    """
    if config_dir is not None:
        return Path(config_dir).expanduser().resolve()

    # The shipped config files that *must* exist for a directory
    # to count as a valid project config dir.  Used to defend
    # against an empty ``./config/`` shadowing the package defaults
    # (e.g. when a CWD happens to contain an unrelated ``config/``
    # subdir).
    _SENTINEL_FILES = ("model_config.yaml", "inference_config.yaml")

    def _is_valid(d: Path) -> bool:
        if not d.is_dir():
            return False
        return any((d / name).is_file() for name in _SENTINEL_FILES)

    cwd = Path.cwd() / "config"
    if _is_valid(cwd):
        return cwd.resolve()

    if getattr(sys, "argv", None) and sys.argv and sys.argv[0]:
        argv0 = Path(sys.argv[0]).expanduser()
        try:
            script_dir = argv0.resolve().parent / "config"
        except OSError:
            script_dir = None
        if script_dir is not None and _is_valid(script_dir):
            return script_dir

    # Walk up from this file's location until we find a `config/`
    # that contains the default files.  ``infrastructure/config_center
    # /_paths.py`` is 4 levels deep, so parents[3] is the package
    # root.
    here = Path(__file__).resolve().parent
    for ancestor in (here, *here.parents):
        candidate = ancestor / "config"
        if _is_valid(candidate):
            return candidate.resolve()
    # Final fallback: the 4-ancestor heuristic that v0.4.x relied on.
    return (here.parents[3] / "config").resolve()


def system_defaults_dir() -> Path:
    """Return the directory that holds the system-shipped YAML defaults.

    The system root is normally the parent of the package
    (``<package_root>/config``) but the
    :data:`ENV_SYSTEM_DIR` env-var can override the location.
    """
    override = os.environ.get(ENV_SYSTEM_DIR)
    if override:
        return Path(override).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "config").resolve()


def user_config_dir() -> Path:
    """Return the per-user config directory, honouring platform conventions.

    * Linux / macOS: ``~/.config/torcha-verse/``
    * Windows: ``%APPDATA%/torcha-verse/``

    Falls back to ``~/.torcha-verse/`` on platforms that do not expose
    the standard XDG/AppData env-vars.
    """
    override = os.environ.get(ENV_USER_DIR)
    if override:
        return Path(override).expanduser().resolve()

    system = platform.system()
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "torcha-verse"
        return Path.home() / "torcha-verse"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "torcha-verse"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "torcha-verse"
    return Path.home() / ".config" / "torcha-verse"


def user_data_dir() -> Path:
    """Return the per-user data directory (cache, model cache, logs).

    * Linux: ``~/.local/share/torcha-verse/``
    * macOS: same ``~/Library/Application Support/torcha-verse/`` as
      the config dir (Apple's HIG reuses Application Support).
    * Windows: ``%LOCALAPPDATA%/torcha-verse/``
    """
    system = platform.system()
    if system == "Windows":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "torcha-verse"
        return Path.home() / "torcha-verse"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "torcha-verse"
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return Path(xdg_data) / "torcha-verse"
    return Path.home() / ".local" / "share" / "torcha-verse"


def run_snapshot_dir() -> Path:
    """Return the directory where the run-time snapshot should be written.

    Honours the :data:`ENV_RUN_DIR` env-var when set; otherwise returns
    a ``config/`` subdirectory under the current working directory.
    """
    override = os.environ.get(ENV_RUN_DIR)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.cwd() / "config").resolve()
