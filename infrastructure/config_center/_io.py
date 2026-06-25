"""YAML / JSON load helpers for :class:`ConfigCenter`.

The :mod:`infrastructure.config_center` sub-package reads its four
layers (system / project / user / run) from disk.  The actual
*file reading* logic (recursive directory walk, file extension
dispatch, deep-merge after each layer) is factored out into this
module so the main :class:`ConfigCenter` core can stay focused on
load / query / snapshot semantics.

Public surface:

* :func:`read_yaml_file` -- read a single YAML file (or JSON when
  the file has a ``.json`` extension), with a clear error message
  when the file is missing or malformed.
* :func:`collect_dir` -- list + read the configuration files in a
  directory, returning a deterministic list of ``(path, data)``
  tuples in alphabetical order.  Missing directories are tolerated
  (or rejected via ``required``).
* :func:`load_dir` -- convenience wrapper that loads every file in
  a directory and returns the deep-merged result.
* :func:`deep_merge` -- the small recursive dict-merge routine
  used at every layer boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import yaml

__all__ = [
    "deep_merge",
    "read_yaml_file",
    "collect_dir",
    "load_dir",
]


def deep_merge(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return ``base``.

    Lists and tuples are *replaced* (not concatenated); this matches
    the policy of the previous single-file implementation and is
    the most intuitive behaviour for configuration files (a
    comma-separated list of comma-overrides should not duplicate the
    defaults).
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def read_yaml_file(
    path: Path,
    required: bool = True,
) -> Optional[Dict[str, Any]]:
    """Read a single YAML or JSON file and return the parsed dict.

    Args:
        path: The file to read.
        required: When ``True`` (default) a missing file is a hard
            error.  When ``False`` a missing file returns ``None``.

    Returns:
        The parsed dict (empty dict if the file exists but is empty
        or contains only comments); or ``None`` if ``required`` is
        ``False`` and the file is missing.

    Raises:
        FileNotFoundError: ``required=True`` and ``path`` does not
            exist.
        yaml.YAMLError: the file is not valid YAML.
        json.JSONDecodeError: the file is a ``.json`` and is invalid.
    """
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Config file not found: {path}")
        return None
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text) if text.strip() else {}
    data = yaml.safe_load(text) if text.strip() else {}
    return data or {}


def collect_dir(
    directory: Path,
    required: bool = False,
    layer: str = "",
    suffix: Union[str, Iterable[str]] = (".yaml", ".yml", ".json"),
) -> List[Tuple[Path, Dict[str, Any]]]:
    """List + read the configuration files in ``directory``.

    Files are returned in alphabetical order, so re-running the
    function produces the same merge order.

    Args:
        directory: The directory to scan.
        required:  When ``True`` a missing directory is a hard
            error.  When ``False`` (default) a missing directory
            returns an empty list.
        layer:     Optional layer label (used in error messages).
        suffix:    A single extension or iterable of extensions to
            accept.  Defaults to ``.yaml`` / ``.yml`` / ``.json``.

    Returns:
        A list of ``(Path, dict)`` tuples, sorted by filename.

    Raises:
        FileNotFoundError: ``required=True`` and the directory is
            missing.
    """
    if not directory.is_dir():
        if required:
            raise FileNotFoundError(
                f"{layer or 'config'} directory not found: {directory}"
            )
        return []
    if isinstance(suffix, str):
        suffix_tuple = (suffix,)
    else:
        suffix_tuple = tuple(suffix)
    files = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix in suffix_tuple
    ]
    files.sort(key=lambda p: p.name)
    return [(p, read_yaml_file(p)) for p in files]


def load_dir(
    directory: Path,
    required: bool = False,
    layer: str = "",
) -> Dict[str, Any]:
    """Load every configuration file in ``directory`` and return the merge.

    Files are loaded in alphabetical order; each file is deep-merged
    into the running result so a later file can override an earlier
    one (which is the convention in the v0.5.x framework).

    Args:
        directory: The directory to load.
        required:  When ``True`` a missing directory is a hard error.
        layer:     Optional layer label (used in error messages).

    Returns:
        The merged configuration dict.
    """
    merged: Dict[str, Any] = {}
    for _path, data in collect_dir(directory, required=required, layer=layer):
        deep_merge(merged, data)
    return merged
