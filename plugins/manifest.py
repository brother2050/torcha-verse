"""Plugin manifest file parsing and validation.

A plugin manifest is a small declarative file (``plugin.toml`` or
``plugin.yaml``) that lives at the root of a plugin directory and
describes the plugin's identity, metadata and the node modules it
contributes.  This module provides :class:`ManifestParser` which turns
such a file into a :class:`plugins.spec.PluginSpec` and validates it.

TOML handling
-------------

The parser prefers the standard-library :mod:`tomllib` (Python 3.11+).
On older Python it transparently falls back to the ``tomli`` backport
when available.  It deliberately does **not** depend on the legacy
third-party ``toml`` package.  When neither ``tomllib`` nor ``tomli`` is
importable, ``.toml`` manifests cannot be parsed and a clear error is
raised -- in that case authors should ship a ``plugin.yaml`` manifest
instead (YAML is always available because ``PyYAML`` is a core
framework dependency).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .spec import PluginSpec, SOURCE_DIRECTORY

__all__ = ["ManifestParser", "ManifestError"]


# ---------------------------------------------------------------------------
# TOML backend selection (tomllib -> tomli -> none).
# ---------------------------------------------------------------------------
_toml_backend: Any = None
try:  # Python 3.11+ standard library.
    import tomllib as _toml_backend  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    try:  # Official backport for Python < 3.11.
        import tomli as _toml_backend  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        _toml_backend = None

#: ``True`` when a TOML parser backend is available.
TOML_AVAILABLE: bool = _toml_backend is not None


# ---------------------------------------------------------------------------
# YAML backend (PyYAML is a core framework dependency).
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml  # type: ignore[import-not-found]
    YAML_AVAILABLE: bool = True
except ModuleNotFoundError:  # pragma: no cover - PyYAML is required
    _yaml = None
    YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Recognised manifest file names (in priority order).
MANIFEST_NAMES: tuple = ("plugin.toml", "plugin.yaml", "plugin.yml")

#: Recognised manifest file suffixes.
MANIFEST_SUFFIXES: tuple = (".toml", ".yaml", ".yml")

#: Top-level keys allowed in a manifest file.
_ALLOWED_KEYS: frozenset = frozenset(
    {
        "name",
        "version",
        "author",
        "description",
        "license",
        "homepage",
        "dependencies",
        "node_modules",
        "on_load",
        "on_unload",
    }
)

#: Minimal semver-ish pattern (X.Y.Z with optional pre-release).
_SEMVER_RE: re.Pattern = re.compile(
    r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?$"
)

#: Valid Python-identifier-ish plugin name pattern.
_NAME_RE: re.Pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ManifestError(ValueError):
    """Raised when a manifest file cannot be parsed or is malformed."""


# ---------------------------------------------------------------------------
# ManifestParser
# ---------------------------------------------------------------------------
class ManifestParser:
    """Parse and validate plugin manifest files.

    The parser is intentionally stateless: every method is a
    ``@staticmethod`` so it can be used without instantiation::

        spec = ManifestParser.parse(Path("my_plugin/plugin.toml"))
        errors = ManifestParser.validate(spec)
    """

    # ------------------------------------------------------------------
    # Discovery helper
    # ------------------------------------------------------------------
    @staticmethod
    def find_manifest(plugin_dir: Union[str, Path]) -> Optional[Path]:
        """Return the manifest file inside ``plugin_dir`` if any.

        Args:
            plugin_dir: Directory that may contain a manifest.

        Returns:
            The :class:`pathlib.Path` of the first recognised manifest
            file (``plugin.toml`` / ``plugin.yaml`` / ``plugin.yml``),
            or ``None`` when the directory holds no manifest.
        """
        directory = Path(plugin_dir)
        if not directory.is_dir():
            return None
        for name in MANIFEST_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    @staticmethod
    def parse(path: Union[str, Path]) -> PluginSpec:
        """Parse a ``plugin.toml`` or ``plugin.yaml`` manifest file.

        The file format is inferred from its suffix.  The returned
        :class:`PluginSpec` has its ``source`` set to ``"directory"`` and
        its ``path`` set to the manifest file's parent directory (so
        that node module paths resolve relative to the plugin root).

        Args:
            path: Path to the manifest file (``plugin.toml`` /
                ``plugin.yaml`` / ``plugin.yml``).

        Returns:
            The parsed :class:`PluginSpec`.

        Raises:
            ManifestError: If the file does not exist, has an unknown
                suffix, the relevant parser backend is missing, or the
                manifest is structurally invalid.
        """
        manifest_path = Path(path)
        if not manifest_path.is_file():
            raise ManifestError(
                "Manifest file does not exist: {}".format(manifest_path)
            )

        suffix = manifest_path.suffix.lower()
        if suffix == ".toml":
            data = ManifestParser._parse_toml(manifest_path)
        elif suffix in (".yaml", ".yml"):
            data = ManifestParser._parse_yaml(manifest_path)
        else:
            raise ManifestError(
                "Unsupported manifest suffix {!r} (expected one of {}). "
                "Use plugin.toml or plugin.yaml.".format(
                    suffix, MANIFEST_SUFFIXES
                )
            )

        if not isinstance(data, dict):
            raise ManifestError(
                "Manifest {} did not produce a mapping (got {}).".format(
                    manifest_path, type(data).__name__
                )
            )

        return ManifestParser._build_spec(data, manifest_path)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_toml(path: Path) -> Dict[str, Any]:
        """Parse a TOML manifest file using the available backend."""
        if _toml_backend is None:
            raise ManifestError(
                "Cannot parse TOML manifest {}: neither 'tomllib' (Python "
                "3.11+) nor the 'tomli' backport is available. Install "
                "'tomli' or ship a plugin.yaml manifest instead.".format(path)
            )
        with open(path, "rb") as fh:
            try:
                return _toml_backend.load(fh)
            except Exception as exc:  # toml parse errors vary by backend
                raise ManifestError(
                    "Failed to parse TOML manifest {}: {}".format(path, exc)
                ) from exc

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_yaml(path: Path) -> Dict[str, Any]:
        """Parse a YAML manifest file."""
        if not YAML_AVAILABLE:
            raise ManifestError(
                "Cannot parse YAML manifest {}: PyYAML is not installed.".format(
                    path
                )
            )
        with open(path, "r", encoding="utf-8") as fh:
            try:
                data = _yaml.safe_load(fh)
            except _yaml.YAMLError as exc:  # type: ignore[union-attr]
                raise ManifestError(
                    "Failed to parse YAML manifest {}: {}".format(path, exc)
                ) from exc
        return data or {}

    # ------------------------------------------------------------------
    @staticmethod
    def _build_spec(data: Dict[str, Any], manifest_path: Path) -> PluginSpec:
        """Build a :class:`PluginSpec` from a parsed manifest mapping."""
        unknown = set(data.keys()) - _ALLOWED_KEYS
        if unknown:
            raise ManifestError(
                "Unknown manifest key(s) in {}: {}. Allowed: {}.".format(
                    manifest_path, sorted(unknown), sorted(_ALLOWED_KEYS)
                )
            )

        name = str(data.get("name", "")).strip()
        version = str(data.get("version", "")).strip()
        author = str(data.get("author", "")).strip()
        description = str(data.get("description", "")).strip()

        if not name:
            raise ManifestError(
                "Manifest {} is missing required 'name' field.".format(
                    manifest_path
                )
            )

        try:
            spec = PluginSpec(
                name=name,
                version=version or "0.0.0",
                author=author or "unknown",
                description=description,
                license=str(data.get("license", "MIT") or "MIT").strip(),
                homepage=str(data.get("homepage", "") or "").strip(),
                dependencies=list(data.get("dependencies", []) or []),
                node_modules=list(data.get("node_modules", []) or []),
                on_load=str(data.get("on_load", "") or "").strip(),
                on_unload=str(data.get("on_unload", "") or "").strip(),
                source=SOURCE_DIRECTORY,
                path=manifest_path.parent,
            )
        except ValueError as exc:
            raise ManifestError(
                "Invalid manifest {}: {}".format(manifest_path, exc)
            ) from exc
        return spec

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def validate(spec: PluginSpec) -> List[str]:
        """Validate a :class:`PluginSpec` and return a list of errors.

        The validation is *non-throwing*: it collects every problem it
        finds into a list of human-readable strings.  An empty list
        means the spec is valid.

        Checks performed:

        * ``name`` is a non-empty Python-identifier-ish string.
        * ``version`` matches a minimal semver pattern.
        * ``author`` and ``description`` are non-empty.
        * ``license`` is a non-empty string.
        * ``dependencies`` and ``node_modules`` are lists of strings.
        * ``on_load`` / ``on_unload``, when set, look like dotted paths.
        * ``source``, when set, is a recognised value.

        Args:
            spec: The :class:`PluginSpec` to validate.

        Returns:
            A list of error strings; empty when the spec is valid.
        """
        errors: List[str] = []

        # name
        if not spec.name or not _NAME_RE.match(spec.name):
            errors.append(
                "name must be a non-empty Python-identifier string "
                "(letters, digits, underscore), got {!r}.".format(spec.name)
            )

        # version
        if not spec.version or not _SEMVER_RE.match(spec.version):
            errors.append(
                "version must look like 'X.Y.Z' (e.g. '0.1.0'), "
                "got {!r}.".format(spec.version)
            )

        # author
        if not spec.author.strip():
            errors.append("author must be a non-empty string.")

        # description
        if not spec.description.strip():
            errors.append("description must be a non-empty string.")

        # license
        if not spec.license.strip():
            errors.append("license must be a non-empty string.")

        # dependencies
        if not isinstance(spec.dependencies, list) or not all(
            isinstance(d, str) and d.strip() for d in spec.dependencies
        ):
            errors.append("dependencies must be a list of non-empty strings.")

        # node_modules
        if not isinstance(spec.node_modules, list) or not all(
            isinstance(m, str) and m.strip() for m in spec.node_modules
        ):
            errors.append("node_modules must be a list of non-empty strings.")

        # on_load / on_unload
        for field_name in ("on_load", "on_unload"):
            value = getattr(spec, field_name)
            if value and not _is_dotted_path(value):
                errors.append(
                    "{} must be a dotted path like 'hooks.on_load', "
                    "got {!r}.".format(field_name, value)
                )

        # source
        if spec.source and spec.source not in (
            "entry_point",
            "directory",
            "code",
        ):
            errors.append(
                "source must be one of 'entry_point', 'directory', 'code', "
                "got {!r}.".format(spec.source)
            )

        return errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_dotted_path(value: str) -> bool:
    """Return ``True`` when ``value`` looks like a dotted callable path.

    Accepts both ``module.sub.attr`` and ``module:attr`` forms.
    """
    if not value:
        return False
    # ``module:attr`` form.
    if ":" in value:
        mod, _, attr = value.partition(":")
        return bool(mod) and bool(attr) and _NAME_RE.match(attr) is not None
    # ``module.sub.attr`` form -- last segment is the callable.
    parts = value.split(".")
    return len(parts) >= 1 and all(
        _NAME_RE.match(p) is not None for p in parts if p
    )
