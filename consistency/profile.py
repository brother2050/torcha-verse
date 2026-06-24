"""Consistency configuration profiles for the TorchaVerse consistency
framework (v0.3.0).

This module defines the *configuration* surface of the consistency
framework: a :class:`ConsistencyProfile` dataclass that captures the
per-axis weights and temporal-consistency knobs, and a
:class:`ConsistencyManager` that provides CRUD operations on named
profiles plus a bridge to the L4 node system via :meth:`apply_to_node`.

A :class:`ConsistencyProfile` is consumed by the
:class:`~consistency.pipeline.ConsistencyPipeline` to decide how much
weight to give each conditioning signal (character / outfit / scene /
depth) when generating an image or video.  For video generation the
profile also controls:

* :attr:`ConsistencyProfile.temporal_consistency` -- whether temporal
  coherence is enforced across frames.
* :attr:`ConsistencyProfile.drift_threshold` -- the CLIP-I frame-to-frame
  distance above which a drift is flagged.
* :attr:`ConsistencyProfile.reframe_interval` -- how often (in frames)
  the pipeline re-writes the conditioning features to prevent drift.

The :class:`ConsistencyManager` keeps an in-memory registry of named
profiles, guarded by a :class:`threading.Lock`, and supports
serialising / deserialising profiles to and from JSON files.

Layering (L1 -> L4):

* L1 ``infrastructure`` -- logging.
* L2 ``assets`` -- asset types (referenced for type hints only).
* L4 ``consistency`` (this module) -- profile configuration.

This module is torch-free; it only depends on the standard library and
the L1 logging layer.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from infrastructure.logger import get_logger

__all__ = ["ConsistencyProfile", "ConsistencyManager"]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Default character-consistency weight.
_DEFAULT_CHARACTER_WEIGHT: float = 0.8

#: Default outfit-consistency weight.
_DEFAULT_OUTFIT_WEIGHT: float = 0.7

#: Default scene-consistency weight.
_DEFAULT_SCENE_WEIGHT: float = 0.6

#: Default depth-conditioning weight.
_DEFAULT_DEPTH_WEIGHT: float = 0.5

#: Default temporal-consistency flag (off for single-image generation).
_DEFAULT_TEMPORAL_CONSISTENCY: bool = False

#: Default CLIP-I drift-detection threshold.
_DEFAULT_DRIFT_THRESHOLD: float = 0.15

#: Default reframe interval (frames between feature re-writes).
_DEFAULT_REFRAME_INTERVAL: int = 8

#: Lower bound for consistency weights.
_WEIGHT_MIN: float = 0.0

#: Upper bound for consistency weights.
_WEIGHT_MAX: float = 1.0

#: Lower bound for the drift threshold.
_DRIFT_MIN: float = 0.0

#: Upper bound for the drift threshold.
_DRIFT_MAX: float = 1.0

#: Lower bound for the reframe interval (frames).
_REFRAME_MIN: int = 1

#: JSON indentation used when serialising profiles to files.
_JSON_INDENT: int = 2

#: Module-level logger.
_logger = get_logger("consistency.profile")


# ---------------------------------------------------------------------------
# ConsistencyProfile
# ---------------------------------------------------------------------------
@dataclass
class ConsistencyProfile:
    """Configuration profile for the consistency pipeline.

    A profile captures the per-axis weights and temporal-consistency
    knobs that the :class:`~consistency.pipeline.ConsistencyPipeline`
    uses to combine character / outfit / scene / depth conditioning
    signals.  Every weight is a float in ``[0, 1]``; higher values mean
    the corresponding conditioning signal is applied more strongly.

    Attributes:
        character_weight: Weight for character-identity conditioning
            (IP-Adapter).  Defaults to ``0.8``.
        outfit_weight: Weight for outfit / garment conditioning
            (LoRA + style embedding).  Defaults to ``0.7``.
        scene_weight: Weight for scene / environment conditioning
            (scene LoRA + ControlNet).  Defaults to ``0.6``.
        depth_weight: Weight for depth-map conditioning
            (ControlNet depth).  Defaults to ``0.5``.
        temporal_consistency: Whether to enforce temporal coherence
            across video frames.  Defaults to ``False`` (single-image).
        drift_threshold: CLIP-I frame-to-frame distance above which a
            drift is flagged.  Defaults to ``0.15``.
        reframe_interval: Number of frames between feature re-writes
            during video generation.  Defaults to ``8``.
    """

    character_weight: float = _DEFAULT_CHARACTER_WEIGHT
    outfit_weight: float = _DEFAULT_OUTFIT_WEIGHT
    scene_weight: float = _DEFAULT_SCENE_WEIGHT
    depth_weight: float = _DEFAULT_DEPTH_WEIGHT
    temporal_consistency: bool = _DEFAULT_TEMPORAL_CONSISTENCY
    drift_threshold: float = _DEFAULT_DRIFT_THRESHOLD
    reframe_interval: int = _DEFAULT_REFRAME_INTERVAL

    def __post_init__(self) -> None:
        """Validate the profile fields after dataclass initialisation."""
        self._validate_weight(
            "character_weight", self.character_weight
        )
        self._validate_weight(
            "outfit_weight", self.outfit_weight
        )
        self._validate_weight(
            "scene_weight", self.scene_weight
        )
        self._validate_weight(
            "depth_weight", self.depth_weight
        )
        self._validate_range(
            "drift_threshold",
            self.drift_threshold,
            _DRIFT_MIN,
            _DRIFT_MAX,
        )
        self._validate_range(
            "reframe_interval",
            self.reframe_interval,
            _REFRAME_MIN,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_weight(name: str, value: float) -> None:
        """Validate that a weight is a float in ``[0, 1]``."""
        if not isinstance(value, (int, float)):
            raise TypeError(
                "{} must be a float, got {}.".format(
                    name, type(value).__name__
                )
            )
        if isinstance(value, bool):
            raise TypeError(
                "{} must be a float, got bool.".format(name)
            )
        if value < _WEIGHT_MIN or value > _WEIGHT_MAX:
            raise ValueError(
                "{} must be in [{}, {}], got {}.".format(
                    name, _WEIGHT_MIN, _WEIGHT_MAX, value
                )
            )

    @staticmethod
    def _validate_range(
        name: str, value: float, lo: float, hi: Optional[float] = None
    ) -> None:
        """Validate that ``value`` is within ``[lo, hi]`` (or ``>= lo``)."""
        if not isinstance(value, (int, float)):
            raise TypeError(
                "{} must be a number, got {}.".format(
                    name, type(value).__name__
                )
            )
        if isinstance(value, bool):
            raise TypeError(
                "{} must be a number, got bool.".format(name)
            )
        if value < lo:
            raise ValueError(
                "{} must be >= {}, got {}.".format(name, lo, value)
            )
        if hi is not None and value > hi:
            raise ValueError(
                "{} must be <= {}, got {}.".format(name, hi, value)
            )

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise this profile to a JSON-serialisable dictionary.

        Returns:
            A dictionary with all seven profile fields.
        """
        return {
            "character_weight": self.character_weight,
            "outfit_weight": self.outfit_weight,
            "scene_weight": self.scene_weight,
            "depth_weight": self.depth_weight,
            "temporal_consistency": self.temporal_consistency,
            "drift_threshold": self.drift_threshold,
            "reframe_interval": self.reframe_interval,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConsistencyProfile":
        """Reconstruct a :class:`ConsistencyProfile` from a serialised dict.

        Missing keys fall back to the dataclass defaults.

        Args:
            d: Dictionary produced by :meth:`to_dict` (or a subset).

        Returns:
            A new :class:`ConsistencyProfile` instance.
        """
        return cls(
            character_weight=float(
                d.get("character_weight", _DEFAULT_CHARACTER_WEIGHT)
            ),
            outfit_weight=float(
                d.get("outfit_weight", _DEFAULT_OUTFIT_WEIGHT)
            ),
            scene_weight=float(
                d.get("scene_weight", _DEFAULT_SCENE_WEIGHT)
            ),
            depth_weight=float(
                d.get("depth_weight", _DEFAULT_DEPTH_WEIGHT)
            ),
            temporal_consistency=bool(
                d.get("temporal_consistency", _DEFAULT_TEMPORAL_CONSISTENCY)
            ),
            drift_threshold=float(
                d.get("drift_threshold", _DEFAULT_DRIFT_THRESHOLD)
            ),
            reframe_interval=int(
                d.get("reframe_interval", _DEFAULT_REFRAME_INTERVAL)
            ),
        )

    def __repr__(self) -> str:
        return (
            "ConsistencyProfile(character={:.2f}, outfit={:.2f}, "
            "scene={:.2f}, depth={:.2f}, temporal={}, "
            "drift={:.2f}, reframe={})".format(
                self.character_weight,
                self.outfit_weight,
                self.scene_weight,
                self.depth_weight,
                self.temporal_consistency,
                self.drift_threshold,
                self.reframe_interval,
            )
        )


# ---------------------------------------------------------------------------
# ConsistencyManager
# ---------------------------------------------------------------------------
class ConsistencyManager:
    """Registry and factory for named :class:`ConsistencyProfile` instances.

    The manager keeps an in-memory dictionary of named profiles, guarded
    by a :class:`threading.Lock` so that it is safe to share across
    threads.  It supports creating, retrieving, listing, saving and
    loading profiles, and provides a bridge to the L4 node system via
    :meth:`apply_to_node`.

    Example::

        mgr = ConsistencyManager()
        profile = mgr.create_profile("cinematic",
                                      character_weight=0.9,
                                      temporal_consistency=True)
        node_inputs = mgr.apply_to_node(my_node, profile)

    Args:
        profiles: Optional initial mapping of name -> profile.  When
            ``None`` an empty registry is created.
    """

    def __init__(
        self,
        profiles: Optional[Dict[str, ConsistencyProfile]] = None,
    ) -> None:
        self._profiles: Dict[str, ConsistencyProfile] = dict(
            profiles
        ) if profiles else {}
        self._lock: threading.Lock = threading.Lock()
        self._logger = _logger

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def create_profile(
        self, name: str, **kwargs: Any
    ) -> ConsistencyProfile:
        """Create and register a new named profile.

        Args:
            name: Unique profile name.
            **kwargs: Keyword arguments forwarded to the
                :class:`ConsistencyProfile` constructor.

        Returns:
            The newly created :class:`ConsistencyProfile`.

        Raises:
            ValueError: If ``name`` is empty or already registered.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Profile name must be a non-empty string.")
        profile = ConsistencyProfile(**kwargs)
        with self._lock:
            if name in self._profiles:
                raise ValueError(
                    "Profile {!r} already exists.".format(name)
                )
            self._profiles[name] = profile
        self._logger.debug("Created profile %r.", name)
        return profile

    def get_profile(self, name: str) -> ConsistencyProfile:
        """Retrieve a previously registered profile by name.

        Args:
            name: The profile name.

        Returns:
            The :class:`ConsistencyProfile` registered as ``name``.

        Raises:
            KeyError: If no profile is registered for ``name``.
        """
        with self._lock:
            if name not in self._profiles:
                raise KeyError(
                    "No profile registered for {!r}.".format(name)
                )
            return self._profiles[name]

    def list_profiles(self) -> List[str]:
        """Return the names of all registered profiles, sorted.

        Returns:
            A sorted list of profile names.
        """
        with self._lock:
            return sorted(self._profiles.keys())

    def remove_profile(self, name: str) -> bool:
        """Remove a profile from the registry.

        Args:
            name: The profile name to remove.

        Returns:
            ``True`` if the profile was found and removed, ``False``
            otherwise.
        """
        with self._lock:
            if name in self._profiles:
                del self._profiles[name]
                self._logger.debug("Removed profile %r.", name)
                return True
            return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_profile(
        self, name: str, path: Union[str, Path]
    ) -> Path:
        """Serialise a named profile to a JSON file.

        Args:
            name: The profile name to save.
            path: Destination file path.

        Returns:
            The resolved :class:`pathlib.Path` of the written file.

        Raises:
            KeyError: If no profile is registered for ``name``.
        """
        profile = self.get_profile(name)
        dest = Path(path).expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": name,
            "profile": profile.to_dict(),
        }
        dest.write_text(
            json.dumps(payload, indent=_JSON_INDENT, ensure_ascii=False),
            encoding="utf-8",
        )
        self._logger.debug("Saved profile %r to %s.", name, dest)
        return dest

    def load_profile(
        self, path: Union[str, Path], name: Optional[str] = None
    ) -> ConsistencyProfile:
        """Load a profile from a JSON file and register it.

        Args:
            path: Source file path (written by :meth:`save_profile`).
            name: Optional override for the registered name.  When
                ``None`` the name stored in the file is used.

        Returns:
            The loaded :class:`ConsistencyProfile`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file is malformed or the name is already
                registered.
        """
        src = Path(path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(
                "Profile file not found: {}".format(src)
            )
        data = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(
                "Profile file {} is not a JSON object.".format(src)
            )
        profile_dict = data.get("profile", data)
        registered_name = name or data.get("name")
        if not registered_name:
            raise ValueError(
                "Profile file {} has no 'name' field and no explicit "
                "name was provided.".format(src)
            )
        profile = ConsistencyProfile.from_dict(profile_dict)
        with self._lock:
            if registered_name in self._profiles:
                raise ValueError(
                    "Profile {!r} already exists; use a different name "
                    "or remove it first.".format(registered_name)
                )
            self._profiles[registered_name] = profile
        self._logger.debug("Loaded profile %r from %s.", registered_name, src)
        return profile

    # ------------------------------------------------------------------
    # Node bridge
    # ------------------------------------------------------------------
    def apply_to_node(
        self,
        node: Any,
        profile: ConsistencyProfile,
    ) -> Dict[str, Any]:
        """Convert a :class:`ConsistencyProfile` into node input parameters.

        The returned dictionary contains the profile's weights and
        temporal knobs keyed by the names the L4 consistency nodes
        (``character_apply``, ``outfit_apply``, ``scene_apply``,
        ``depth_condition``) expect.  When ``node`` is a
        :class:`~nodes.base.BaseNode` with a ``spec`` attribute, only
        the inputs declared in the spec are included; otherwise all
        profile-derived parameters are returned.

        Args:
            node: A :class:`~nodes.base.BaseNode` instance, a node
                type string, or any object with a ``spec.inputs`` dict.
            profile: The :class:`ConsistencyProfile` to convert.

        Returns:
            A dictionary of node input parameters derived from the
            profile.
        """
        params: Dict[str, Any] = {
            "character_weight": profile.character_weight,
            "outfit_weight": profile.outfit_weight,
            "scene_weight": profile.scene_weight,
            "depth_weight": profile.depth_weight,
            "temporal_consistency": profile.temporal_consistency,
            "drift_threshold": profile.drift_threshold,
            "reframe_interval": profile.reframe_interval,
        }

        # When the node exposes a spec with declared inputs, filter to
        # only the relevant keys so that unknown parameters are not
        # passed through.
        spec = getattr(node, "spec", None)
        declared_inputs = getattr(spec, "inputs", None)
        if isinstance(declared_inputs, dict) and declared_inputs:
            return {
                k: v
                for k, v in params.items()
                if k in declared_inputs
            }
        return params

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        with self._lock:
            count = len(self._profiles)
        return "ConsistencyManager(profiles={})".format(count)
