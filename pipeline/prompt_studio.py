"""Prompt studio for the TorchaVerse pipeline layer (L5).

This module provides the prompt-authoring toolkit that sits alongside the
pipeline composer:

* :class:`PromptTemplate` -- a renderable, serialisable prompt template
  treated as an :class:`~assets.base.Asset`-style artefact (it owns an id,
  name, tags and revision-free content).  Templates use ``{{var}}`` slots
  filled by simple string substitution.
* :class:`PromptEnhancer` -- augments a raw prompt with quality-boosting
  vocabulary according to a named :class:`StylePreset`, and loads a
  negative-prompt library from configuration.
* :class:`SeedManager` -- records and recalls generation seeds keyed by a
  prompt hash, enabling reproducible and searchable generation history.
* :class:`StylePreset` -- a dataclass describing a named style's positive
  / negative boost vocabulary and recommended sampler parameters.

The module is dependency-free with respect to :mod:`torch`; the only
optional integration point is :class:`~infrastructure.config_center.ConfigCenter`,
which :meth:`PromptEnhancer.load_negative_prompts` consults for a user-
supplied negative-prompt library, falling back to a built-in default set.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "PromptTemplate",
    "PromptEnhancer",
    "SeedManager",
    "StylePreset",
    "BUILTIN_STYLE_PRESETS",
]


# ---------------------------------------------------------------------------
# Module-level logger (stdlib only -- this layer must not import torch).
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger("pipeline.prompt_studio")


#: Regex matching ``{{var}}`` slots inside a prompt template.
_VAR_PATTERN: re.Pattern[str] = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

#: Default style used by :meth:`PromptEnhancer.enhance` when none is given.
_DEFAULT_STYLE: str = "cinematic"

#: Configuration key (under :class:`~infrastructure.config_center.ConfigCenter`)
#: from which the negative-prompt library is loaded.
_NEGATIVE_PROMPTS_CONFIG_KEY: str = "image.negative"

#: Upper bound on the bit-width of a generated seed (32-bit unsigned).
_SEED_BIT_WIDTH: int = 32
_SEED_MODULUS: int = 2 ** _SEED_BIT_WIDTH


# ---------------------------------------------------------------------------
# StylePreset
# ---------------------------------------------------------------------------
@dataclass
class StylePreset:
    """A named prompt-enhancement style.

    Attributes:
        name: Unique style name (e.g. ``"cinematic"``).
        positive_boost: Vocabulary appended to a prompt to boost quality in
            this style.
        negative_boost: Vocabulary added to the negative prompt for this
            style.
        recommended_steps: Suggested sampler step count.
        recommended_cfg: Suggested classifier-free-guidance scale.
    """

    name: str
    positive_boost: str
    negative_boost: str
    recommended_steps: int = 30
    recommended_cfg: float = 7.5

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this preset to a JSON-serialisable dictionary."""
        return {
            "name": self.name,
            "positive_boost": self.positive_boost,
            "negative_boost": self.negative_boost,
            "recommended_steps": self.recommended_steps,
            "recommended_cfg": self.recommended_cfg,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StylePreset":
        """Reconstruct a :class:`StylePreset` from a serialised dict."""
        return cls(
            name=d["name"],
            positive_boost=d.get("positive_boost", ""),
            negative_boost=d.get("negative_boost", ""),
            recommended_steps=int(d.get("recommended_steps", 30)),
            recommended_cfg=float(d.get("recommended_cfg", 7.5)),
        )

    def __repr__(self) -> str:
        return "StylePreset(name={!r}, steps={}, cfg={})".format(
            self.name, self.recommended_steps, self.recommended_cfg
        )


#: The immutable catalogue of built-in :class:`StylePreset` instances shipped
#: with the framework.  These are content defaults (not configuration) and are
#: intended to be extended, not replaced, by user configuration.
BUILTIN_STYLE_PRESETS: List[StylePreset] = [
    StylePreset(
        name="cinematic",
        positive_boost=(
            "cinematic lighting, dramatic composition, shallow depth of "
            "field, film grain, 8k, highly detailed, masterpiece"
        ),
        negative_boost=(
            "low quality, blurry, overexposed, flat lighting, watermark"
        ),
        recommended_steps=30,
        recommended_cfg=7.5,
    ),
    StylePreset(
        name="anime",
        positive_boost=(
            "anime style, vibrant colors, clean line art, detailed eyes, "
            "studio quality, high resolution"
        ),
        negative_boost=(
            "realistic, photo, 3d, low quality, deformed, extra fingers"
        ),
        recommended_steps=28,
        recommended_cfg=7.0,
    ),
    StylePreset(
        name="photoreal",
        positive_boost=(
            "photorealistic, ultra detailed, professional photography, "
            "natural lighting, sharp focus, 8k, dslr"
        ),
        negative_boost=(
            "illustration, painting, cartoon, low quality, blurry, noise"
        ),
        recommended_steps=40,
        recommended_cfg=8.0,
    ),
    StylePreset(
        name="digital_art",
        positive_boost=(
            "digital painting, concept art, trending on artstation, "
            "intricate detail, vivid colors, artstation hd"
        ),
        negative_boost=(
            "photo, low quality, jpeg artifacts, signature, text"
        ),
        recommended_steps=35,
        recommended_cfg=7.5,
    ),
    StylePreset(
        name="fantasy",
        positive_boost=(
            "epic fantasy, magical atmosphere, ethereal lighting, "
            "highly detailed, ornate, masterpiece"
        ),
        negative_boost=(
            "modern, mundane, low quality, blurry, deformed"
        ),
        recommended_steps=32,
        recommended_cfg=7.5,
    ),
]


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------
class PromptTemplate:
    """A renderable, serialisable prompt template (an asset-like artefact).

    A :class:`PromptTemplate` owns a ``template`` string containing
    ``{{var}}`` slots.  The declared ``variables`` list documents which slots
    are expected; :meth:`render` performs simple string substitution and
    raises if a required variable is missing.

    Although it does not inherit from :class:`~assets.base.Asset` (to keep the
    pipeline layer free of the asset-store dependency), it follows the same
    ``to_dict`` / ``from_dict`` serialisation contract so it can be stored as
    an :class:`~assets.types.AssetType.PROMPT_TEMPLATE` asset by the L2 layer.

    Attributes:
        id: Unique template identifier.
        name: Human-readable display name.
        template: The template string with ``{{var}}`` slots.
        variables: Declared slot names (auto-derived from ``template`` when
            not supplied).
        negative: Whether this is a negative-prompt template.
        tags: Free-form tags.
    """

    def __init__(
        self,
        id: str,
        name: str,
        template: str,
        variables: Optional[List[str]] = None,
        negative: bool = False,
        tags: Optional[List[str]] = None,
    ) -> None:
        if not id or not isinstance(id, str):
            raise ValueError("PromptTemplate 'id' must be a non-empty string.")
        if not name or not isinstance(name, str):
            raise ValueError("PromptTemplate 'name' must be a non-empty string.")
        if not isinstance(template, str):
            raise TypeError("PromptTemplate 'template' must be a string.")

        self.id: str = id
        self.name: str = name
        self.template: str = template
        self.negative: bool = bool(negative)
        self.tags: List[str] = list(tags) if tags else []
        self.variables: List[str] = (
            list(variables) if variables is not None else self._extract_variables(template)
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_variables(template: str) -> List[str]:
        """Return the ordered, de-duplicated list of ``{{var}}`` slots."""
        seen: set[str] = set()
        variables: List[str] = []
        for match in _VAR_PATTERN.finditer(template):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                variables.append(name)
        return variables

    # ------------------------------------------------------------------
    def render(self, **kwargs: Any) -> str:
        """Render the template by substituting ``{{var}}`` slots.

        Args:
            **kwargs: Values for the template variables.

        Returns:
            The rendered prompt string.

        Raises:
            ValueError: If a declared variable has no value supplied.
        """
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise ValueError(
                "Missing template variables: {}".format(", ".join(missing))
            )

        def _replace(match: re.Match[str]) -> str:
            return str(kwargs.get(match.group(1), match.group(0)))

        return _VAR_PATTERN.sub(_replace, self.template)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise this template to a JSON-serialisable dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "template": self.template,
            "variables": list(self.variables),
            "negative": self.negative,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PromptTemplate":
        """Reconstruct a :class:`PromptTemplate` from a serialised dict."""
        return cls(
            id=d["id"],
            name=d["name"],
            template=d["template"],
            variables=list(d.get("variables") or []),
            negative=bool(d.get("negative", False)),
            tags=list(d.get("tags") or []),
        )

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return "PromptTemplate(id={!r}, name={!r}, vars={})".format(
            self.id, self.name, self.variables
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PromptTemplate):
            return NotImplemented
        return self.id == other.id and self.template == other.template

    def __hash__(self) -> int:
        return hash((self.id, self.template))


# ---------------------------------------------------------------------------
# PromptEnhancer
# ---------------------------------------------------------------------------
class PromptEnhancer:
    """Augments raw prompts with style-specific quality vocabulary.

    The enhancer maintains a registry of :class:`StylePreset` instances.  A
    raw prompt is enhanced by appending the preset's ``positive_boost`` (or
    prepending the ``negative_boost`` for negative prompts).  The negative-
    prompt library can be loaded from :class:`~infrastructure.config_center.ConfigCenter`
    or from a built-in default.

    Args:
        presets: Optional explicit list of :class:`StylePreset`.  When
            ``None`` the built-in catalogue (:data:`BUILTIN_STYLE_PRESETS`)
            is used.
        config_center: Optional :class:`~infrastructure.config_center.ConfigCenter`
            used by :meth:`load_negative_prompts`.
    """

    def __init__(
        self,
        presets: Optional[List[StylePreset]] = None,
        config_center: Any = None,
    ) -> None:
        self._presets: Dict[str, StylePreset] = {}
        self._config_center = config_center
        self._lock: threading.RLock = threading.RLock()
        for preset in (presets if presets is not None else BUILTIN_STYLE_PRESETS):
            self.register_preset(preset)

    # ------------------------------------------------------------------
    # Preset management
    # ------------------------------------------------------------------
    def register_preset(self, preset: StylePreset) -> None:
        """Register (or replace) a :class:`StylePreset`."""
        with self._lock:
            self._presets[preset.name.lower()] = preset

    def get_preset(self, style: str) -> StylePreset:
        """Return the :class:`StylePreset` named ``style``.

        Raises:
            KeyError: If the style is not registered.
        """
        with self._lock:
            preset = self._presets.get(style.lower())
        if preset is None:
            raise KeyError("Unknown style preset {!r}.".format(style))
        return preset

    def list_styles(self) -> List[str]:
        """Return the names of all registered styles (sorted)."""
        with self._lock:
            return sorted(self._presets.keys())

    # ------------------------------------------------------------------
    # Enhancement
    # ------------------------------------------------------------------
    def enhance(self, prompt: str, style: str = _DEFAULT_STYLE) -> str:
        """Enhance ``prompt`` with the vocabulary of ``style``.

        The style's ``positive_boost`` is appended to the prompt, separated
        by a comma.  If the style is unknown the prompt is returned unchanged
        (with a debug log) so that enhancement never hard-fails.

        Args:
            prompt: The raw prompt to enhance.
            style: The name of the :class:`StylePreset` to apply.

        Returns:
            The enhanced prompt string.
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")
        with self._lock:
            preset = self._presets.get(style.lower())
        if preset is None:
            _logger.debug("Unknown style %r; returning prompt unchanged.", style)
            return prompt
        boost = preset.positive_boost.strip()
        if not boost:
            return prompt
        if not prompt.strip():
            return boost
        return "{}, {}".format(prompt.strip().rstrip(","), boost)

    def enhance_negative(self, prompt: str, style: str = _DEFAULT_STYLE) -> str:
        """Enhance a negative prompt with the style's ``negative_boost``.

        Args:
            prompt: The raw negative prompt.
            style: The style whose ``negative_boost`` to apply.

        Returns:
            The enhanced negative prompt string.
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")
        with self._lock:
            preset = self._presets.get(style.lower())
        if preset is None:
            return prompt
        boost = preset.negative_boost.strip()
        if not boost:
            return prompt
        if not prompt.strip():
            return boost
        return "{}, {}".format(prompt.strip().rstrip(","), boost)

    # ------------------------------------------------------------------
    # Negative-prompt library
    # ------------------------------------------------------------------
    def load_negative_prompts(self) -> List[str]:
        """Load the negative-prompt library.

        The library is read from :class:`~infrastructure.config_center.ConfigCenter`
        (when configured) under the ``image.negative`` key, which typically
        holds a comma- or newline-separated string.  When no configuration
        is available a built-in default list is returned.

        Returns:
            A list of negative-prompt tokens.
        """
        raw: Any = None
        if self._config_center is not None:
            try:
                raw = self._config_center.get(_NEGATIVE_PROMPTS_CONFIG_KEY)
            except Exception:  # pragma: no cover - defensive
                _logger.debug("ConfigCenter lookup failed for negative prompts.", exc_info=True)
                raw = None
        if raw is None:
            return list(_DEFAULT_NEGATIVE_PROMPTS)
        if isinstance(raw, (list, tuple)):
            return [str(item).strip() for item in raw if str(item).strip()]
        if isinstance(raw, str):
            # Split on commas and newlines.
            parts = re.split(r"[,\n]", raw)
            return [p.strip() for p in parts if p.strip()]
        return list(_DEFAULT_NEGATIVE_PROMPTS)

    def __repr__(self) -> str:
        return "PromptEnhancer(styles={})".format(len(self._presets))


#: Built-in default negative-prompt library used when no configuration is
#: available.  Mirrors the ``image.negative`` entry shipped in
#: ``config/prompt_templates.yaml``.
_DEFAULT_NEGATIVE_PROMPTS: List[str] = [
    "low quality",
    "blurry",
    "distorted",
    "watermark",
    "signature",
    "extra limbs",
    "bad anatomy",
    "jpeg artifacts",
]


# ---------------------------------------------------------------------------
# SeedManager
# ---------------------------------------------------------------------------
class SeedManager:
    """Records and recalls generation seeds keyed by a prompt hash.

    Every ``(prompt, seed, model, params)`` tuple is stored under the sha256
    hash of the prompt text, so that a given prompt's generation history can
    be replayed or audited later.  The store is thread-safe and in-memory;
    persistence is left to the caller (the records are plain dicts).

    Args:
        config_center: Optional :class:`~infrastructure.config_center.ConfigCenter`
            (reserved for future persistence hooks).
    """

    def __init__(self, config_center: Any = None) -> None:
        self._records: Dict[str, List[Dict[str, Any]]] = {}
        self._config_center = config_center
        self._lock: threading.RLock = threading.RLock()
        self._rng_seed: int = 0

    # ------------------------------------------------------------------
    @staticmethod
    def prompt_hash(prompt: str) -> str:
        """Return the sha256 hex digest of ``prompt`` (the storage key)."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    def record(
        self,
        prompt: str,
        seed: int,
        model: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a generation seed for ``prompt``.

        Args:
            prompt: The prompt text that was used.
            seed: The generation seed.
            model: The model identifier that was used.
            params: Optional sampler / generation parameters.

        Returns:
            The stored record dictionary.
        """
        record: Dict[str, Any] = {
            "prompt": prompt,
            "prompt_hash": self.prompt_hash(prompt),
            "seed": int(seed),
            "model": str(model),
            "params": dict(params) if params else {},
            "timestamp": _now_iso(),
        }
        with self._lock:
            self._records.setdefault(record["prompt_hash"], []).append(record)
        return record

    def recall(self, prompt_hash: str) -> List[Dict[str, Any]]:
        """Return all records stored under ``prompt_hash``.

        Args:
            prompt_hash: A sha256 hex digest (see :meth:`prompt_hash`) or a
                raw prompt text (hashed automatically).

        Returns:
            A list of matching record dictionaries (newest last).
        """
        key = prompt_hash
        # Auto-hash raw prompts (a 64-char hex digest is unlikely to be a
        # real prompt, so we hash anything shorter).
        if len(prompt_hash) != 64 or not re.fullmatch(r"[0-9a-f]{64}", prompt_hash):
            key = self.prompt_hash(prompt_hash)
        with self._lock:
            return [dict(r) for r in self._records.get(key, [])]

    def random_seed(self) -> int:
        """Return a non-negative 32-bit random seed.

        Uses :mod:`secrets` when available for cryptographic quality,
        falling back to :mod:`random` otherwise.
        """
        try:
            import secrets

            value = secrets.randbelow(_SEED_MODULUS)
        except Exception:  # pragma: no cover - defensive
            import random

            value = random.randrange(0, _SEED_MODULUS)
        return int(value)

    # ------------------------------------------------------------------
    def count(self) -> int:
        """Return the total number of recorded seeds."""
        with self._lock:
            return sum(len(records) for records in self._records.values())

    def clear(self) -> None:
        """Forget every recorded seed."""
        with self._lock:
            self._records.clear()

    def __repr__(self) -> str:
        return "SeedManager(records={}, prompts={})".format(
            self.count(), len(self._records)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (stdlib only)."""
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat()
