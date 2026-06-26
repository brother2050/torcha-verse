"""Abstract :class:`BaseNode` and the resource-estimation coefficients.

The estimation coefficients (base VRAM / RAM / time, megapixel
scalars, per-step time) are module-level overridable constants
so subclasses can re-tune them per domain.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Dict, List

from ._context import NodeContext
from ._constants import _logger
from ._spec import NodeSpec

# Re-export the type checker helper for callers that import
# ``is_optional`` from the old ``nodes.base`` location.
from ..type_system import is_optional  # noqa: F401

__all__ = [
    "BaseNode",
    # Estimation coefficients (overridable by subclasses).
    "_BASE_VRAM_GB",
    "_BASE_RAM_GB",
    "_BASE_TIME_S",
    "_MEGAPIXEL_PIXELS",
    "_VRAM_PER_MEGAPIXEL_GB",
    "_RAM_PER_MEGAPIXEL_GB",
    "_REFERENCE_PIXELS",
    "_TIME_PER_STEP_S",
]


# ---------------------------------------------------------------------------
# Estimation coefficients (module-level, overridable by subclasses)
# ---------------------------------------------------------------------------
#: Base VRAM overhead (GB) assumed for any node before scaling.
_BASE_VRAM_GB: float = 0.5
#: Base host-RAM overhead (GB) assumed for any node before scaling.
_BASE_RAM_GB: float = 0.25
#: Base wall-clock time (s) assumed for any node before scaling.
_BASE_TIME_S: float = 1.0
#: Number of pixels in one megapixel (used to normalise spatial estimates).
_MEGAPIXEL_PIXELS: float = 1_000_000.0
#: Additional VRAM (GB) per megapixel of output resolution.
_VRAM_PER_MEGAPIXEL_GB: float = 0.25
#: Additional host RAM (GB) per megapixel of output resolution.
_RAM_PER_MEGAPIXEL_GB: float = 0.10
#: Reference resolution (512x512) used to normalise per-step time.
_REFERENCE_PIXELS: float = 512.0 * 512.0
#: Wall-clock seconds per denoising step at the reference resolution.
_TIME_PER_STEP_S: float = 0.05


class BaseNode(abc.ABC):
    """Abstract base class for every TorchaVerse capability node.

    A node is the smallest unit of generative capability.  Subclasses
    declare their contract through the ``spec`` class attribute (a
    :class:`NodeSpec`) and implement :meth:`execute`.  The base class
    provides real, reusable implementations of
    :meth:`validate_inputs` and :meth:`estimate_resources` that
    operate on ``spec.inputs``; subclasses typically extend them
    with domain-specific checks.

    Class attributes:
        spec: The :class:`NodeSpec` describing this node.  Subclasses
            *must* assign a :class:`NodeSpec` instance.
    """

    #: Declarative node contract.  Subclasses assign a :class:`NodeSpec`.
    spec: ClassVar[NodeSpec]

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def execute(self, ctx: NodeContext, **inputs: Any) -> Dict[str, Any]:
        """Run the node and return its outputs.

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: Keyword inputs matching ``spec.inputs``.

        Returns:
            A dictionary mapping output names (per ``spec.outputs``)
            to their produced values.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Safe-execute wrapper (S2-4)
    # ------------------------------------------------------------------
    def _safe_execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Wrap :meth:`execute` with consistent error logging.

        The wrapper validates inputs first (failure raises
        :class:`ValueError` and is *not* re-logged as an execution
        error) and, on any exception raised by :meth:`execute`, logs
        an error-level message on the context's logger (or the
        module-level :data:`_logger` as a fallback) before
        re-raising.  The pipeline layer can then preserve partial
        results (R0-7).

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: Same keyword inputs as :meth:`execute`.

        Returns:
            The node's output dict.

        Raises:
            Exception: Whatever :meth:`execute` raised.
        """
        # Validate inputs first; a validation failure raises
        # ValueError *outside* the try-block, so it is not logged
        # as an "execution failure".
        errors = self.validate_inputs(inputs)
        if errors:
            raise ValueError(
                "Input validation failed: {}".format(errors)
            )
        try:
            return self.execute(ctx, **inputs)
        except Exception as exc:
            logger = getattr(ctx, "logger", None) or _logger
            logger.error(
                "节点 %s 执行失败 (%s): %s",
                getattr(self.spec, "type", self.__class__.__name__),
                type(exc).__name__,
                exc,
            )
            raise

    # ------------------------------------------------------------------
    # Reusable validation
    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate ``inputs`` against :attr:`spec`.

        The base implementation checks that every *required* input
        (one whose declared type string is not wrapped in
        ``Optional[...]``) is present and not ``None``.  Because
        port types are opaque strings (e.g. ``"IMAGE"``, ``"INT"``),
        runtime ``isinstance`` checks are no longer possible; the
        value is only checked for ``None``.  Unknown inputs are
        ignored (lenient) so pipelines can pass extra metadata
        through a node without erroring.

        Subclasses are expected to call
        ``super().validate_inputs(inputs)`` first and then append
        any domain-specific errors.

        Args:
            inputs: The input dictionary to validate.

        Returns:
            A list of human-readable error strings; empty when
            valid.
        """
        errors: List[str] = []
        spec = self.spec
        for name, type_str in spec.inputs.items():
            optional = is_optional(type_str)
            if name not in inputs:
                if not optional:
                    errors.append(
                        "Missing required input {!r} for node {!r}.".format(
                            name, spec.type
                        )
                    )
                continue
            value = inputs[name]
            if value is None and not optional:
                errors.append(
                    "Required input {!r} for node {!r} is None.".format(
                        name, spec.type
                    )
                )
        return errors

    # ------------------------------------------------------------------
    # Reusable resource estimation
    # ------------------------------------------------------------------
    def estimate_resources(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        """Estimate the resources this node would consume for ``inputs``.

        Returns a dictionary with three keys:

        * ``vram_gb`` -- estimated GPU memory in gigabytes.
        * ``ram_gb`` -- estimated host memory in gigabytes.
        * ``time_s`` -- estimated wall-clock time in seconds.

        The base implementation applies a generic heuristic: a
        small base overhead plus, when the inputs carry spatial
        dimensions (``width`` / ``height``) and a step count
        (``steps``), a pixel-and-step scaling term.  Subclasses
        override with domain-specific formulas (see e.g.
        :class:`nodes.image.ImageTxt2ImgNode`).

        Args:
            inputs: The input dictionary the node would be executed
                with.

        Returns:
            A ``{"vram_gb", "ram_gb", "time_s"}`` dictionary.
        """
        vram_gb: float = _BASE_VRAM_GB
        ram_gb: float = _BASE_RAM_GB
        time_s: float = _BASE_TIME_S

        width = inputs.get("width")
        height = inputs.get("height")
        steps = inputs.get("steps")
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            pixels = float(width) * float(height)
            megapixels = pixels / (_MEGAPIXEL_PIXELS)
            vram_gb += megapixels * _VRAM_PER_MEGAPIXEL_GB
            ram_gb += megapixels * _RAM_PER_MEGAPIXEL_GB
            if isinstance(steps, (int, float)) and steps > 0:
                time_s += float(steps) * _TIME_PER_STEP_S * (
                    pixels / (_REFERENCE_PIXELS)
                )

        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return "<{cls} type={type!r} name={name!r}>".format(
            cls=self.__class__.__name__,
            type=self.spec.type,
            name=self.spec.name,
        )
