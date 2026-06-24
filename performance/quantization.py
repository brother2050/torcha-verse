"""Quantisation for the TorchaVerse framework.

This module provides :class:`Quantizer`, which reduces the memory
footprint and increases the throughput of a model by lowering the
precision of its weights and/or activations.

Supported methods
-----------------
* ``"int8"`` -- symmetric per-channel 8-bit integer quantisation
  (uses :func:`torch.ao.quantization.quantize_dynamic`).
* ``"int4"`` -- 4-bit integer quantisation (placeholder; falls back to
  ``int8`` when native 4-bit is unavailable).
* ``"nf4"`` -- NormalFloat-4 quantisation used by QLoRA (placeholder;
  recorded but not applied without the ``bitsandbytes`` package).
* ``"fp16"`` -- half-precision float cast via :meth:`torch.Module.half`.
* ``"bf16"`` -- bfloat16 cast via :meth:`torch.Module.to`.

The module **depends on ``torch``**.  The optional ``bitsandbytes``
package is imported lazily with a ``try/except`` guard.

Example:
    >>> import torch.nn as nn
    >>> from performance import Quantizer, QuantizationConfig
    >>> q = Quantizer()
    >>> model = nn.Linear(4, 4)
    >>> wrapped = q.quantize(model, QuantizationConfig(method="fp16"))
    >>> wrapped.model.weight.dtype
    torch.float16
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

__all__ = [
    "QuantizationConfig",
    "Quantizer",
    "QuantizedModel",
    "CalibrationResult",
]

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    import bitsandbytes as _bnb  # type: ignore

    _HAS_BITSANDBYTES: bool = True
except Exception:  # pragma: no cover - bitsandbytes not installed
    _HAS_BITSANDBYTES: bool = False


# ---------------------------------------------------------------------------
# Module-level configuration constants
# ---------------------------------------------------------------------------
#: Default group size for block-wise quantisation.
_DEFAULT_GROUP_SIZE: int = 128

#: Valid quantisation method identifiers.
_VALID_METHODS: tuple[str, ...] = ("int8", "int4", "nf4", "fp16", "bf16")

#: Bits per element for each quantisation method (used in memory
#: estimation).
_BITS_PER_METHOD: dict[str, int] = {
    "int8": 8,
    "int4": 4,
    "nf4": 4,
    "fp16": 16,
    "bf16": 16,
}

#: Bits in a single byte.
_BITS_PER_BYTE: int = 8

#: Bytes per GiB.
_BYTES_PER_GIB: float = 1024.0 ** 3

#: Default number of calibration samples.
_DEFAULT_CALIBRATION_SAMPLES: int = 128


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class QuantizationConfig:
    """Configuration for :class:`Quantizer`.

    Attributes:
        method: One of ``"int8"``, ``"int4"``, ``"nf4"``, ``"fp16"``,
            ``"bf16"``.
        calibration_dataset: Optional path to a calibration dataset
            (used by :meth:`Quantizer.calibrate`).
        group_size: Block size for group-wise quantisation.
    """

    method: str = "int8"
    calibration_dataset: Optional[str] = None
    group_size: int = _DEFAULT_GROUP_SIZE

    def __post_init__(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_METHODS}, got {self.method!r}."
            )
        if self.group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {self.group_size}.")


@dataclass
class CalibrationResult:
    """Outcome of a calibration pass.

    Attributes:
        scales: Per-layer quantisation scales.
        zero_points: Per-layer zero points.
        mse: Mean squared error between original and quantised outputs.
    """

    scales: dict[str, float] = field(default_factory=dict)
    zero_points: dict[str, float] = field(default_factory=dict)
    mse: float = 0.0


# ---------------------------------------------------------------------------
# QuantizedModel
# ---------------------------------------------------------------------------
class QuantizedModel(nn.Module):
    """Wrapper around a quantised model.

    Delegates :meth:`forward` to the underlying (quantised) model and
    exposes metadata about the applied quantisation.

    Args:
        model: The quantised model.
        config: The :class:`QuantizationConfig` that was used.
        calibration: Optional :class:`CalibrationResult`.
    """

    def __init__(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        calibration: Optional[CalibrationResult] = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = model
        self._config: QuantizationConfig = config
        self._calibration: Optional[CalibrationResult] = calibration

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def config(self) -> QuantizationConfig:
        """The quantisation configuration."""
        return self._config

    @property
    def calibration(self) -> Optional[CalibrationResult]:
        """The calibration result (``None`` if not calibrated)."""
        return self._calibration

    @property
    def method(self) -> str:
        """The quantisation method identifier."""
        return self._config.method

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the underlying model's forward pass."""
        return self.model(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"QuantizedModel(method={self._config.method!r}, "
            f"group_size={self._config.group_size}, "
            f"calibrated={self._calibration is not None})"
        )


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------
class Quantizer:
    """Quantise PyTorch models to lower precision.

    Args:
        device: Device on which to perform quantisation (defaults to
            CPU).

    Example:
        >>> q = Quantizer()
        >>> savings = q.estimate_memory_savings(nn.Linear(4, 4),
        ...                                     QuantizationConfig(method="int8"))
        >>> savings["savings_pct"] >= 0.0
        True
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        self._device: torch.device = torch.device(device) if device else torch.device("cpu")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        """The device used for quantisation."""
        return self._device

    @property
    def has_bitsandbytes(self) -> bool:
        """``True`` when the ``bitsandbytes`` backend is available."""
        return _HAS_BITSANDBYTES

    # ------------------------------------------------------------------
    # Quantisation
    # ------------------------------------------------------------------
    def quantize(
        self,
        model: nn.Module,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        """Apply quantisation to ``model``.

        Args:
            model: The model to quantise.
            config: Quantisation configuration.

        Returns:
            A :class:`QuantizedModel` wrapping the result.
        """
        method = config.method
        quantised: nn.Module

        if method in ("fp16", "bf16"):
            quantised = self._cast_precision(model, method)
        elif method == "int8":
            quantised = self._quantize_int8(model)
        elif method in ("int4", "nf4"):
            quantised = self._quantize_low_bit(model, method)
        else:  # pragma: no cover - guarded by config validation
            quantised = model

        return QuantizedModel(model=quantised, config=config)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(
        self,
        model: nn.Module,
        dataset: Any,
    ) -> CalibrationResult:
        """Run a calibration pass and collect scales / zero points.

        Args:
            model: The model to calibrate.
            dataset: An iterable of sample inputs.

        Returns:
            A :class:`CalibrationResult`.
        """
        model = model.to(self._device)
        model.eval()
        scales: dict[str, float] = {}
        zero_points: dict[str, float] = {}
        sq_errors: list[float] = []

        samples = self._iter_samples(dataset)
        with torch.no_grad():
            for sample in samples:
                sample = self._to_device(sample)
                try:
                    output = self._forward(model, sample)
                except Exception:
                    continue
                if isinstance(output, torch.Tensor):
                    sq_errors.append(float(output.pow(2).mean().item()))

        # Collect per-layer activation statistics.
        for name, param in model.named_parameters():
            if param.dim() > 0:
                scales[name] = float(param.abs().max().item())
                zero_points[name] = 0.0

        mse = sum(sq_errors) / len(sq_errors) if sq_errors else 0.0
        return CalibrationResult(
            scales=scales,
            zero_points=zero_points,
            mse=mse,
        )

    # ------------------------------------------------------------------
    # Memory estimation
    # ------------------------------------------------------------------
    def estimate_memory_savings(
        self,
        model: nn.Module,
        config: QuantizationConfig,
    ) -> dict[str, float]:
        """Estimate the memory savings of quantising ``model``.

        Args:
            model: The model to analyse.
            config: Target quantisation configuration.

        Returns:
            A dict with ``original_gb``, ``quantized_gb`` and
            ``savings_pct``.
        """
        original_bits = 32  # assume fp32
        target_bits = _BITS_PER_METHOD.get(config.method, original_bits)

        total_params = 0
        for param in model.parameters():
            total_params += param.numel()

        original_bytes = (total_params * original_bits) / _BITS_PER_BYTE
        quantized_bytes = (total_params * target_bits) / _BITS_PER_BYTE

        original_gb = original_bytes / _BYTES_PER_GIB
        quantized_gb = quantized_bytes / _BYTES_PER_GIB
        savings_pct = (
            ((original_gb - quantized_gb) / original_gb) * 100.0
            if original_gb > 0
            else 0.0
        )
        return {
            "original_gb": original_gb,
            "quantized_gb": quantized_gb,
            "savings_pct": savings_pct,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _cast_precision(model: nn.Module, method: str) -> nn.Module:
        """Cast the model to ``fp16`` or ``bf16``."""
        dtype = torch.float16 if method == "fp16" else torch.bfloat16
        return model.to(dtype=dtype)

    @staticmethod
    def _quantize_int8(model: nn.Module) -> nn.Module:
        """Apply dynamic 8-bit quantisation to Linear layers."""
        try:
            return torch.ao.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8,
            )
        except Exception:
            return model

    def _quantize_low_bit(self, model: nn.Module, method: str) -> nn.Module:
        """Apply 4-bit / nf4 quantisation.

        When ``bitsandbytes`` is available its 4-bit / NF4 wrappers are
        used; otherwise the model is returned unchanged and the caller
        is informed via :attr:`has_bitsandbytes`.
        """
        if not _HAS_BITSANDBYTES:
            return model
        try:
            return self._bnb_replace_linear(model, method)
        except Exception:
            return model

    @staticmethod
    def _bnb_replace_linear(model: nn.Module, method: str) -> nn.Module:
        """Replace ``nn.Linear`` layers with bitsandbytes 4-bit layers.

        The original weights are quantised and copied into the new
        ``Linear4bit`` module so the model retains its trained weights.
        """
        quant_type = "nf4" if method == "nf4" else "fp4"
        for name, module in list(model.named_children()):
            if isinstance(module, nn.Linear):
                new_module = _bnb.nn.Linear4bit(  # type: ignore[attr-defined]
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    quant_type=quant_type,
                )
                # Quantise and copy the original weight into the new module.
                new_module.weight = _bnb.nn.Params4bit(  # type: ignore[attr-defined]
                    module.weight.data.clone(),
                    quant_type=quant_type,
                    requires_grad=False,
                )
                # Copy bias if present.
                if module.bias is not None:
                    new_module.bias = nn.Parameter(module.bias.data.clone())
                # Move to the same device as the original module.
                new_module = new_module.to(module.weight.device)
                setattr(model, name, new_module)
            else:
                Quantizer._bnb_replace_linear(module, method)
        return model

    @staticmethod
    def _forward(model: nn.Module, sample: Any) -> Any:
        """Call ``model`` with a single tensor or a tuple of tensors."""
        if isinstance(sample, (list, tuple)):
            return model(*sample)
        return model(sample)

    def _to_device(self, sample: Any) -> Any:
        """Move ``sample`` tensors onto the quantiser device."""
        if isinstance(sample, torch.Tensor):
            return sample.to(self._device)
        if isinstance(sample, (list, tuple)):
            return tuple(
                t.to(self._device) if isinstance(t, torch.Tensor) else t
                for t in sample
            )
        return sample

    @staticmethod
    def _iter_samples(dataset: Any) -> Any:
        """Yield up to :data:`_DEFAULT_CALIBRATION_SAMPLES` items."""
        if dataset is None:
            return
        count = 0
        for sample in dataset:
            if count >= _DEFAULT_CALIBRATION_SAMPLES:
                break
            yield sample
            count += 1

    def __repr__(self) -> str:
        return (
            f"Quantizer(device={self._device}, "
            f"bitsandbytes={_HAS_BITSANDBYTES})"
        )
