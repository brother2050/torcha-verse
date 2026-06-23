"""Model registry and discovery center for TorchaVerse.

This module provides :class:`ModelRegistry`, a singleton registry that
catalogues every model architecture available in the framework.  Models are
registered at import time (either explicitly or via the
:func:`register_model` decorator) and can subsequently be loaded by name
with optional checkpoint weights and device/dtype placement.

The :class:`BaseModel` abstract base class defines the contract that every
registered model must honour: a :meth:`~BaseModel.forward` method for the
forward pass and a :meth:`~BaseModel.generate` method for autoregressive
generation.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn

from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

__all__ = [
    "BaseModel",
    "ModelRegistry",
    "register_model",
]


# ---------------------------------------------------------------------------
# BaseModel
# ---------------------------------------------------------------------------
class BaseModel(nn.Module, abc.ABC):
    """Abstract base class for all TorchaVerse models.

    Every concrete model must subclass :class:`BaseModel` and implement
    :meth:`forward` and :meth:`generate`.  Subclasses receive their
    configuration through ``__init__`` and are expected to be
    serialisable via the standard ``state_dict`` / ``load_state_dict``
    mechanism.

    Args:
        config: Model configuration dictionary.  The exact schema is
            defined by the model's ``config_schema`` registered with
            :class:`ModelRegistry`.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.config: Dict[str, Any] = dict(config) if config else {}
        self.model_name: str = self.config.get("name", self.__class__.__name__)

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the forward pass.

        Concrete subclasses define their own signature.  For language
        models this typically accepts ``input_ids`` and optional
        ``attention_mask``; for diffusion models it accepts the noisy
        latent, timestep, and conditioning.

        Args:
            *args: Positional arguments forwarded by the subclass.
            **kwargs: Keyword arguments forwarded by the subclass.

        Returns:
            The model output (logits, latents, etc.).
        """
        ...

    @abc.abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate output autoregressively or via sampling.

        For text models this produces token sequences; for image/audio
        models it produces the generated media tensor.

        Args:
            *args: Positional arguments forwarded by the subclass.
            **kwargs: Keyword arguments forwarded by the subclass.

        Returns:
            The generated output.
        """
        ...

    # ------------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the total number of parameters.

        Args:
            trainable_only: When ``True`` only parameters with
                ``requires_grad=True`` are counted.

        Returns:
            The parameter count.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_config_schema(self) -> Dict[str, Any]:
        """Return the configuration schema for this model.

        Subclasses may override this to provide a richer schema.  The
        default implementation returns the stored ``config`` dictionary.
        """
        return self.config

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.model_name!r}, "
            f"params={self.num_parameters():,})"
        )


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------
class _ModelEntry:
    """Internal record stored for each registered model."""

    __slots__ = ("name", "model_class", "config_schema")

    def __init__(
        self,
        name: str,
        model_class: Type[BaseModel],
        config_schema: Optional[Dict[str, Any]],
    ) -> None:
        self.name: str = name
        self.model_class: Type[BaseModel] = model_class
        self.config_schema: Dict[str, Any] = config_schema or {}


class ModelRegistry:
    """Singleton model registry implementing the registry pattern.

    All model architectures are registered at startup (either explicitly
    or via the :func:`register_model` decorator).  The registry then
    provides a uniform ``load`` API that instantiates the model, optionally
    loads checkpoint weights from a local path, and places the model on
    the requested device with the requested dtype.

    Example:
        >>> @register_model("my-text-model")
        ... class MyModel(BaseModel):
        ...     def forward(self, input_ids):
        ...         ...
        ...     def generate(self, **kwargs):
        ...         ...
        >>> registry = ModelRegistry()
        >>> "my-text-model" in registry.list_available()
        True
        >>> model = registry.load("my-text-model", config={...})
    """

    _instance: Optional["ModelRegistry"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._registry: Dict[str, _ModelEntry] = {}
        self._logger = get_logger(self.__class__.__name__)
        self._device_manager: DeviceManager = DeviceManager()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        model_class: Type[BaseModel],
        config_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a model class under ``name``.

        Args:
            name: Unique identifier for the model (case-insensitive).
            model_class: A subclass of :class:`BaseModel`.
            config_schema: Optional dictionary describing the expected
                configuration keys and their types.

        Raises:
            TypeError: If ``model_class`` is not a subclass of
                :class:`BaseModel`.
            ValueError: If ``name`` is empty.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Model name must be a non-empty string.")
        if not (isinstance(model_class, type) and issubclass(model_class, BaseModel)):
            raise TypeError(
                f"model_class must be a subclass of BaseModel, got "
                f"{model_class!r}."
            )

        key = name.strip().lower()
        self._registry[key] = _ModelEntry(name, model_class, config_schema)
        self._logger.debug("Registered model '%s' -> %s", key, model_class.__name__)

    def unregister(self, name: str) -> bool:
        """Remove a model from the registry.

        Args:
            name: Registered model name.

        Returns:
            ``True`` if the model was removed, ``False`` if it was not
            found.
        """
        key = name.strip().lower()
        if key in self._registry:
            del self._registry[key]
            self._logger.debug("Unregistered model '%s'.", key)
            return True
        return False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def is_registered(self, name: str) -> bool:
        """Return ``True`` if ``name`` is a registered model."""
        return name.strip().lower() in self._registry

    def list_available(self) -> List[str]:
        """Return a sorted list of all registered model names."""
        return sorted(entry.name for entry in self._registry.values())

    def get_config_schema(self, name: str) -> Dict[str, Any]:
        """Return the configuration schema for model ``name``.

        Args:
            name: Registered model name.

        Returns:
            A copy of the registered ``config_schema`` dictionary.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        entry = self._get_entry(name)
        return dict(entry.config_schema)

    def get_model_class(self, name: str) -> Type[BaseModel]:
        """Return the model class registered under ``name``.

        Args:
            name: Registered model name.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        return self._get_entry(name).model_class

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(
        self,
        name: str,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        config: Optional[Dict[str, Any]] = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> BaseModel:
        """Instantiate and optionally load a registered model.

        Args:
            name: Registered model name.
            checkpoint_path: Optional path to a weights file
                (``.pt``, ``.safetensors``) or a checkpoint directory.
                When provided the weights are loaded after instantiation.
            device: Target device.  Defaults to the :class:`DeviceManager`
                active device.
            dtype: Target dtype.  Resolved through the dtype policy.
            config: Configuration dictionary forwarded to the model
                constructor.
            strict: Whether to enforce exact key matching when loading
                weights.
            **kwargs: Additional keyword arguments forwarded to the model
                constructor.

        Returns:
            The instantiated (and optionally loaded) model placed on the
            target device.

        Raises:
            KeyError: If ``name`` is not registered.
            FileNotFoundError: If ``checkpoint_path`` does not exist.
        """
        entry = self._get_entry(name)
        model_config = {**(entry.config_schema), **(config or {})}
        model_config.update(kwargs)

        self._logger.info("Instantiating model '%s' (%s).", name, entry.model_class.__name__)
        model = entry.model_class(config=model_config)

        # Load checkpoint weights when provided.
        if checkpoint_path is not None:
            self._load_weights(model, checkpoint_path, strict=strict)

        # Place on device with the requested dtype.
        target_device = self._resolve_device(device)
        target_dtype = self._device_manager.dtype_policy.resolve(dtype, target_device)
        model = self._device_manager.to_device(model, target_device, target_dtype)  # type: ignore[assignment]
        model.eval()

        self._logger.info(
            "Model '%s' loaded on %s with dtype %s (%s params).",
            name,
            target_device,
            target_dtype,
            f"{model.num_parameters():,}",
        )
        return model

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_entry(self, name: str) -> _ModelEntry:
        """Return the registry entry for ``name`` or raise ``KeyError``."""
        key = name.strip().lower()
        if key not in self._registry:
            raise KeyError(
                f"Model '{name}' is not registered. "
                f"Available: {', '.join(self.list_available()) or '(none)'}."
            )
        return self._registry[key]

    def _resolve_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Resolve a device specification, falling back to the manager."""
        if device is None:
            return self._device_manager.get_device()
        if isinstance(device, str):
            return torch.device(device)
        return device

    @staticmethod
    def _load_weights(
        model: BaseModel,
        checkpoint_path: Union[str, Path],
        strict: bool = True,
    ) -> None:
        """Load weights from ``checkpoint_path`` into ``model``.

        Supports both ``.safetensors`` files and legacy ``.pt`` / ``.bin``
        files.  When the path is a directory it is searched for a weights
        file.
        """
        path = Path(checkpoint_path).expanduser().resolve()

        # Resolve directory to a weights file.
        if path.is_dir():
            for candidate_name in ("model.safetensors", "model.pt", "pytorch_model.bin"):
                candidate = path / candidate_name
                if candidate.exists():
                    path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"No weights file found in checkpoint directory {checkpoint_path}."
                )

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load state dict.
        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file as _safetensors_load

                state_dict = _safetensors_load(str(path))
            except ImportError:
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
        else:
            try:
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                state_dict = torch.load(path, map_location="cpu", weights_only=False)

        # Handle nested state dicts (e.g. {"state_dict": ...}).
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict, strict=strict)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
        cls._initialized = False


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------
def register_model(
    name: str,
    config_schema: Optional[Dict[str, Any]] = None,
) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
    """Class decorator that registers a model with the global registry.

    Usage::

        @register_model("my-model")
        class MyModel(BaseModel):
            ...

    Args:
        name: Unique registry name for the model.
        config_schema: Optional configuration schema dictionary.

    Returns:
        The original class (unchanged) after registration.
    """

    def _decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        ModelRegistry().register(name, cls, config_schema)
        return cls

    return _decorator
