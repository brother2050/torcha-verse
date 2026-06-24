"""Core layer for the TorchaVerse framework.

This package groups the domain-specific core components that sit above
the infrastructure layer: the module assembly bus, model registry,
tokenizer hub, KV cache management, diffusion scheduling, vocoder
management, memory management, inference scheduling, and the tool
registry.

Import policy
--------------
To keep ``import core`` lightweight and free of optional heavy
dependencies (e.g. ``torch``), the torch-backed submodules are imported
*lazily* on first attribute access (see :pep:`562`).  Only
:mod:`core.module_bus` -- which has no third-party dependencies -- is
imported eagerly, so the :class:`~core.module_bus.ModuleBus` registry is
always available even in minimal environments.

Concretely::

    import core                       # succeeds without torch
    from core import ModuleBus        # succeeds without torch
    from core import BaseModel        # lazily imports torch on access
"""

from __future__ import annotations

import importlib
from typing import Any

# Eagerly import the dependency-free module bus so it is always available.
from .module_bus import ModuleBus, ModuleSpec, register_module

__all__ = [
    # module_bus (eager, dependency-free)
    "ModuleBus",
    "ModuleSpec",
    "register_module",
    # model_registry
    "BaseModel",
    "ModelRegistry",
    "register_model",
    # tokenizer_hub
    "AudioTokenizer",
    "BaseTokenizer",
    "ImageTokenizer",
    "TextTokenizer",
    "TokenizerHub",
    "VideoTokenizer",
    # kv_cache_manager
    "CacheStrategy",
    "KVCacheManager",
    # diffusion_scheduler
    "BaseSampler",
    "ConsistencySampler",
    "DDIMSampler",
    "DDPMSampler",
    "DiffusionScheduler",
    "DPMSolverSampler",
    "EulerSampler",
    "GuidanceController",
    "NoiseSchedule",
    "SAMPLER_REGISTRY",
    "StepController",
    # vocoder_manager
    "BaseVocoder",
    "HiFiGANVocoder",
    "VocoderManager",
    # memory_manager
    "MemoryInfo",
    "MemoryManager",
    "MemoryPool",
    # inference_scheduler
    "Future",
    "InferenceRequest",
    "InferenceResult",
    "InferenceScheduler",
    "RequestStatus",
    # tool_registry
    "BaseTool",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "register_tool",
    "validate_params",
]

#: Mapping of public attribute name -> submodule providing it.  These are
#: imported lazily on first access so that ``import core`` does not pull in
#: optional heavy dependencies such as ``torch``.
_LAZY_IMPORTS: dict[str, str] = {
    # model_registry
    "BaseModel": "model_registry",
    "ModelRegistry": "model_registry",
    "register_model": "model_registry",
    # tokenizer_hub
    "AudioTokenizer": "tokenizer_hub",
    "BaseTokenizer": "tokenizer_hub",
    "ImageTokenizer": "tokenizer_hub",
    "TextTokenizer": "tokenizer_hub",
    "TokenizerHub": "tokenizer_hub",
    "VideoTokenizer": "tokenizer_hub",
    # kv_cache_manager
    "CacheStrategy": "kv_cache_manager",
    "KVCacheManager": "kv_cache_manager",
    # diffusion_scheduler
    "BaseSampler": "diffusion_scheduler",
    "ConsistencySampler": "diffusion_scheduler",
    "DDIMSampler": "diffusion_scheduler",
    "DDPMSampler": "diffusion_scheduler",
    "DiffusionScheduler": "diffusion_scheduler",
    "DPMSolverSampler": "diffusion_scheduler",
    "EulerSampler": "diffusion_scheduler",
    "GuidanceController": "diffusion_scheduler",
    "NoiseSchedule": "diffusion_scheduler",
    "SAMPLER_REGISTRY": "diffusion_scheduler",
    "StepController": "diffusion_scheduler",
    # vocoder_manager
    "BaseVocoder": "vocoder_manager",
    "HiFiGANVocoder": "vocoder_manager",
    "VocoderManager": "vocoder_manager",
    # memory_manager
    "MemoryInfo": "memory_manager",
    "MemoryManager": "memory_manager",
    "MemoryPool": "memory_manager",
    # inference_scheduler
    "Future": "inference_scheduler",
    "InferenceRequest": "inference_scheduler",
    "InferenceResult": "inference_scheduler",
    "InferenceScheduler": "inference_scheduler",
    "RequestStatus": "inference_scheduler",
    # tool_registry
    "BaseTool": "tool_registry",
    "Tool": "tool_registry",
    "ToolRegistry": "tool_registry",
    "ToolResult": "tool_registry",
    "register_tool": "tool_registry",
    "validate_params": "tool_registry",
}


def __getattr__(name: str) -> Any:
    """Lazily import torch-backed core submodules on first access.

    Args:
        name: The attribute being accessed on the :mod:`core` package.

    Returns:
        The requested attribute, imported from its submodule.

    Raises:
        AttributeError: If ``name`` is not a known public attribute.
    """
    submodule = _LAZY_IMPORTS.get(name)
    if submodule is None:
        raise AttributeError(
            "module {!r} has no attribute {!r}".format(__name__, name)
        )
    module = importlib.import_module("." + submodule, __name__)
    value = getattr(module, name)
    # Cache in the module namespace so subsequent accesses are direct.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the public names of the package (for ``dir(core)``)."""
    return sorted(set(globals()) | set(__all__))
