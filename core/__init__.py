"""Core layer for the TorchaVerse framework.

This package groups the domain-specific core components that sit above
the infrastructure layer: the module assembly bus, paged KV cache,
runtime scheduler, memory pool, unified sampler, diffusion scheduler,
vocoder manager, and the tool registry.

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
    from core import PagedKVCache     # lazily imports torch on access
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
    # kv_cache_v2
    "EvictionResult",
    "KVCacheConfig",
    "KVCacheBlock",
    "PagedKVCache",
    # runtime_scheduler
    "TaskPriority",
    "Task",
    "Future",
    "RuntimeScheduler",
    # memory_pool
    "OffloadStrategy",
    "MemoryBlock",
    "MemoryPool",
    "MemoryEstimate",
    "MemoryEstimator",
    "ModelOffloader",
    # sampler (unified sampler abstraction)
    "SamplerConfig",
    "SamplerRegistry",
    "register_sampler",
    # diffusion_scheduler
    "NoiseSchedule",
    "BaseSampler",
    "DDPMSampler",
    "DDIMSampler",
    "EulerSampler",
    "DPMSolverSampler",
    "ConsistencySampler",
    "GuidanceController",
    "StepController",
    "DiffusionScheduler",
    "SAMPLER_REGISTRY",
    # vocoder_manager
    "BaseVocoder",
    "HiFiGANVocoder",
    "VocoderManager",
    # tool_registry
    "BaseTool",
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "register_tool",
    "validate_params",
]

#: Mapping of public attribute name -> submodule providing it.  These are
#: imported lazily on first access so that ``import core`` does not pull in
#: optional heavy dependencies such as ``torch``.
_LAZY_IMPORTS: dict[str, str] = {
    # kv_cache_v2
    "EvictionResult": "kv_cache_v2",
    "KVCacheConfig": "kv_cache_v2",
    "KVCacheBlock": "kv_cache_v2",
    "PagedKVCache": "kv_cache_v2",
    # runtime_scheduler
    "TaskPriority": "runtime_scheduler",
    "Task": "runtime_scheduler",
    "Future": "runtime_scheduler",
    "RuntimeScheduler": "runtime_scheduler",
    # memory_pool
    "OffloadStrategy": "memory_pool",
    "MemoryBlock": "memory_pool",
    "MemoryPool": "memory_pool",
    "MemoryEstimate": "memory_pool",
    "MemoryEstimator": "memory_pool",
    "ModelOffloader": "memory_pool",
    # sampler
    "SamplerConfig": "sampler",
    "SamplerRegistry": "sampler",
    "register_sampler": "sampler",
    # diffusion_scheduler
    "NoiseSchedule": "diffusion_scheduler",
    "BaseSampler": "diffusion_scheduler",
    "DDPMSampler": "diffusion_scheduler",
    "DDIMSampler": "diffusion_scheduler",
    "EulerSampler": "diffusion_scheduler",
    "DPMSolverSampler": "diffusion_scheduler",
    "ConsistencySampler": "diffusion_scheduler",
    "GuidanceController": "diffusion_scheduler",
    "StepController": "diffusion_scheduler",
    "DiffusionScheduler": "diffusion_scheduler",
    "SAMPLER_REGISTRY": "diffusion_scheduler",
    # vocoder_manager
    "BaseVocoder": "vocoder_manager",
    "HiFiGANVocoder": "vocoder_manager",
    "VocoderManager": "vocoder_manager",
    # tool_registry
    "BaseTool": "tool_registry",
    "Tool": "tool_registry",
    "ToolResult": "tool_registry",
    "ToolRegistry": "tool_registry",
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
