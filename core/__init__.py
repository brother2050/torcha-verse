"""Core layer for the TorchaVerse framework.

This package groups the domain-specific core components that sit above
the infrastructure layer: model registry, tokenizer hub, KV cache
management, diffusion scheduling, vocoder management, memory management,
inference scheduling, and the tool registry.
"""

from __future__ import annotations

from .diffusion_scheduler import (
    BaseSampler,
    ConsistencySampler,
    DDIMSampler,
    DDPMSampler,
    DiffusionScheduler,
    DPMSolverSampler,
    EulerSampler,
    GuidanceController,
    NoiseSchedule,
    SAMPLER_REGISTRY,
    StepController,
)
from .inference_scheduler import (
    Future,
    InferenceRequest,
    InferenceResult,
    InferenceScheduler,
    RequestStatus,
)
from .kv_cache_manager import CacheStrategy, KVCacheManager
from .memory_manager import MemoryInfo, MemoryManager, MemoryPool
from .model_registry import BaseModel, ModelRegistry, register_model
from .tokenizer_hub import (
    AudioTokenizer,
    BaseTokenizer,
    ImageTokenizer,
    TextTokenizer,
    TokenizerHub,
    VideoTokenizer,
)
from .tool_registry import (
    BaseTool,
    Tool,
    ToolRegistry,
    ToolResult,
    register_tool,
    validate_params,
)
from .vocoder_manager import BaseVocoder, HiFiGANVocoder, VocoderManager

__all__ = [
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
