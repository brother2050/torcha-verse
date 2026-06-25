"""Infrastructure layer for the TorchaVerse framework.

This package groups the foundational, non-domain-specific utilities used
across the framework: configuration management (including the v0.3.0 tiered
:class:`ConfigCenter`), device/distributed abstractions, checkpoint
lifecycle, logging, error handling, rate limiting, caching, audit logging,
resource budgeting, and multi-source model fetching.
"""

from __future__ import annotations

from .audit_log import AuditEvent, AuditLogger, EventType, Severity
from .cache_store import CacheStore
from .checkpoint_manager import CheckpointManager
from .config_center import ConfigCenter
from .config_manager import ConfigManager, get_config
from .defaults import (
    DIFFUSION_ETA,
    DIFFUSION_GUIDANCE_SCALE,
    DIFFUSION_SCHEDULER,
    DIFFUSION_STEPS,
    SAMPLING_REPETITION_PENALTY,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_K,
    SAMPLING_TOP_P,
)
from .device_manager import DeviceManager, DTypePolicy, PipelineParallel, TensorParallel
from .error_handler import ErrorHandler, with_error_handler
from .logger import Logger, get_logger, set_log_level
from .rate_limiter import RateLimiter
from .resource_budget import (
    AllocationHandle,
    BudgetExceededError,
    BudgetTracker,
    FeasibilityEstimate,
    ResourceBudget,
)
from .source_fetcher import (
    FetchError,
    HuggingFaceSource,
    LicenseRef,
    LocalSource,
    ModelScopeSource,
    ModelersSource,
    SourceFetcher,
    SourceRegistry,
)

__all__ = [
    # config
    "ConfigCenter",
    "ConfigManager",
    "get_config",
    # inference defaults
    "DIFFUSION_STEPS",
    "DIFFUSION_GUIDANCE_SCALE",
    "DIFFUSION_SCHEDULER",
    "DIFFUSION_ETA",
    "SAMPLING_TEMPERATURE",
    "SAMPLING_TOP_K",
    "SAMPLING_TOP_P",
    "SAMPLING_REPETITION_PENALTY",
    # device / distributed
    "DeviceManager",
    "DTypePolicy",
    "TensorParallel",
    "PipelineParallel",
    # logging & errors
    "Logger",
    "get_logger",
    "set_log_level",
    "ErrorHandler",
    "with_error_handler",
    # lifecycle utilities
    "CacheStore",
    "CheckpointManager",
    "RateLimiter",
    # audit
    "AuditEvent",
    "AuditLogger",
    "EventType",
    "Severity",
    # resource budget
    "ResourceBudget",
    "BudgetTracker",
    "AllocationHandle",
    "BudgetExceededError",
    "FeasibilityEstimate",
    # source fetching
    "SourceFetcher",
    "SourceRegistry",
    "LocalSource",
    "HuggingFaceSource",
    "ModelScopeSource",
    "ModelersSource",
    "LicenseRef",
    "FetchError",
]
