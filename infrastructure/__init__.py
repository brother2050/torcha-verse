"""Infrastructure layer for the TorchaVerse framework.

This package groups the foundational, non-domain-specific utilities used
across the framework: configuration management, device/distributed
abstractions, checkpoint lifecycle, logging, error handling, rate
limiting, and caching.
"""

from __future__ import annotations

from .cache_store import CacheStore
from .checkpoint_manager import CheckpointManager
from .config_manager import ConfigManager, get_config
from .device_manager import DeviceManager, DTypePolicy
from .error_handler import ErrorHandler, with_error_handler
from .logger import Logger, get_logger
from .rate_limiter import RateLimiter

__all__ = [
    "CacheStore",
    "CheckpointManager",
    "ConfigManager",
    "DeviceManager",
    "DTypePolicy",
    "ErrorHandler",
    "Logger",
    "RateLimiter",
    "get_config",
    "get_logger",
    "with_error_handler",
]
