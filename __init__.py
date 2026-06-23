"""
TorchaVerse: A pure PyTorch all-modal generative AI framework.

This framework provides end-to-end generative AI capabilities including
text, image, audio, video generation, multimodal fusion, RAG, and agents,
all built on native PyTorch without high-level wrappers.
"""

__version__ = "0.1.0"
__author__ = "TorchaVerse Team"

# Core imports for convenience. Relative imports keep the package importable
# both as ``import torcha_verse`` and when running from the source directory.
from .infrastructure.config_manager import ConfigManager
from .infrastructure.device_manager import DeviceManager
from .infrastructure.logger import get_logger

__all__ = [
    "ConfigManager",
    "DeviceManager",
    "get_logger",
    "__version__",
]
