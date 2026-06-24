"""
TorchaVerse: A pure PyTorch all-modal generative AI framework.

This framework provides end-to-end generative AI capabilities including
text, image, audio, video generation, multimodal fusion, RAG, and agents,
all built on native PyTorch without high-level wrappers.
"""

__version__ = "0.3.0-alpha"
__author__ = "TorchaVerse Team"

# ---------------------------------------------------------------------------
# v0.3.0: Lazy imports to avoid hard dependency on torch at package level.
# The framework's L1 infrastructure (ConfigCenter, AuditLogger, etc.) and
# L2 asset layer (AssetStore) and ModuleBus are usable without torch.
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    """Lazy attribute access for top-level exports."""
    # L1 Infrastructure
    if name == "ConfigManager":
        from .infrastructure.config_manager import ConfigManager
        return ConfigManager
    if name == "ConfigCenter":
        from .infrastructure.config_center import ConfigCenter
        return ConfigCenter
    if name == "DeviceManager":
        from .infrastructure.device_manager import DeviceManager
        return DeviceManager
    if name == "get_logger":
        from .infrastructure.logger import get_logger
        return get_logger
    if name == "AuditLogger":
        from .infrastructure.audit_log import AuditLogger
        return AuditLogger
    if name == "ResourceBudget":
        from .infrastructure.resource_budget import ResourceBudget
        return ResourceBudget
    if name == "BudgetTracker":
        from .infrastructure.resource_budget import BudgetTracker
        return BudgetTracker
    if name == "SourceRegistry":
        from .infrastructure.source_fetcher import SourceRegistry
        return SourceRegistry

    # L2 Assets
    if name == "AssetStore":
        from .assets import AssetStore
        return AssetStore
    if name == "AssetRef":
        from .assets import AssetRef
        return AssetRef
    if name == "ModelAsset":
        from .assets import ModelAsset
        return ModelAsset
    if name == "CharacterAsset":
        from .assets import CharacterAsset
        return CharacterAsset

    # L3 Core
    if name == "ModuleBus":
        from .core.module_bus import ModuleBus
        return ModuleBus
    if name == "register_module":
        from .core.module_bus import register_module
        return register_module

    raise AttributeError(f"module 'torcha_verse' has no attribute '{name}'")


__all__ = [
    # L1 Infrastructure
    "ConfigManager",
    "ConfigCenter",
    "DeviceManager",
    "get_logger",
    "AuditLogger",
    "ResourceBudget",
    "BudgetTracker",
    "SourceRegistry",
    # L2 Assets
    "AssetStore",
    "AssetRef",
    "ModelAsset",
    "CharacterAsset",
    # L3 Core
    "ModuleBus",
    "register_module",
    # Meta
    "__version__",
]
