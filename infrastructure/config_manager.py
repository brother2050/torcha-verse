"""配置管理器的薄别名模块（已合并至 :class:`ConfigCenter`）。

历史上 ``ConfigManager`` 是独立的单例配置加载器。在 v0.3.0 的 R1-2 重构中，
``ConfigManager`` 的全部功能已合并进 :class:`~infrastructure.config_center.ConfigCenter`
（四级配置合并：System < Project < User < Run）。本模块仅保留为薄别名，
使现有引用 ``from infrastructure.config_manager import ConfigManager`` 的代码
无需修改即可继续工作。

新代码应直接使用 :class:`~infrastructure.config_center.ConfigCenter`。
"""

from __future__ import annotations

from .config_center import (
    ConfigCenter as ConfigManager,
    _deep_merge,
    _resolve_config_dir,
    get_config,
)

__all__ = ["ConfigManager", "get_config", "_deep_merge", "_resolve_config_dir"]
