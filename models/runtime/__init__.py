"""Local runtime: 自研的 "transformers 风格" 本地模型加载与推理串联层 (v0.10.0)。

项目自 v0.4.x 以来一直保持 **零依赖** 的核心:不依赖
``transformers`` / ``tokenizers`` / ``diffusers`` / ``huggingface_hub``。
但项目使用方依然期望一个类似 ``transformers.AutoModel.from_pretrained``
+ ``transformers.pipeline("text-generation"|"text-to-image"|...)`` 的
**统一入口** 来"下载 → 加载 → 推理"。

本包正是填补这个缺口:

* :mod:`models.runtime.local_loader` -- 类似
  ``transformers.AutoModel`` + ``AutoTokenizer`` 的本地加载统一入口
  (LocalModelHub / load_model_and_tokenizer / LocalModelFor* 类)。
* :mod:`models.runtime.local_pipeline` -- 类似 ``transformers.pipeline``
  的轻量推理管道 (LocalTextGenerationPipeline /
  LocalImageGenerationPipeline / LocalAudioPipeline)。
* :mod:`models.runtime.runtime_config` -- 一行配置:
  :func:`enable_local_runtime()` 把 "自研加载 + 真推理循环" 注入
  :class:`core.module_bus.ModuleBus`,让 39 个 L4 节点从默认的 echo
  工厂切到 **真模型真生成**。
* :mod:`models.runtime.device_planner` -- CPU / GPU / MPS / multi-GPU
  自动分配,无外部依赖 (比 ``accelerate`` 简单但够用)。

设计原则 (与 v0.8.0 + v0.9.0 路线图保持一致):

1. **零外部依赖**:不依赖 ``transformers`` / ``tokenizers`` /
   ``diffusers`` / ``accelerate``。仅用 ``torch`` + 标准库 + 项目内部
   的 :mod:`core.checkpoint_loader` / :mod:`models.text.*_tokenizer` /
   :mod:`nodes._helpers._backends`。
2. **与现有 API 兼容**:走 :class:`models.base.ModelMixin.from_pretrained`
   + key_renames (HUNYUAN_DIT_KEY_MAP / FLUX / SD3 / WAN2 / MUSICGEN)
   路径,**不破坏** v0.8.x 的 1157+ 测试。
3. **可插拔**:每条 pipeline 都接收一个 ``backend`` 工厂 (零参 callable),
   因此可以接 v0.4.x P0 的 ``LocalTorchTextProvider``、v0.8.x 的真
   HunyuanDiT、v1.0.0 的真 FLUX,无需改 pipeline 代码。
4. **生产友好**:支持 ``device_map`` / ``torch_dtype`` / ``variant`` /
   ``key_renames`` / ``strict`` 5 维参数,与 diffusers 行为对齐。
5. **测试 0 回归**:新模块的占位 / not-implemented 全部在
   ``docs/placeholder_registry.md`` 登记。

公共 API (按使用频率倒序):

* :func:`load_model_and_tokenizer`  -- 一行 "本地加载模型 + tokenizer"
* :func:`pipeline`                  -- 类似 ``transformers.pipeline``
  的多模态推理管道工厂
* :func:`enable_local_runtime`      -- 注入 39 节点 (一行)
* :class:`LocalModelHub`            -- 类似 ``transformers.Hub`` 的本地 hub
* :class:`LocalTextGenerationPipeline`
* :class:`LocalImageGenerationPipeline`
* :class:`LocalAudioPipeline`

完整使用示例见 :mod:`examples.local_transformers_demo` 与
:mod:`docs.local_transformers`。
"""
from __future__ import annotations

from .local_loader import (
    LocalModelHub,
    LocalModelForCausalLM,
    LocalModelForTextToImage,
    LocalModelForTextToSpeech,
    LocalModelForMusic,
    ModelFamily,
    TokenizerBundle,
    load_model_and_tokenizer,
    detect_model_family,
)
from .local_pipeline import (
    LocalTextGenerationPipeline,
    LocalImageGenerationPipeline,
    LocalAudioPipeline,
    pipeline,
    list_supported_tasks,
    PipelineOutput,
)
from .runtime_config import (
    RuntimeConfig,
    enable_local_runtime,
    disable_local_runtime,
    is_local_runtime_enabled,
    get_active_config,
)
from .device_planner import (
    DevicePlan,
    plan_device,
    pick_default_device,
    get_device_map,
    is_cuda_available,
    is_mps_available,
)

__all__ = [
    # local_loader
    "LocalModelHub",
    "LocalModelForCausalLM",
    "LocalModelForTextToImage",
    "LocalModelForTextToSpeech",
    "LocalModelForMusic",
    "ModelFamily",
    "load_model_and_tokenizer",
    "detect_model_family",
    # local_pipeline
    "LocalTextGenerationPipeline",
    "LocalImageGenerationPipeline",
    "LocalAudioPipeline",
    "pipeline",
    # runtime_config
    "RuntimeConfig",
    "enable_local_runtime",
    "disable_local_runtime",
    "is_local_runtime_enabled",
    # device_planner
    "DevicePlan",
    "plan_device",
    "pick_default_device",
    "get_device_map",
]
