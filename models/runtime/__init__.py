"""自研的 "transformers 风格" 本地模型加载与推理串联层 (v0.10.0)。

项目自 v0.4.x 以来一直保持 **零依赖** 的核心:不依赖
``transformers`` / ``tokenizers`` / ``diffusers`` / ``huggingface_hub``。
但项目使用方依然期望一个类似 ``transformers.AutoModel.from_pretrained``
+ ``transformers.pipeline("text-generation"|"text-to-image"|...)`` 的
**统一入口** 来"下载 → 加载 → 推理"。

本包正是填补这个缺口:

* :mod:`models.runtime.loader` -- 类似
  ``transformers.AutoModel`` + ``AutoTokenizer`` 的本地加载统一入口
  (:class:`ModelHub` / :func:`load_model_and_tokenizer` /
  :class:`ModelFor*` 类)。
* :mod:`models.runtime.pipeline` -- 类似 ``transformers.pipeline``
  的轻量推理管道 (:class:`TextGenerationPipeline` /
  :class:`ImageGenerationPipeline` / :class:`AudioPipeline`)。
* :mod:`models.runtime.runtime_config` -- 一行配置:
  :func:`enable_local_runtime` 把 "自研加载 + 真推理循环" 注入
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
* :class:`ModelHub`                 -- 类似 ``transformers.Hub`` 的本地 hub
* :class:`TextGenerationPipeline`
* :class:`ImageGenerationPipeline`
* :class:`AudioPipeline`

向后兼容 (v0.10.0 -> v0.10.1 命名重整):

为了让名字更准确 (不重复 ``Local`` 前缀 / 避免 ``local_loader``
嵌套在 ``models.runtime`` 下的语义重复),我们在 v0.10.1 把模块
文件和公共类都重命名了。**v0.10.0 旧名仍然作为 alias 保留**,所以
``from models.runtime import LocalModelHub`` /
``from models.runtime import LocalTextGenerationPipeline`` 在
v0.10.1+ 仍然能工作。

详细使用示例见 :mod:`docs.local_transformers` (顶层 docs 入口)。
"""
from __future__ import annotations

from .loader import (
    ModelHub,
    ModelForCausalLM,
    ModelForTextToImage,
    ModelForTextToSpeech,
    ModelForMusic,
    ModelFamily,
    TokenizerBundle,
    load_model_and_tokenizer,
    detect_model_family,
)
from .pipeline import (
    TextGenerationPipeline,
    ImageGenerationPipeline,
    AudioPipeline,
    PipelineOutput,
    pipeline,
    list_supported_tasks,
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

# ---------------------------------------------------------------------------
# Backward-compatible aliases (v0.10.0 旧名, v0.10.1+ 仍可用)
# ---------------------------------------------------------------------------
# The old name ``Local*`` carried the "local" prefix because the
# module was originally called ``local_loader.py`` /
# ``local_pipeline.py`` -- that file name has been simplified to
# ``loader.py`` / ``pipeline.py`` since "runtime" already implies
# local.  The old class names are kept as aliases so v0.10.0 user
# code continues to work unmodified.
LocalModelHub = ModelHub
LocalModelForCausalLM = ModelForCausalLM
LocalModelForTextToImage = ModelForTextToImage
LocalModelForTextToSpeech = ModelForTextToSpeech
LocalModelForMusic = ModelForMusic
LocalTextGenerationPipeline = TextGenerationPipeline
LocalImageGenerationPipeline = ImageGenerationPipeline
LocalAudioPipeline = AudioPipeline


__all__ = [
    # loader (canonical names)
    "ModelHub",
    "ModelForCausalLM",
    "ModelForTextToImage",
    "ModelForTextToSpeech",
    "ModelForMusic",
    "ModelFamily",
    "TokenizerBundle",
    "load_model_and_tokenizer",
    "detect_model_family",
    # pipeline (canonical names)
    "TextGenerationPipeline",
    "ImageGenerationPipeline",
    "AudioPipeline",
    "PipelineOutput",
    "pipeline",
    "list_supported_tasks",
    # runtime_config
    "RuntimeConfig",
    "enable_local_runtime",
    "disable_local_runtime",
    "is_local_runtime_enabled",
    "get_active_config",
    # device_planner
    "DevicePlan",
    "plan_device",
    "pick_default_device",
    "get_device_map",
    "is_cuda_available",
    "is_mps_available",
    # backward-compat aliases (v0.10.0 -> v0.10.1)
    "LocalModelHub",
    "LocalModelForCausalLM",
    "LocalModelForTextToImage",
    "LocalModelForTextToSpeech",
    "LocalModelForMusic",
    "LocalTextGenerationPipeline",
    "LocalImageGenerationPipeline",
    "LocalAudioPipeline",
]
