# Local Transformers Runtime (v0.10.0)

> **Status**: v0.10.0 design contract. Backed by
> `models/runtime/` (4 个子模块, ~1100 行)。
>
> **零外部依赖**:不依赖 `transformers` / `tokenizers` / `diffusers` /
> `huggingface_hub` / `accelerate`。只依赖 `torch` + 项目自有的 L1-L6 模块。

This document is the single source of truth for torcha-verse v0.10.0's
"**自研 transformers 风格**" 本地运行时:一个不依赖第三方包的、但
API 与 diffusers / transformers 高度兼容的"下载 → 加载 → 推理"串联层。

---

## 1. Why a new runtime layer?

V0.4.x → v0.9.6 期间,项目已经累积了:

| 能力 | 文件 | 状态 |
|---|---|---|
| 自研 `ModelMixin` + `from_pretrained` 协议 | `models/base.py` | ✅ v0.8.0 |
| 5 个 upstream → local key rename table | `core/checkpoint_loader.py` | ✅ v0.9.0 |
| HunyuanDiT-Tiny / HunyuanVideo 真权重 + key map | `models/image/dit.py` + `papers/adapters/` | ✅ v0.8.5 |
| 自研 CLIP BPE + T5 SentencePiece tokenizer | `models/text/clip_tokenizer.py` / `t5_tokenizer.py` | ✅ v0.8.0 |
| 5 种 sampler + 3 种 schedule | `core/schedulers/` | ✅ v0.9.0 |
| 真 30 步去噪循环 | `nodes/_helpers/_backends.py::call_diffusion_loop_backend` | ✅ v0.8.0 |
| 项目自有 mirror / dedup / integrity 的 model fetcher | `models/source/` | ✅ v0.4.x P2+ |
| 39 个 L4 节点 | `nodes/` | ✅ v0.6.x |

但调用方 (39 节点 / examples / CLI) 依然要自己写 5+ 行 boilerplate:

```python
# 5+ 行 boilerplate, 每个调用方都要重写
model = HunyuanDiT.from_pretrained(path, key_renames=HUNYUAN_DIT_KEY_MAP,
                                    torch_dtype=torch.float16)
tokenizer = SimpleByteBPETokenizer(vocab_path, merges_path)
text_embeds = model.encode_text(prompt)
latents = call_diffusion_loop_backend(...)
```

**`models/runtime/`** 填补这个缺口:把以上 5+ 行打包成 **2 行**:

```python
# 2 行
model, tok, family = load_model_and_tokenizer(path)
pipe = pipeline("text-to-image", model=LocalModelForTextToImage(model, tok, family))
```

并且提供:

* :func:`load_model_and_tokenizer` — `transformers.AutoModel + AutoTokenizer` 的本地等价
* :func:`pipeline` — `transformers.pipeline(...)` 的本地等价
* :func:`enable_local_runtime` — 一行把 39 节点从 echo 切到真模型
* :class:`LocalModelHub` — `transformers.Hub` 的本地等价
* 4 个 TaskHead — `LocalModelFor{CausalLM, TextToImage, TextToSpeech, Music}`

---

## 2. Public API

```python
from models.runtime import (
    # 统一入口 (一行)
    load_model_and_tokenizer,    # 类似 AutoModel + AutoTokenizer
    pipeline,                    # 类似 transformers.pipeline

    # 一行运行时开关
    enable_local_runtime,        # 注入 39 节点
    disable_local_runtime,
    is_local_runtime_enabled,
    get_active_config,
    RuntimeConfig,

    # 4 个 TaskHead
    LocalModelForCausalLM,
    LocalModelForTextToImage,
    LocalModelForTextToSpeech,
    LocalModelForMusic,

    # 3 个 Pipeline
    LocalTextGenerationPipeline,
    LocalImageGenerationPipeline,
    LocalAudioPipeline,

    # Hub + 设备规划
    LocalModelHub,
    ModelFamily,                 # HUNYUAN_DIT / FLUX / SD3 / WAN2 / MUSICGEN / TINY_TRANSFORMER
    TokenizerBundle,
    DevicePlan,                  # (device, dtype, device_map, notes)
    plan_device,
    pick_default_device,
    get_device_map,
)
```

---

## 3. End-to-end examples

### 3.1 Text generation

```python
from models.runtime import (
    LocalModelForCausalLM, LocalTextGenerationPipeline,
    TokenizerBundle, ModelFamily,
)
from models.providers.tiny_transformer import TINY_CONFIG, build_tiny_transformer

model, tok = build_tiny_transformer(TINY_CONFIG)
head = LocalModelForCausalLM(
    model, TokenizerBundle(byte=tok),
    family=ModelFamily.TINY_TRANSFORMER,
)
pipe = LocalTextGenerationPipeline(head)

out = pipe(["the quick brown fox", "lorem ipsum"], max_new_tokens=24)
for rec in out:
    print(rec["prompt"], "->", rec["generated_text"])
```

### 3.2 Image generation

```python
from models.runtime import (
    load_model_and_tokenizer,
    LocalModelForTextToImage,
    LocalImageGenerationPipeline,
)

# 1) 一行加载 (auto-detect family + auto key-rename + auto device)
model, tok, family = load_model_and_tokenizer(
    "/path/to/hunyuan-dit-tiny",
    torch_dtype=torch.float16,
    device="cpu",
)

# 2) 包一层 TaskHead
head = LocalModelForTextToImage(model, tok, family)
pipe = LocalImageGenerationPipeline(head)

# 3) 推理
out = pipe(
    "a serene mountain landscape at sunset",
    height=512, width=512,
    num_inference_steps=30,
    guidance_scale=4.5,
    sampler="flow_match_euler",
)
for rec in out:
    print(rec["prompt"], "-> latents shape:", tuple(rec["latents"].shape))
```

### 3.3 Audio / TTS / Music

```python
from models.runtime import (
    load_model_and_tokenizer,
    LocalModelForTextToSpeech, LocalModelForMusic,
    LocalAudioPipeline,
)

# TTS
model, tok, family = load_model_and_tokenizer("/path/to/f5tts", device="cpu")
head = LocalModelForTextToSpeech(model, tok, family)
pipe = LocalAudioPipeline(head)
out = pipe("Hello world.", sample_rate=22050)
for rec in out:
    print(rec["mel"].shape, rec["sample_rate"])

# MusicGen
model, tok, family = load_model_and_tokenizer("/path/to/musicgen", device="cpu")
head = LocalModelForMusic(model, tok, family)
pipe = LocalAudioPipeline(head)
out = pipe("a funky beat", duration_s=8.0, sample_rate=32000)
```

### 3.4 `pipeline()` factory

```python
# 一行构造管道 (内部自动 load)
from models.runtime import pipeline

pipe = pipeline("text-to-image", model_path="/path/to/dit", device="cpu")
out = pipe("a tiny cat", num_inference_steps=20)

# 或者预加载 model + tokenizer 后再传 TaskHead
from models.runtime import load_model_and_tokenizer, LocalModelForTextToImage
model, tok, family = load_model_and_tokenizer("/path/to/dit")
head = LocalModelForTextToImage(model, tok, family)
pipe = pipeline("text-to-image", model=head)
```

### 3.5 Inject into the 39 L4 nodes (one call)

```python
from models.runtime import enable_local_runtime, disable_local_runtime, RuntimeConfig

# 一行把 39 节点从 echo 切到真模型
cfg = RuntimeConfig(
    prefer_local_text=True,
    prefer_local_image=True,
    prefer_local_video=True,
    prefer_local_audio=True,
    prefer_local_multimodal=True,
    use_real_diffusion_loop=True,
    device="cpu",
)
enable_local_runtime(cfg)

# 现在所有 39 节点都走真模型
from pipeline.composer import PipelineBuilder
from nodes.base import NodeContext
pipe = PipelineBuilder("demo").node("text_chat", id="t", prompt="hi").build()
result = pipe.run(NodeContext())

# 还原
disable_local_runtime()
```

---

## 4. Supported model families

| `ModelFamily` | 上游项目 | `key_renames` 来源 | TaskHead |
|---|---|---|---|
| `HUNYUAN_DIT` | Tencent HunyuanDiT | `HUNYUAN_DIT_KEY_MAP` (50+ 条) | `LocalModelForTextToImage` |
| `FLUX` | Black Forest Labs FLUX.1-dev | `FLUX_KEY_MAP` (49 条) | `LocalModelForTextToImage` |
| `SD3` | Stability AI SD3-Medium | `SD3_KEY_MAP` (37 条) | `LocalModelForTextToImage` |
| `WAN2` | Wan-2.1 video | `WAN2_KEY_MAP` (36 条) | `LocalModelForTextToImage` (视频) |
| `MUSICGEN` | Meta MusicGen | `MUSICGEN_KEY_MAP` (32 条) | `LocalModelForMusic` / `LocalModelForTextToSpeech` |
| `TINY_TRANSFORMER` | torcha-verse 内部 | (无) | `LocalModelForCausalLM` |
| `UNKNOWN` | — | `ModelMixin.from_pretrained` fallback | (无) |

Family detection: :func:`detect_model_family` 通过扫描
`state_dict.keys()` 的前 64 个前缀,按 5 个 family 各自的特征签名
(`img_in.proj` / `double_blocks` / `joint_transformer_blocks` /
`patch_embedding` / `text_encoder.transformer`) 自动判断。用户在
`load_model_and_tokenizer` / `LocalModelHub.load` 时也可以显式
`family=...` 覆盖。

---

## 5. Configuration knobs

### 5.1 `load_model_and_tokenizer` 接受的 11 个 kwargs

| Kwargs | 类型 | 默认 | 说明 |
|---|---|---|---|
| `path` | `str \| Path` | (必填) | 本地路径:目录 或 `.safetensors` 文件 |
| `repo_id` | `str` | `None` | HF `org/repo` id (需 `download=True`) |
| `revision` | `str` | `"main"` | HF git revision |
| `family` | `ModelFamily` | auto | 强制 family (跳过自动检测) |
| `torch_dtype` | `torch.dtype` | auto | dtype (CPU → fp32, CUDA → fp16) |
| `device` | `str` | auto | "cpu" / "cuda" / "cuda:0" / "mps" |
| `device_map` | `dict \| str` | `None` | diffusers 风格 per-layer 分片 |
| `variant` | `str` | `None` | diffusers 风格 `fp16` / `bf16` sibling |
| `strict` | `bool` | `False` | diffusers 默认 lenient |
| `num_blocks` | `int` | auto | DiT / Transformer block 数 (per-block key 展开) |
| `download` | `bool` | `False` | 是否走 hub.download |

### 5.2 `RuntimeConfig` 字段

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `prefer_local_text` | `bool` | `True` | 文本节点切真后端 |
| `prefer_local_image` | `bool` | `True` | 图像节点切真后端 |
| `prefer_local_video` | `bool` | `True` | 视频节点切真后端 |
| `prefer_local_audio` | `bool` | `True` | 音频节点切真后端 |
| `prefer_local_multimodal` | `bool` | `True` | 多模态节点切真后端 |
| `torch_dtype` | `torch.dtype` | `None` | 全局 dtype |
| `device` | `str` | `"cpu"` | 全局 device |
| `max_memory_per_backend_gb` | `float` | `None` | 单后端显存上限 |
| `use_real_diffusion_loop` | `bool` | `True` | 是否走 v0.8.x 真 30 步循环 |
| `tags` | `list[str]` | `[]` | 用户自定义标签 (用于日志 / 监控) |

---

## 6. Zero external dependencies

本包及它的依赖关系:

```
models.runtime
├── models.base           # 自研 ModelMixin (v0.8.0)
├── core.checkpoint_loader  # 5 个 KEY_MAP (v0.9.0)
├── models.text.clip_tokenizer  # 自研 BPE (v0.8.0)
├── models.text.t5_tokenizer    # 自研 SentencePiece (v0.8.0)
├── models.providers.tiny_transformer  # 自研 Transformer (v0.4.x P0)
├── core.module_bus       # L3 模块总线 (v0.4.x)
├── nodes._helpers._backends  # 39 节点 backend 注册 (v0.4.x P0)
├── models.source         # mirror / dedup / integrity (v0.4.x P2+)
└── infrastructure.logger
```

**不**依赖:

* ❌ `transformers`
* ❌ `tokenizers`
* ❌ `diffusers`
* ❌ `huggingface_hub`
* ❌ `accelerate`
* ❌ `sentencepiece` (T5 tokenizer 有原生 fallback)

只依赖 `torch` (硬依赖) + 标准库 + 项目自有的 L1-L6 模块。

---

## 7. End-to-end demo

```bash
# 跑全部 demo (不需要任何外部权重, 全部 random init)
python examples/local_transformers_demo.py

# 只跑 text-generation
python examples/local_transformers_demo.py --task text-generation

# 只跑 image
python examples/local_transformers_demo.py --task text-to-image

# 只跑 TTS
python examples/local_transformers_demo.py --task text-to-speech

# 也跑"注入 39 节点" demo
python examples/local_transformers_demo.py --inject-runtime
```

---

## 8. CI guards (v0.10.0)

| Test | What it checks |
|---|---|
| `tests/test_local_transformers.py::TestDevicePlanner` | 设备规划 (11 个 test) |
| `tests/test_local_transformers.py::TestDetectModelFamily` | family 自动检测 (4 个) |
| `tests/test_local_transformers.py::TestLocalModelHub` | LocalModelHub 缓存 + 加载 (5 个) |
| `tests/test_local_transformers.py::TestTaskHeads` | 4 个 TaskHead (5 个) |
| `tests/test_local_transformers.py::TestLoadModelAndTokenizer` | 顶层入口 (3 个) |
| `tests/test_local_transformers.py::TestLocalTextGenerationPipeline` | text pipeline (2 个) |
| `tests/test_local_transformers.py::TestLocalImageGenerationPipeline` | image pipeline (2 个) |
| `tests/test_local_transformers.py::TestLocalAudioPipeline` | audio pipeline (2 个) |
| `tests/test_local_transformers.py::TestPipelineFactory` | `pipeline()` factory (5 个) |
| `tests/test_local_transformers.py::TestRuntimeConfig` | runtime 开关 (7 个) |
| `tests/test_local_transformers.py::TestEndToEnd` | e2e smoke (4 个) |

合计 ~50 个 test, 全部 < 2 s 跑完, 零网络, 零 GPU。

其它既有 CI 守卫 (placeholder / hardcoding / degrade_logging) 保持开启;
新增 `pass` / `NotImplementedError` 全部在
[`docs/placeholder_registry.md`](placeholder_registry.md) 登记。

---

## 9. Migration from the pre-v0.10 5-line boilerplate

**Before** (5+ 行 boilerplate, 每个调用方都要重写):

```python
from core.checkpoint_loader import HUNYUAN_DIT_KEY_MAP, load_hunyuan_dit
from models.text.clip_tokenizer import SimpleByteBPETokenizer
from nodes._helpers._backends import call_diffusion_loop_backend
from core.module_bus import ModuleBus
import torch

model = load_hunyuan_dit(
    "/path/to/dit",
    torch_dtype=torch.float16,
    device="cpu",
    num_blocks=20,
    strict=False,
)
tokenizer = SimpleByteBPETokenizer(
    vocab_path="/path/to/dit/vocab.json",
    merges_path="/path/to/dit/merges.txt",
)
bus = ModuleBus()
out = call_diffusion_loop_backend(
    bus=bus,
    name="hunyuan_dit",
    model=model,
    text_embeds=tokenizer(["a cat"])["input_ids"],
    latents=torch.randn(1, 4, 64, 64),
    num_inference_steps=30,
    guidance_scale=4.5,
    sampler="flow_match_euler",
)
images = out["latents"]
```

**After** (2 行):

```python
from models.runtime import (
    load_model_and_tokenizer,
    LocalModelForTextToImage, LocalImageGenerationPipeline,
)
model, tok, family = load_model_and_tokenizer(
    "/path/to/dit", torch_dtype=torch.float16, device="cpu",
)
images = LocalImageGenerationPipeline(LocalModelForTextToImage(model, tok, family))(
    "a cat", height=512, width=512, num_inference_steps=30,
)
```

**或更短的 1 行** (用 `pipeline()` factory):

```python
from models.runtime import pipeline
out = pipeline("text-to-image", model_path="/path/to/dit", device="cpu")(
    "a cat", num_inference_steps=30,
)
```

---

## 10. Limitations + future work

| 限制 | 当前 | 计划 |
|---|---|---|
| HF Hub 在线下载 (e.g. `repo_id`) | 走项目自有 `models.source` mirror / dedup | 完整 `huggingface_hub.snapshot_download` 集成 (可选 dep) |
| Tied weight detection / key auto-mapping | 依赖手动 `*_KEY_MAP` | v0.11.0: 自动从 `state_dict` 推断 |
| `accelerate` 风格的 `disk_offload` | 暂无 | v0.11.0: 添加 disk offload |
| `safetensors` 写时压缩 | 默认不压缩 | v0.11.0: 支持 `compress=True` |
| FLUX / SD3 真权重集成 | key map 就位, 缺真权重 | v1.0.0: 与 `tencent/FLUX.1-dev` 等对齐 |
| 多 LoRA 堆叠 | 已支持 (v0.8.5) | — |
| 视频 `text-to-video` 专用 TaskHead | 共用 `LocalModelForTextToImage` | v0.11.0: 拆出 `LocalModelForTextToVideo` |

---

## 11. Why this matters

用户的两个核心痛点:

1. **"框架的基础设施基本齐全了,但很多未串联"**
   → v0.4.x P0 → v0.9.6 期间累积的 ~10000 行模块散落在
   `models/` / `core/` / `nodes/` / `papers/`,**没有**一个统一
   入口把它们串起来。本包(`models/runtime/`)的
   `load_model_and_tokenizer` + `pipeline` + `enable_local_runtime`
   **3 个** 一行函数,把"下载 → 加载 → 推理 → 注入 39 节点"打通。

2. **"本地模型下载后使用,使用自己写的 transformers 在本地加载模型
   并使用的方式实现还没有完成"**
   → v0.8.0 §11.1 的设计目标 (不依赖 `transformers` / `diffusers`,
   自研 BPE / safetensors 解析) 已部分落地 (ModelMixin + 5 个
   KEY_MAP + 自研 BPE / T5 tokenizer),但**没有**一个
   `transformers.AutoModel.from_pretrained` 风格的一行入口。
   本包的 `load_model_and_tokenizer` + `LocalModelHub` + 4 个
   `LocalModelFor*` TaskHead + 3 个 `Local*Pipeline` + `pipeline()`
   factory **完全自研**,**不依赖** `transformers` / `diffusers`,
   但 API 表面 diffusers / transformers 高度兼容 (5 维 from_pretrained
   kwargs / 3 种 pipeline / 6 种 model family / 多 device / 多 dtype)。

---

**相关文档**:

* [`docs/model_loading.md`](model_loading.md) — v0.8.0 `from_pretrained` 协议
* [`docs/V0.8_UPGRADE_PLAN.md`](V0.8_UPGRADE_PLAN.md) — v0.8.x 升级设计方案
* [`docs/architecture.md`](architecture.md) — 六层架构
* [`docs/operations.md`](operations.md) — 部署 / 监控 / checkpoint
* [`docs/placeholder_registry.md`](placeholder_registry.md) — 占位登记
