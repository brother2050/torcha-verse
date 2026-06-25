# TorchaVerse Examples Catalog

> 每个示例的详细说明:节点映射、依赖、真实模型路径、失败排查、相关测试。
>
> 总览索引见 [`examples/README.md`](../examples/README.md)。本文件按
> 1 个 example 1 个小节组织,信息深度大于 README。
>
> 最近一次更新: 2026-06-25(v0.4.x 收尾)

---

## 1. `basic_text_gen.py` — 1 节点 echo 演示

**目标**:用最小代码演示 v0.3 架构下的 L4 节点 + PipelineBuilder。

**关键 API**:
- `nodes.base.NodeContext` — 节点执行上下文(空 dict 即可)
- `pipeline.composer.PipelineBuilder` — 链式构造 pipeline
  - `.node(name, id, **kwargs)` 注册一个节点
  - `.build()` 编译
  - `.run(ctx)` 执行, 返回 `{node_id: output_dict}`

**节点映射**:
- L4 `text_chat` (`nodes/text_chat.py`)

**Backend**:
- 默认走 `nodes/_helpers.py` 里的 echo factory, 不调任何真实模型
- 走 `core.module_bus.ModuleBus` 解析 backend, 输出是 echo

**期望输出**:
```
[output keys] ['text', 'usage']
[text]        <echo of the prompt>
[usage]       {'input_tokens': N, 'output_tokens': N}
```

**跑法**:
```bash
python examples/basic_text_gen.py
```

**CI smoke**: `tests/test_examples_import.py::TestExamplesImport`

---

## 2. `real_text_chat.py` — v0.4.0 P0 headline demo

**目标**:**真跑**项目自有 tiny Transformer LM(纯 torch, 零外部依赖) +
L4 `text_chat` 节点端到端。

**关键 API**:
- `models.providers.TINY_CONFIG` / `SMALL_CONFIG` — 模型超参 preset
- `models.providers.TrainConfig` — 训练超参
- `models.providers.train_tiny_transformer` — 训练函数
- `models.providers.LocalTorchTextProvider` — 文本 provider (满足
  `LLMProvider` 协议, 暴露 `generate(prompt, **kwargs) -> str`)
- `models.providers.fetch_and_load_text` — 从 `ModelCache` 取
  checkpoint(若已存在), 否则 build 一个 random-init provider
- `nodes._helpers.register_default_text_backend` — 把 backend
  factory 注册到 ModuleBus
- `nodes._helpers.call_text_backend` — 直调 backend (绕过 Pipeline)
- `pipeline.composer.PipelineBuilder` + `nodes.base.NodeContext`

**节点映射**:
- L4 `text_chat` (`nodes/text_chat.py`)

**Backend**:
- 项目自有 `LocalTorchTextProvider` (纯 torch, 字节级 tokenizer)
- 训练数据: 内存小语料 (~1k tokens)
- tiny preset: 0.3M 参数, 训练 ~2s
- small preset: 10M 参数, 训练 ~30s (CPU)

**两条 prompt**(证明字节级 tokenizer 无需手动预处理):
1. `Hello, who are you?` (English)
2. `用一句话介绍 TorchaVerse 框架` (Chinese)

**跑法**:
```bash
# 跑 tiny preset, 跳过训练(用项目仓库里的 checkpoint)
python examples/real_text_chat.py --preset tiny --skip-pretrain

# 跑 small preset(默认), 含 ~30s 训练
python examples/real_text_chat.py --preset small

# 用自己的 checkpoint
python examples/real_text_chat.py --skip-pretrain \
    --checkpoint assets/checkpoints/tiny-transformer-small.pt
```

**失败排查**:
- `ModuleNotFoundError: models.providers.pretrain_tiny` — 没装项目:
  `pip install -e .`
- `RuntimeError: checkpoint not found` — 加 `--skip-pretrain` 跳过
  检查, 或先 `train_tiny_transformer()` 跑一次

**CI smoke**: `tests/test_examples_import.py::TestExamplesImport` 只
验 import-clean; `main()` 跑通与否靠 `tests/test_models.py` /
`tests/test_models_source.py` 间接覆盖。

---

## 3. `image_gen.py` — UNet + VAE 真模型端到端

**目标**:L4 `image_txt2img` 节点 + 项目自有 `LocalTorchImageProvider`
跑通真模型前向(64×64, ~3-5s CPU)。

**关键 API**:
- `nodes._helpers.register_default_image_backend` — 注册
  `LocalTorchImageProvider` factory
- `nodes.image.image_txt2img` (L4 节点)
- 节点参数: `prompt`, `width`, `height`, `steps`, `guidance_scale`,
  `seed`

**节点映射**:
- L4 `image_txt2img` (`nodes/image.py`)

**Backend**:
- `models.providers.local_image.LocalTorchImageProvider`
- 架构: UNet (小) + VAE (小) — 纯 torch, 随机初始化即可出图

**期望输出**:
```
[1] image_txt2img (64x64) via LocalTorchImageProvider...
    output keys:  ['image', 'seed', 'steps', ...]
    image tensor: shape=(3, 64, 64)
    seed:         42
    steps:        4
    elapsed:      ~3-5s
```

**跑法**:
```bash
python examples/image_gen.py
```

**已知限制**: 第二段 "chained pipeline" (`image_txt2img -> image_upscale`)
因 `image_upscale` 上游的 tensor/bool 类型检查未完全适配 P0 真模型
的 tensor 路径, 当前**禁用**; 单节点 `image_txt2img` 路径已可用。

**CI smoke**: import-clean via `TestExamplesImport`。

---

## 4. `video_gen.py` — VideoDiT + VideoVAE 真模型端到端

**目标**:L4 `video_txt2vid` 节点 + 项目自有 `LocalTorchVideoProvider`
跑通真模型前向(短视频片段, 2-4 帧)。

**关键 API**: 同 `image_gen.py` (image provider 换 video provider)。

**节点映射**:
- L4 `video_txt2vid` (`nodes/video.py`)

**Backend**: `models.providers.local_video.LocalTorchVideoProvider`
- 架构: VideoDiT (小) + VideoVAE (小) — 纯 torch
- 输出: `(T, 3, H, W)` tensor

**跑法**: `python examples/video_gen.py`

**CI smoke**: import-clean via `TestExamplesImport`。

---

## 5. `audio_tts.py` — TTS-Transformer + HiFi-GAN 真模型端到端

**目标**:L4 `audio_tts` 节点 + 项目自有 `LocalTorchAudioProvider`
跑通真模型前向(短 mel-spectrogram → waveform)。

**节点映射**:
- L4 `audio_tts` (`nodes/audio.py`)

**Backend**: `models.providers.local_audio.LocalTorchAudioProvider`
- 架构: TTS-Transformer + HiFi-GAN — 纯 torch
- 输出: `(samples,)` waveform tensor, sample rate 由 provider 给

**跑法**: `python examples/audio_tts.py`

**CI smoke**: import-clean via `TestExamplesImport`。

---

## 6. `consistency_character.py` — L6 一致性 + L4 角色图

**目标**:演示 L6 `consistency` 子系统: `character_apply` 角色化生图 +
`character_five_view` 五视图扩展。

**关键 API**:
- L4 `image_txt2img` / `image_character_apply` / `image_five_view`
  (`nodes/image.py`)
- L6 `consistency.character` (profile + asset references)

**节点映射**:
- L4 `image_character_apply` (受 L6 profile 约束)
- L4 `image_five_view` (5 视角展开)
- L4 `image_txt2img` (上游 base image)

**Backend**: 项目自有 `LocalTorchImageProvider` (与 `image_gen.py` 同)。

**跑法**: `python examples/consistency_character.py`

**CI smoke**: import-clean via `TestExamplesImport`。

---

## 7. `dh_lipsync.py` — 数字人口型同步

**目标**:L4 `dh_lip_sync` 节点 + 项目自有 `LocalTorchVideoProvider`
跑通真模型前向(把 driving audio 的口型 reanimate 到目标 video)。

**节点映射**:
- L4 `dh_lip_sync` (`nodes/digital_human.py`)

**Backend**: `LocalTorchVideoProvider` (与 `video_gen.py` 同)。

**跑法**: `python examples/dh_lipsync.py`

**CI smoke**: import-clean via `TestExamplesImport`。

---

## 8. `agent_demo.py` — ReAct agent + 工具调用 + 多 agent 编排

**目标**:演示 L4 + agents 子系统: 工具注册 → 工具执行 → ReAct 循环
(模拟) → 多 agent sequential flow (模拟)。

**关键 API**:
- `tools.calculator.CalculatorTool` — 数学表达式求值
- `tools.python_executor.PythonExecutorTool` — 子进程沙箱执行 Python
- `tools.file_ops.FileOpsTool` — 读 / 写 / 删文件
- `core.tool_registry.ToolRegistry` — 工具注册中心
  - `.register_tool(name, func, description, parameter_schema)`
  - `.execute_tool(name, params) -> Any`
  - `.get_tool_descriptions() -> List[Dict]`

**流程**:
1. 注册 3 个工具(calculator / python_executor / file_ops)
2. 测试 calculator: `2 + 3 * 4`, `sqrt(144) + 10`, `sin(pi/2)`
3. 测试 python_executor: 跑 `sum(range(100))`
4. 测试 file_ops: 写 + 读 `data/outputs/agent_test.txt`
5. 模拟 ReAct loop: `10! → sqrt(10!)`
6. 模拟 multi-agent sequential flow: Researcher → Writer → Reviewer

**期望输出**: 各步执行结果 + 最终答案 + agent 流串接文本

**跑法**: `python examples/agent_demo.py`

**真实模型依赖**: 无 (完全离线)

**CI smoke**: import-clean + main() 可调 via `TestExamplesImport`。

---

## 9. `rag_demo.py` — RAG 4 步链路演示

**目标**:演示 `rag/` 子系统的端到端: 文档摄取 → chunking → 向量存储
→ 检索 → context 组装。

**关键 API**:
- `rag.loaders.document_loader.DocumentLoaderFactory` / `Document` —
  文档加载抽象 + 数据类
- `rag.chunkers.text_chunker.RecursiveChunker` — 递归字符切分
  (chunk_size=100, overlap=20)
- `rag.vectorstore.vector_store.InMemoryVectorStore` — 内存向量库
  (cosine 距离)
- `rag.retrievers.retriever.VectorRetriever` + `ContextAssembler` —
  检索器 + context 组装器

**流程**:
1. 构造 4 个 `Document`(项目自描述: 概述 / 架构 / 核心层 / 推理)
2. `RecursiveChunker.chunk(content)` 切成块
3. 用 `torch.randn` 做 dummy embedding 写入 `InMemoryVectorStore`
4. `VectorRetriever.retrieve(query, top_k=3)` 检索
5. `ContextAssembler.assemble(results)` 组装 context
6. 打印 "基于检索 context 的 RAG 答案" (硬编码示例)

**期望输出**: 4 docs → 4-6 chunks → 3 retrieve results → context 串

**跑法**: `python examples/rag_demo.py`

**真实模型依赖**: 无 (embeddings 是 `torch.randn`, 不是真模型)

**CI smoke**: import-clean via `TestExamplesImport`。

---

## 10. `model_download.py` — 模型源子系统完整 surface

**目标**:展示 v0.4.x P2+ / P2++ 的 `ModelFetcher` API: 镜像列表 →
健康检查 → 进度回调 → 跨镜像去重 → token 解析 → sha256 pin → 401/403
处理。

**关键 API**:
- `models.source.MirrorSet` — 镜像集合
- `models.source.check_mirror_health` — 健康探测
- `models.source.DownloadProgress` — 进度回调 dataclass
- `models.source.ModelFetcher` — 统一 fetcher
  - `.fetch(source, repo_id, ...)` — 主入口
  - 支持 token / expected_sha256s / 进度回调
- `models.source.auth.{resolve_token, auth_headers, GatedRepoError,
  ChecksumMismatch}` — token 解析 + 异常类
- `models.source.HttpTransport` + 内存实现 `FakeTransport`
  (demo 自带, 用于零网络跑通)

**5 个演示场景**:
1. **镜像列表构造** — 从默认 catalog 构建 `MirrorSet`
2. **镜像健康检查** — 探测每个 mirror, 失败的不参与后续 fetch
3. **进度回调** — 文件级 tick + bytes / total / elapsed
4. **跨镜像去重** — 同一 file set 用不同 `(repo_id, revision)`
   fetch 两次, 第二次应当 short-circuit (`from_cache=True`)
5. **token + sha256 pin + 401/403** — `GatedRepoError` 和
   `ChecksumMismatch` 都演示

**Backend**:
- 默认走 demo 自带的 `FakeTransport`, 零网络
- 加 `--real` 切到 `UrllibTransport` 走真镜像 (需要网络)

**跑法**:
```bash
python examples/model_download.py             # 零网络
python examples/model_download.py --real      # 走真镜像
```

**已知 fail 模式**:
- `--real` 模式报 401 → 设 `$HF_TOKEN` (或 `$HUGGING_FACE_HUB_TOKEN`)
- `--real` 模式报 ChecksumMismatch → 镜像被污染, 改用其他 mirror

**CI smoke**: import-clean via `TestExamplesImport`; fake-transport 路径
在 `tests/test_model_source_integrity.py` 里有更深入的覆盖。

---

## 全局约定

### Run with:: 与 Usage::

examples 的 docstring 里两种约定都被接受, `tests/test_examples_import.py`
同时认两种。 `Run with::` 是新代码的首选, 老代码(比如
`real_text_chat.py`) 沿用 `Usage::`。

### sys.path 注入

每个 example 顶部都有这一行:
```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```
让 `python examples/<name>.py` 在任意 cwd 都能 import `nodes.*` /
`models.*`。CI 里跑测试时是 `pip install -e .`, 不依赖这行。

### `if __name__ == "__main__":` 守卫

所有 example 的入口都包了 `if __name__ == "__main__":`, 保证
`import examples.<name>` 不会真跑 demo。 这条是
`tests/test_examples_import.py::TestExamplesImport::test_module_has_main_guard`
的强制约束。

### 默认离线

所有 example 设计为**默认离线 + 默认 echo / fake backend**。
真模型 / 真网络入口都通过显式 flag 打开 (`--real` / `--preset small`
/ 不传 `--skip-pretrain` 等)。
