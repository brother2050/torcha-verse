# 架构设计

本文档是 TorchaVerse 的唯一设计文档:六层分层、横切层、依赖规则、配置系统、节点系统与安全设计。

## 架构概述

六层分层架构,每层只依赖其下层,禁止反向依赖。

```
┌─────────────────────────────────────────────┐
│  L6 Consistency  角色/服装/场景引擎 + 评分   │
├─────────────────────────────────────────────┤
│  L5 Pipeline     DAG / 构建器 / 模板 / 画布  │
├─────────────────────────────────────────────┤
│  L4 Nodes        39 个能力节点              │
├─────────────────────────────────────────────┤
│  L3 Core         模块总线 / 采样器 / 缓存    │
├─────────────────────────────────────────────┤
│  L2 Assets       资产模型 / 版本化存储       │
├─────────────────────────────────────────────┤
│  L1 Infrastructure 配置 / 设备 / 日志 / 审计 │
└─────────────────────────────────────────────┘
        横切层:Security | Plugins
```

| 层级 | 名称 | 核心职责 |
|------|------|----------|
| L1 | Infrastructure | 配置中心、设备、日志、审计、资源预算、限流、检查点、模型下载、推理默认值 |
| L2 | Assets | 统一资产模型 + 三级 (热/温/冷) 分层存储 |
| L3 | Core | 模块装配总线、采样器、扩散调度器、内存池、KV 缓存、运行时调度、工具注册 |
| L4 | Nodes | 39 个可组合能力节点,统一 `BaseNode` + `NodeSpec` 契约 |
| L5 | Pipeline | DAG、流式构建器、连接校验、模板、Prompt 工作室、可视化画布 |
| L6 | Consistency | 角色/服装/场景引擎 + 评分计算器 |

---

## 分层详解

### L1 Infrastructure

`infrastructure/` 目录。无第三方依赖,只使用 Python 标准库 + PyYAML。

| 模块 | 说明 |
|------|------|
| `ConfigCenter` | 单例配置中心,四级合并 (System < Project < User < Run) + 点号访问 |
| `DeviceManager` | CUDA / MPS / CPU 自动检测 |
| `Logger` | `get_logger(name)` 工厂 |
| `AuditLogger` | 安全与运维事件审计 |
| `ResourceBudget` | VRAM / RAM / 磁盘 / 并发 硬性预算 |
| `CacheStore` | TTL + LRU 缓存 |
| `RateLimiter` | 令牌桶限流 |
| `CheckpointManager` | 训练/推理 检查点 |
| `SourceFetcher` | HuggingFace / ModelScope / GitHub 模型下载 |
| `defaults` | 推理默认值单一数据源,从 `inference_config.yaml` 懒加载 |

`defaults.py` 通过延迟导入 `ConfigCenter` 打破循环依赖。

### L2 Assets

`assets/` 目录。仅依赖 L1。资产基类 `Asset` 携带元数据/版本/许可证,5 个具体子类:

| 子类 | 用途 |
|------|------|
| `ModelAsset` | 模型权重 + 架构/格式/配置 |
| `CharacterAsset` | 角色 (reference_images / 5-view / embedding) |
| `OutfitAsset` | 服装 (style embedding + LoRA) |
| `SceneAsset` | 场景 (LoRA + ControlNet + depth) |
| `DepthAsset` | 深度图 (源图 + 路径 + 估算方法) |

`AssetStore` 提供三级存储:
- **热 (Hot)** — 进程内 LRU 缓存 (`HotCache`)
- **温 (Warm)** — 内容寻址本地对象存储 + WAL SQLite 元数据索引
- **冷 (Cold)** — S3 / OSS / MinIO 协议 (`ColdStorageProtocol` 抽象)

不可变 `AssetRef` 句柄携带 `(asset_id, revision, content_hash)`,可安全跨层传递。

### L3 Core

`core/` 目录。依赖 L1 + L2。

| 模块 | 说明 |
|------|------|
| `ModuleBus` | 线程安全单例注册表,`(kind, name)` 命名空间,工厂缓存 |
| `Sampler` | temperature / top-k / top-p / repetition_penalty |
| `DiffusionScheduler` | DDPM / DDIM / Euler / DPM-Solver / Consistency |
| `MemoryPool` | 张量分配与复用 |
| `PagedKVCache` | 分页 KV 缓存 + CPU 卸载 |
| `RuntimeScheduler` | 并发模型加载与请求排队 |
| `ToolRegistry` | Agent 工具发现与执行 |

`ModuleBus` 仅用标准库,无 `infrastructure.logger` 依赖,改用 `logging`。

### L4 Nodes

`nodes/` 目录。依赖 L1-L3。导入 `nodes` 包即触发所有节点子模块的 `@register_node` 装饰器,节点随即注册到 `ModuleBus` 的 `"node"` kind。

| 基类 | 说明 |
|------|------|
| `BaseNode` | 抽象基类,提供 `validate_inputs` / `estimate_resources` / `_safe_execute` |
| `NodeSpec` | 声明式契约 (`type` / `name` / `inputs` / `outputs` / `tags`) |
| `NodeContext` | 运行期上下文 (L4 + L5 统一) |
| `NodeRegistry` | 节点发现与实例化 facade |
| `register_node` | 类装饰器,挂到 `ModuleBus` |

**39 节点清单**:

| 域 | 节点 |
|------|------|
| 文本 | `text_chat` / `text_complete` |
| 图像 | `image_txt2img` / `image_img2img` / `image_upscale` / `image_inpaint` |
| 视频 | `video_txt2vid` / `video_interpolate` / `video_stitch` |
| 音频 | `audio_tts` / `audio_music` |
| 字幕 | `subtitle_generate` / `subtitle_translate` / `subtitle_burn` / `subtitle_export` |
| 一致性 | `character_apply` / `outfit_apply` / `scene_apply` / `depth_condition` / `character_five_view` |
| 数字人 | `dh_lip_sync` / `dh_talking_head` / `dh_portrait_animate` / `dh_full_body` / `dh_face_enhance` / `dh_voice_clone` |
| 导出 | `export_image` / `export_video` / `export_audio` |
| RAG | `rag_ingest` / `rag_query` / `rag_delete` / `rag_list_indices` |
| Agent | `agent_react` / `agent_plan_solve` |

### L5 Pipeline

`pipeline/` 目录。依赖 L1-L4。**刻意不导入 `torch`**。

| 模块 | 说明 |
|------|------|
| `DAG` / `DAGNode` / `DAGEdge` | 有向无环图数据结构 |
| `PipelineBuilder` | 流式 `.node() / .connect() / .build()` |
| `Pipeline` | 可执行流水线 (`run` / `dry_run` / `validate`) |
| `ConnectionValidator` | 8 点端口校验 |
| `TemplateRegistry` | 12 个开箱即用模板 + 目录加载 |
| `PromptStudio` | 提示词工程与版本管理 |
| `Canvas` / `AutoDirector` | 可视化画布 + 自动编排 |

执行器解析顺序:
1. 显式 `ctx.executors` 映射
2. `ModuleBus` 的 `"node"` kind
3. passthrough (返回合并输入)

### L6 Consistency

`consistency/` 目录。依赖 L1-L5。

| 模块 | 说明 |
|------|------|
| `CharacterEngine` / `OutfitEngine` / `SceneEngine` | 身份/穿搭/场景引擎 |
| `ScoreCalculator` | 一致性量化评分 |
| `ConsistencyPipeline` | 三引擎 + 评分编排 |

---

## 横切层

### Security

纵深防御 4 道关卡,纯 Python 不依赖 `torch`:

| 关卡 | 模块 |
|------|------|
| 1. 输入消毒 | `InputSanitizer` (NFC 规范化 + 控制字符剥离 + 路径遍历检测 + 提示词注入检测) |
| 2. 沙箱执行 | `Sandbox` (AST 静态分析 + 受限环境 + 超时) |
| 3. 输出过滤 | `OutputFilter` (毒性/NSFW/音频内容) |
| 4. 供应链审计 | `SupplyChainAudit` (漏洞扫描 + CycloneDX SBOM + 许可证合规) |

集成在 `serving/app.py` 每个端点:限流 → 输入消毒 → Pipeline 执行 → 输出过滤 → 响应。

### Plugins

基于 `ModuleBus` 的声明式发现,三种来源:
1. 入口点 (`torcha_verse.plugins`)
2. 插件目录 (`plugin.toml` / `plugin.yaml`)
3. 编程注册 (`PluginManager.register()`)

---

## 依赖规则

- **高层依赖低层**:L6 → L5 → L4 → L3 → L2 → L1,箭头方向即依赖方向
- **禁止反向依赖**:L1 不得导入 L2+,依此类推
- **同层可互依赖**,但应避免循环

循环依赖 2 个标准模式:
```python
# 模式 1: TYPE_CHECKING 守卫 (类型注解)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nodes.base import NodeContext
def run(self, ctx: "NodeContext"): ...

# 模式 2: 延迟导入 (运行时)
def some_method(self):
    from nodes.base import NodeContext
    ctx = NodeContext()
```

---

## 配置系统

`ConfigCenter` 四级合并,优先级递增:

| 层级 | 位置 |
|------|------|
| System | `config/_defaults/` (随包发布) |
| Project | `./config/*.yaml` |
| User | `~/.config/torcha-verse/` |
| Run | `config_snapshot.json` (运行快照,可重放) |

```python
from infrastructure.config_center import ConfigCenter
cc = ConfigCenter()
cc.get("sampling.default.temperature")    # 0.7
cc.get("diffusion.default_steps")         # 30
cc.set("default.dtype", "fp16")
```

`infrastructure/defaults.py` 是推理默认值**唯一数据源**,所有模块必须从这里 import,首次访问时从 `ConfigCenter` 懒加载:

```python
from infrastructure.defaults import (
    DIFFUSION_STEPS, DIFFUSION_GUIDANCE_SCALE,
    SAMPLING_TEMPERATURE, SAMPLING_TOP_K, SAMPLING_TOP_P,
)
```

回退值 (传给 `get` 的第二参数) 仅在配置文件缺失时生效,必须与 YAML 一致。

---

## 节点系统

### 声明式定义

```python
from nodes.base import BaseNode, NodeSpec, register_node

@register_node("image_txt2img")
class ImageTxt2ImgNode(BaseNode):
    spec = NodeSpec(
        type="image_txt2img",
        name="Text to Image",
        inputs={"prompt": "TEXT", "width": "INT", ...},
        outputs={"image": "IMAGE"},
        tags=["image", "generation"],
    )
```

### _safe_execute 流程

```
_safe_execute(ctx, **inputs)
  ├─ 1. validate_inputs(inputs)    # 执行前校验 (try 块外,失败 → ValueError)
  ├─ 2. try: execute(ctx, **inputs)
  │     └─ 捕获异常 → ctx.logger.error() + 重新抛出
  └─ 3. 返回输出字典
```

`validate_inputs` 检查必填输入(类型字符串未被 `Optional[...]` 包裹)是否存在且非 `None`;端口类型是 opaque string,不做 `isinstance` 检查;未知输入忽略(宽松策略)。
