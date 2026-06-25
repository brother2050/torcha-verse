# 架构设计

本文档是 TorchaVerse 的唯一设计文档,描述六层分层架构、横切层、依赖规则、配置系统、节点系统与安全设计。面向新用户,无需了解历史版本。

## 架构概述

TorchaVerse 采用**六层分层架构**,每一层只依赖其下层,禁止反向依赖。这种设计带来三个好处:

1. **可测试性** — 底层无第三方依赖,可在最小 CI 环境中导入与测试。
2. **可组合性** — 节点是原子能力单元,通过流水线自由编排。
3. **可扩展性** — 新能力只需新增节点,无需改动既有层。

```
┌─────────────────────────────────────────────┐
│  L6 Consistency  角色/服装/场景引擎 + 评分   │
├─────────────────────────────────────────────┤
│  L5 Pipeline     DAG / 构建器 / 模板 / 画布  │
├─────────────────────────────────────────────┤
│  L4 Nodes        29 个能力节点              │
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
| L1 | Infrastructure | 配置中心、设备管理、日志、审计、资源预算、缓存、限流、检查点、资源获取 |
| L2 | Assets | 统一资产模型与版本化存储 |
| L3 | Core | 模块装配总线、采样器、扩散调度器、内存池、KV 缓存、运行时调度、工具注册 |
| L4 | Nodes | 29 个可组合能力节点与类型系统 |
| L5 | Pipeline | DAG、流水线构建器、连接校验、模板注册表、Prompt 工作室、可视化画布 |
| L6 | Consistency | 角色引擎、服装引擎、场景引擎、评分计算器、一致性流水线 |

---

## 分层详解

### L1 Infrastructure

**职责**:为所有上层提供配置、设备、日志、审计、资源管理与基础设施服务。本层无第三方依赖,仅使用 Python 标准库与 PyYAML。

**核心模块**:

| 模块 | 说明 |
|------|------|
| `ConfigCenter` | 单例配置中心,四级合并加载(System < Project < User < Run),支持点号分隔键访问与运行快照 |
| `DeviceManager` | 设备管理,自动检测 CUDA / MPS / CPU,提供设备信息查询 |
| `Logger` | 统一日志器工厂,`get_logger(name)` 返回配置好的 logger |
| `AuditLogger` | 审计日志器,记录安全与运维事件 |
| `ResourceBudget` | 硬性资源预算,限制 VRAM / RAM / 磁盘 / 并发数 |
| `CacheStore` | 通用缓存存储,支持 TTL 与 LRU 淘汰 |
| `RateLimiter` | 令牌桶限流器,保护 API 端点 |
| `CheckpointManager` | 检查点管理,保存与恢复训练/推理状态 |
| `SourceFetcher` | 资源获取器,从 HuggingFace / ModelScope / GitHub 下载模型 |
| `defaults` | 推理默认值单一数据源,从 `inference_config.yaml` 懒加载 |

**依赖规则**:本层是架构最底层,不依赖任何其他业务层。`defaults.py` 通过延迟导入 `ConfigCenter` 避免循环依赖。

### L2 Assets

**职责**:将框架中所有版本化产物(模型权重、LoRA、角色、服装、场景、深度图等)统一为单一 `Asset` 抽象,由 `AssetStore` 持久化。跨层引用通过不可变 `AssetRef` 句柄进行,避免版本漂移。

**核心模块**:

| 模块 | 说明 |
|------|------|
| `AssetType` | 资产类型枚举(MODEL / CHARACTER / OUTFIT / SCENE / DEPTH 等) |
| `AssetRef` | 不可变资产引用句柄,携带 `(type, name, version)` 三元组 |
| `Asset` | 资产基类,定义元数据、版本、许可证等公共字段 |
| `ModelAsset` | 模型权重资产,记录架构、精度、路径 |
| `CharacterAsset` | 角色资产,记录外貌、身份特征 |
| `OutfitAsset` | 服装资产,记录穿搭定义 |
| `SceneAsset` | 场景资产,记录环境与光照 |
| `DepthAsset` | 深度图资产,作为一致性条件输入 |
| `AssetStore` | 分层资产存储,支持热/冷存储与版本检索 |

**依赖规则**:仅依赖 L1。`AssetRef` 是值对象,可在层间安全传递。

### L3 Core

**职责**:提供模块装配、采样、调度、内存与缓存等运行时核心服务。本层是连接基础设施与能力节点的桥梁。

**核心模块**:

| 模块 | 说明 |
|------|------|
| `ModuleBus` | 线程安全单例注册表,统一发现/实例化/缓存所有可插拔组件,替代散落单例 |
| `Sampler` | 采样器,实现 temperature / top-k / top-p / repetition_penalty 等策略 |
| `DiffusionScheduler` | 扩散调度器,支持 DDPM / DDIM / Euler / DPM-Solver / Consistency |
| `MemoryPool` | 内存池,管理张量分配与复用 |
| `PagedKVCache` | 分页 KV 缓存,支持静态与分页策略、CPU 卸载 |
| `RuntimeScheduler` | 运行时调度器,协调并发模型加载与请求排队 |
| `ToolRegistry` | 工具注册表,为 Agent 提供可调用工具的发现与执行 |

**依赖规则**:依赖 L1 与 L2。`ModuleBus` 仅使用标准库,可在任何环境导入。组件以工厂形式注册在 `(kind, name)` 命名空间下,工厂至多调用一次并缓存实例。

### L4 Nodes

**职责**:提供 29 个可组合的能力节点,每个节点声明类型化的输入/输出契约,执行单一明确的操作。节点是流水线编排的原子单元。

**核心模块**:

| 模块 | 说明 |
|------|------|
| `BaseNode` | 节点抽象基类,提供 `validate_inputs`、`estimate_resources` 与 `_safe_execute` |
| `NodeSpec` | 节点的声明式描述(type / name / inputs / outputs / tags),附加为类属性 `spec` |
| `NodeContext` | 运行期上下文,传递 ModuleBus、AssetStore、ResourceBudget、日志器、审计器、运行配置 |
| `NodeRegistry` | 节点发现与实例化门面,基于 ModuleBus 的 `"node"` kind |
| `register_node` | 类装饰器,将节点注册到全局 ModuleBus |

**29 个节点清单**:

| 类别 | 节点 |
|------|------|
| 文本 | `text_completion`、`text_chat` |
| 图像 | `image_txt2img`、`image_img2img`、`image_upscale`、`image_inpaint` |
| 视频 | `video_txt2vid`、`video_interpolate`、`video_stitch` |
| 音频 | `audio_tts`、`audio_music` |
| 字幕 | `subtitle_generate`、`subtitle_translate`、`subtitle_burn`、`subtitle_export` |
| 一致性 | `character_apply`、`outfit_apply`、`scene_apply`、`depth_condition`、`five_view` |
| 数字人 | `dh_lip_sync`、`dh_talking_head`、`dh_portrait_animate`、`dh_full_body`、`dh_face_enhance`、`dh_voice_clone` |
| 导出 | `export_image`、`export_video`、`export_audio` |

**依赖规则**:依赖 L1、L2、L3。导入 `nodes` 包会急切导入所有节点子模块,使 `@register_node` 装饰器执行,节点随即出现在 ModuleBus 上。

### L5 Pipeline

**职责**:将 L4 节点编排为可执行的有向无环图(DAG),支持并发执行、输出传递、连接校验、模板复用与可视化画布。本层刻意不导入 `torch`。

**核心模块 — Pipeline**:

| 模块 | 说明 |
|------|------|
| `DAG` | 有向无环图数据结构,由 `DAGNode` 与 `DAGEdge` 组成 |
| `PipelineBuilder` | 流式构建器,链式调用 `.node()` / `.connect()` / `.build()` 生成 Pipeline |
| `Pipeline` | 可执行流水线,支持 `run` / `dry_run` / `validate`、YAML 序列化、取消/暂停 |
| `ConnectionValidator` | 连接校验器,8 点校验确保端口类型兼容与无环 |
| `TemplateRegistry` | 模板注册表,内置 12 个开箱即用的流水线模板,支持目录加载与搜索 |
| `PromptStudio` | Prompt 工作室,提示词工程与版本管理 |

**核心模块 — Canvas**:

| 模块 | 说明 |
|------|------|
| `Canvas` | 可视化画布,节点的空间化表示,可序列化、版本化、分享 |
| `CanvasNode` | 画布节点,携带 2D 坐标与端口信息 |
| `AutoDirector` | 自动导演,根据自然语言描述自动生成画布与流水线 |

**依赖规则**:依赖 L1-L4。节点执行器通过 `NodeContext` 懒解析 — 显式注册的执行器优先,其次 ModuleBus 的 `"node"` kind,最后回退到 passthrough(返回合并输入)。当无执行器注册时仍可完整演练编排逻辑。

### L6 Consistency

**职责**:保证跨镜头与跨时间的角色、服装、场景身份一致性。组合 L2 资产层与 L4 节点系统,提供引擎、评分与一致性流水线。

**核心模块**:

| 模块 | 说明 |
|------|------|
| `CharacterEngine` | 角色引擎,维护角色身份特征并应用到生成过程 |
| `OutfitEngine` | 服装引擎,保证穿搭定义在跨镜头间一致 |
| `SceneEngine` | 场景引擎,维护环境与光照一致性 |
| `ScoreCalculator` | 评分计算器,量化一致性程度 |
| `ConsistencyPipeline` | 一致性流水线,编排三引擎与评分,产出带一致性分数的生成结果 |

**依赖规则**:依赖 L1-L5。位于架构顶层,组合下层能力。`ConsistencyProfile` 定义权重(如 `character_weight`),`ConsistencyManager` 管理配置文件生命周期。

---

## 横切层

横切层不归属六层中的任何一层,而是为所有层提供横切关注点服务。

### Security(安全)

实现纵深防御的四道顺序关卡:

| 关卡 | 模块 | 说明 |
|------|------|------|
| 关卡 1 — 输入消毒 | `InputSanitizer` | NFC 规范化、控制字符剥离、路径遍历检测、路径白名单、提示词注入检测 |
| 关卡 2 — 沙箱执行 | `Sandbox` | 基于 AST 的静态分析 + 受限执行环境(超时与内存限制) |
| 关卡 3 — 输出过滤 | `OutputFilter` | 模型输出的毒性、NSFW、音频内容筛查 |
| 关卡 4 — 供应链审计 | `SupplyChainAudit` | 依赖漏洞扫描、CycloneDX SBOM 生成、许可证合规、审计日志聚合 |

安全层为纯 Python 实现,不依赖 `torch`,所有可选第三方后端(RestrictedPython、Detoxify、NudeNet、pip-audit、safety)均延迟导入并带 `try/except` 守卫。

### Plugins(插件)

提供声明式插件发现与加载机制,基于 ModuleBus 构建:

| 模块 | 说明 |
|------|------|
| `PluginManager` | 插件发现 / 加载 / 卸载 / 启用 / 禁用 |
| `PluginSpec` | 插件声明式描述 |
| `ManifestParser` | 解析与校验 `plugin.toml` / `plugin.yaml` 清单 |
| `BasePlugin` | 插件开发 SDK 基类 |

插件可从三种独立来源发现:
1. **入口点** — `pip` 安装的包声明 `torcha_verse.plugins` 入口点。
2. **目录** — 放入插件目录的带清单文件夹。
3. **代码** — 通过 `PluginManager.register()` 编程注册。

---

## 依赖规则

### 分层依赖原则

- **高层依赖低层**:L6 → L5 → L4 → L3 → L2 → L1,箭头方向即依赖方向。
- **禁止反向依赖**:L1 不得导入 L2 及以上;L3 不得导入 L4 及以上,以此类推。
- **同层可互依赖**:同一层内的模块可相互引用,但应避免循环。

### 循环依赖处理

当出现不可避免的循环引用时,采用两种标准模式:

**模式 1 — `TYPE_CHECKING` 守卫**(用于类型注解):

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodes.base import NodeContext  # 仅类型检查时导入,运行时不执行

class Pipeline:
    def run(self, ctx: "NodeContext") -> dict: ...
```

**模式 2 — 延迟导入**(用于运行时调用):

```python
def some_method(self):
    from nodes.base import NodeContext  # 在方法内部导入,打破循环
    ctx = NodeContext()
```

`ModuleBus` 作为单一装配点,本身仅使用标准库,不导入 `infrastructure.logger`(避免传递性拉入 `torch`),改用标准 `logging` 模块。

---

## 配置系统

### 四级合并

`ConfigCenter` 是单例配置中心,按以下顺序深度合并四个层级,优先级递增:

| 层级 | 位置 | 说明 |
|------|------|------|
| System | `config/_defaults/` | 随包发布的不可变默认值,CI 中作为黄金文件快照 |
| Project | `./config/*.yaml` | 仓库提交的项目配置 |
| User | `~/.config/torcha-verse/` | 用户偏好、API 密钥、本地路径 |
| Run | `config_snapshot.json` | 每次运行生成的快照,保证可重放 |

```
System (最低) → Project → User → Run (最高)
```

### 点号分隔访问

```python
from infrastructure.config_center import ConfigCenter

cc = ConfigCenter()
cc.get("sampling.default.temperature")    # 0.7
cc.get("diffusion.default_steps")          # 30
cc.set("default.dtype", "fp16")
cc.has("kv_cache.enabled")                # True
```

### defaults.py 单一数据源

`infrastructure/defaults.py` 是推理默认值的**唯一数据源**。所有模块必须从此导入默认值,而非硬编码。它在首次属性访问时从 `ConfigCenter` 懒加载,因此导入 `infrastructure` 不会触发 ConfigCenter 初始化:

```python
from infrastructure.defaults import (
    DIFFUSION_STEPS,           # 30
    DIFFUSION_GUIDANCE_SCALE,  # 7.5
    SAMPLING_TEMPERATURE,      # 0.7
    SAMPLING_TOP_K,            # 50
    SAMPLING_TOP_P,            # 0.9
)
```

回退值(传给 `get` 的第二参数)仅在配置文件缺失时生效,与 YAML 值保持一致。

---

## 节点系统

### NodeSpec 声明式定义

每个节点通过 `NodeSpec` 声明其契约,作为类属性 `spec` 附加:

```python
from nodes.base import BaseNode, NodeSpec, register_node

@register_node("image_txt2img")
class ImageTxt2ImgNode(BaseNode):
    spec = NodeSpec(
        type="image_txt2img",
        name="Text to Image",
        description="Generate an image from a text prompt.",
        inputs={
            "prompt": "TEXT",
            "negative_prompt": "Optional[TEXT]",
            "width": "INT",
            "height": "INT",
            "steps": "Optional[INT]",
            "seed": "Optional[SEED]",
        },
        outputs={"image": "IMAGE"},
        tags=["image", "generation"],
    )
```

`NodeSpec` 是节点身份、类型化输入/输出契约与标签的**唯一真相来源**,被 L5 流水线与画布用于校验与渲染。

### validate_inputs 校验

`BaseNode.validate_inputs` 检查每个**必填**输入(类型字符串未被 `Optional[...]` 包裹)是否存在且非 `None`。由于端口类型是不透明字符串(如 `"IMAGE"`、`"INT"`),运行时不再做 `isinstance` 检查,仅检查 `None`。未知输入被忽略(宽松策略),允许流水线传递额外元数据:

```python
def validate_inputs(self, inputs: dict) -> list[str]:
    errors = []
    for name, type_str in self.spec.inputs.items():
        optional = is_optional(type_str)
        if name not in inputs:
            if not optional:
                errors.append(f"Missing required input {name!r}.")
            continue
        if inputs[name] is None and not optional:
            errors.append(f"Required input {name!r} is None.")
    return errors
```

### _safe_execute 执行流程

`_safe_execute` 是包裹 `execute` 的安全执行包装,统一异常处理与日志记录:

```
_safe_execute(ctx, **inputs)
    │
    ├─ 1. validate_inputs(inputs)        # 执行前校验(try 块之外)
    │     └─ 失败 → 抛 ValueError(不被当作"执行失败")
    │
    ├─ 2. try: execute(ctx, **inputs)    # 执行节点逻辑
    │     └─ 捕获 OSError / RuntimeError / MemoryError
    │           ├─ 在 ctx.logger 记录 error 级别日志
    │           └─ 重新抛出(供上层 Pipeline 处理部分结果保留)
    │
    └─ 3. 返回输出字典
```

执行器解析顺序(`NodeContext.resolve_executor`):
1. 显式 `executors` 映射。
2. ModuleBus 的 `"node"` kind(若结果是 BaseNode 实例,包装为适配器闭包,将 `(inputs, ctx)` 转为 `execute(ctx, **inputs)`)。
3. 返回 `None`(passthrough,返回合并输入)。

---

## 安全设计

TorchaVerse 采用**纵深防御**策略,通过四道顺序关卡保护系统:

### 1. 输入消毒(InputSanitizer)

- **NFC 规范化**:统一 Unicode 表示,防止同形异义攻击。
- **控制字符剥离**:移除不可见控制字符。
- **路径遍历检测**:拦截 `../` 等路径注入。
- **路径白名单**:仅允许访问授权路径。
- **提示词注入检测**:识别"忽略以上指令"等注入模式。

```python
from security import InputSanitizer

s = InputSanitizer()
clean = s.sanitize_text(user_input)          # 消毒
result = s.detect_prompt_injection(prompt)   # 检测注入
if result.is_injected:
    raise ValueError("Prompt injection detected")
```

### 2. 沙箱执行(Sandbox)

- **AST 静态分析**:执行前分析不可信 Python 代码的抽象语法树。
- **受限环境**:限制可访问的模块与内建函数。
- **超时与内存限制**:防止资源耗尽攻击。

```python
from security import SandboxExecutor, SandboxConfig

executor = SandboxExecutor(SandboxConfig(timeout=10, memory_limit=512))
result = executor.run(untrusted_code)
```

### 3. 输出过滤(OutputFilter)

- **毒性检测**:筛查生成文本的有害内容。
- **NSFW 检测**:筛查图像的成人内容。
- **音频内容筛查**:筛查音频输出。

```python
from security import OutputFilter

f = OutputFilter()
result = f.filter_text(generated_text)
if not result.passed:
    # 拦截输出
    return error_response("Output filtered: " + result.action)
```

### 4. 供应链审计(SupplyChainAudit)

- **依赖漏洞扫描**:扫描已安装依赖的已知漏洞。
- **CycloneDX SBOM 生成**:生成软件物料清单。
- **许可证合规**:检查依赖许可证兼容性。
- **审计日志聚合**:汇总安全事件供审计追溯。

### 安全集成位置

安全关卡集成在 API 服务层(`serving/app.py`)的每个端点中:

```
请求到达
  │
  ├─ 限流检查 (RateLimiter)
  ├─ 关卡 1: 输入消毒 (sanitize_text) + 注入检测
  ├─ 调用 PipelineService → 节点执行
  ├─ 关卡 3: 输出过滤 (filter_text / filter_image)
  └─ 返回响应
```

关卡 2(沙箱)在执行不可信代码时启用,关卡 4(审计)持续运行记录所有安全事件。
