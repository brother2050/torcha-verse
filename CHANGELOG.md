# Changelog

本文件记录 TorchaVerse 框架各版本的变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.3.2] - 2026-06-25

### 架构清理 — 移除过时代码与文档

本轮聚焦于移除历史遗留的孤儿模块、兼容层与过时文档，并补齐架构与操作文档，
在保持全部测试通过的前提下完成代码库瘦身与文档对齐。

#### 删除过时代码

- 删除 `evaluation/` 孤儿模块（无引用的废弃评估代码）。
- 删除 `config_manager.py` 别名（统一使用 `config_center.py`）。
- 删除 `api_server.py` 兼容层（历史 API 入口已迁移）。
- 删除 `consistency/pipeline.py` 中的 deprecated 方法。

#### 删除过时文档

- 删除 `docs/plans/` 历史规划文档（含 4.2MB 二进制字体与 JS 资源），
  对应版本均已完成。

#### 新增文档

- 创建 `architecture.md` 设计文档。
- 创建 `operations.md` 操作文档。
- 更新 `README.md`。

#### 配置与文档引用修正

- 修复 `.dockerignore`：移除不存在的 `r*-deep-review/` 条目。
- 修复 `.gitignore`：移除未使用的 `config.dev.yaml` / `config.prod.yaml` 条目。
- 修复 `rag/retrievers/retriever.py` docstring 中对不存在
  `torcha_verse.engines.TextEngine` 的交叉引用。
- 修复 `agents/tool_call_agent.py` docstring 中对不存在
  `torcha_verse.engines.text_engine.ToolCall` 的交叉引用。
- 更新 `examples/rag_demo.py` 示例文档内容：四层架构描述更新为六层架构
  （L1 Infrastructure → L2 Assets → L3 Core → L4 Nodes → L5 Pipeline → L6 Consistency），
  旧组件名（ModelRegistry/TokenizerHub/KVCacheManager/MemoryManager）更新为当前组件名
  （ModuleBus/Sampler/MemoryPool/PagedKVCache）。

## [0.3.1] - 2026-06-24

### 第二轮修复 (R0-R3) — 基础设施 / 资产 / 安全 / 插件层

本轮聚焦于基础设施层、资产层、安全层与插件层的加固与完善，在保持 331 项测试全部通过的前提下
补齐依赖、文档、安全集成与并发安全。

#### R0 文档与依赖补全

- **R0-1 README 示例修复**
  - 所有 `p.run()` 调用改为 `p.run(NodeContext())`，并补充 `from nodes.base import NodeContext` 导入。
  - 将 `subtitle_burn` 连接到 `image_upscale` 的示例改为类型兼容的 `IMAGE -> IMAGE` 连接
    （改用 `image_img2img` 节点的 `input_image` 端口）。
  - 测试数量说明由 301 更新为 331。
  - 连接校验点数由 "7-point" 更新为 "8-point"。
  - 一致性框架描述由 "四套件" 更新为 "三引擎 + Depth 节点"。

- **R0-2 setup.py 依赖补全**
  - `install_requires` 新增 `pydantic>=2.0.0`、`gradio>=4.0.0`、`faiss-cpu>=1.7.4`。
  - 新增 `extras_require`：`quantization`（bitsandbytes）与 `dev`（pytest）可选依赖组。

- **R0-3 CHANGELOG 创建**
  - 新建本文件，记录第一轮 (P0-P3) 与第二轮 (R0-R3) 全部修复。

- **R0-4 文档注释修正**
  - `nodes/base.py`：删除 NodeContext 概述中 "(future)" 前缀（L5 管线层已落地）。
  - `consistency/pipeline.py`：模块与类 docstring 中 "four engines" 修正为 "three engines"
    （Depth 仅为条件化输入，非独立引擎）。
  - `core/module_bus.py`：docstring 中 `node.image_txt2img` 命名空间示例修正为扁平 `node` kind。
  - `pipeline/composer.py`：`NodeContext` 类概述补充 `strict_mode` / `budget` 字段描述。

#### R0 安全集成

- **R0-5 InputSanitizer 集成**
  - `nodes/export.py`：`export_image` / `export_video` / `export_audio` 三个节点的 `execute()`
    在使用 `path` 输入前调用 `InputSanitizer.sanitize_path()` 进行路径净化，拒绝越界路径。
  - `nodes/subtitle.py`：`subtitle_generate`（`media_path`）与 `subtitle_export`（`path`）
    节点的 `execute()` 同样接入路径净化。

- **R0-6 插件沙箱 AST 分析**
  - `plugins/manager.py`：`_import_file()` 在 `importlib` 执行插件源码前，使用
    `security.sandbox.ASTAnalyzer.analyze()` 对源码进行静态分析，命中危险调用 / 危险导入 /
    敏感文件访问时抛出 `PluginError` 拒绝加载。

#### S1 并发安全重构（锁外执行 I/O）

- **S1-2 PluginManager 三阶段锁外 I/O**
  - `plugins/manager.py` `load()` 重构为三阶段：
    - Phase 1（锁内）：检查状态、标记加载中。
    - Phase 2（锁外）：执行 `import_module` 与 `_call_hook`，`sys.path` 修改在锁外完成并回滚。
    - Phase 3（锁内）：更新元数据、处理重复加载竞争。

- **S1-3 AssetStore 三阶段锁外 I/O**
  - `assets/store.py` `put()` 重构为三阶段：
    - Phase 1（锁内）：检查状态、生成临时路径。
    - Phase 2（锁外）：执行文件复制与哈希计算。
    - Phase 3（锁内）：原子重命名、更新数据库与热缓存元数据。

#### S2 资产层与基础设施层增强

- **S2-5 AssetStore 哈希优化**
  - `put()` 中先比较文件大小与已有记录，大小不同则跳过哈希计算直接存储，避免重复哈希开销。

- **S2-6 AssetStore 事务保护**
  - `put()` 的内容存储、资产加载与资产保存操作包裹在显式 SQLite 事务中，保证原子性。

- **S2-7 AssetStore close 标志**
  - 新增 `_closed` 标志，`close()` 后置位；所有公共方法检查标志并在已关闭时抛出
    `RuntimeError("AssetStore is closed")`。

- **S2-8 SourceFetcher 进度跟踪**
  - `infrastructure/source_fetcher.py` `_parallel_fetch()` 中的 `downloaded["bytes"] += 0`
    占位符替换为真实进度累加逻辑，按分片完成字节数聚合更新总进度。

---

### 第一轮修复 (P0-P3) — 全面审计优化

第一轮基于 v0.3.1 审计完成 21 项修复，测试由 301 增至 331（新增 `tests/test_v03_templates.py` 30 项）。
节点现已通过总线解析执行（而非 passthrough）。

#### P0 关键修复 (4)

- `pipeline/composer.py`：修复 L4-L5 命名空间不匹配 —— 管线现以 `kind='node'` 解析节点
  （原为 `node.<type>`）。
- `pipeline/composer.py`：修复执行器签名不兼容 —— 适配闭包将 `(inputs, ctx)` 转换为
  `node.execute(ctx, **inputs)`。
- `pipeline/composer.py`：passthrough 日志级别由 debug 提升为 warning，并新增 `strict_mode`
  参数在缺失执行器时抛出 `RuntimeError`。
- `nodes/consistency.py`：修复 `ConsistencyProfile` 桥接 —— 一致性节点 spec 新增权重输入
  （`character_weight` / `outfit_weight` / `scene_weight` / `depth_weight`）使 `apply_to_node()`
  参数可透传。

#### P1 严重修复 (3)

- `pipeline/templates.py`：将全部 35 个模板的 `node_type` 对齐到已注册的 29 个节点，并修正端口键
  （`frames -> image`、`images -> image` 等）。
- `core/module_bus.py` / `papers/registry.py` / `infrastructure/config_*.py`：以类级 `Lock` +
  双重检查锁定 (DCL) 修复单例初始化竞争。
- `nodes/base.py`：新增 `_NODE_CLASSES_LOCK`（RLock）保护全局节点类索引字典。

#### P2 中等修复 (9)

- `infrastructure/config_center.py` / `config_manager.py`：新增 RLock 保护全部 `_config` 访问
  （get / set / load / merge / snapshot）。
- `canvas/canvas.py`：新增 `_replace_state()` 方法；`canvas/versioning.py` 的 revert / merge
  改用该方法，不再直接修改 `_state`。
- `assets/store.py`：将文件复制移出锁（三阶段：检查 / 复制 / 更新），通过 `os.replace` 原子重命名。
- `plugins/manager.py`：将模块导入与钩子调用移出锁（三阶段：检查 / 导入 / 更新），含重复加载检测。
- `pipeline/dag.py`：在 `topological_sort` 与 `parallel_groups` 中预构建反向邻接表，
  将复杂度由 O(V*E) 降至 O(V+E)。
- `canvas/canvas.py`：`_load_specs()` 新增基于 TTL（5s）的 spec 缓存。
- `performance/optimizer.py`：SDPA 通过 `torch.backends.cuda` 启用 Flash / 内存高效后端
  （而非仅设置标签）。
- `performance/quantization.py`：`_bnb_replace_linear` 通过 `Params4bit` 复制原始权重，
  并将 bias 拷贝到新的 `Linear4bit` 模块。
- `canvas/canvas.py`：`state` 属性与 `_load_specs` 改进。

#### P3 低优先级修复 (5)

- `canvas/sharing.py`：使用 `hmac.compare_digest` 进行常数时间密码比较（防时序侧信道）。
- `consistency/pipeline.py`：修复漂移检测占位符 —— 距离现跨度 [0, 0.18]，可超过默认阈值 0.15。
- `canvas/canvas.py`：`to_pipeline()` 在空画布时抛出 `ValueError`。
- `nodes/type_system.py`：移除兼容性矩阵中的冗余自引用（`is_compatible` 已处理自匹配）。
- `pipeline/composer.py`：`NodeContext` 新增 `budget` 字段，并在 `_run_node` 中加入资源预检
  （VRAM 不足时告警）。
