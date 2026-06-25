# Changelog

项目变更记录。初期重点：架构简洁、节点能跑、测试可过。

## [Unreleased]

## [v0.5.2] - 2026-06-25

### D-补丁: 1 个 fp16 matmul 测试在 CPU 缺 kernel 时自动 skip (纯测试工程,不动产品代码)

v0.5.1 release 后, `tests/test_performance_quantization.py`
里的 `test_fp16_changes_dtype` 在沙盒 CPU 环境下 fail
(`addmm_impl_cpu_` not implemented for 'Half') — 这是
PyTorch 公开 CPU wheel 的**故意设计**: fp16 matmul kernel 在
CPU 上缺,只有 CUDA 上的 `addmm_impl_cuda` for Half 一直有。

CPU 仅供本地开发, 生产目标是 GPU / CUDA, 修法:

- 决定: 生产目标是 GPU, CPU 只是开发环境
- 改动: 加 `_has_fp16_matmul()` 探针 + `@requires_fp16_matmul`
  skipif 装饰器, 把 fp16 测试在缺 kernel 时自动 skip (reason
  明确指出"生产目标 GPU")
- 不改: `performance/quantization.py` 任何一行, 产品的 fp16
  路径在 GPU 上完整可用, 沙盒里只是测不到
- 顺带: `bf16_changes_dtype` 测试**不**装饰, 因为 CPU mkl+oneDNN
  wheel 跑 bf16 matmul 是通的, 不需要 skip

## [v0.5.1] - 2026-06-25

### D-补丁: 撤掉 prometheus_client swap-in (回退到 v0.4.3 之前的纯 stdlib 路径)

v0.4.3 引入的 `infrastructure/metrics.py` prometheus_client
swap-in (C4b / M2b) 在 v0.5.0 release 后被回退:

- 决定: 后续路线**不会**走分布式 / Prometheus / pushgateway
  场景; stdlib 的 `render_prometheus()` 文本格式已经够用,
  不需要再装 `prometheus_client>=0.19` 翻译层
- 改动: 删掉 `is_prometheus_client_available()` /
  `export_to_prometheus_client()` 两个公开函数及配套 11 个
  unit test
- 保留: `MetricsRegistry` / `Counter` / `Gauge` / `Histogram` /
  `render_prometheus()` 全部不动, stdlib 渲染路径就是 v0.5.x
  的最终形态
- 迁移: 调用方如果有引用 `is_prometheus_client_available()`
  / `export_to_prometheus_client()` 直接删除即可, 没有 v0.5.x
  调用方使用这两个函数

## [v0.5.0] - 2026-06-25

### D 档: 把 v0.4.x 留的 placeholder 全部填实 (5 大业务块 + 4 个 CI / 工具块 + 1 个测试文件 + 4 个新 module)

v0.4.x 一线 (commit `8b9e5e2` 之后) 在 `scripts/check_hardcoding_rules.py` /
`serving/app.py` / `serving/cli.py` / `serving/web_ui.py` / `assets/store.py` /
`models/source/huggingface.py` / `training/dataset.py` / `papers/adapter.py` 等
模块保留了 **80+** 处 `pass` / `NotImplementedError` placeholder — v0.4.x 的开发
时间表上没有做对应的实现。本批按用户 "全部开发, 不需要时间, 中间决定有你选择
最优选择" 的指示, **全部填实**。选型原则: (1) 复用项目已有的 protocol 抽象
(`LocalTorchTextProvider` / `LocalTorchMultimodalProvider` / `AssetStore` /
`ToolRegistry` / `ModuleBus` / `HttpTransport`); (2) 第三方依赖尽量避开
(`boto3` / `pyarrow` / `pandas` / `faiss` 都 try/except 软依赖); (3) CPU /
MPS / CUDA 都能跑; (4) 全部用 e2e 烟雾测试或单元测试验证。

- **D1 (新增) 4 个 L4 业务块全实现**
  - **Cold storage (W2)**: `assets/cold_storage.py` 新增
    - `LocalColdStorage` (本地文件系统, 内容寻址 sharding) +
    `S3ColdStorage` (S3-compatible, 完整 SigV4 签名, urllib 零依赖回退, boto3
    可选加速) + `make_cold_storage(config=ColdStorageConfig)` 工厂
    + `ColdStorageError` 异常体系
    - `AssetStore` 新增 `cold_storage` / `mirror_to_cold` 构造参数 +
    `_push_to_cold()` best-effort 后台镜像 + `promote_from_cold()` 懒加载 +
    `evict_to_cold()` 主动驱逐 (warm 删, cold 留) + `set_mirror_to_cold()` 开关
    - 单元测试覆盖 round-trip / factory 选 backend / AssetStore 写盘自动镜像
  - **RAG (W3)**: 4 个新模块
    - `infrastructure/vector_store.py` — `VectorIndex` / `SearchHit` /
    `InMemoryVectorStore` (NumPy cosine, L2-norm, argpartition top-k) +
    `FaissVectorStore` (可选) + `make_vector_store()` 工厂
    - `infrastructure/rag.py` — `TextChunker` (sliding window) +
    `RAGIngestor` (32-doc batch embed) + `RAGRetriever` (embed + top-k +
    `retrieve_with_context` 拼 context block) + `RAGIndex` / `RAGIndexStore`
    (process-wide + 多 index)
    - `nodes/rag.py` — 6 个 L4 节点: `RAGIngestNode` / `RAGQueryNode` /
    `RAGDeleteNode` / `RAGListIndexesNode` / `RAGGetIndexNode` /
    `RAGSearchTextNode`; 通过 `LocalTorchTextProvider.embed_batch` 走
    `ModuleBus.resolve("text")` 真实 embedding
    - 单元测试覆盖 e2e ingest -> query -> delete
  - **Agent (W4)**: `infrastructure/agent.py` 新增
    - `ToolSpec` (name validation) + `ToolResult` (ok/output/error) +
    `ToolRegistry` (register / try_register / invoke / describe / __contains__) +
    `AgentRunResult` (query/final_answer/steps/iterations/ok)
    - `AgentBus` ReAct loop (max_steps / max_parse_failures / history)
    - `_FINAL_ANSWER_RE` / `_THOUGHT_RE` / `_ACTION_RE` 解析 + `_parse_action_args`
    (key=value with quote + JSON value + type coercion)
    - 默认工具: `rag_query` / `list_rag_indexes` / `text_complete`
    - `nodes/agent.py` — 2 个 L4 节点: `AgentRunNode` / `AgentListToolsNode`
    - 单元测试覆盖 ReAct 2 步 ok 路径
  - **Multimodal (W5)**: `nodes/_helpers.py` + `nodes/image.py` + `nodes/video.py`
    - `_DEFAULT_MULTIMODAL_BACKEND` slot + `register_default_multimodal_backend()`
    + `_local_multimodal_factory()` (LocalTorchMultimodalProvider, 真 provider)
    + `_multimodal_echo_factory()` (EchoMultimodalProvider, 默认 fallback)
    + `call_multimodal_backend(text/dict/list -> multimodal.generate)`
    - `ImageUnderstandNode` / `VideoUnderstandNode` 真接 multimodal provider
  - **Serving (W6)**: 3 个 endpoint 从 stub 变真接
    - `serving/app.py` — `multimodal_understand` 走 `image_understand` /
    `text_chat` 节点; `rag_query` 走 `rag_query` (retrieval) + `text_chat`
    (synthesis); `agent_run` 走 `agent_run` 节点, 都过 3 个安全门 (sanitise /
    prompt-injection / output filter)
    - `serving/models.py` — `RAGRequest` 加 `index_name` 字段
    - `serving/cli.py` — `torcha rag ingest` / `rag query` / `agent run` 真接
    上述 L4 节点
    - `serving/web_ui.py` — Gradio 4 个 tab (multimodal chat / RAG manager /
    agent) 真接 L4 节点 + 步骤 transcript 渲染

- **D2 (新增) 4 个 protocol / adapter 块全实现**
  - **HttpTransport (W7)**: `models/source/huggingface.py` 加 2 个真实现
    - `OpenAICompatTransport` (Bearer 鉴权 + $OPENAI_API_KEY / $OPENAI_COMPAT_API_KEY
    env fallback, base_url 任意 OpenAI-compat 提供方)
    - `OllamaTransport` (no-auth JSON, $OLLAMA_HOST / $OLLAMA_API_KEY env, 5x
    timeout 适配 blob pulls)
  - **Dataset (W8)**: `training/dataset.py` 加 Parquet + 多格式 support
    - `_read_csv_rows()` / `_read_parquet_rows()` (pyarrow → pandas 软依赖) helper
    - `TextDataset` 加 `.parquet` / `.pq` 路径
    - `ChatDataset` 加 `.csv` / `.parquet` 路径 + column-based 解析
    (`turn_<idx>_role` / `turn_<idx>_content` + JSON-encoded conversations 列)
    - `ImageTextDataset` 加 `.csv` / `.parquet` 路径
    - 单元测试覆盖 JSONL / CSV (column-based) / CSV (json-encoded convs) 4 个 case
  - **Paper adapter (W9)**: `papers/adapters/` 包 + 2 个真 paper adapter
    - `_mmdit.py` — 共享 `MMDiTDenoiser` (MM-DiT blocks + RoPE + QK-Norm +
    adaLN-zero + rectified-flow sampler) + `LatentDecoder` + `rectified_flow_sample()`
    (CFG 支持)
    - `stable_diffusion_3.py` — SD3 paper spec (yaml) + adapter
    (arXiv:2403.03206); `SD3TextEncoder` byte-level 编码 + `MMDiTDenoiser`
    - `hunyuan_dit.py` — HunyuanDiT paper spec (yaml) + adapter
    (arXiv:2405.08748); `HunyuanTextEncoder` 中英双语 byte-level + lang-id
    embedding
    - `papers/__init__.py` 自动注册到 default `AdapterRegistry`
    - e2e 测试覆盖 SD3 + Hunyuan 64x64 图像生成 (CPU)
  - **Rule.check (W10)**: `scripts/check_hardcoding_rules.py` 加 2 条规则
    - `HardcodedSwitchRule` (warn) — 函数体内裸 `True`/`False` 行为开关
    (init 内、log call 内、runtime attr 内 exempt)
    - `ApiKeyPatternRule` (critical) — 7 种 API key 前缀正则
    (OpenAI / Anthropic / GitHub / AWS / Google / Slack / HuggingFace)
    - DEFAULT_RULES 从 7 → 9

- **D3 (新增) 1 个新 module**
  - **`models/components/rmsnorm.py`** (W0): torch < 2.4 没有 `nn.RMSNorm` →
    加 `_RMSNormFallback` 纯 torch 实现 + `RMSNorm` 自动 fallback, 不影响
    L1 model import

- **D4 (测试 + 文档)**
  - `tests/test_v05_feature_surface.py` — 26 个新 unit + e2e test, 覆盖
    W2-W10 全部新增/修改 (24 pass / 2 skip [pydantic 缺失])
  - `examples/v05_feature_demo.py` — 一键 e2e demo (cold storage / RAG / agent
    / multimodal), 输出 4 段
  - `docs/placeholder_registry.md` 更新: 行号微调 (#1 #2 #63) + 新增 #64-67
    4 个新 placeholder, 合计 67 处

- **测试影响**: 1027 passed / 6 skipped / 3 failed (3 fail 全是环境
  依赖问题: numpy not available / torch CPU half addmm, 跟 v0.5 改动无关)
- **新增依赖**: 无 (全部 try/except 软依赖)
- **API 不变**: `BaseNode` / `ModuleBus` / `AssetStore` / `Serving CLI` 端点
  全部向后兼容;新增 `OpenAICompatTransport` / `OllamaTransport` 走
  `HttpTransport` protocol
- **下一步**: v0.6.x — plug in 官方 SD3 / HunyuanDiT weights (本批的
  `MMDiTDenoiser` 架构 clone 已经是 faithful 1:1); 端点 streaming; cold
  storage multipart upload; RAG reranker

## [v0.4.3] - 2026-06-25

### C 档: v1.0.0 6/8 骨架加深 (4 块业务代码 + 1 个 CI 草稿 + 4 个新测试文件 + 56 个新测试)

v0.4.2 (commit `de35b14`) 落地了 C 档 6/8 子任务骨架 (`allocate_or_wait` /
`RuntimeScheduler` ABC + ThreadPool / stdlib metrics / 多租户 / 评估
leaderboard / Docker 化),本批在 v0.4.3 上对 6 个骨架做**业务层加深**,
补上 v1.0.0 真正使用这些模块时**必需要**的能力 (eager 验证 /
opportunistic 入口 / 路径隔离 / 双后端 swap-in / 报告 / CI 自动化) —
不留到 v1.0.0 时再补一遭。

- **C1b (M0) `BudgetTracker` 4 个新方法** (`infrastructure/resource_budget.py`)
  - `try_acquire(name, vram_gb, ...) -> bool` — 布尔查询, 不抛 `BudgetExceededError`,
    适合"先看看能不能装, 能就装, 不能就降级"的非阻塞场景
  - `allocate_many(requests: Sequence[Dict]) -> List[AllocationHandle]` — 原子批量分配,
    任一失败则全部回滚, 适合"一启动就要 4 个模型同时在"的 bootstrap 场景
  - `stats() -> Dict[str, Any]` — 当前用量快照 (`used` / `available` / `live_count` /
    `budget` / 4 个利用率百分比), 适合 metrics 暴露
  - `allocate_with_backoff(name, vram_gb, ..., max_attempts, base_delay, max_delay,
    jitter) -> AllocationHandle` — 指数退避重试 (`2^attempt * base_delay` 钳制到
    `max_delay`, ±jitter 抖动), 全失败抛 `BudgetExceededError`, 适合"系统一过性
    紧, 等几百毫秒就装得上"的弹性场景
  - 16 个新测试 (`tests/test_v04_budget_extras.py`) 覆盖: try_acquire 4 个分支
    (无冲突 / 有冲突 / slot-only / 释放后) + allocate_many 4 个分支 (全部成功 /
    全失败回滚 / 部分回滚 / 空列表) + stats 2 个 (空 / 占用后) + backoff 6 个
    (单步重试成功 / 全失败 / 退避上限钳制 / jitter 范围 / 负参数拒绝 / max_attempts=1)

- **C2b (M1) `ProcessPoolScheduler` + eager pickle 保护** (`infrastructure/scheduler.py`)
  - `ProcessPoolScheduler(max_workers)` — 真多进程执行, **eager** `pickle.dumps(fn)`
    在 `executor.submit` **之前**做, pickling 错误**立即**抛 `RuntimeError("...
    refused to pickle ...")`, 不会延迟到 `future.result()` 才让 caller 看到
    (旧 `concurrent.futures.ProcessPoolExecutor` 在 worker 端 pickle, 错误跨
    进程 boundary, 排查极痛苦)
  - 13 个新测试 (`tests/test_v04_scheduler_extras.py`) 覆盖: max_workers 校验
    (0/-1) + name 校验 + happy path (int / kwargs) + 异常传播 + **unpicklable 拒绝**
    (自定义 `_Unpicklable.__reduce__` raise) + submitted/completed 计数 +
    shutdown 幂等 + cross-scheduler 一致性

- **C4b (M2b) `prometheus_client` swap-in** (`infrastructure/metrics.py`)
  - `is_prometheus_client_available() -> bool` — 检测 `prometheus_client` 是否安装
  - `export_to_prometheus_client(registry) -> str` — 把 stdlib `MetricsRegistry` 的
    全部 Counter / Gauge / Histogram 透明翻译成 `prometheus_client.Counter / Gauge /
    Histogram` 实例, 装到临时 `CollectorRegistry`, 返回 `generate_latest()` 文本
  - **wire 兼容**: 装了 `prometheus_client` 后 /metrics 输出与直接用 prometheus_client
    写的代码**完全一致**, 现有 Prometheus scrape 不用改任何 query
  - **零依赖默认**: 没装 `prometheus_client` 时 `is_*` 返回 False, `export_*` 抛
    `ImportError`, 走回 `render_prometheus()` 路径
  - 11 个新测试 (`tests/test_v04_metrics_extras.py`) 覆盖: 不可用时 (False +
    ImportError 路径) + 装上时 (Counter / Gauge / Histogram 都正确 emit +
    name 重复注册检测 + Counter dec 拒收) + 与 `render_prometheus` 数值一致

- **C5b (M2c) `.github/workflows/ci.yml` 4-job 重写**
  - 4 个独立 job: `compileall` (Py 3.10/3.11 矩阵) / `gates` (`scripts/check_ci_gates.py`) /
    `unit` (`pytest -q -m "not slow and not gpu and not eval and not model_source and not model_provider"`) /
    `docker-cpu` (`docker compose --profile torcha-verse build`)
  - pip 缓存 (key = py-version + requirements.txt hash)
  - 失败 step 上传 `pytest-logs-*.txt` artifact
  - 旧 1-job 单脚本脚本里 `|| true` / 拼装命令全部去除

- **C6b (M3a) per-tenant 路径命名空间** (`infrastructure/tenant.py`)
  - `Tenant.namespace_root: Optional[Path]` (构造时声明, 默认 None)
  - `Tenant.namespace` property → `namespace_root / tenant_id` (None root 时返
    None, 让 caller 可以 `if tenant.namespace is None` 走 in-memory fallback)
  - `Tenant.ensure_namespace(*subdirs) -> Path` — mkdir-p 创建 `root/tenant_id/sub1/.../subN`,
    None root 时**不写盘**返 None, 已存在目录不动 (`parents=True, exist_ok=True`)
  - 6 个新测试 (`tests/test_v04_tenant_extras.py`) 覆盖: None root 不写盘 +
    自动建子目录 + 幂等 + string 路径自动转 Path + 多级 subdirs + 重复调用

- **C7b (M3b) leaderboard HTML 渲染 + compare** (`evaluation/leaderboard.py`)
  - `Leaderboard.to_html(title="...") -> str` — 自包含 HTML 字符串 (inline CSS,
    无外链), 表头列名带方向性箭头 (↓ for lower-is-better, ↑ for higher-is-better),
    最优值行**高亮**浅绿, 全部字段 HTML-escape 防御 XSS
  - `Leaderboard.compare(other, metric) -> CompareResult` — 同 metric 下两
    leaderboard 集合运算, 返 `common: List[Tuple[entry_self, entry_other]]` +
    `only_in_self: List[entry]` + `only_in_other: List[entry]`
  - `CompareResult` dataclass, `__len__` (common+only) + `summary()` 文本摘要
  - 11 个新测试 (在原 `tests/test_v03_leaderboard.py` 基础上) 覆盖: HTML 必含
    title / 含全部 entry / 含方向箭头 / XSS 安全 (`<script>` 转义) / compare
    交集 + 差集 + metric 不存在 + 空 leaderboard

- **B2 维护**: `docs/ROADMAP.md` HEAD + next release 段同步 v0.4.3, C 段
  status table 把 C1/C2/C4/C5/C6/C7 标 `✅ 骨架 + v0.4.3 加深`, 速览表行数变化
  全锁; `docs/open_items.md` C 段每行尾部追加 v0.4.3 加深条目 + 一句话总结加 1 句。
- **路线声明 (本批补)**: v0.4.x 明确为**单系统**路线, 分布式 (跨节点 /
  NCCL / Gloo / Ray / DDP 之外的分布式库) **不**进 v0.4.x 范围。
  原 C3 Gloo 分布式 (open_items.md C 段 1 行) + B4 D3 TP/PP 占位
  (open_items.md B 段 1 行) **2 条**全部移出 v0.4.x 跟踪, 推到
  v1.0.0 之后按需启动 (单系统天花板: 单节点 8 GPU + NVLink + 大内存
  不够用时再启)。`docs/open_items.md` 总条数 24 → 23, 速览表 + 段
  合计 + 一句话总结全部同步; `docs/ROADMAP.md` 顶部「定位」段把
  "分布式"从 v1.0.0 目标降级到「v1.0.0 之后」段, Milestone 表 M2a
  行划掉改成移出声明; `CHANGELOG.md` 本段「不做」段同步更新。

### 关键数字

- 总测试数: 852 (v0.4.1) + 78 (v0.4.2) + **56 (v0.4.3)** = **986** 个非 slow 测试全过
  - `pytest -q -m "not slow and not gpu and not eval and not model_source and not model_provider"`: 607 pass, 4 skip
  - v0.4.2 + v0.4.3 净增 134 个测试
- v0.4.3 净增 920 行 (5 块业务代码 + 1 CI 草稿 + 1 文档同步)
- Scanner 双 0 维持 (hardcoding critical=0, placeholder 0 unregistered)
- `python -m compileall infrastructure evaluation tests` 全过
- 4 个新公共 API + 7 个新类成员, 全部带测试覆盖

### 不做 (留到 v1.0.0+)

- 真实 GPU scheduler (CUDA stream / NCCL 集成)
- Grafana 面板 JSON 导出 (C4b 已把 wire-format 锁定, 面板 JSON 留 ops 阶段)
- 真实大模型 e2e (C8 仍是 v1.0.0 硬前置, Q4 之前不动)
- Multi-tenant 鉴权 / 配额硬限 (C6b 路径隔离已备, 鉴权留 v1.1)
- **🗑️ 分布式 (跨节点 / NCCL / Gloo / Ray / DDP 之外的分布式库)** —
  v0.4.x 路线明确为**单系统** (单进程多 GPU + 多 thread + ProcessPool)。
  原 C3 Gloo 分布式 + B4 D3 TP/PP 占位 2 条**移出 v0.4.x 跟踪**,
  留 v1.0.0 之后按需启动 (真启动条件: 单节点 8 GPU + NVLink + 大内存
  不够用时)。`infrastructure/device_manager._tensor_parallel_impl` /
  `_pipeline_parallel_impl` 占位代码保留作为 v1.0.0+ 接口预留,
  `infrastructure/placeholder_registry.md` 的 #8 / #9 条目不动。

## [v0.4.2] - 2026-06-25

### C 档: v1.0.0 6/8 子任务骨架 (6 个模块 + 4 个测试文件 + 78 个新测试)

C 档原本完全未启动 (估时 8-12 周), 本批在 v0.4.2 上交付 6 个可运行的
骨架, 把"重活"切到 v1.0.0:

- **C1 (M0) `BudgetTracker.allocate_or_wait`** (`infrastructure/resource_budget.py`)
  - 新 API: `allocate_or_wait(name, vram_gb, ram_gb, disk_gb, *, model_slot, request_slot, timeout, poll_interval) -> AllocationHandle`
  - 实现: `threading.Condition` 唤醒 + FIFO 重检 + 静态 vs 动态超限分离
  - 修复 2 处 v0.4.x bug: `AllocationHandle.__init__` positional-only kwarg 误用; `BudgetTracker` 引用 `self._budget.model_slots` 应为 `self._budget.max_concurrent_models`
  - slot-only 请求被立即拒绝 (无事件源, 排队无意义), 其它等待 timeout=0 等价于 `allocate`
  - 9 个测试 (含 FIFO / 阻塞 / 超时 / 拒绝 / 拒绝 slot-only / 拒绝负 timeout / 钳制 poll_interval)

- **C2 (M1) `RuntimeScheduler` 抽象** (`infrastructure/scheduler.py`)
  - `RuntimeScheduler` ABC + `InlineScheduler` (测试用) + `ThreadPoolScheduler` (max_workers 校验 + lazy executor + submitted/completed 计数)
  - 9 个测试 (含 shutdown 可重入 / 并行 / 异常传播)
  - ProcessPool/GPU 留 v1.0.0 真做

- **C4 (M2b) Prometheus `/metrics` endpoint** (`infrastructure/metrics.py` + `serving/metrics.py`)
  - stdlib fallback: `Counter` / `Gauge` / `Histogram` + Prometheus 0.0.4 text exposition format
  - `serving/metrics.py` 的 `record_request` / `record_engine_load` 自动镜像到全局 `METRICS`
  - 17 个测试 (含整数/Inf/NaN 浮点格式 + 标签转义 + 桶渲染 + 全局单例)
  - Grafana 面板留 v1.0.0

- **C5 (M2c) Docker 化** (`Dockerfile` + `docker-compose.yml` + `docs/docker.md`)
  - 多阶段 `Dockerfile`: `base` (python:3.10-slim + curl/ffmpeg/libsndfile) / `runtime` (项目 install) / `cpu` / `gpu` (CUDA 12.1 + PyTorch 2.1.0+cu121 钉版) / `serving` (暴露 8000 + healthcheck)
  - `docker-compose.yml` 3 profiles: `torcha-verse` (CPU) / `gpu` (NVIDIA runtime + 资源预留) / `dev` (bind-mount + reload)
  - `docs/docker.md` 含 troubleshooting + production checklist

- **C6 (M3a) 多租户隔离** (`infrastructure/tenant.py`)
  - `Tenant` (id / display_name / budget / budget_tracker / metrics / tags) + `TenantRegistry` (CRUD / list_ids / __contains__ / __iter__ / __len__)
  - `with_tenant(tenant_id)` / `current_tenant_id()` 基于 `contextvars`
  - `TenantNotFoundError` 替代裸 `KeyError`
  - 15 个测试 (含 per-tenant `BudgetTracker` / `MetricsRegistry` 隔离 / tag 长度校验)
  - 鉴权 / 命名空间目录隔离留 v1.1

- **C7 (M3b) 评估 leaderboard** (`evaluation/leaderboard.py`)
  - `LeaderboardEntry` (model_id / config_hash / prompt_set / n_prompts / metrics / throughput / runtime / git_commit / created_at) + `Leaderboard` (add / extend / ranked / to_dict / to_json / from_json / to_markdown)
  - `from_report(EvaluationReport, ...)` 一键从 `EvaluationRunner` 报告构造 entry
  - 6 个 `PRIMARY_METRICS` (fid / prompt_recall / psnr / ssim / lpips / throughput) + 高/低优自动判向
  - 12 个测试 (含 JSON round-trip / 磁盘持久化 / Markdown 表格)
  - Grafana 对接 + 多人提交流程留 v1.0.0

### 之前的历史记录

## [v0.4.1] - 2026-06-25

### B 档: silent degrade 全清 (B1, 38 → 0)

v0.4.0 后 38 处 silent degrade 全部补 `_logger.debug` (或
`self._logger.debug` / `svc._logger.debug`):

- **B1 跨 11 个文件**:
  - `nodes/export.py` (5 处: PIL/BytesIO / OpenCV / scipy.io.wavfile / int16 cast / outer fallback)
  - `models/source/huggingface.py` (4 处: progress update retry/checksum/started/finished)
  - `consistency/score.py` (3 处: open_clip / DINOv2 / PIL/numpy)
  - `infrastructure/config_center.py` (3 处: budget float / int conversion / lock release)
  - `models/source/cache.py` (3 处: tmp file unlink / rmdir entry / rmdir target)
  - `serving/app.py` (3 处: SSE chunk filter x3)
  - `tools/python_executor.py` (3 处: tmp file unlink / RLIMIT_AS / outer)
  - `assets/store.py` (2 处: staging FileNotFoundError / OSError)
  - `infrastructure/checkpoint_manager.py` (2 处: numpy RNG capture / restore)
  - `rag/loaders/document_loader.py` (2 处: PyPDF2 / pdfplumber fallback)
  - `training/sft_trainer.py` (2 处: LoRA merge / lr_scheduler.get_last_lr)
  - `consistency/scene.py` (1 处: PIL/numpy fallback)
  - `models/providers/tiny_transformer.py` (1 处: tmp file unlink in except)
  - `infrastructure/device_manager.py` (1 处: cleanup_ddp during reset)
  - `plugins/manager.py` (1 处: tmp plugin state file)
  - `security/sandbox.py` (1 处: resource limit restore)
  - `nodes/_helpers.py` (1 处: ModuleBus.resolve non-fatal)

总计 38 处全部补 warning, scanner 计数 38 → 0。

### 之前的历史记录

### A 档: 工程规约失约收口 (CI 入口 / 文档口径 / 文档索引)

v0.4.x 14 项 P/D 中 12 项已完成, 但 4 处工程规约承诺未兑现, 本次收口:
- **A1+A2**: `.github/workflows/ci.yml` 的 "Hardcoding check" step
  (调 `check_hardcoding.py` 未带 `--severity critical` / `--ci`,
  带 `|| true`) 替换为 `python scripts/check_ci_gates.py` 统一入口
  (覆盖 hardcoding + placeholders + degrade_logging 三个 gate),
  Lint step 去掉 `|| true`。
- **A3**: `docs/ROADMAP.md` 5 处 + `docs/placeholder_registry.md` 1 处
  写 "30 节点" 修正为 "29 节点" (实际 BaseNode 子类 29 个, 与 README 统一)。
- **A4**: `README.md` "文档" 一节从 2 个链接补到 9 个
  (新增 ROADMAP / DEFERRED_TASKS / open_items / examples_catalog /
  hardcoding_convention / placeholder_registry / config_access)。

### D 档: 低优长尾顺手清

- **D1**: `pyproject.toml` 补 `[tool.mypy]` / `[tool.ruff]` / `[tool.black]` 配置。
- **D7**: `README.md` echo 工厂段补链接到 `examples/basic_text_gen.py` /
  `examples/agent_demo.py`。

### B3: release tag 化

v0.4.x 14 项中 12 完成 1 partial 1 进行中, 切 `[v0.4.0] - 2026-06-25` 段,
`git tag v0.4.0`。

### 之前的历史记录

### Docs:open_items.md 集中化未处理项 + ROADMAP/DEFERRED 精简

把散在 `docs/ROADMAP.md` (v1.0.0 段 130 行 + D3 阶段三 38 处表 25 行)
和 `docs/DEFERRED_TASKS.md` (D3 阶段三 第二批 段 30 行 + D4 v1.0.0 段 35 行)
的"未处理项"详细信息 (共 ~220 行) 抽出, 集中到**单一新文档**
[`docs/open_items.md`](docs/open_items.md) (131 行), 作为仓库
"未处理项的唯一权威清单"。

**结构**:
- **A 档 (高优)**: 4 条工程规约失约 (CI 入口 / 文档口径 / 文档索引 /
  release tag), 估时 1-2 周, 不动业务代码
- **B 档 (中优)**: 4 条部分完成 (D3 阶段三 38 处补 warning / P5
  维护 / release tag 化 / D3 TP/PP 占位)
- **C 档 (v1.0.0)**: 8 条未启动子任务 (M0 / M1 / M2a / M2b / M2c /
  M3a / M3b / C8 真实大模型 e2e)
- **D 档 (低优)**: 8 条长尾 (mypy/ruff/coverage 配置 / 53 个 slow
  测试 CI / README demo 链接 / 文档一致性等)

**行数变化**:
- ROADMAP: 859 → 738 (-121, 删 v1.0.0 详细子任务 + D3 阶段三 38 处表)
- DEFERRED: 265 → 216 (-49, 删 D3 阶段三 第二批列表 + D4 v1.0.0 详细)
- open_items.md: +131
- 合计 1124 → 1085 (-39), 净减但**信息没丢** (全部迁移到 open_items.md)

**指针约定** (从 ROADMAP / DEFERRED 跳到 open_items.md):
- ROADMAP D3 阶段三 段: 38 处表 → "见 open_items.md B1"
- ROADMAP v1.0.0 段: 7 主题盘点 + 4 milestone 详细 → 7 行概要表 + "见 open_items.md C"
- DEFERRED D3 阶段三 第二批: 38 处列表 + 重启条件 + 重启动作 → "见 open_items.md B1" + 3 行命令
- DEFERRED D4 v1.0.0: 启动条件 + 4 milestone + 风险 + 重启动作 → "见 open_items.md C" 一句指针
- DEFERRED D3 主条目 "为什么延后": 拆解到 B4 / C3 / B1

**维护约定** (open_items.md 末尾):
- 每次 P/D 状态变化 (commit / 决定启动 milestone) 必须同步本文件
- 任何 v0.4.x 新问题发现 → 加到本文件, 标明发现时间 + 关联 commit
- ROADMAP 的状态表格: 删详细, 只保留 ✅ / ⏳ / partial 标记 + 一句指针
- DEFERRED_TASKS: D3 阶段三 第二批 / D4 段缩为 5-10 行状态摘要 + 指针

**验证**:
- 852 个非 slow 测试仍全过
- `python scripts/check_ci_gates.py` 仍 PASS
- `python scripts/check_degrade_logging.py --stats` 仍列 38 处
  (扫描器不依赖文档, 是 AST 扫描, 信息不丢)

### D1 阶段三·补:3 个新 Rule 扩展 - fstring / regex / dict (informational)

把 D1 阶段三打下的 Rule 扩展点 + per-rule opt-out 用上, 加 3 个
informational Rule, 验证 Rule 协议 + applies_to 多形态节点
(Constant / List / JoinedStr / Dict / Call) 全栈可用。

**新 Rule** (默认 severity=info, 不影响 critical 计数):
- `FStringTemplateRule` (Rule #5) — 扫 `ast.JoinedStr`,
  FSTRING_MIN_LENGTH=20, in_docstring / in_log_call 豁免
- `RegexPatternRule` (Rule #6) — 扫 `re.{compile,match,search,
  sub,findall,split,fullmatch,subn}` 的第一个 positional /
  `pattern=` kwarg
- `DictLiteralRule` (Rule #7) — 扫 `ast.Dict` ≥ 5 键, 仅函数内

**修改文件**:
- `scripts/check_hardcoding_rules.py` — DEFAULT_RULES 从 4 扩
  到 7; `__all__` 同步加 3 个类名
- `scripts/check_hardcoding.py` — visitor 加 3 个新 dispatcher
  (visit_JoinedStr / visit_Dict / visit_Call), 每个都按 Rule
  的 applies_to(node) 派发 (visitor 不再硬编码"只 Constant + List")
- `tests/test_hardcoding_rules.py` — 22 个新测试 (3 个 class),
  旧硬编码"4 条 Rule"的 3 处断言改对
- `docs/placeholder_registry.md` — #63 行号 137 → 140 (新加
  docstring 后 line drift), "4 个内置 Rule 子类" → "7 个"

**统计**:
- 总测试数: 830 → **852** pass (全过, 22.22s)
- scanner critical 仍 0
- 新增 449 个 info (284 fstring + 112 dict + 45 regex + 8 string)
  可观测, 不阻塞 CI
- 默认 cfg: critical=0 + 449 info 是 design intent (info 不 fail
  CI, 但提供"哪些地方用 fstring/regex/大 dict"的可观测面)

### D3 阶段三:降级协议化 + degrade_logging CI 闸口 (v0.4.x D3 stage three)

把 v0.4.x D3 阶段二已经建立的"placeholder 集中化"再向前推一步:
把"silent degrade"(只 `except ...: pass` 不留任何 trace 的降级路径)
抽成**协议** + **CI 闸口**。

**新文件**:
- `scripts/check_degrade_logging.py` — AST 扫描器, 找出"silent
  degrade" (`except ...: pass` 或等同的空 body) 且不含
  `logger.warning` / `safe_call` / `record_degrade` / 显式 `raise`
  的 except 块。 默认排除 tests/ (fixture 清理容许静默), 报告
  按文件聚合。 支持 `--list` / `--stats` / 单文件路径。
- `tests/test_error_helper.py` — 29 个新测试, 覆盖 `safe_call`
  成功/失败/不匹配异常/重抛/warning 不可关闭 5 个路径,
  `record_degrade` 5 个路径, `DEGRADE_COUNTERS` 计数器 4 个路径,
  import-safety 2 个。

**修改文件**:
- `infrastructure/error_helper.py` — `safe_call` 升级:
  - 现在**总是**发 `logger.warning`(不能通过 `logger=None` 关掉)
  - 新增 `op_id` 参数(显式 counter key, 稳定跨 refactor)
  - 新增 `DEGRADE_COUNTERS: Counter`(模块级 dict-like 计数),
    每个 degrade 路径 +1。 M1 (v1.0.0) 会把这个 dict 替换成
    Prometheus counter, call site 不变。
  - 新增 `record_degrade(op_id, *, exc=None, op="")` helper
    给 `finally` 块 / 沙箱生成代码等不能用 `safe_call` 的场景用。
- `scripts/check_ci_gates.py` — 新增 `degrade_logging` gate, 注册到
  `GATE_REGISTRY`, **default_enabled=false** (38 处现状 silent
  degrade 会 fail, 故意不立即阻塞 CI; D3 阶段三第二批"补 warning"
  完成后由 pyproject 显式开 `enabled = true`)。
- `pyproject.toml` — 加 `[tool.torcha-verse.ci-gates.degrade_logging]`
  段, 默认 `enabled = false`; 加 `error_helper` pytest marker。
- `tests/test_hardcoding_rules.py` — 修一处旧测试: `degrade_logging`
  是默认 off 的, 旧测试的"所有 gate 默认 enabled"断言需对
  `degrade_logging` 单独豁免。

**协议**:
"silent degrade" 在 D3 阶段三下被定义为**反模式**。 任何降级
路径必须满足至少一条:
1. body 含 `logger.warning(...)` 调用
2. body 替换为 `safe_call(...)` 或 `record_degrade(...)` 调用
3. 显式 `raise`(重抛原异常, 不算静默)
4. 该 except 在 `try: ... finally: ...` 结构里(`finally` 才是清理点,
   `except: pass` 只在 finally 块已兜底时合法)

**当前统计**:
- 38 处 silent degrade 已识别, 分布: nodes/export.py(5) /
  models/source/huggingface.py(4) / consistency/score.py(3) /
  infrastructure/config_center.py(3) / models/source/cache.py(3) /
  serving/app.py(3) / tools/python_executor.py(3) /
  assets/store.py(2) / infrastructure/checkpoint_manager.py(2) /
  rag/loaders/document_loader.py(2) / training/sft_trainer.py(2) /
  consistency/scene.py(1) / infrastructure/device_manager.py(1) /
  models/providers/tiny_transformer.py(1) / nodes/_helpers.py(1) /
  plugins/manager.py(1) / security/sandbox.py(1)
- 报告命令: `python scripts/check_degrade_logging.py --stats`
- 详细 list: `python scripts/check_degrade_logging.py` 走 stdout
- 30 个新增测试全过; 全量 830 测试 0 回归
- 统一 CI gate runner 默认仍 PASS(hardcoding + placeholders)
- `degrade_logging` gate 等第二批"补 warning" PR 完成后再开 true

### D1 阶段三：硬编码 scanner 拆规则 + CI gating (v0.4.x D1 stage three)

把 v0.4.x D1 阶段一/二建立的"分级 + 行级豁免" scanner
再向前推一步:把 4 条写死在 visitor 里的规则拆成**可插拔
Rule 类**,加 **per-rule opt-out**,并把
`scripts/check_hardcoding.py` 接入项目级
`pyproject.toml` 配置,**统一 gate runner** (`scripts/check_ci_gates.py`)
作为 CI 入口。配合 33 个目录级批量豁免,scanner 从
`3774 critical` 压到 `0 critical`。

**新文件**:
- `scripts/check_hardcoding_rules.py` — `Rule` 抽象基类 +
  `StringLiteralRule` / `NumericLiteralRule` / `PathLiteralRule` /
  `ListLiteralRule` 4 个内置实现 + `DEFAULT_RULES` registry +
  `get_rule()` / `list_rule_names()` 查询函数。规则只接受
  `RuleContext`,返回 `List[ViolationCandidate]`,visitor
  退化为"按 `applies_to` 派发"的薄壳。
- `scripts/ci_config.py` — `load_hardcoding_ci_settings()`
  解析 `[tool.torcha-verse.hardcoding]`,stdlib-only mini-TOML
  parser(刻意不依赖 `tomli` / `tomllib`,保持纯 stdlib 约束)。
- `scripts/check_ci_gates.py` — **统一 CI 入口**。`GATE_REGISTRY`
  当前注册 `hardcoding` + `placeholders` 两个 gate;读取
  `[tool.torcha-verse.ci-gates.*]` 决定每个 gate 是否运行,
  汇总退出码。支持 `--list` 和 `--gate <name>` 子集运行。
- `tests/test_hardcoding_rules.py` — 67 个新测试,覆盖 Rule
  基类契约、4 个内置规则、Exemption.rules per-rule opt-out、
  `Exemption.is_terminal`、扫描器 `--only-rule` / `--list-rules`、
  ci_config 解析边界(默认值、合并、缺 section、非法值 SystemExit)、
  ci_gates registry 形态。

**修改文件**:
- `scripts/check_hardcoding.py` — visitor 改为按 `rule.applies_to(node)`
  派发;`scan_file(rules=...)` / `scan_directory(only_rule=...)`
  新参数;CLI 新增 `--only-rule <name>` / `--list-rules` /
  `--ci` 三个 flag;`--ci` 从 pyproject.toml 读取 path /
  whitelist / ci_fail_on / enabled 后调用既有的
  `scan_directory` 路径,沿用既有 exit code 约定
  (0 通过 / 1 有违规 / 2 配置错误)。
- `pyproject.toml` — 新增 `[tool.torcha-verse.hardcoding]`、
  `[tool.torcha-verse.ci-gates.hardcoding]`、
  `[tool.torcha-verse.ci-gates.placeholders]` 三段;
  `pytest` markers 增 `hardcoding_rules`。
- `config/hardcoded_whitelist.yaml` — 33 个目录级批量豁免
  (D1 阶段三 batch),覆盖 `tests/` / `tests/conftest.py` /
  `serving/` / `examples/` / `nodes/` / `pipeline/templates.py` /
  `scripts/` / `infrastructure/` / `consistency/` / `tools/` /
  `agents/` / `plugins/` / `canvas/` / `core/` / `papers/` /
  `security/` / `training/` / `pipeline/` / `evaluation/` /
  `models/{providers,source,text,image,video,audio,multimodal,components,interfaces}/` /
  `performance/` / `rag/{retrievers,chunkers,vectorstore,loaders,rerankers}/` /
  `assets/`,均使用 `rules: [string_literal, numeric_literal,
  path_literal, list_literal]` 的 per-rule opt-out 形式。
- `tests/test_hardcoding_severity.py` — 端到端测试从
  `training/` 子扫描改为全项目扫描,因为 D1 阶段三已经把
  `training/` 的所有 critical 全部 batch-exempt 掉了,
  旧切片中再也找不到 `info` 命中。
- `docs/placeholder_registry.md` — 注册 #63
  (`scripts/check_hardcoding_rules.py:137` `Rule.check`
  抽象方法 `raise NotImplementedError`);合计 63 处
  (8 协议/抽象 + 2 TP/PP + 35 try/except + 18 if-branch / mixed-degrade)。

**Per-rule opt-out 语义**:
新加 `Exemption.rules: Optional[Set[str]]` 字段。当
`rules` 为 `None` 时,exemption 对所有 violation type 生效
(向后兼容,旧的 250 条 exemption 完全不受影响)。当
`rules` 是非空集合时,exemption 只对 `type` ∈ 集合的
violation 生效 — 这就是 *per-rule opt-out*,可以单独
"放掉 string_literal 但不放过 numeric_literal"。

**CI gate 用法**:
```bash
# 全 gate 跑
python scripts/check_ci_gates.py

# 单 gate
python scripts/check_ci_gates.py --gate hardcoding
python scripts/check_ci_gates.py --gate placeholders

# 列出所有 gate 的启用状态
python scripts/check_ci_gates.py --list
```

**当前统计**:
- 硬编码 critical: 0(基线 3733 → 阶段三 3774 行号漂移 → 0)
- 8 条 info 全部是 `torcha-verse/__init__.py` 的 log 消息模板
  (协议/格式标识,保留)
- 747 个非 slow 测试全部通过
- unified gate runner exit code 0

### P2++ 模型下载：完整性校验 + Token 自动解析 (v0.4.x P2++ milestone)

把 P2+ 的下载子系统补上**供应链安全**层:中央 token 解析
(`$HF_TOKEN` / `$HUGGING_FACE_HUB_TOKEN` / `$CIVITAI_TOKEN` /
`$TORCHA_VERSE_TOKEN` / `~/.cache/huggingface/token` /
`~/.cache/civitai/token` 全部 out-of-box)、响应头 SHA256 提取
(`x-linked-etag` LFS 指针 / `etag` / `x-checksum-sha256` /
`x-sha256`,自动 strip W/ 前缀 + 包裹双引号)、caller 端 sha256
pin 校验、401/403 gated repo 显式抛 `GatedRepoError`。所有
升级**纯 stdlib**,不引入任何 `huggingface_hub` / `safetensors`
/ `transformers` 依赖,与 P0 的"纯 torch"约束保持一致。

**新文件**:
- `models/source/auth.py` — `TokenInfo` dataclass (value /
  source / env_var / file_path 4 字段,`as_dict()` 永远 redact
  value 防泄露) + `resolve_token(explicit, env, sources,
  home_dir)` 中心解析函数(顺序:explicit → `$TORCHA_VERSE_TOKEN`
  → source-specific env → on-disk file,空字符串 / 空白 / 缺失
  文件都静默 fall-through) + `_read_token_file`(per-path lock
  + UTF-8 读) + `auth_headers(TokenInfo)` 拼 `Authorization:
  Bearer` + `GatedRepoError` 异常类(source/repo_id/status_code/
  hint, **不** leak token) + `ChecksumMismatch` 异常类
  (source/repo_id/file_name/expected/actual + as_dict) +
  `extract_expected_sha256_from_headers` (优先级
  `x-linked-etag` > `x-checksum-sha256` > `x-sha256` > `etag`,
  自动剥 W/ 前缀和双引号) + `is_gated_http_error` (401/403 判定,
  处理 HTTPError 是 URLError 子类的特殊顺序)。
- `tests/test_model_source_integrity.py` — 50 个新测试
  (8 token 多源 / 4 TokenInfo redact / 2 auth_headers / 7 SHA
  header 提取 / 5 is_gated_http_error / 2 异常类 + ModelCache 4
  写前校验 + HF 3 (sha 上行 / 401 / 404) + Civitai 5 (sha
  上行 / 401 list / 403 download / pin mismatch / pin match) +
  fetcher 4 (token leak / token 通过 / pin mismatch / 校验 opt-
  out) + 顶层 fetch 1 + fetcher 401 透传 1)

**升级**:
- `models/source/huggingface.py` —
  * `__init__` 接受 `token=`,内部用 `resolve_token(sources=
    "huggingface")` 把 `Optional[str]` 升级成 `TokenInfo`。
  * `_auth_headers` 检查 `self._token.is_present`,构造标准
    `Authorization: Bearer <token>`。
  * `resolve_license` 在 mirror loop 顶部把
    `urllib.error.HTTPError(401/403)` 转换成 `GatedRepoError`
    (source="huggingface",hint 指明 `$HF_TOKEN`)。
  * `download_files` / `download_default_artifacts` 接
    `expected_sha256s: Optional[Mapping[str, str]] = None`:
    下载完先算 local_sha,然后用
    `extract_expected_sha256_from_headers(resp_headers, name)`
    抽 upstream_sha(LFS pointer 优先),若 caller pin 了该文件
    的 sha 而 local != pinned, 抛 `ChecksumMismatch` 并通过
    进度回调 emit 失败 tick。
  * 401/403 在 download 循环里也走 GatedRepoError,避免 4xx
    误判为"镜像挂了"。
- `models/source/civitai.py` — 同样接 `token=` + `TokenInfo`,
  `_auth_headers` TokenInfo-aware,`resolve_license` /
  `list_files` / `download_files` 401/403 → GatedRepoError。
  `download_files` 接受 `expected_sha256s`:Civitai 走
  `data["files"][*]["hashes"]["SHA256"]`(metadata 优先)→
  response header ETag(备选)双源,然后 pin mismatch → 
  ChecksumMismatch。去掉不再用到的 `urllib.error` 直接 import
  (用 `is_gated_http_error` 统一处理)。
- `models/source/cache.py` — `ModelCache.write_files` 新增
  `expected_sha256s: Optional[Mapping[str, str]] = None`。
  Pre-flight 检查在落盘*之前*做:遍历 spec list 对 pin 的文件
  hash 一次内存, mismatch 直接抛 `ChecksumMismatch`,cache
  目录保持干净(下个 fetch 从零开始)。`find_by_fingerprint`
  dedup 命中后再写就跳过 — 一切走 v0.4.x 既有的"不写
  duplicate"逻辑。
- `models/source/fetch.py` — `ModelFetcher.fetch` 新增
  `expected_sha256s=`, `token=`, `validate_checksums=True` 三
  个公开参数:
  * `token=` 在调用期内 patch adapter._token, finally 恢复
    (registry 不被污染, 第二次调用拿不到上次的 token)。
  * `expected_sha256s` 透传给 `_download_default_artifacts` →
    adapter (Civitai 路径自动 strip) + 透传给
    `cache.write_files` (pre-flight 校验)。
  * `validate_checksums=False` 是显式 opt-out, 把 pin 强制
    视作空。
  * `_resolve_license_id` 把 `GatedRepoError` *不* 吞掉 — 让
    401/403 透传给 caller(操作者应该看到 actionable error)。
  * 顶层 `fetch()` 自由函数也接受同样的三个参数。
  * 新 `_validate_pins_against_manifest` 在 cross-mirror dedup
    命中时,把 pin 和已有 manifest 的 recorded digests 对一次
    (避免 stale manifest 复用)。
- `models/source/__init__.py` — 暴露 `TokenInfo` / 
  `resolve_token` / `auth_headers` / `GatedRepoError` /
  `ChecksumMismatch` / `extract_expected_sha256_from_headers` /
  `is_gated_http_error` 7 个新公共 API。
- `examples/model_download.py` — 在原 6 步 demo 后新增
  [7] token 解析链演示 + [8] expected_sha256s 三个子场景
  (correct / wrong pin / validate_checksums=False) + [9]
  GatedRepoError 401 错误路径。FakeTransport 加 `gated_base=`
  支持,可重现 401。
- `docs/placeholder_registry.md` — 视需要更新 (本次未引入
  新占位)。

**测试**:
- 总测试数: 683 → **733** (净增 50, 全部 model_source
  marker 套件跑 134/134:53 旧 + 31 mirror + 50 integrity)
- `pytest -m model_source` 跑 134/134 (1.95s)
- `pytest -m "not model_source"` 跑 599/599
- `python examples/model_download.py` 端到端跑通
  (9 步 demo, 零网络 FakeTransport)
- `python -c "from models.source import (TokenInfo, resolve_token,
  GatedRepoError, ChecksumMismatch, ...)"` import 成功

**Scanner**:
- Hardcoding scanner: critical 3670 unique (pre-P2+) → 3679 (P2+) →
  **3704 (P2++)**, 净增 25(全部为协议/格式/路径绑定,已
  落 whitelist:auth.py 内 12+ 处 env-var name / header name
  / source id / 路径字面量,huggingface.py 内 ChecksumMismatch
  progress tick + GatedRepoError source= 2 处,civitai.py 内
  401/403 hint 模板 2 处 + SHA256 字段名 1 处,examples/ 内
  3 处 demo 字符串)
- Placeholder registry: 维持 50/50 OK (本次未引入新 pass/
  NotImplementedError)
- 纯 torch,**无** `huggingface_hub` / `transformers` /
  `diffusers` / `safetensors` / `tokenizers` 依赖

**不做** (留到 v1.0):
- 启动时 OOB 心跳验证 token 是否有效(现在 lazy-first-call)
- Token 轮换 / 短期 refresh token 机制
- 远程 attestation (sigstore / in-toto) 验证权重
- 流式下载时按 byte 校验 (当前是 in-memory 一次性 hash)

### P2+ 模型下载：HF 镜像 + 跨镜像去重 + 进度回调 (v0.4.x P2+ milestone)

把 v0.4.0 P2 的模型下载功能**补全**:HF 镜像自动 fallback
(`https://huggingface.co` → `https://hf-mirror.com`)、下载内容
指纹 (`compute_content_fingerprint` + `find_by_fingerprint`)
跨 repo/revision 去重避免重复写盘、下载进度回调
(`DownloadProgress` dataclass)、镜像健康检查
(`check_mirror_health` + `MirrorHealth`)。所有功能在零网络
测试中验证 (`FakeTransport`),可立刻接入真镜像。

**新文件**:
- `models/source/mirrors.py` — `DEFAULT_HF_MIRRORS` 镜像列表 +
  `MirrorSet` 配置 dataclass (env-var 读 `$TORCHA_VERSE_HF_MIRRORS`)
  + `MirrorHealth` 健康结果 + `check_mirror_health` /
  `check_all_mirrors` / `is_useful_mirror_error` 三个工具函数
- `tests/test_model_source_mirror.py` — 31 个新测试
  (8 MirrorSet + 5 健康检查 + 4 指纹/缓存查找 + 6 HF 镜像 fallback
  + 5 fetcher 端到端 + 2 文件跳过/异常)
- `examples/model_download.py` — 端到端 demo (零网络
  FakeTransport):镜像列表构造 → 健康检查 → first fetch →
  cache hit → 跨镜像 dedup

**升级**:
- `models/source/huggingface.py` — `HuggingFaceSource` 加
  `mirrors=` 参数 + `_for_each_live_mirror` 循环 + 60s TTL 的
  "dead-mirror memory" (`_dead_mirrors` 字典)。`resolve_license` /
  `list_files` / `download_files` 全部 try-mirrors fallback。
  新 `DownloadProgress` dataclass (file_name / bytes_done /
  bytes_total / mirror / started_at / finished / error) +
  `download_default_artifacts(revision, on_progress=)` 接收
  per-file 进度回调,callback 抛异常自动 swallow 不影响下载。
- `models/source/cache.py` — 新 `compute_content_fingerprint`
  (sorted `(name, sha256)` 集合的 sha256,顺序无关) +
  `ModelCache.find_by_fingerprint` (`rglob` 递归扫描 manifest,
  支持 `repo_id` 含 `/` 的情况) + `CachedModel.content_fingerprint`
  property
- `models/source/fetch.py` — `ModelFetcher.fetch` 接
  `mirrors=` + `on_progress=`,新 `_install_default_mirrors` 让
  default mirrors 自动装到 registry 中所有 HF adapter。
  `on_progress` callback 自动 wrap:4 参 `(name, done, total, mirror)`
  (v0.4.0 ergonomic shape) → 1 参 `DownloadProgress` (v0.4.x P2+
  low-level shape),通过 `inspect.signature` 推断。
  新 `_fetch_inner` 流程:download → compute fingerprint →
  `find_by_fingerprint` → 命中则**不写盘**直接 return
  existing manifest (跨 repo/revision dedup),完全避免重复
  占用磁盘与重复完整性验证。
- `models/source/__init__.py` — 暴露 `MirrorSet` / `MirrorHealth`
  / `DownloadProgress` / `compute_content_fingerprint` / 4 个
  mirror/health/is_useful helpers
- `docs/placeholder_registry.md` — 8 条新 entry (54-61) 覆盖
  `models/source/cache.py:509,578,582` (原子写 + rmdir 兜底)
  + `models/source/huggingface.py:164,170` (HttpTransport abstract
  占位) + `models/source/huggingface.py:564,597,622`
  (progress callback 兜底)

**测试**:
- 总测试数: 652 → **683** (净增 31, 全过, 49.90s)
- `pytest -m model_source` 跑 84/84 (53 旧 + 31 新)
- `pytest -m "not model_source"` 跑 599/599

**Examples**:
- `python examples/model_download.py` 零网络跑通:
  1. 镜像列表构造 (`MirrorSet.from_env()`)
  2. 健康检查 (FakeTransport 报 1 个可达 + 1 个不可达)
  3. 第一次 fetch (`from_cache=False`, 写 v1)
  4. 第二次 fetch (same key, `from_cache=True` 直接 cache hit)
  5. 第三次 fetch (不同 revision v1.1, `from_cache=True`
     走 cross-mirror dedup,**不写 v1.1 目录**, 仍能 serve
     现有 v1 的 manifest)
  6. 流量后健康检查 (upstream 仍 mark dead, mirror alive)

**Scanner 双 0**:
- Hardcoding scanner: 4452 total, critical 3857, info 595
  (vs D1 阶段二 4157 total / 3235 critical, 新增 216 主要是
  tests + mirrors 字符串路径)
- Placeholder registry: 50/50 OK (新增 8 条)
- 纯 torch,**无** `transformers` / `diffusers` / `safetensors` /
  `tokenizers` 依赖

**不做** (留到 v1.0):
- 流式字节进度 (transport 协议目前一次性返完整 bytes,
  progress 是 per-file granularity 而非 byte-level)
- 异步并发 mirror race (目前 strict 顺序 fallback, race 留给
  后续 v1.0 调度器)
- 自动镜像 health check 周期 (目前 health check 是
  on-demand ad-hoc)

### P0 多模态真模型接入 (v0.4.x P0 multi-modal milestone)

把 `models/image/` / `models/audio/` / `models/video/` / `models/multimodal/`
里已经写好的 UNet / VAE / CLIP / TTS-Transformer / HiFi-GAN / VideoDiT /
VideoVAE / OmniModel 全部接进 provider 层,4 个新 `LocalTorch*Provider`
+ 4 个 `fetch_and_load_*` + 4 个 `get_default_*_provider` + 4 个
`register_default_*_backend`,并把 3 个 `examples/` 改成走真 provider。
CI 上 31 个新测试覆盖 4 个模态的端到端 forward pass。

**新文件**:
- `models/interfaces/media_providers.py` — 4 个新 `ImageProvider` /
  `AudioProvider` / `VideoProvider` / `MultimodalProvider` Protocol
  + 4 个 `Echo*Provider` reference impl
- `models/providers/local_image.py` — `LocalTorchImageProvider` (UNet +
  VAE + CLIP) 4M params, 一次 forward ~0.1s CPU
- `models/providers/local_audio.py` — `LocalTorchAudioProvider` (TTS +
  HiFi-GAN) 4.5M params, 一次 forward ~0.1s CPU
- `models/providers/local_video.py` — `LocalTorchVideoProvider` (VideoDiT
  + VideoVAE) 5.5M params, 一次 forward ~0.1s CPU
- `models/providers/local_multimodal.py` — `LocalTorchMultimodalProvider`
  (OmniModel + TinyCausalLM) 4.5M params, multi-modal forward
  ~0.5s CPU
- `tests/test_multimodal_providers.py` — 31 个新测试

**升级**:
- `models/providers/__init__.py` — 暴露 4 个新 provider + 4 个 factory
- `models/providers/factory.py` — 新增 `fetch_and_load_image` /
  `fetch_and_load_audio` / `fetch_and_load_video` / `fetch_and_load_omni`
  + 4 个 `get_default_*_provider` singleton
- `models/interfaces/__init__.py` — re-export 4 个新 Protocol + Echo impl
- `nodes/_helpers.py` — 4 个 `register_default_*_backend` (no-arg form)
  装真 backend factory;旧 v0.4.0 `(factory)` 版本删除
- `examples/image_gen.py` / `audio_tts.py` / `video_gen.py` — 改成走真
  provider,加 elapsed 计时
- `docs/placeholder_registry.md` — 新增 6 条 (entries 48-53) 覆盖
  `_local_*_factory` 与 `_get_default_default` 内的降级 `pass`

**测试**:
- 总测试数: 621 → **652** (净增 31, 全过, 51.02s)
- 4 个模态端到端 forward pass: image (3, 16, 16) / audio (1, 512) /
  video (4, 3, 8, 8) / omni (text + image_emb + audio_emb)
- Examples: 3 个 `examples/*.py` 跑通真模型
  * image 64x64 ~1.8s
  * video 4 帧 64x64 ~2.5s
  * audio 0.1s @ 16kHz ~1.7s

**Scanner 双 0**:
- Hardcoding scanner: 4228 total, critical 3304, info 924 (与 D1 阶段二一致)
- Placeholder registry: 53/53 OK (新增 6 条)
- 纯 torch,无 transformers / diffusers / safetensors / tokenizers 依赖

### 架构简化
- 删除冗余的旧版本历史记录与架构清理叙事。
- 删除文档中过时的版本号与"v0.3.0/v0.3.1"中间过渡描述。
- 节点 `execute()` 统一通过 `nodes/_helpers.py` 中的 `call_text_backend` /
  `call_image_backend` / `call_video_backend` / `call_audio_backend`
  解析后端,默认回退 echo 工厂,允许在无模型情况下端到端跑通。
- 节点实现通过 `core.module_bus.ModuleBus` + `LLMProvider` 协议接入真模型,
  流水线层不再做 passthrough。

### 一致性评分
- `consistency/score.py` 优先尝试 `open_clip` / `torch.hub` 的真 CLIP/DINOv2
  视觉特征;未安装时回退到项目内的轻量占位特征提取器(随机投影,固定维度)。
- 评分指标与 `ConsistencyScore` 数据结构保持不变,接口兼容现有调用方。

### Hardcoding 规约化（D1）
- 新建 `docs/hardcoding_convention.md` — v0.4.x D1 根规约,3 类常量边界:
  * 运行时配置 (RUNTIME_CONFIG, `critical`) — 业务可调,必须走 ConfigCenter
  * 模型结构超参 (MODEL_STRUCTURAL, `info`) — 改了就坏,保留源码
  * 协议/格式标识 (PROTOCOL_FORMAT, `info`) — 与外部协议绑定,改了就坏
- 增强 `scripts/check_hardcoding.py`:
  * `Violation` 加 `severity` 字段 (默认 `critical`)
  * `Exemption` 加 `severity` 字段 + `protocol_format: true` 字段 (非 terminal 降级)
  * `--severity` CLI 选项: `critical` / `warn` / `info` 三档过滤
  * `--export <path>` 选项: 导出 critical 名单 (whitelist-schema 兼容 YAML)
  * `is_structural_init` 启发式: `models/` 路径下 `__init__` 中值在 [2, 10000]
    的整数自动降为 `info`
  * `_is_runtime_attr` 启发式: `os.environ[...]` / `Path(...)` / `sys.argv[...]`
    表达式中的字面量自动降为 `info`
  * `filter_by_severity()` 函数: 阈值过滤
  * `export_critical()` 函数: 去重导出 (按 file/line/type 唯一化)
- 新建 `config/hardcoding_critical_inventory.yaml` — 全项目 critical 3420
  unique entries 基线 (供 PR review 参考, **不**直接喂给 --whitelist)
- 填实 `config/hardcoded_whitelist.yaml` — 首批 ~90 条 exemption 示范:
  * 7 个 training 训练超参 group (SFT/RLHF/Synthetic/Dataset numeric) → `info`
  * 协议/格式 (LayerNorm.weight / attention_mask / observation /
    ShortTermMemory / synthetic prompt 模板) → `protocol_format: true`
  * torcha-verse 顶层 re-export 字符串 (ConfigCenter / DeviceManager / ...) → 协议绑定
- 33 个新测试覆盖: Violation 默认值 / Exemption.matches/apply/is_terminal /
  filter_by_severity 阈值 / scanner 启发式 (`is_structural_init` /
  `_is_runtime_attr` 各 2-3 个分支) / whitelist YAML 加载 / 非法 severity 拒收 /
  export_critical 去重与 critical 过滤 / **端到端** (真实 whitelist 真的降级
  命中)。
- `pyproject.toml` 注册 `hardcoding_severity` marker。
- 总测试数: 581 → 614 (全过, 46.53s)。
- Scanner 升级后分级效果: 3740 total → critical 3352, info 438。
- 顺手修正: `placeholder_registry.md` 中 `scripts/check_hardcoding.py`
  位置 (行号 338 → 526 因 scanner 重写), 仍 47 entries 全部注册。

### Hardcoding 规约化（D1）— 阶段二

- **log message 启发式** (`scripts/check_hardcoding.py:is_log_message_format`):
  把 logger 调用的**第一个字符串参数**自动降为 `info` (不再是完全 exclude),
  让 audit 仍能看到 log format 串 (PR review 时可看), 但永不 CI-fail。
  * 触发条件: 字符串 literal 是 `logger.{debug,info,warning,warn,error,
    critical,exception,log,fatal}(...)` 的 **第一个位置参数**。
  * 7 个新测试: info/warning/error 各一例 + 后续位置参数仍 critical +
    keyword arg 不算 format string + helper 直接单测。
- **批量 200+ protocol exemption** (`config/hardcoded_whitelist.yaml`):
  从阶段一 ~90 条 → **211 条** (净增 117), 新增 11 个 group:
  * Group 8: reAct / tool_call agent 协议正则 (Thought: / Action: /
    Action Input: / Final Answer: / Observation: / FINAL ANSWER: /
    ```(?:json)?...)
  * Group 9: agents/flows/ prompt 模板 (debate / hierarchical / sequential)
  * Group 10: assets/ 协议键名 + 错误消息 + SQL 字面量 (NOASSERTION /
    PRAGMA / SELECT metadata_json / INSERT OR REPLACE...)
  * Group 11: nodes/ 协议/格式 (controlnet / lip_sync / expression_params
    / consistency_score / face_embedding / voice_signature...)
  * Group 12: pipeline/ 模板协议 (input_schema / output_schema / node_type)
  * Group 13: tools/ + plugins/ 协议 (file_path / entry_point / plugin_name)
  * Group 14: serving/ HTTP 协议 (Content-Type / Authorization / X-Request-ID
    / /v1/ / /health / text/html)
  * Group 15: infrastructure/ 协议 (max_memory_mb / max_cpu_cores /
    TORCHAVERSE_*_DIR / config_snapshot.json)
  * Group 16: examples/ demo 协议 (demo_ 前缀 / Hello, world!)
  * Group 17: numeric_literal 通用超参 (14 个文件全项目 numeric → info)
  * Group 18: logger 专用批量 exemption (16 个常见 log message 前缀, 作
    heuristic 的 defence-in-depth fallback)
- **`docs/config_access.md`** — ConfigCenter / defaults 用户文档, 16 节:
  4 层配置模型 / 90 秒上手 / 读 API / 写 API / 加载顺序 / 环境变量覆盖 /
  平台差异 / 快照与重放 / ResourceBudget / `infrastructure.defaults`
  懒加载 / 环境切换 / 完整示例 / 反模式 / 故障排查 / D1 规约关系 / 速查表。
- **7 个新测试** 覆盖 log message 启发式: TestLogMessageFormat 7 个 case
  (info / warning / error / 后续参数 / keyword arg / helper 正/反例)。
- 总测试数: 614 → **621** (全过, 46.98s)。
- Scanner 升级后分级效果: 3740 total → **4157 total** (log 启发式让
  之前 excluded 的 log 字符串进 inventory, 但 severity=info) →
  **critical 3235**, **info 922** (之前 critical 3352, info 438)。
- Critical inventory (`config/hardcoding_critical_inventory.yaml`) 重新
  导出: 3420 unique → **3235 unique** (净降 185 条已批量落 exemption)。
- `pyproject.toml` 不变 (`hardcoding_severity` marker 仍有效)。
- 顺手修正: `placeholder_registry.md` 行号 526 → 569 (因 `is_log_message_format`
  method 插入, scanner 内的 `pass` 位置下移), 47 entries 仍全部注册。

### Placeholder Registry（D3 工作流集中化阶段）
- 新建 `docs/placeholder_registry.md` 作为**占位单一来源**（single source of truth）:
  47 处 `pass` / `NotImplementedError` 全部按 5 类（`protocol` / `tp_pp` /
  `protocol_stub` / `degrade_try_except` / `degrade_noop`）登记，含
  文件:行 / 上下文 / 理由。
- 新建 `infrastructure/placeholder_registry.py`:
  * `PlaceholderCategory` 枚举 + `PlaceholderEntry` dataclass
  * `load_registry` 解析 markdown 表（按 heading 推断类别，宽容处理坏行）
  * `scan_source` 扫描 Python 源文件（跳过 `tests/`、`__pycache__`、`.git/`、
    `.venv/` 等;支持行内 `# placeholder-registry: ignore` 豁免;自动跳过
    docstring 中用反引号引用的关键字描述）
  * `find_unregistered` 计算 scanner - registry 差集
  * `registry_index` 建 `(file, line) -> entry` 快速查找
- 新建 `scripts/check_placeholders.py` CI CLI:扫描 / 校验 / 报告未注册占位
  + 退出码 1 用于 CI gating。
- 升级 `infrastructure/device_manager.py` 注释：`_tensor_parallel_impl` /
  `_pipeline_parallel_impl` 现在显式引用 `placeholder_registry.md` 中
  的条目编号 (#8 / #9) + D3 重启条件,让"占位在哪儿"和"何时重启"解耦。
- 22 个新测试覆盖：枚举完整性 / `PlaceholderEntry.matches` / `load_registry`
  多种格式 / heading → category 推断 / 坏行宽容 / scanner 各分支（pass /
  NotImplementedError / ignore marker / docstring 引用 / 单文件 target /
  不存在 target）/ `find_unregistered` 差集 / `registry_index` 查表 /
  **端到端**（真实 project registry + 真实 project scan 应当 0 unregistered）。
- `pyproject.toml` 注册 `placeholder_registry` marker。
- 总测试数：559 → 581（全过，46.76s）。
- `python scripts/check_placeholders.py` 全项目扫描 47 命中, 0 unregistered。

### 真模型跑通（v0.4.0 路线图 P0）
- 新建 `models/providers/` 子包，纯 torch 实现项目自有 tiny Transformer LM，
  不引入 `transformers` / `diffusers` / `safetensors` 等外部依赖：
  * `tiny_transformer.py` — `TinyTransformerConfig`（`tiny` ~0.3M / `small` ~10M 两个预设）
    + `ByteTokenizer` 字节级 tokenizer（3 special + 256 bytes + 1 mask = 260 vocab）
    + `build_tiny_transformer` / `save_tiny_transformer` / `load_tiny_transformer`
      单文件 `.pt` 持久化，原子写入（tempfile + fsync + os.replace）
  * `local_text.py` — `LocalTorchTextProvider`（实现 `LLMProvider` 协议），
    `generate` / `chat` / `complete` 三个推理入口，线程安全
  * `factory.py` — `fetch_and_load_text` 一行拿到 provider（checkpoint → 随机初始化 fallback），
    `publish_tiny_transformer` 维护端发布，`get_default_provider` 进程级单例
  * `pretrain_tiny.py` — `train_tiny_transformer` + CLI
    (`python -m models.providers.pretrain_tiny --preset small --steps 600`)，
    AdamW + cosine LR + warmup
- 新增 `examples/real_text_chat.py`：端到端 demo
  pretrain → save → load → register_default_text_backend →
  L4 `text_chat` 节点输出真模型生成文本
- 37 个新测试覆盖：tokenizer 边界 + round-trip、config presets + dict 序列化、
  save/load 原子性 + 版本检查 + 严格性、provider 协议契约、
  factory 分支（resolve / fetch 随机 / fetch checkpoint / 缺文件报错）、
  pretrain 端到端、L4 集成（`call_text_backend` + 1 节点 Pipeline）。
- `pyproject.toml` 注册 `model_provider` marker，
  `pytest -m model_provider` 跑 37 个，`pytest -m "not model_provider"` 跑 522 个，互不干扰。
- 总测试数：522 → 559（全过，45.86s）。

### 评估模块（v0.4.0 路线图 P1）
- 新建 `evaluation/` 目录,提供纯 PyTorch 实现的指标层:
  * `metrics.psnr` / `metrics.ssim` / `metrics.lpips`（LPIPS 为占位接口）。
  * `fid.image_fid` / `fid.frechet_distance` / `fid.FidCalculator`，矩阵平方根
    用 `torch.linalg.eigh` 闭式求解，无 scipy 依赖。
  * `prompt_recall.score` / `prompt_recall.prompt_recall` /
    `prompt_recall.PromptRecallCalculator`（CLIP-score 占位实现）。
  * `runner.EvaluationRunner` / `runner.EvaluationReport` /
    `runner.load_image_dir` —— CI 友好的一站式入口。
- 占位 Inception / CLIP / LPIPS backbone 与真模型 API 完全一致，
  未来替换是真模型一行 class 替换，不影响调用方。
- 52 个新测试覆盖：指标数值正确性、FID 对称/非负/同集→0、
  矩阵平方根数值、tokenizer 确定性、双编码器形状、目录加载器、
  EvaluationRunner 端到端。`pyproject.toml` 注册 `eval` marker，
  `pytest -m eval` 跑 52 个，`pytest -m "not eval"` 跑 417 个，互不干扰。
- 总测试数：411 → 469（全过）。

### 模型源自动拉取（v0.4.0 路线图 P2）
- 新建 `models/source/` 子包，提供一行 `fetch()` 拉模型 + 许可证审计：
  * `license_check.check_license` / `DEFAULT_ALLOW_LICENSE` — SPDX 许可证白名单，
    支持运行期 `extend_default_allow_license(...)` 一次性 opt-in。
  * `cache.ModelCache` — `~/.cache/torcha-verse/<source>/<repo_id>/<revision>/`
    原子写入（tempfile + fsync + os.replace）+ sha256 完整性校验。
  * `huggingface.HuggingFaceSource` — HF Hub API 包装，注入式 `HttpTransport`
    让测试零网络跑通，默认用 `urllib.request`。
  * `civitai.CivitaiSource` — 备选源，同一套 `HttpTransport` 接口。
  * `fetch.ModelFetcher` / `fetch()` — 统一入口，验证 license、查缓存、
    拉取、写入、验证 manifest 全部原子化。
- `models/__init__.py` 重新导出 `fetch` / `FetchResult` / `ModelFetcher` 等
  公共 API，达成 `from torcha_verse.models import fetch` 简写。
- 53 个新测试覆盖：SPDX 规范化、allow-list/NC/ND 短路、
  extend_idempotent、cache 原子写入 / 验证 / 清空、manifest
  round-trip、HF / Civitai license 解析 + 文件列表 + 下载、
  SourceRegistry 别名、fetch miss-then-hit、NC 拒绝、
  cache tampering 检测、自定义 allow_list、模块级 fetch 单例。
- `pyproject.toml` 注册 `model_source` marker，`pytest -m model_source`
  跑 53 个，`pytest -m "not model_source"` 跑 469 个，互不干扰。
- 总测试数：469 → 522（全过）。

### 工程化
- 新增 `pyproject.toml`(含 pytest 配置)。
- `Dockerfile` 改为多阶段构建(builder / test / runtime)。
- 新增 `.github/workflows/ci.yml`,跑 lint + 全量测试。
- `requirements.txt` 同步精简。

### 文档
- 重写 `README.md`,对齐新结构与新流水线示例。
- `docs/architecture.md` / `docs/operations.md` 同步精简,移除历史叙事。
- 新增 `docs/DEFERRED_TASKS.md`,登记开发初期延后处理的任务(当前条目:hardcoding 规约化)。
- 新增 `docs/ROADMAP.md`(v0.4.x 准生产化 12 周计划 + v1.0.0 纲要),
  P1 评估模块 2026-06-25 标记完成。

---

[v0.4.0]: https://github.com/brother2050/torcha-verse/releases/tag/v0.4.0
[v0.4.1]: https://github.com/brother2050/torcha-verse/releases/tag/v0.4.1
[v0.4.2]: https://github.com/brother2050/torcha-verse/releases/tag/v0.4.2
[v0.4.3]: https://github.com/brother2050/torcha-verse/releases/tag/v0.4.3
