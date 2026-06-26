# Changelog

> 项目变更记录。架构简洁、节点能跑、测试可过。

## [Unreleased]

## [v0.6.0] - 2026-06-26

### R-* 重构 (R-3 ~ R-15 + R-19) — 65+ 聚焦子模块

13 个 PR 把 9,419 行的单文件拆成 65+ 子模块,**0 回归**:

| PR | 范围 | 行数 | 子文件数 |
|---|---|---:|---:|
| R-3 | `checkpoint_manager` 拆 | 631 | 7 |
| R-4 | `agent` 拆 | 692 | 4 |
| R-5 | `training/dataset` 拆 | 1063 | 6 |
| R-6 | `huggingface` 拆 | 983 | 9 |
| R-7 | `models/source/fetch` 拆 | 837 | 6 |
| R-8 | `serving/app` 拆 | 1208 | 8 |
| R-9 | `serving/cli` 拆 | 962 | 9 |
| R-10 | `serving/service` 拆 + PipelineService | 848 | 8 |
| R-11 | `scripts/check_*` 合并 | 2634 | 11 (子包) |
| R-12 | `assets/store` 拆 + 协议分层 | 861 | 8 |
| R-13 | `assets/model_asset` 拆 | 473 | 6 |
| R-14 | `nodes/base` 拆 | 850 | 7 |
| R-15 | `nodes/_helpers` 拆 | 710 | 5 |
| R-19 | 撤 shim + MD 重写 | - | - |

## [v0.6.1] - 2026-06-26

### R-18 — papers 懒化

`import papers` 不再触发 1000+ 行 torch 模块加载:
- **Cold import**: 1288 ms → **47 ms (-96%)**
- **import 后 `sys.modules` 中 0 个 `papers.adapters.*` 模块**

变更:
- `papers/__init__.py` — 移除 `from .adapters import ...` 的 eager 加载;
  - 加 `_ADAPTER_NAME_TO_MODULE` 字典 + `_loaded_adapters` 模块级 cache
  - `AdapterRegistry.get` / `has` 加 lazy fallback (monkey-patch): bundled name 未注册时,先 import 模块再 register
  - `__getattr__` (PEP 562) 懒导出 `StableDiffusion3Adapter` / `HunyuanDiTAdapter` / `cli` (用 module-level cache,不写 `globals()`,支持测试 purge)
  - `__dir__` 暴露懒导出给 IDE
  - `TYPE_CHECKING` 守卫 import 给静态分析器
- `papers/adapters/__init__.py` — 改为 lazy: `PaperAdapter` 仍 eager (无 torch),`StableDiffusion3Adapter` / `HunyuanDiTAdapter` 改 `__getattr__` 懒加载子模块
- `tests/test_r18_lazy.py` — 11 个新测试: import 不加载 torch / `has` 不触发 import / `get` 触发 / cache hit / 未知名字抛 `AdapterNotFoundError` / PEP 562 懒导出 / 子包懒 / `PaperAdapter` 仍 eager

### R-16 — 性能优化

优化 `NodeContext` / `NodeRegistry` hot path,微基准:
- `get_output` (100k): **85 → 66 ms (-23%)** — 拆 fast path 无锁
- `resolve_executor` cache hit (100k): **255 → 22 ms (-91%)** — LRU 1024
- `resolve_executor` mixed 5 types (100k): **255 → 66 ms (-74%)**
- `registry.list` (1k): **39 → 0.4 ms (-99%)** — 缓存到 register/unregister
- `set_output` (100k): **87 → 84 ms** — `RLock` 改 `Lock`

变更:
- `nodes/base/_context.py` — `RLock` 改双 `Lock` (`_outputs_lock` / `_executors_lock`),
  保留 `_lock` 兼容 alias,`get_output` / `has_output` 走 GIL 保护无锁 fast path,
  `resolve_executor` 加 FIFO 1024 LRU,`register_executor` 失效缓存,
  负结果也缓存 (unregistered type 不再每次打 bus)
- `nodes/base/_registry.py` — `list()` 缓存到 `register` / `unregister`
- `tests/test_r16_perf.py` — 11 个新测试 (并发不破 + 缓存 + 失效 + 负缓存 + 驱逐)

### R-19 — 撤 shim + MD 重写

**撤 shim** (5 个):
- `scripts/check_hardcoding.py` / `check_hardcoding_rules.py` / `check_placeholders.py` / `check_degrade_logging.py` / `check_ci_gates.py` 全删
- 测试 3 处 `from scripts.check_*` → `from scripts.check.<subpkg>` 子包路径
- `_cli.py` 加 `if __name__ == "__main__": sys.exit(main())` 入口
- `hardcoding/__init__.py` 把 `main` 改成 PEP 562 `__getattr__` 懒导出,避免循环

**重写 MD** (13 个):
- `README.md` / `docs/architecture.md` / `docs/operations.md` / `docs/ROADMAP.md` / `docs/DEFERRED_TASKS.md` / `docs/open_items.md` / `docs/examples_catalog.md` / `docs/hardcoding_convention.md` / `docs/placeholder_registry.md` / `docs/config_access.md` / `docs/docker.md` / `examples/README.md` / `CHANGELOG.md`

**Bug 顺手修** (3 个,非重构):
- CWD config 兜底: `_paths.py` 加 sentinel files 验证
- `torcha models` rich 渲染: `_info.py` 改用 dict 字段
- `image_txt2img` PIL 转换: `_image.py` 加 `_to_pil()` 支持 4 种类型

## [v0.5.2] - 2026-06-25

### D-补丁: 1 个 fp16 matmul 测试在 CPU 缺 kernel 时自动 skip

`tests/test_performance_quantization.py` 里 `test_fp16_changes_dtype`
在沙箱 CPU 环境下 fail (`addmm_impl_cpu_` not implemented for 'Half') —
PyTorch 公开 CPU wheel 故意不实现 fp16 matmul kernel。

- 加 `_has_fp16_matmul()` 探针 + `@requires_fp16_matmul` skipif
- `bf16_changes_dtype` 不装饰 (CPU bf16 mkl+oneDNN wheel 跑得通)

## [v0.5.1] - 2026-06-25

### D-补丁: 撤掉 prometheus_client swap-in (回退 v0.4.3 之前的纯 stdlib 路径)

swap-in 后引入 4 个新 `pass` 兜底 (TCP socket / file descriptors),
与 v0.4.x D1 关闭的 silent-degrade 战略冲突。`StdoutHandler` 即可
对接 ELK / Loki。C3 metrics 阶段会重新评估。

## [v0.5.0] - 2026-06-25

### Added (v0.5.x 功能扩展)

- **L2 资产子系统** — `Asset` / `AssetRef` / `AssetStore` + 5 子类
  (Model / Character / Outfit / Scene / Depth) + 三级存储 (hot/warm/cold)
- **L6 一致性** — `CharacterEngine` / `OutfitEngine` / `SceneEngine`
  + `ScoreCalculator`
- **L5 Pipeline 模板** — `TemplateRegistry` (12 个开箱即用模板)
- **L5 Prompt Studio** — 提示词工程与版本管理
- **L5 Canvas** — 可视化画布 + `AutoDirector` 自动编排
- **安全** — `OutputFilter` (毒性 / NSFW)
- **v0.5 feature demo** — `examples/v05_feature_demo.py`

测试: 986 → 1053 (+67)。

## [v0.4.3] - 2026-06-25

### C1b-C7b 加深

详见 git log `b032082`:
- C1b `BudgetTracker` 排队 + 超时 stub
- C2b `RuntimeScheduler` 抽象 + stub
- C3b 监控端点 stub
- C4b Dockerfile + compose stub
- C5b 多租户 stub
- C6b leaderboard stub
- C7b Docker / k8s 配置

测试: 747 → 830。

## [v0.4.2] - 2026-06-25

### C1-C7 v1.0 骨架

`de35b14`。详见 ROADMAP 历史段。

## [v0.4.1] - 2026-06-25

### B1 silent-degrade 清零

`cec3e5b`。所有 `pass` 改为 `logging.WARN` + 登记到
`docs/placeholder_registry.md`,CI 扫描器强制登记。

## [v0.4.0] - 2026-06-25

### P0 真模型 + P0 多模态 + P1 评估 + P2 模型源

详见 ROADMAP 历史段。测试: 369 → 559。

## [v0.3.0] - 2026-06-24

### 架构骨架 (L1-L5)

6 层分层、ModuleBus、Pipeline DAG、NodeSpec、ConfigCenter、
ResourceBudget、AuditLogger、Security 中游件。
