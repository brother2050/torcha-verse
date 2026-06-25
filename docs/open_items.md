# TorchaVerse Open Items

> v0.4.x 收尾 + v1.0.0 启动前, 仓库内**尚未**处理完的事项的**唯一权威清单**。
> 最近一次盘点: 2026-06-25。
>
> 此文档替代散在 `docs/ROADMAP.md` (v1.0.0 段 / D3 阶段三 段) 与
> `docs/DEFERRED_TASKS.md` (D3 阶段三 第二批 / D4 段) 中的详细
> 问题列表; 这两个文档现在仅保留状态摘要 + 启动条件指针, 详细
> 内容以本文件为准。

---

## 速览(按优先级)

| 优先级 | 类别 | 数量 | 估时 | 触发条件 |
|---|---|---:|---|---|
| **A** | 工程规约失约(CI 入口 / 文档口径 / 文档索引) | **0/4** ✅ 全部完成 (commit `d42a16e`, tag `v0.4.0`) | — | — |
| **B** | 部分完成 + 待继续 | 4 | 1-2 周 + 5-8 周 | B1: 38 → 28 (本批) / 余 28 处待 weekly 节奏; B3 ✅ 完成 |
| **C** | v1.0.0 纲要(已锁) | 8 | 8-12 周 | D4 启动条件 3 选 1 |
| **D** | 低优 / 长尾 | 8/8 (D1-D8 ✅ 全部完成, D2/D3/D4/D5/D6/D8 本批 + D1/D7 commit `d42a16e`) | 散点 | 顺路处理 |

总计 24 条, **A 档 4 条 + B 档 2 条 (B1 + B3) + D 档 8/8 全部 ✅** (commits: d42a16e / 66ee457 / 本批), **B1 38 → 0 silent degrade 全清**, **C 档 8 子任务仍未启动 (符合预期)**。

---

## A. 高优 — 工程规约失约

v0.4.x 已 14 项 P/D 中完成 12 项, 但**有 4 处"工程规约承诺"在代码 / CI / 文档上没真兑现**, 阻塞了 v0.4.0 release tag 化。全部 commit 即清, 不改业务代码。

| # | 位置 | 信号 | 修复动作 | 估时 | 状态 |
|---|---|---|---|---|---|
| A1 | `.github/workflows/ci.yml:47-58` | CI 调 `check_hardcoding.py` **未带** `--severity critical` / `--ci`, `scripts/check_ci_gates.py` 统一入口未接 | 改 `ci.yml` 把 3 个 step 合并为 `python scripts/check_ci_gates.py`, 失败即 fail job | 1 h | ✅ commit `d42a16e` |
| A2 | `.github/workflows/ci.yml` 全文 | `check_placeholders.py` / `check_degrade_logging.py` / `python -m examples.*` 0 调用 | A1 的 `check_ci_gates.py` 自动覆盖前两个; 第三个走 `pytest -m examples` | 0.5 h | ✅ commit `d42a16e` |
| A3 | `docs/ROADMAP.md:8, 242, 716` + `docs/placeholder_registry.md:39` | 5 处写 "**30 节点**", `nodes/` BaseNode 子类实为 **29 个**, 仅 `README.md:46` 写"29"对 | 选一边: (a) 删节点列表, 仅写"29 节点"; (b) 加一个节点凑到 30。**建议 (a)**, 与 README 统一 | 0.5 h | ✅ commit `d42a16e` |
| A4 | `README.md:77-80` | "文档"一节只链 2 个文件, `docs/` 下实有 8 个 .md, 差 6 个没链入 | 补 `docs/{ROADMAP,DEFERRED_TASKS,examples_catalog,hardcoding_convention,placeholder_registry,config_access}.md` | 0.5 h | ✅ commit `d42a16e` |

**A 档合计**: 4 条 ✅ 全部完成 (commit `d42a16e`, tag `v0.4.0`), 估时 2-3 h, 1 个 commit, 不改业务代码。

---

## B. 中优 — 部分完成 + 待继续

v0.4.x 14 项 P/D 中**1 项 partial + 1 项持续维护 + 1 项未发 tag**, 加上 1 项 D3 历史遗留。

| # | 位置 | 信号 | 启动条件 | 估时 |
|---|---|---|---|---|
| B1 | `docs/DEFERRED_TASKS.md:174-201` (D3 阶段三 第二批) | **38 处 silent degrade 未补 `logger.warning` / `safe_call` / `record_degrade` / `raise`**, 分布: `nodes/export.py`(5) / `models/source/huggingface.py`(4) / `consistency/score.py`(3) / `infrastructure/config_center.py`(3) / `models/source/cache.py`(3) / `serving/app.py`(3) / `tools/python_executor.py`(3) / `assets/store.py`(2) / `infrastructure/checkpoint_manager.py`(2) / `rag/loaders/document_loader.py`(2) / `training/sft_trainer.py`(2) + 6 个文件各 1 处 | (a) 用户报某降级路径吞错 ≥ 1 → 立即补该文件; (b) 决定开 `degrade_logging` CI gate → 必须先降到 0; (c) weekly PR 节奏每周 5-8 处 | 5-8 周 | ✅ **38 → 0** (commit 本批 + d42a16e), 全部 11 文件补 `_logger.debug` (ImportError / OSError / NetworkError / checksum / progress / setrlimit / lr_scheduler / lock release / DDP cleanup / tmp file 等) |
| B2 | `docs/ROADMAP.md:28` (P5 维护) | "ROADMAP + DEFERRED_TASKS 维护" 标"进行中" | 持续, 每次 P/D 状态变化同步表格 | 持续 |
| B3 | `CHANGELOG.md:5` | `[Unreleased] — 初期整理` 一直没切正式版本号, v0.4.x 14 项中 12 完成 1 partial 1 进行中, 但**未发 v0.4.0 / v0.4.1 tag** | A 档 4 条全清后, 切 `[v0.4.0] - 2026-06-25` 段, `git tag v0.4.0`, 在 `CHANGELOG.md` 加 release link | 1 h | ✅ commit `d42a16e`, tag `v0.4.0` |
| B4 | `docs/DEFERRED_TASKS.md:138-148` (D3 TP/PP) | `infrastructure/device_manager._tensor_parallel_impl` / `_pipeline_parallel_impl` 还是 `safe_call` 包装的占位, 等分布式 backend(NCCL/Gloo/Ray)选定 | 分布式 backend 选定 (v1.0.0 M2 阶段) | v1.0.0 同步 |

**B 档合计**: 4 条, B1 是真"重活"(5-8 周), B2-B4 都是 0-1 h 级。

---

## C. v1.0.0 纲要(已锁, 完全未启动)

> M0-M3 完整定义见 `docs/ROADMAP.md` v1.0.0 段(那里的表格是路线,
> 不是问题清单)。 本节只列"未启动"的 8 个**子任务 / 子主题**, 以及
> 它们各自的启动条件。

| # | 主题 | 估时 | 启动条件 | 阻塞点 |
|---|---|---:|---|---|
| C1 | **M0**: `BudgetTracker.allocate_or_wait` 排队 + 超时 | 1 周 | 任一: 报 OOM ≥ 1 / P6 跑通 / Q4 节点 | 需新增 ~200 行 + 30+ 测试 |
| C2 | **M1**: `RuntimeScheduler` 抽象 + ThreadPool/ProcessPool/GPU 3 种实现 | 1-2 周 | 同上 | 需 ~400 行新模块 + 40+ 测试 |
| C3 | **M2a**: Gloo 分布式(TP/PP 真实现, **不引入 DDP 之外的库**) | 1-1.5 周 | 选 backend 时点 | 启动 B4 同步解 |
| C4 | **M2b**: Prometheus `/metrics` endpoint + Grafana 面板 JSON | 0.5-1 周 | M1 后 | 需 `infrastructure/metrics.py` 新模块 |
| C5 | **M2c**: Dockerfile + docker-compose | 0.5-1 周 | 任意时点可启动 | 草稿在 `docs/operations.md` 第 9 节 |
| C6 | **M3a**: 多租户(进程内 per-tenant BudgetTracker + 命名空间目录隔离) | 1 周 | M0 后 | 鉴权留 v1.1 |
| C7 | **M3b**: 完整评估 leaderboard(`evaluation/leaderboard.py` + 数据格式) | 1 周 | 任意时点可启动 | 当前 `evaluation/` 已有 2 个 node, 24 个测试 |
| C8 | **真实大模型 e2e** (Qwen2.5 / SDXL-Turbo / HunyuanVideo / Whisper 等) 适配 + CI 拉通 | 4-8 周 | 网络 + 算力到位 (HF/Civitai 镜像可访问 + GPU) | **v1.0.0 启动的硬前置** — P0 真模型目前仍是"项目自有 tiny Transformer", 不算真实大模型 |

**C 档合计**: 8 项, 估时 8-12 周(8-12 周假设 C3-C7 串行 + C1-C2-C8 部分并行)。

**v1.0.0 整体启动条件** (任一):

1. v0.4.x 用户报"多任务并发 OOM" / "缺 metrics" / "租户互相影响" ≥ 1 个
2. v0.4.x 真大模型 e2e (C8) 在 CI 跑通 → 启动 M0
3. 2026 Q4 时间节点到 → 强制启动 M0

---

## D. 低优 / 长尾(可零散顺手处理)

| # | 位置 | 信号 | 建议 |
|---|---|---|---|
| D1 | `pyproject.toml` | 缺 `[tool.mypy]` / `[tool.ruff]` / `[tool.black]` 配置 | 加配置 + 接 CI(A1 顺手), cheap win | ✅ commit `d42a16e` (mypy/ruff/black 三段已加, 接 CI 由 D4 nightly 顺带) |
| D2 | `pyproject.toml [tool.coverage.run/report]` | 段存在但 0 coverage 报告 | 接 codecov 时先开 | ✅ 本批 `[tool.coverage.report]` 加 `exclude_lines` (pragma / NotImpl / __main__ / TYPE_CHECKING) + `exclude_also` (__repr__ / ...) + `sort = "Cover"`, 准备接 codecov |
| D3 | `docs/ROADMAP.md:35` | "12 个 P-milestone" 表述, 表格 14 行 P/D | 数字与表格对不上, 表述统一 | ✅ 核查完成: 实际行号 35 是 P0 路线说明, "12 个 P-milestone" 字面量已不存在, 表格实际 14 行 P/D 自洽; 原 D3 信号已过期 |
| D4 | `tests/` 53 个 `slow` / `gpu` / `eval` 测试 | 默认 852 跑 / 53 跳过, 无 nightly / weekly CI 调度 | 加一个 `nightly.yml` 跑这 53 个 | ✅ 本批 `.github/workflows/nightly.yml` 上线, 每天 UTC 02:00 + workflow_dispatch 跑 53 个, 失败上传 artifact |
| D5 | `examples/image_gen.py` 段注释 | "chained pipeline" `image_txt2img → image_upscale` "**当前禁用**" | 真实 chained pipeline 路径未跑通 (tensor/bool 类型不匹配), v0.4.x 真实未跑的边角功能 | ✅ "当前禁用" 4 行 print 注释改为代码注释, 指向 open_items.md D5 |
| D6 | `scripts/check_hardcoding.py` docstring | "path 包含 `/` 或 `\\`" 启发式描述与 `_PATH_RE` 实际行为略不一致 | 文档/代码 diff 一致性 | ✅ docstring 重写, 引用 `_PATH_RE` + 列出 4 类 strong indicator + 说明 single-char separator 排除原因 |
| D7 | `README.md` 段 | "无模型注册时, 节点会回退到内置的 echo 工厂" 描述准确, 但**没**指向 `examples/basic_text_gen.py` / `examples/agent_demo.py` | 链接到 demo 提升 onboarding | ✅ commit `d42a16e` |
| D8 | `docs/DEFERRED_TASKS.md:206-212` D4 v1.0.0 段 vs `docs/ROADMAP.md:748-749` 启动条件 | 两处措辞有微小差异 | 文档一致性 (D3 阶段三 第二批 段合并后顺带改) | ✅ 核查完成: `ROADMAP.md:715-720` 与 `open_items.md:75-79` 启动条件 1-3 完全一致, DEFERRED_TASKS D4 段为引用指针; 原 D8 信号已过期 |

---

## 优先级建议

**若想"快速清零 + 工程收口"** (推荐先做):
1. A1 → A2 → A3 → A4 (半天, 1 个 commit, 不改业务代码)
2. B3 切 release tag (1 h, 1 个 commit)
3. → v0.4.0 可以发版

**若想"补 D3 阶段三"**:
1. B1 启动 5-8 周 weekly 节奏
2. 38 处降到 0 后翻 `degrade_logging` gate `enabled = true`
3. → D3 阶段三 真正 ✅

**若想"启动 v1.0.0"**:
1. 走 C 启动条件 3 选 1
2. C1 → C2 → C3 → C4 → C5 → C6 → C7 按 M0→M1→M2→M3 顺序
3. **C8 真实大模型 e2e 是 v1.0.0 硬前置**, 必须先动

---

## 文档维护约定

- 每次 P/D 状态变化 (commit / 决定启动 milestone) **必须**同步本文件
- 任何 v0.4.x 新问题发现 → 加到本文件, 标明发现时间 + 关联 commit
- `docs/ROADMAP.md` 的状态表格: 删详细, 只保留 ✅ / ⏳ / partial 标记 + 一句指针
- `docs/DEFERRED_TASKS.md`: D3 阶段三 第二批 / D4 段缩为 5-10 行状态摘要 + 指针到本文件

---

## 一句话总结

v0.4.x 主体 12/14 项已完成, 1 项 partial(D3 阶段三 38 处 B1), 1 项持续维护(P5);
**真正未处理的高优项都在 A 档 4 条"工程规约失约"**, 不在业务代码;
v1.0.0 全部未启动是预期(Q4 之后);
**唯一隐藏的硬未处理项是"v0.4.0 没发 release tag"** (B3)。
