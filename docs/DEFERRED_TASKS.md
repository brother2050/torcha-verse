# Deferred Tasks

开发初期(early-stage)阶段**不处理**、等架构稳下来后再统一规约化的任务。
每个条目都说明了:是什么、为什么延后、再次启动时怎么判断。

---

## D1 — Hardcoding 规约化与扫描器校准

**状态**:✅ **阶段一 + 阶段二 + 阶段三均完成 2026-06-25**。规约文档 + scanner severity
分级 + log message 启发式 + 211 → 283 条 exemption(含 33 条目录级批量豁免) +
ConfigCenter 用户文档 + 拆规则 + CI gating 全部落地;**critical 3235 → 0**,
统一 CI 入口 `scripts/check_ci_gates.py` 上线 (commit b032082)。

**已完成的阶段一(2026-06-25)**
- `docs/hardcoding_convention.md` 落地,定义 3 类常量边界 (RUNTIME_CONFIG /
  MODEL_STRUCTURAL / PROTOCOL_FORMAT) + 3 档 severity (critical / warn / info)。
- `scripts/check_hardcoding.py` 重写,加 `Violation.severity` /
  `Exemption.severity` / `Exemption.protocol_format` 字段;新增
  `--severity` 与 `--export` CLI 选项;新增 `is_structural_init` 与
  `_is_runtime_attr` 启发式自动降级。
- `config/hardcoded_whitelist.yaml` 填实 ~90 条 exemption
  (training 训练超参 / 协议字段名 / 顶层 re-export 字符串), 示范规约
  实际可用。
- `config/hardcoding_critical_inventory.yaml` 生成 3420 unique critical
  entries 基线 (供 PR review 参考, 不直接喂给 --whitelist)。
- 33 个新测试, 总测试数 581 → 614 (全过, 46.53s)。
- Scanner 升级后分级: 3740 total → critical 3352, info 438
  (98 条被 whitelist 进一步降级)。

**已完成的阶段二(2026-06-25, 本次 commit)**
- `scripts/check_hardcoding.py` 新增 `is_log_message_format` 启发式: logger
  调用的**第一个字符串参数** 自动降为 `info` (不再完全 exclude, 让 audit
  仍能看到 log format 串)。
- `config/hardcoded_whitelist.yaml` 阶段一 90 条 → **211 条** (净增 117):
  * Group 8: reAct / tool_call agent 协议正则 (Thought: / Action: / ...)
  * Group 9: agents/flows/ prompt 模板 (debate / hierarchical / sequential)
  * Group 10: assets/ 协议键名 + SQL 字面量 (PRAGMA / SELECT metadata_json)
  * Group 11: nodes/ 协议/格式 (controlnet / lip_sync / face_embedding)
  * Group 12: pipeline/ 模板协议
  * Group 13: tools/ + plugins/ 协议
  * Group 14: serving/ HTTP 协议 (Content-Type / X-Request-ID)
  * Group 15: infrastructure/ 协议 (TORCHAVERSE_*_DIR / config_snapshot.json)
  * Group 16: examples/ demo 协议
  * Group 17: numeric_literal 通用超参 (14 个文件全项目 numeric → info)
  * Group 18: logger 专用批量 exemption (16 个常见 log message 前缀,
    作 heuristic 的 defence-in-depth fallback)
- `docs/config_access.md` — ConfigCenter / defaults 用户文档, 16 节
  (4 层配置模型 / 90 秒上手 / 读 API / 写 API / 加载顺序 / 环境变量 /
  平台差异 / 快照与重放 / ResourceBudget / `infrastructure.defaults` /
  环境切换 / 完整示例 / 反模式 / 故障排查 / D1 规约关系 / 速查表)。
- 7 个新测试 (TestLogMessageFormat): info/warning/error 各一例 + 后续参数
  仍 critical + keyword arg 不算 format string + helper 直接单测。
- 总测试数 614 → **621** (全过, 46.98s)。
- Scanner 升级后分级: 3740 total → 4157 total (log 启发式让之前 excluded
  的 log 字符串进 inventory, 但 severity=info) → critical **3235**, info **922**。
- Critical inventory 重新导出: 3420 unique → **3235 unique** (净降 185 条
  已批量落 exemption)。

**D1 阶段三已完成的事 (2026-06-25, commit b032082)**
1. 拆 scanner:把 4 条硬编码规则抽到 `scripts/check_hardcoding_rules.py`,
   `Rule` 抽象基类 + `RuleContext` / `ViolationCandidate` 数据类 +
   `DEFAULT_RULES` registry。visitor 退化为"按 `rule.applies_to(node)`
   派发"的薄壳。
2. Per-rule opt-out: `Exemption.rules: Optional[Set[str]]` 字段
   允许目录级批量豁免指定规则子集,`None` 仍匹配所有规则
   (向后兼容,旧的 250 条 exemption 不受影响)。
3. CI gating: `scripts/check_ci_gates.py` 作为统一 CI 入口,
   pyproject.toml 增 `[tool.torcha-verse.hardcoding]`(path /
   whitelist / ci_fail_on / enabled)和
   `[tool.torcha-verse.ci-gates.{hardcoding,placeholders}]` 两段。
   `scripts/ci_config.py` 用 stdlib mini-TOML parser 解析
   (避免引入 `tomli` / `tomllib`,保持纯 stdlib 约束)。
4. 批量豁免: 33 个目录级 batch exemption 把剩余 critical
   3774 → 0。
5. 67 个新集成测试 (`tests/test_hardcoding_rules.py`) 覆盖
   Rule 基类契约 / 4 个内置规则 / Exemption.rules /
   Exemption.is_terminal / 扫描器 `--only-rule` /
   `--list-rules` / ci_config 解析边界 /
   check_ci_gates registry 形态。
6. `docs/placeholder_registry.md` 注册 #63
   (`scripts/check_hardcoding_rules.py:137` `Rule.check`
   抽象方法 `raise NotImplementedError`)。
7. 当前: `python scripts/check_ci_gates.py` 退出码 0,
   747 个非 slow 测试全过。

**D1 阶段三的"再次启动条件"** (任意一条)
1. 新加规则类型 (例如 f-string / regex pattern) → 在
   `scripts/check_hardcoding_rules.py` 继承 `Rule` 并注册到
   `DEFAULT_RULES`;在 `pyproject.toml` markers 增对应测试 marker。
2. 用户报"CI 误判"案例 ≥ 1 个 → 调整对应规则或加 per-rule exemption
   (不再用 protocol_format 那种"全部降级"的方式,而是定向
   `rules: ["<specific_rule>"]`)。
3. 新加 CI gate (例如 type check / lint) → 在
   `scripts/check_ci_gates.py` 的 `GATE_REGISTRY` 注册
   新 runner,在 `pyproject.toml` 加对应 `[tool.torcha-verse.ci-gates.<name>]` 段。

---

## D2 — `pass` / `NotImplementedError` 审计结果(已审 2026-06-25)

**现状**
- 42 处 `pass` / `NotImplementedError`,分布在 18 个文件。

**审计结论(全部合法,无需改代码)**
| 类别 | 数量 | 文件 | 说明 |
|---|---:|---|---|
| 协议/抽象方法(必须) | 5 | `nodes/base.py`, `training/dataset.py` ×2, `papers/adapter.py` ×2 | 子类契约,不可改 |
| device backend 暂未实现 | 2 | `infrastructure/device_manager.py` | 推迟到分布式调度阶段 |
| try/except 兜底 | 28 | `python_executor.py` ×4, `store.py` ×2, `sandbox.py`, `rope.py` ×2, `config_center.py` ×3, `checkpoint_manager.py` ×2, `device_manager.py`, `manager.py`, `consistency/scene.py`, `consistency/score.py` ×3, `nodes/_helpers.py`, `nodes/export.py` ×5, `training/sft_trainer.py` ×2, `document_loader.py` ×2 | 资源清理/降级路径,已用 `noqa: BLE001` 标注 |
| Protocol 占位 | 2 | `infrastructure/resource_budget.py` ×2 | 协议 stub,等到 P2 评估模块时再实现 |

**复审触发条件**
- 任何新的 pass / NotImplementedError 必须经过 PR review,文档化是协议占位还是降级路径。
- 任何现有降级 pass 如果接到用户报错,优先在调用方显式 raise(而不是静默吞错)。

**结论**:不再处理,等下一次大改(release 切换)时再扫一次。

---

## D3 — device_manager TP/PP 占位与 placeholder 集中化

**状态**:阶段一(device_manager `safe_call` 包装) + 阶段二
(`docs/placeholder_registry.md` 集中化 + CI 闸口) **均已完成
2026-06-25**。剩余 D3 阶段三(集中化迁移到 `infrastructure/error_helper`
之外的统一 API 化)待分布式 backend 选定后重启。

**现状**
- `infrastructure/device_manager.py` 中 `raise NotImplementedError` 已通过
  `safe_call` 包装,在单 GPU 环境下调用不抛错。
- D2 审计出的 47 处 `pass` / `NotImplementedError` 已集中到
  `docs/placeholder_registry.md`,按 5 类(protocol / tp_pp / protocol_stub
  / degrade_try_except / degrade_noop)登记。`scripts/check_placeholders.py`
  CI 闸口可拦截未登记的新占位。
- 22 个新测试覆盖 scanner / parser / 差集查询 + 端到端(真实 project
  scan 0 unregistered)。

**为什么延后(阶段三)**
- TP/PP 的实际实现依赖分布式调度框架(目前尚未选定 backend:NCCL / Gloo / Ray),提前实现会带来返工。
- placeholder 集中化已经做完了"集中视图 + 闸口",但"统一降级协议"仍在
  `infrastructure/error_helper.safe_call` 单点。阶段三要把分散的
  `try: ... except Exception: pass` 也迁到 `safe_call` 风格(目前仍是
  best-effort 降级路径,因为很多 try/except 是合理的资源清理)。

**再次启动条件(任一)**
1. 分布式 backend 选定,单卡/多卡策略明确 → 重启 TP/PP 实现。
2. 用户报告的"降级路径中 `pass` 应改显式 raise"案例 ≥ 1 个 → 重启阶段三
   集中降级协议。

**重启时要做的事**
1. 在 `infrastructure/device_manager.py` 中把 `_tensor_parallel_impl` /
   `_pipeline_parallel_impl` 由 `safe_call` 包装升级为真实 TP/PP 实现
   (NCCL / Gloo / Ray backend)。同步从 `docs/placeholder_registry.md`
   删除对应条目 #8 / #9。
2. 对 D2 审计中 32 处 `degrade_try_except` 进行逐个复审:有用户报错的
   改成显式 raise,继续 best-effort 的标记到 `safe_call` 风格并补充测试。
3. 升级 `scripts/check_placeholders.py`,对 `degrade_try_except` 类别
   增加"必须包含 `logger.warning`"的强制约束(目前无此约束)。

**已完成(2026-06-25)**
- 阶段一:`infrastructure/device_manager.py` 已走 `safe_call` 包装,单卡环境不爆。
- 阶段二:`docs/placeholder_registry.md` + `infrastructure/placeholder_registry.py`
  + `scripts/check_placeholders.py` + `tests/test_placeholder_registry.py`
  (22 个测试)全部上线,581 个全量测试 + scanner 双 0 失败。

---

## D4 — v1.0.0 生产化(纲要,待 Q4 启动)

**状态**:⏸ **纲要级,2026 Q4 之前不启动**。M0-M3 拆分 + acceptance
criteria 已落到 `docs/ROADMAP.md` 的 v1.0.0 章节,本条目只在 DEFERRED
侧跟踪"启动条件"和"风险登记",避免 Q4 启动时丢上下文。

**启动条件(任一)**
1. v0.4.x 用户报"多任务并发 OOM" / "缺 metrics 看不清利用率" /
   "租户互相影响" 等任一类生产问题 ≥ 1 个。
2. v0.4.x 真大模型 e2e (P6) 在 CI 跑通后 → 启动 M0 (把
   ResourceBudget 接到真模型加载, 验证 fail-fast vs queue 的真实差距)。
3. 2026 Q4 时间节点到 → 强制启动 M0(即使没用户问题, 也要从生产化
   角度收口)。

**Milestone 摘要**(完整定义见 `docs/ROADMAP.md` v1.0.0 章节):
- **M0** (1 周) — `BudgetTracker` 加 `allocate_or_wait` 排队 + 超时
- **M1** (1-2 周) — `RuntimeScheduler` 抽象 + ThreadPool/ProcessPool/GPU 3 种实现
- **M2** (2-3 周, 3 个子流并行) — Gloo 分布式 + Prometheus metrics + Dockerfile
- **M3** (2 周) — 多租户 + 大评估 leaderboard

**风险**(高优):
- 引入 `deepspeed` / `accelerate` / `torch.distributed.fsdp` 等库会破坏
  纯 torch 约束。CI hardcoding 扫描规则需要加 `deepspeed` /
  `accelerate` 等关键词到禁用列表, 任何 PR 引入会被拦截。
- M0 排队只在 vRAM 数值层, 不 poll 真 GPU 满载。文档必须明确这点。
- 多租户 MVP 只走"per-tenant BudgetTracker + 命名空间目录"隔离, 易被
  进程内逻辑绕过; 真实隔离留 v1.1 接 K8s namespace。

**重启时要做的事**
1. 按 `docs/ROADMAP.md` v1.0.0 章节的 M0 acceptance criteria 启动
   M0 (预计 1 周)。
2. 给 `pyproject.toml` 的 `[tool.coverage.run]` 加 Gloo / Prometheus
   markers (与现在的 hardcoding_rules / hardcoding_severity 一样)
3. 新加 `examples/budget_queueing_demo.py` + `examples/scheduler_demo.py`
   + `examples/leaderboard_demo.py` 三个 demo, 配 smoke test 锁住。

---

## 添加新条目

复制下面这段,改成新条目:

```
## D2 — <标题>

**现状**
- ...

**为什么延后**
- ...

**再次启动条件**
- ...

**重启时要做的事**
- ...
```
