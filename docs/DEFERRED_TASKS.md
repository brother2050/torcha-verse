# Deferred Tasks

开发初期(early-stage)阶段**不处理**、等架构稳下来后再统一规约化的任务。
每个条目都说明了:是什么、为什么延后、再次启动时怎么判断。

---

## D1 — Hardcoding 规约化与扫描器校准

**现状**
- `scripts/check_hardcoding.py` 扫出 **3209** 条违规(2693 string + 398 numeric + 64 path + 54 list)。
- 分布在 `models/`、`core/`、`infrastructure/`、`training/`、`nodes/` 等多个子包。

**为什么延后**
- 误报率高:模型结构超参(`d_model=768`、`num_layers=12`、LoRA `r=16` 等)不是"用户可调配置",不应被规约。
- 收益低:单纯把 `r=16` 提成 `cfg.lora.r` 不带来功能价值,反而扩大配置面。
- 需要先有边界:没有"什么算配置、什么算结构超参"的统一规约,扫描器无法稳定运行。

**再次启动条件(任一)**
1. L4 节点层稳定至少 1 个 release(无破坏性接口变更)。
2. `infrastructure/config_center.py` 与 `infrastructure/defaults.py` 文档化,用户可见。
3. 准备好投入 ≥ 2 个工作日做"分类 → 校准 → 灰度"。

**重启时要做的事**
1. 写规约文档 `docs/hardcoding_convention.md`,明确定义:
   - 哪些常量属于"运行时配置"(提进 YAML / ConfigCenter)
   - 哪些属于"模型结构超参"(保留在源码)
   - 哪些属于"协议/格式标识"(白名单)
2. 按规约重写 `config/hardcoded_whitelist.yaml`,分批落 exemption。
3. 拆 scanner:把 `string_literal` / `numeric_literal` 拆成两个独立规则,允许结构超参 opt-out。
4. 在 `pyproject.toml` 里把"critical 违规"设为 CI 必过项,non-critical 仍 warn。

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
