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

**现状**
- `infrastructure/device_manager.py` 中存在 `raise NotImplementedError`,在单 GPU 环境下调用会直接抛错,影响开发期体验。
- D2 审计出的 42 处 `pass` / `NotImplementedError` 分散在 18 个文件,缺少集中视图,后续维护成本高。

**为什么延后**
- TP/PP 的实际实现依赖分布式调度框架(目前尚未选定 backend:NCCL / Gloo / Ray),提前实现会带来返工。
- placeholder 集中化需要先有统一的 `infrastructure/error_helper.py` 的 `safe_call` API,以及降级协议约定。

**再次启动条件(任一)**
1. 分布式 backend 选定,单卡/多卡策略明确。
2. `safe_call` API 稳定并文档化,接入 ≥ 1 个真实模块。
3. 准备好投入 ≥ 1 个工作日做"集中化 → 灰度 → 收口"。

**重启时要做的事**
1. 在 `infrastructure/device_manager.py` 中把 `raise NotImplementedError` 替换为 `from .error_helper import safe_call` + 原样返回 `model` + 警告日志,保证单 GPU 环境下调用不爆。
2. 建立 `docs/placeholder_registry.md`,集中列出所有 `pass` / `NotImplementedError`,按"协议占位 / 分布式未实现 / 降级路径"分类。
3. 统一相关 docstring,统一指向 D3,让所有占位都有单一来源的说明。
4. 在 `pyproject.toml` 里把"未在 placeholder_registry 注册的 pass/NotImplementedError"设为 lint 必过项。

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
