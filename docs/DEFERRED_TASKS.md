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
