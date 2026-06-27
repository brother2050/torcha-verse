# Open Items

仓库已知未处理 / 部分处理的事项,按优先级倒序。

> **最近更新**: 2026-06-27

---

## A. 已完成 (v0.7.x 收尾)

| ID | 项 | 启动条件 | 完成 |
|---|---|---|---|
| A1 | R-16 性能优化 (`NodeContext` lock 细化 / cache / batch) | 必做 | R-16 ✅ 2026-06-26 |
| A2 | R-17 CLI 增强 (`--config` / JSON log / request-id) | 必做 | R-17 ✅ 2026-06-26 |
| A3 | R-18 `nodes/papers` 懒化 | 必做 | R-18 ✅ 2026-06-26 |
| A4 | R-19 撤 shim + MD 重写 | 必做 | R-19 ✅ 2026-06-26 |

## B. 启动条件待定

| ID | 项 | 启动条件 |
|---|---|---|
| B1 | Gloo 分布式 (TP/PP) | 用户报"多卡多节点"需求 ≥ 1 次 |
| B2 | ONNX / GGUF 导出 | 用户提 issue 反馈 ≥ 3 次 |
| B3 | Agent 多 LLM 后端 | 用户提 issue 反馈 ≥ 3 次 |

## C. v1.0.0 路线 (生产化) — 见 ROADMAP

| ID | 项 | 估时 | 启动条件 |
|---|---|---:|---|
| C1 | `BudgetTracker` 真实调度 (排队 + 超时) | 1 周 | (A 启动后) |
| C2 | `RuntimeScheduler` 抽象 + 3 实现 | 1-2 周 | (A 启动后) |
| C3 | Prometheus metrics | 0.5-1 周 | (C2 之后) |
| C4 | Dockerfile + compose | 0.5-1 周 | (C3 之后) |
| C5 | 多租户 (per-tenant BudgetTracker) | 1 周 | (C1 之后) |
| C6 | 评估 leaderboard | 1 周 | (C2 之后) |
| C7 | 真实大模型 e2e (Qwen / SDXL / HunyuanVideo) | 4-8 周 | 用户提 ≥ 1 次 |

## D. 历史已关闭

| ID | 项 | 关闭于 |
|---|---|---|
| D1 | silent-degrade 清零 (B1) | v0.4.1 (commit cec3e5b) |
| D2 | placeholder 集中化 (B2) | v0.4.3 (commit 7c9cff2) |
| D3 | device_manager TP/PP + safe_call | 维持,延后到 v1.0.0 启动 |

## E. 设计决策记录 (ADR)

### E1: 单系统路线 (单卡多租户 vs 多节点)

**决定**: 走单系统路线,1 节点 N 租户,跨节点 Gloo 分布式延后。

**原因**: v0.4.x 真实 e2e 跑下来,所有 PoC 场景在 1 个 L40S / H100 上
完全 hold 得住,ResourceBudget + RateLimiter 即可切租户预算。跨
节点 (TP/PP) 需 NCCL + 多进程 + 状态同步,投入产出比 (单系统已能
服务 50+ 业务方) 偏低,延后到 v1.0.0 启动。

**影响**:
- C2 (RuntimeScheduler) 删去分布式实现,只做单系统多租户
- `infrastructure.device_manager` 的 TP/PP 接口保留但 `pass`
- 节点 e2e 流程不依赖跨节点

### E2: 撤销 `prometheus_client` swap-in (v0.5.1 补丁)

**决定**: 回到纯 stdlib 路径。

**原因**: swap-in 后引入 4 个新 `pass` 兜底 (TCP socket / file
descriptors),与 v0.4.x D1 关闭的 silent-degrade 战略冲突。
v0.4.3 之前的 `StdoutHandler` 即可对接 ELK / Loki。

**影响**: C3 metrics 阶段会重新评估,优先 stdlib `http.server`
暴露 `/metrics`。

### E3: fp16 matmul 测试在 CPU 缺 kernel 时自动 skip (v0.5.2 补丁)

**决定**: 加 `_has_fp16_matmul()` 探针 + `@requires_fp16_matmul`
skipif 装饰器;不改产品代码。

**原因**: PyTorch 公开 CPU wheel 故意不实现 `addmm_impl_cpu_ for
Half`,产品路径在 GPU 上完整可用,沙盒 CPU 测不到是正常。

**影响**: 测试 `test_fp16_changes_dtype` 在沙箱自动 skip,
`test_bf16_changes_dtype` 不装饰 (CPU bf16 kernel 完整)。

## F. 复审节奏

- **每周一**: 扫 A 段
- **每月初**: 扫 B + C 段,评估是否新增"启动条件"已达成
- **每 release 切换**: 扫 D 段是否真的可关闭
- **ADR 决策** (E 段): 季度评审

---

## 与其它文档的关系

| 文档 | 关注 |
|---|---|
| [ROADMAP.md](ROADMAP.md) | "什么时候做" + 整体进度 |
| [DEFERRED_TASKS.md](DEFERRED_TASKS.md) | "为什么延后" + 重启条件 |
| [placeholder_registry.md](placeholder_registry.md) | "占位在哪儿" + 95 条登记 |
| [operations.md](operations.md) | 日常运维 (部署 / 监控 / checkpoint) |
