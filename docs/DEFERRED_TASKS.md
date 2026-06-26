# DEFERRED_TASKS

延后到 v1.0.0 之后或满足特定启动条件才会重启的任务清单。

> **最近更新**: 2026-06-26

延后≠删除。每条都标明:**为什么延后 / 何时重启 / 关联占位**。

## 任务表

| ID | 任务 | 分类 | 重启条件 |
|---|---|---|---|
| D1 | `pass` / `NotImplementedError` 静默降级审计 | 质量 | 每 release 切换 |
| D2 | device_manager TP/PP 真实实现 | 性能 | 跨节点需求 ≥ 1 |
| D3 | Agent 多 LLM 后端 (Anthropic / Google) | 业务 | 用户 issue ≥ 3 |
| D4 | 模型 ONNX / GGUF 导出 | 工程 | 用户 issue ≥ 3 |
| D5 | 评估 leaderboard 站点 | 业务 | C6 启动 |

> **D1 / D2** 在 v0.4.x 阶段被视为"基础质量门"已重新关闭,
> 重启后看 v1.0.0 需求。

---

## D1 — `pass` / `NotImplementedError` 静默降级审计 (v0.4.1 已完成关闭)

**原因**: v0.4.0 静默降级路径过多,用户报"看不出错在哪儿"。

**v0.4.1 关闭方法**:
1. 100% 静默降级改为 logging (LEVEL=WARN),保留 best-effort 语义
2. 所有占位登记到 [`placeholder_registry.md`](placeholder_registry.md)
3. CI 扫描器 `scripts/check/placeholders.py` 强制登记
4. 用户触发时: warn 日志 + audit log + 不阻塞流

**重启条件**: 任何 release 切换前,跑 `scripts/check/placeholders.py`
扫一次,差集非空则 fail。

## D2 — device_manager TP/PP 真实实现

**原因**: v0.4.x 路线明确走"单系统多租户",跨节点分布式不紧急。

**接口已留**:
```python
# infrastructure/device_manager.py:42
def _tensor_parallel_impl(...): pass
# infrastructure/device_manager.py:58
def _pipeline_parallel_impl(...): pass
```

走 `safe_call` 包装后单卡环境不抛;分布式 backend 选定后重启。

**重启条件**:
- 用户报"多卡多节点" ≥ 1 次
- 或 v1.0.0 启动后,P95 延迟 > 1s 持续 1 周 → 启动 M0/M2b
- 关联占位 #8 / #9 / 部分 #15 / #16

## D3 — Agent 多 LLM 后端 (Anthropic / Google)

**原因**: ReAct agent 已在 v0.4.x 跑通,但只接项目自有 tiny
Transformer + echo factory。Anthropic / Google 后端的 LLMProvider
实现留在 v0.5.x 路线外。

**重启条件**:
- 用户 issue ≥ 3 (例如 "ReAct agent 不能接 Claude 3.5")
- 估算: 1-2 周 (2 个 LLMProvider 子类 + 测试 + 文档)

## D4 — 模型 ONNX / GGUF 导出

**原因**: 当前 5 个 `export_*` 节点用 STUB bytes 兜底
(占位 #37-41);真实 onnxruntime / gguf 导出未实现。

**重启条件**:
- 用户 issue ≥ 3 (例如 "我想要把 Pipeline 导出成 ONNX 部署到 edge")
- 估算: 1-2 周 (3 个 exporter + 6 个测试 + 文档)

## D5 — 评估 leaderboard 站点

**原因**: v0.4.x ScoreCalculator 已能跑出量化分数,leaderboard 站点
未建,缺运营 + UI 投入。

**重启条件**:
- C6 (评估 leaderboard) 启动,见 [ROADMAP.md](ROADMAP.md)
- 估算: 1 周 (FastAPI 站点 + SQLite + 5 个测试)

---

## 复审节奏

| 频率 | 动作 |
|---|---|
| 每周一 | 扫本表,看 A 段是否新增"启动条件"达成 |
| 每月初 | 评估 D1 重新激活;扫 D 段所有条目 |
| 每 release 切换 | 跑 placeholder 扫描器 (D1 重启) |
| 季度 | 评估 E 段 (设计决策) 是否需调整 |

## 关联文档

| 文档 | 关注 |
|---|---|
| [placeholder_registry.md](placeholder_registry.md) | 占位在哪儿 |
| [open_items.md](open_items.md) | 启动条件 + 决策记录 |
| [ROADMAP.md](ROADMAP.md) | 整体路线图 |
