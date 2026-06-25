# Hardcoding Convention

> TorchaVerse 工程规约:源码中的常量如何分类、哪些走配置、哪些保留在源码。
>
> **本文档是 D1 (Hardcoding 规约化) 的根规约**。所有 PR 引入新的"看起来像
> 配置"的常量时,必须先读本文档确认分类。任何 scanner (见
> `scripts/check_hardcoding.py`) 的豁免决策都基于本文档。
>
> 最近一次更新:2026-06-25

---

## 1. 三类常量边界

源码里的字符串、数字、路径、列表字面量都归为以下三类之一。

### 1.1 运行时配置 (RUNTIME_CONFIG) — `severity: critical`

**定义**:会影响**生产推理行为**的、用户/运维**应当可调**的常量。
**典型例子**:
- 采样参数 `temperature=0.7`, `top_p=0.9`, `repetition_penalty=1.1`
- 扩散参数 `num_inference_steps=30`, `guidance_scale=7.5`
- 业务阈值 `chunk_size=512`, `max_concurrent_requests=8`
- 路径前缀 `cache_dir="~/.cache/torcha-verse"`, `output_dir="/data/output"`

**必须**走 `infrastructure/config_center.ConfigCenter` (读自
`config/inference_config.yaml`) 或 `infrastructure/defaults.py` (范式参考)。

**scanner 行为**:`severity=critical` 命中,CI 必过。
**不允许**在函数体里直接写字面量(除了 `_defaults/` YAML 文件本身)。

### 1.2 模型结构超参 (MODEL_STRUCTURAL) — `severity: info`

**定义**:**改变会破坏模型**、**用户不应自行调节**、**仅作者层维护**的常量。
**典型例子**:
- Transformer 结构:`d_model=768`, `num_layers=12`, `num_heads=12`
- 注意力窗口:`max_position_embeddings=2048`, `rope_theta=10000.0`
- 离散化层:`vocab_size=50257`, `pad_token_id=0`
- 卷积结构:`kernel_size=3`, `stride=2`, `groups=8`

**保留**在源码里,作为模型定义的一部分。**不**走 ConfigCenter。

**scanner 行为**:`severity=info`,不计入 CI 失败指标,但生成报告供审计。
**豁免**:通过 `config/hardcoded_whitelist.yaml` 的 `structural_hyperparam: true`
行级豁免;或文件级豁免(`file: "models/audio/audio_codec.py"` + `severity: info`)。

### 1.3 协议/格式标识 (PROTOCOL_FORMAT) — `severity: info`

**定义**:与**外部协议/格式**绑定的字面量,改了不工作。
**典型例子**:
- reAct prompt 的 `'Thought:\\s*(.*?)...'` 正则
- 文件魔数 `'\\x89PNG'`, MIME `'image/png'`
- 日志 tag `'[TEXT]', '[IMAGE]'`
- HTTP 头 `'Content-Type: application/json'`
- 错误信息 `'Unknown strategy: %s'`

**保留**在源码里,绑定协议契约。**不**走 ConfigCenter。

**scanner 行为**:`severity=info`,同上。
**豁免**:文件级 `whitelist` 加 `protocol_format: true` 标记。

---

## 2. severity 等级

| 等级 | 含义 | CI 影响 | 适用类别 |
|---|---|---|---|
| `critical` | 必须修,生产风险 | 计入 `--severity critical` 失败 | 1.1 运行时配置 |
| `warn` | 应当修,代码质量 | 默认报告但不 fail | 边界 case(目前未使用) |
| `info` | 可豁免,留作审计 | 仅报告 | 1.2 / 1.3 模型结构 / 协议 |

**CI 调用约定**:
```bash
# 默认 — 输出全报告, exit code = (critical > 0)
python scripts/check_hardcoding.py --path .

# 只关心 critical — 用于 PR review 必过项
python scripts/check_hardcoding.py --path . --severity critical

# 审计 — 列所有 info
python scripts/check_hardcoding.py --path . --severity info
```

---

## 3. 识别规则 (scanner 内置启发式)

scanner 在打 severity 标签前,会先按以下顺序判断是否**已经是合规的**:

| 启发式 | 说明 | 影响类别 |
|---|---|---|
| `_HARDCODE_EXEMPT_MODULES` | 模块名出现在内置豁免集 | 全部 |
| `imports_defaults` | `from infrastructure.defaults import X` 后续 `X` 引用 | string |
| `imports_config_center` | `from infrastructure.config_center import get_config, ConfigCenter` 后续 `cfg.get(...)` 引用 | string |
| `attribute_access` | `os.environ[...]`, `Path(...).expanduser()`, `sys.argv[...]` 表达式 | string / path |
| `log_method_call` | 调用 `.info()`/`.warning()` 等 logger 方法的 string 实参 | string |
| `is_structural_init` | 数值出现在 `__init__` 且**明显是结构超参**(值 ≥ 2 且 ≤ 10000 且位于 `models/` 路径) | numeric |

**豁免优先级**:文件级 whitelist > 行级 whitelist > 启发式规则 > 默认 severity。

---

## 4. whitelist YAML 扩展

`config/hardcoded_whitelist.yaml` 现在支持 4 个新字段:

```yaml
exemptions:
  # 1) 老语法 — 行为不变
  - file: "core/model_registry.py"
    type: "string_literal"

  # 2) 行级 + severity 降级
  - file: "models/audio/audio_codec.py"
    line: 77
    severity: "info"     # 不论原 severity 是什么,降为 info

  # 3) 文件级 + 类别(批量落)
  - file: "models/audio/audio_codec.py"
    type: "numeric_literal"
    reason: "audio codec structure"

  # 4) 协议/格式豁免(显式标记)
  - file: "agents/react_agent.py"
    protocol_format: true
    type: "string_literal"
    reason: "reAct prompt template — protocol-defined"
```

字段含义:
- `severity`:把命中降级到指定级别(`info` / `warn` / `critical`)。
- `protocol_format`:显式声明"这是协议绑定",scanner 标记为 `info`。
- `reason`:人类可读理由,出现在报告中。

---

## 5. 维护规则

1. **新增结构超参**(如新模型层)→ 不需改 whitelist,scanner 自动识别 `is_structural_init`。
2. **新增运行时配置** → 必须改 `config/inference_config.yaml` + `infrastructure/defaults.py`。
3. **新增协议标识** → 改 whitelist 加 `protocol_format: true`,不需改代码。
4. **CI 必过项**:`--severity critical` 失败 = PR 必拒。
5. **月度复审**:`--severity info` 报告扫一遍,新增 ≥ 50 条时考虑规约修订。

---

## 6. 与 D1 重启条件的关系

D1 再次启动条件之一是"`infrastructure/config_center.py` 与
`infrastructure/defaults.py` 文档化"。本文档是规约侧文档;ConfigCenter /
defaults API 文档作为衍生任务 **D1.4**,可独立排期,见
`docs/DEFERRED_TASKS.md` D1 章节末。

---

## 7. 首批 critical 名单

见 `config/hardcoding_critical.yaml`(由 `python scripts/check_hardcoding.py
--severity critical --export config/hardcoding_critical.yaml` 生成,首批
数据基于 2026-06-25 扫描)。critical 命中数应当**逐渐减少**,新代码引入
critical 必须同时提供迁移到 ConfigCenter 的方案。
