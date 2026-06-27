# 硬编码约定 (Hardcoding Convention)

> **最近更新**: 2026-06-27 · 9 个可插拔规则 · 4 级严重性

## 概述

硬编码扫描器 (`scripts/check/hardcoding/`) 静态分析 Python 源,
找出应走配置 / 注入的"魔法值"——字符串字面量、数字字面量、路径、
f-string 模板、Regex pattern、Dict literal、硬编码 switch 语句、
API key 模式。所有命中以 violation 形式报告,可按严重性 / 路径
/ 豁免规则过滤。

## 启动方式

```bash
# 子包路径(无 shim,v0.6.x 起)
python -m scripts.check.hardcoding --path <PATH>

# 退出码: 0 = 无违规 / 1 = 有违规
```

Python API:
```python
from scripts.check.hardcoding import scan_directory, filter_by_severity
violations = scan_directory("/path/to/src")
critical = filter_by_severity(violations, "CRITICAL")
```

## 严重性

| Level | 含义 | 例子 |
|---|---|---|
| `CRITICAL` | 默认 fail CI | 硬编码 API key 模式 (`sk-...`) |
| `WARN` | 默认 warn,可在 PR 中豁免 | 长字符串字面量 (>= 32 字符) |
| `INFO` | 默认 info,常见 | 短字符串字面量 (< 32 字符) |
| `DEBUG` | 仅 IDE 提示 | docstring 中的字符串 |

## 9 个可插拔规则

| Rule | 触发条件 | 严重性 |
|---|---|---|
| `StringLiteralRule` | 任意 Python string AST | `INFO`/`WARN` |
| `NumericLiteralRule` | 任意 int/float literal | `INFO` |
| `PathLiteralRule` | 像路径的字面量 (含 `/` / `\`) | `WARN` |
| `ListLiteralRule` | list 字面量超阈值元素数 | `INFO` |
| `FStringTemplateRule` | f-string 中含变量插值 | `WARN` |
| `RegexPatternRule` | 形如 regex 的字符串 | `INFO` |
| `DictLiteralRule` | dict literal 在业务路径 | `INFO` |
| `HardcodedSwitchRule` | `if x == "literal":` 链 | `WARN` |
| `ApiKeyPatternRule` | API key 模式 (`sk-...` 等) | `CRITICAL` |

每条规则都是一个独立 Python 类,定义在
`scripts/check/hardcoding_rules/` 子包,可通过 YAML 豁免或
命令行 `--disable-rule` 关闭单条。

## 豁免机制

### 文件内 (in-line)

```python
password = "sk-abc123"  # noqa:hardcoding-cryptic
```

支持的 `noqa` tag 列表(按严重性):
- `# noqa:hardcoding-cryptic` — 加密 / API key 类
- `# noqa:hardcoding-string` — 字符串字面量
- `# noqa:hardcoding-path` — 路径类
- `# noqa:hardcoding-numeric` — 数字类
- `# noqa:hardcoding` — 全部

### 配置文件

`hardcoding.yaml`:
```yaml
# 整个目录豁免
- path_regex: "tests/.*"
  reason: "测试夹具允许硬编码"

# 单文件豁免某条规则
- path_regex: "^models/.*\\.py$"
  rules: ["StringLiteralRule"]
  reason: "模型配置来自 YAML"

# 整行豁免某条规则
- pattern: '^\s*LOG_FORMAT = .*'
  rules: ["StringLiteralRule"]
  reason: "log format 是常量"
```

CI: 优先级 **文件内 `noqa` > 项目级 `hardcoding.yaml` > 内置规则**。

## 架构 (v0.6.x 拆分后)

```
scripts/check/
├── ci_gates.py             # 5 段门禁
├── degrade_logging.py      # 降级日志审计
├── placeholders.py         # placeholder 行号扫描
├── hardcoding/             # 扫描器主体
│   ├── __init__.py         # facade
│   ├── _cli.py             # argparse CLI
│   ├── _constants.py       # 严重性 + 默认阈值
│   ├── _ast_helpers.py     # runtime_attr / log_format / etc
│   ├── _visitor.py         # AST visitor
│   ├── _formatters.py      # text / json / 关键摘要
│   ├── _scan.py            # scan_file / scan_directory
│   ├── _whitelist.py       # 加载 + 过滤
│   └── _test_bench.py      # 内置测试集
└── hardcoding_rules/       # 9 个 Rule 类
    ├── __init__.py         # DEFAULT_RULES 列表 + get_rule
    ├── _protocol.py        # Rule protocol + RuleContext
    ├── _constants.py       # STRING_MIN_LENGTH etc
    ├── _string.py          # StringLiteralRule
    ├── _numeric.py         # NumericLiteralRule
    ├── _path.py            # PathLiteralRule
    ├── _list.py            # ListLiteralRule
    ├── _fstring.py         # FStringTemplateRule
    ├── _regex.py           # RegexPatternRule
    ├── _dict.py            # DictLiteralRule
    ├── _switch.py          # HardcodedSwitchRule
    └── _api_key.py         # ApiKeyPatternRule
```

## 与 placeholder_registry 的差异

| 工具 | 关注 |
|---|---|
| `hardcoding` | 业务代码中"应该是配置"的硬编码值 |
| `placeholders` | `pass` / `NotImplementedError` 的静默路径 |

两者扫描不同 AST 形态,缺一不可。

## 常见豁免样例

```python
# 测试 fixture
sample_text = "hello world test fixture"  # noqa:hardcoding-string

# 编译期 regex
_EMAIL_RE = re.compile(r"^[^@]+@[^@]+$")  # noqa:hardcoding-path

# 已知 URL
PUBLIC_CDN = "https://cdn.example.com"  # noqa:hardcoding-string
```

更多示例见 `tests/test_hardcoding_rules.py`。
