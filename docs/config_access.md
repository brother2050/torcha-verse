# 配置中心使用指南

> **最近更新**: 2026-06-27

## 概述

`infrastructure/config_center/ConfigCenter` 单例,四级合并 + 点号
访问。`infrastructure/defaults.py` 是**推理默认值的唯一数据源**,所有
模块必须从这里 import,首次访问时从 `ConfigCenter` 懒加载。

## 四级合并 (优先级递增)

| 层级 | 路径 | 用途 |
|---|---|---|
| System | `config/_defaults/*.yaml` (随包发布) | 项目级默认值 |
| Project | `./config/*.yaml` (CWD) | 部署/项目覆盖 |
| User | `~/.config/torcha-verse/*.yaml` | 用户偏好 |
| Run | `config_snapshot.json` (运行快照) | 运行时覆盖 + 可重放 |

## 公共 API

```python
from infrastructure.config_center import ConfigCenter
cc = ConfigCenter()

# 获取 (点号访问)
cc.get("sampling.default.temperature")    # 0.7
cc.get("diffusion.default_steps")         # 30

# 设置 (运行时)
cc.set("default.dtype", "fp16")
cc.set("user.history", [1, 2, 3])

# 保存 / 加载快照
cc.save_snapshot("/path/to/snapshot.json")
cc.load_snapshot("/path/to/snapshot.json")
```

## defaults.py — 唯一数据源

```python
# infrastructure/defaults.py
from infrastructure.config_center import ConfigCenter

def _get(name, default):
    cc = ConfigCenter()
    return cc.get(name, default)

# 推理默认值
DIFFUSION_STEPS = _get("diffusion.default_steps", 30)
DIFFUSION_GUIDANCE_SCALE = _get("diffusion.default_guidance_scale", 7.5)
SAMPLING_TEMPERATURE = _get("sampling.default.temperature", 0.7)
SAMPLING_TOP_K = _get("sampling.default.top_k", 50)
SAMPLING_TOP_P = _get("sampling.default.top_p", 0.9)
SAMPLING_REPETITION_PENALTY = _get("sampling.default.repetition_penalty", 1.05)
```

`ConfigCenter` 单例 + 延迟导入机制打破循环依赖:
- `defaults.py` 首次 import 时创建/获取 `ConfigCenter` 单例
- `ConfigCenter` 不 import `defaults.py`
- 任何业务模块都可以 `from infrastructure.defaults import ...` 拿到全局一致的值

## 配置中心目录解析

`ConfigCenter` 按以下顺序解析 `config/` 目录 (R-19 已加 CWD 兜底):

1. 显式 `config_dir` 参数
2. CWD 下 `config/`(验证含 sentinel `model_config.yaml` / `inference_config.yaml`)
3. `sys.argv[0]` 所在目录的 `config/`(同上验证)
4. 沿 `__file__` 向上找含 sentinel 的 `config/`

因此 **`torcha` 命令必须从项目根或装了包的目录运行**(CWD 兜底后从
`/tmp` / `~` 任意目录都跑通)。

## 加载流程

```python
ConfigCenter._load()
  ├─ 1. System:  load_yaml("config/_defaults/*.yaml")
  ├─ 2. Project: load_yaml("config/*.yaml")     [CWD 兜底]
  ├─ 3. User:    load_yaml("~/.config/torcha-verse/*.yaml")
  └─ 4. deep merge (4 → 3 → 2 → 1)
```

合并规则: 字典深合并 (D1),列表**覆盖**(D2),标量**覆盖**(D3)。
`__file__` 路径解析与 sentinel 验证见 `ConfigCenter._resolve_root`。

## 点号访问 (dot access)

```yaml
# config/inference_config.yaml
diffusion:
  default_steps: 30
  default_guidance_scale: 7.5

sampling:
  default:
    temperature: 0.7
    top_k: 50
    top_p: 0.9
```

访问:
```python
cc.get("diffusion.default_steps")           # 30
cc.get("sampling.default.temperature")      # 0.7
cc.get("sampling.default.top_k")            # 50
```

## Run 快照 (snapshot)

每次启动可写一份:
```bash
python -m tools.config_snapshot --output /tmp/snap.json
```

重放:
```python
cc = ConfigCenter()
cc.load_snapshot("/tmp/snap.json")
```

## 常见反模式

```python
# ❌ 不要在自己的模块里写常量
DIFFUSION_STEPS = 30  # 重复定义,改一个忘一个

# ✅ 从 defaults 拿
from infrastructure.defaults import DIFFUSION_STEPS
```

```python
# ❌ 不要绕开 ConfigCenter
cc.set("sampling.default.temperature", 1.2)  # OK 写是 OK 的
# ✅ 但默认值改动要走 defaults.py + inference_config.yaml
```
