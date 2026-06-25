# 配置访问指南 (Configuration Access Guide)

> 适用于 TorchaVerse v0.3.x+。本文档面向框架使用者:模型作者 / 节点作者
> / 部署 / 单元测试。如果你想知道 *"我的代码应该从哪里拿配置?"* /
> *"用户配置存在哪?"* / *"怎么重放一次过去的运行?"*,请读这一篇。

**配套文档**:
- `docs/hardcoding_convention.md` — 哪些常量应该放进 config,哪些留在源码。
- `infrastructure/config_center.py` — 完整 API reference (本文件是
  概念 + 例子,源文件是 reference)。

---

## 1. 设计目标

把"配置"提升为一等公民。`ConfigCenter` 是一个 **单例**,负责按递增
优先级合并 4 层配置:

| 层级 | 路径 | 用途 | 谁能改 |
|------|------|------|--------|
| **System** (最低) | `<package>/config/_defaults/*.yaml` | 框架级 immutable baseline | 框架维护者 |
| **Project** | `<project_root>/config/*.yaml` | 项目内提交的配置 | 项目作者 |
| **User** | `~/.config/torcha-verse/*.yaml` (Linux/macOS) / `%APPDATA%/torcha-verse/*.yaml` (Windows) | 用户偏好、API key、本地路径 | 用户 |
| **Run** (最高) | `~/.local/share/torcha-verse/runs/<timestamp>/config_snapshot.json` | 每次运行的快照,用于重放 | 框架自动写 |

**合并规则**: 4 层按 *System < Project < User < Run* 顺序深度合并 —
高优先级层覆盖低优先级层,字典按 key 递归合并,标量直接覆盖。

---

## 2. 90 秒上手

```python
from infrastructure.config_center import ConfigCenter

cc = ConfigCenter()  # 单例: 第一次调用时自动加载 4 层

# 1) 读 — 点号分隔的 key,带 fallback
temperature = cc.get("sampling.default.temperature", 0.7)
dtype       = cc.get("default.dtype", "bf16")
steps       = cc.get("diffusion.default_steps", 30)

# 2) 写 — 动态修改 (测试 / 临时 override)
cc.set("sampling.default.temperature", 0.0)

# 3) 合并 — 一次性塞入一个 dict
cc.merge({"sampling": {"default": {"temperature": 0.3}}})

# 4) 检查存在性
if cc.has("kv_cache.enabled"):
    use_kv = cc.get("kv_cache.enabled")

# 5) ResourceBudget — 资源约束从 config 直接出
budget = cc.resource_budget()
print(budget.vram_gb, budget.max_concurrent_models)
```

> `ConfigCenter` 是 **单例 + 线程安全** (内部 `RLock`)。任何模块任何
> 线程调用 `ConfigCenter()` 拿到的是同一个实例。

---

## 3. 读 API

### 3.1 `cc.get(key, default=None)`

点号分隔的 key,逐层下钻。**键不存在 → 返回 `default`**。

```python
# 命中 4 层合并后的最终值
val = cc.get("sampling.default.top_p")           # → 0.9
val = cc.get("sampling.creative.top_p")          # → 0.95
val = cc.get("does.not.exist", "fallback")       # → "fallback"
```

**返回值特性**:
- 标量 (int / float / str / bool): 直接返回。
- 容器 (dict / list): 返回 **深拷贝**,可放心修改不会污染内部状态。

### 3.2 `cc.has(key)`

`True` 当且仅当 `key` 在合并后的配置中存在(任何层都行):

```python
if cc.has("kv_cache.enabled"):
    ...
```

### 3.3 `cc.to_dict()`

整个配置字典的 **深拷贝**。调试时常用:

```python
import json
print(json.dumps(cc.to_dict(), indent=2, ensure_ascii=False, default=str))
```

### 3.4 `cc.loaded_files`

`List[Path]`,按加载顺序列出 4 层中**实际读到的** YAML 文件(可能是
空 — 如果某层目录不存在)。便于排查"为什么我的配置没生效"。

---

## 4. 写 API

### 4.1 `cc.set(key, value)`

点号分隔的 key,自动创建缺失的中间字典:

```python
cc.set("custom.experiment.tag", "abl-2026-06-25")
cc.set("default.dtype", "fp16")
```

**注意**: `set` 只影响**当前进程内的单例状态**,不会回写 YAML。持久化
请用 `save_run_snapshot()` 或自己写文件。

### 4.2 `cc.merge(*sources)`

把多个 dict 合并到当前配置。**后面的源覆盖前面** (同 `set` 的覆盖
方向,但一次接受多源):

```python
cc.merge(
    {"sampling": {"default": {"temperature": 0.5}}},
    {"sampling": {"default": {"top_p": 0.85}}},
)
```

### 4.3 临时覆盖 — `reset_context()` 上下文管理器

**测试场景**: 改一下配置跑代码,跑完恢复。`reset_context` 会在
`with` 块退出时把配置恢复到进入前:

```python
def test_low_temperature_works():
    with ConfigCenter().reset_context():
        ConfigCenter().set("sampling.default.temperature", 0.0)
        result = generate("hello world")
        assert result.temperature_used == 0.0
    # 退出 with 块: temperature 自动恢复
    assert ConfigCenter().get("sampling.default.temperature") == 0.7
```

### 4.4 `ConfigCenter.reset()` — 销毁单例

> **测试专用**,生产代码不要用。`reset()` 把单例 + `_initialized`
> 标记一起清掉,下次 `ConfigCenter()` 调用会重新走加载流程。

```python
@pytest.fixture(autouse=True)
def reset_config_singleton():
    yield
    ConfigCenter.reset()
```

---

## 5. 四层配置的加载顺序

```python
cc = ConfigCenter(environment="dev", auto_load=True, include_user=True, include_run=True)
```

实际加载顺序(看 `infrastructure/config_center.py:306` `load` 方法):

1. **System**: 读 `<package>/config/_defaults/*.yaml`(immutable baseline)。
2. **Project**: 读 `DEFAULT_CONFIG_FILES` 里的 4 个文件 —
   `model_config.yaml`, `inference_config.yaml`, `training_config.yaml`,
   `prompt_templates.yaml`;然后**如果存在** `config.{environment}.yaml`
   也读(给 `dev` / `prod` 差异用)。
3. **User**(若 `include_user=True`): 读 `user_dir` 下所有 `*.yaml`。
4. **Run**(若 `include_run=True`): 调用 `save_run_snapshot()`,把当前
   合并结果作为快照写盘。

**覆盖关系**:`System < Project < User < Run`。所以:

- Project 加新 key → 不会覆盖 System 同名 key(除非 Project 显式赋值)。
- User 加偏好(例如 `sampling.default.temperature=0.0`)→ 覆盖 Project。
- Run snapshot 的所有 key 都是当时 Project + User 的合并值,
  加载后**最高优先级** — 这就是"运行可重放"。

---

## 6. 环境变量覆盖

不想改文件也能临时换配置:

| 变量 | 作用 | 默认 |
|------|------|------|
| `TORCHAVERSE_CONFIG_DIR` | 覆盖 Project 层目录 | `<package>/config` |
| `TORCHAVERSE_SYSTEM_CONFIG_DIR` | 覆盖 System 层目录 | `<package>/config/_defaults` |
| `TORCHAVERSE_USER_CONFIG_DIR` | 覆盖 User 层目录 | `~/.config/torcha-verse` (Linux/macOS), `%APPDATA%/torcha-verse` (Windows) |
| `TORCHAVERSE_RUN_DIR` | 覆盖 Run snapshot 目录 | `~/.local/share/torcha-verse` (Linux/macOS), `%LOCALAPPDATA%/torcha-verse` (Windows) |

```bash
# CI 环境用临时配置目录
export TORCHAVERSE_CONFIG_DIR=/tmp/torcha-verse-test
pytest tests/
```

---

## 7. 平台差异

| 平台 | User config | Run snapshot / data |
|------|-------------|---------------------|
| Linux | `~/.config/torcha-verse/` | `~/.local/share/torcha-verse/` |
| macOS | `~/.config/torcha-verse/` | `~/.local/share/torcha-verse/` |
| Windows | `%APPDATA%/torcha-verse/` | `%LOCALAPPDATA%/torcha-verse/` |

`APPDATA` / `LOCALAPPDATA` 没设时,fallback 到 `~/AppData/{Roaming,Local}/`。

---

## 8. 快照 (Snapshot) — 运行可重放

### 8.1 `cc.snapshot()`

返回当前配置的 **深拷贝 + JSON 兼容** 字典(所有 `Path` 转 `str`,
tuple 转 list)。**只读、不会影响配置**。

```python
data = cc.snapshot()
json.dump(data, open("/tmp/snap.json", "w"), indent=2)
```

### 8.2 `cc.save_run_snapshot(path=None)`

**自动在每次 `load()` 时调用** (`include_run=True` 是默认)。把
当前配置 + 元数据(`framework` / `version` / `created_at` / `platform` /
`python` / `environment`) 一起写到 `config_snapshot.json`。

- `path=None` → 写到 `~/.local/share/torcha-verse/runs/<timestamp>/config_snapshot.json`。
- `path=/some/where.json` → 写到指定位置。

返回实际写入的 `Path` 对象。

### 8.3 `cc.load_run_snapshot(path)`

**重放** 一次过去运行的配置。snapshot 的内容被合并为 **最高优先级**
的 Run 层, 覆盖当下任何 Project / User 配置。

```python
# 复现 2026-06-25 09:00 那次推理
cc.load_run_snapshot("~/.local/share/torcha-verse/runs/20260625-090000/config_snapshot.json")
# 现在 cc.get("sampling.default.temperature") 一定是当时跑的值
```

### 8.4 快照 envelope schema

```json
{
  "framework": "TorchaVerse",
  "version": "0.3.1",
  "created_at": "2026-06-25 09:00:00",
  "timestamp": 1719283200.0,
  "platform": "Linux-5.15.0-x86_64",
  "python": "3.10.12",
  "environment": "dev",
  "config": {
    "sampling": { "default": { "temperature": 0.7, ... } },
    "diffusion": { ... }
  }
}
```

`config` 字段是合并后的最终值 — `load_run_snapshot` 只读这个字段。

---

## 9. ResourceBudget — 资源约束

```python
budget = cc.resource_budget()
# ResourceBudget(vram_gb=24, ram_gb=64, disk_gb=200,
#                max_concurrent_models=2, max_concurrent_requests=4,
#                kv_cache_gb=0, activations_gb=0, offload_to='none')
```

字段读取顺序(由 `config_center.py:689` 决定):
1. 优先读 `resource_budget.<field>` (Project / User 覆盖)。
2. 否则读 `resource_budget.default_<field>` (System 层 `default_*`
   字段, 用于种子默认值)。
3. 最后 fallback 到代码里 hardcoded 的 default(只是兜底, 不推荐)。

**示例** — 在 `config/inference_config.yaml` 里加:

```yaml
resource_budget:
  vram_gb: 16               # 覆盖 default
  max_concurrent_models: 1   # 覆盖 default
```

下次 `cc.resource_budget()` 就会拿到 `vram_gb=16, max_concurrent_models=1`。

---

## 10. `infrastructure.defaults` — 懒加载默认常量

`infrastructure/defaults.py` 提供模块级常量,内部走 `ConfigCenter` 懒
加载 — **导入 `infrastructure` 不会触发 ConfigCenter 初始化**,只有
第一次访问常量时才会读 config。

```python
from infrastructure.defaults import (
    DIFFUSION_STEPS,            # → cfg.get("diffusion.default_steps", 30)
    DIFFUSION_GUIDANCE_SCALE,   # → cfg.get("diffusion.default_guidance_scale", 7.5)
    DIFFUSION_SCHEDULER,        # → cfg.get("diffusion.scheduler", "dpm_solver")
    DIFFUSION_ETA,              # → cfg.get("diffusion.eta", 0.0)
    SAMPLING_TEMPERATURE,       # → cfg.get("sampling.default.temperature", 0.7)
    SAMPLING_TOP_K,             # → cfg.get("sampling.default.top_k", 50)
    SAMPLING_TOP_P,             # → cfg.get("sampling.default.top_p", 0.9)
    SAMPLING_REPETITION_PENALTY,# → cfg.get("sampling.default.repetition_penalty", 1.1)
)
```

**所有模块必须从这里导入推理默认值**,**不要**自己 hardcode 数字。
D1 规约的"运行时配置"类常量,集中在这里。详见
`docs/hardcoding_convention.md` 第 1.1 节。

**fallback 值的意义**: 只有当 `config/inference_config.yaml` **缺失**
(例如最小测试环境)时, `get` 的第二个参数才生效。YAML 存在时,fallback
被忽略。

---

## 11. 环境切换

```python
cc = ConfigCenter()                            # 默 dev
cc.switch_environment("prod")                  # 重新加载,读 config.prod.yaml
```

`SUPPORTED_ENVIRONMENTS = ("dev", "prod")`。其他值抛 `ValueError`。

---

## 12. 完整示例: 训练 / 推理 / 部署

### 12.1 模型作者 — 在 `models/my_model.py` 里读推理超参

```python
from infrastructure.defaults import DIFFUSION_STEPS, SAMPLING_TEMPERATURE

class MyModel:
    def __init__(self, steps: int | None = None, temperature: float | None = None):
        self.steps = steps or DIFFUSION_STEPS
        self.temperature = temperature or SAMPLING_TEMPERATURE

    def generate(self, prompt: str) -> str:
        return f"generated@T={self.temperature},steps={self.steps}: {prompt}"
```

### 12.2 节点作者 — 在 `nodes/image.py` 里读 batch / kv_cache 配置

```python
from infrastructure.config_center import ConfigCenter

cc = ConfigCenter()
BATCH = cc.get("batch.max_batch_size", 1)
KV = cc.get("kv_cache", {"enabled": False})

class ImageNode:
    def run(self, prompts):
        for i in range(0, len(prompts), BATCH):
            batch = prompts[i:i+BATCH]
            yield self._process(batch, use_kv=KV.get("enabled", False))
```

### 12.3 部署 — 通过环境变量切换配置目录

```bash
# staging 环境
export TORCHAVERSE_CONFIG_DIR=/etc/torcha-verse/staging
export TORCHAVERSE_USER_CONFIG_DIR=/var/lib/torcha-verse/user
python -m serving.cli --host 0.0.0.0 --port 8000
```

### 12.4 测试 — 临时改配置 + 自动恢复

```python
def test_with_low_temperature():
    cc = ConfigCenter()
    original = cc.get("sampling.default.temperature")
    with cc.reset_context():
        cc.set("sampling.default.temperature", 0.0)
        result = generate("hello")
        assert result.deterministic
    # 上下文退出,自动恢复
    assert cc.get("sampling.default.temperature") == original
```

### 12.5 复现一次历史推理

```python
from pathlib import Path
from infrastructure.config_center import ConfigCenter

# 找到目标 run 的 snapshot
snap_dir = Path("~/.local/share/torcha-verse/runs/20260625-090000")
cc = ConfigCenter()                                # 加载默认 4 层
cc.load_run_snapshot(snap_dir / "config_snapshot.json")  # 加上 Run 层
# 现在的配置和当时完全一致
result = generate("replay this prompt")
```

---

## 13. 反模式 (Anti-patterns)

按 D1 规约,这些写法**不要**出现在新代码里:

| 反模式 | 为什么错 | 正确做法 |
|--------|----------|----------|
| `DIFFUSION_STEPS = 30` 在模块顶部 hardcoded | 改动需要发版 | `from infrastructure.defaults import DIFFUSION_STEPS` |
| `cfg["sampling.default.temperature"]` 直接 dict 访问 | 不走 API 入口,`reset_context` 不可见 | `cc.get("sampling.default.temperature", 0.7)` |
| `os.environ["MY_KEY"]` 直接读 | 绕过 ConfigCenter 的快照重放 | 放进 yaml, 用 `cc.get("my.key")` |
| `cc.set("default.dtype", "fp16")` 持久化为源码常量 | `set` 只在进程内生效 | 改 `config/inference_config.yaml` 并 commit |
| 在 hot path 里反复 `cc.get(...)` | 每一次都走 RLock + split | 局部变量缓存一次 |
| 写新 key 不用命名空间 | 污染 `default.*` 等保留段 | 用业务前缀,例如 `my_app.cache_size` |

---

## 14. 故障排查

### 14.1 "我改了 YAML, 没生效"

```python
print(ConfigCenter().loaded_files)         # 看实际加载的路径
print(ConfigCenter().to_dict()["sampling"]) # 看实际合并后的值
```

确认你的 YAML 路径在 `loaded_files` 列表里,以及 `env` 没把
`TORCHAVERSE_CONFIG_DIR` 指到别处。

### 14.2 "我的 User 覆盖没生效"

```python
print(ConfigCenter().user_dir)             # 应是 ~/.config/torcha-verse
print(list(Path(ConfigCenter().user_dir).glob("*.yaml")))
```

User 层的目录不存在时**静默跳过**;要么创建目录,要么通过
`TORCHAVERSE_USER_CONFIG_DIR` 指到别处。

### 14.3 "Run snapshot 写到哪里了?"

```python
print(ConfigCenter().run_snapshot_path)
```

或直接 `ls ~/.local/share/torcha-verse/runs/`。

### 14.4 "我想清掉所有运行时修改, 重新加载"

```python
ConfigCenter.reset()                       # 销毁单例
cc = ConfigCenter()                        # 重新加载
```

仅限测试 / 启动期使用。运行时频繁 reset 会丢快照。

---

## 15. 与 D1 规约的关系

`ConfigCenter` 是 D1 规约里"运行时配置"(RUNTIME_CONFIG 类)的**权威
存放点**。任何被规约识别为 `critical` 的源码常量,如果有用户可调
价值,都应当:

1. 挪到 `config/*.yaml` 对应 section。
2. 在代码里改成 `cc.get("...")` 或 `infrastructure.defaults.*`。
3. 把 hardcode 删掉。

而 *协议 / 格式标识*(PROTOCOL_FORMAT)和 *模型结构超参*
(MODEL_STRUCTURAL)不需要挪,前者进 `hardcoded_whitelist.yaml` 加
`protocol_format: true`, 后者由 `is_structural_init` 启发式自动
识别为 `info`。详见 `docs/hardcoding_convention.md`。

---

## 16. 速查表

| 操作 | API |
|------|-----|
| 读单值 | `cc.get("a.b.c", default)` |
| 批量读 | `cc.to_dict()` |
| 检查存在 | `cc.has("a.b.c")` |
| 改单值(临时) | `cc.set("a.b.c", value)` |
| 合并 dict | `cc.merge({"a": {"b": 1}})` |
| 临时改 + 自动恢复 | `with cc.reset_context(): cc.set(...)` |
| 重置单例 | `ConfigCenter.reset()` (测试用) |
| 读 inference 缺省 | `from infrastructure.defaults import DIFFUSION_STEPS` |
| 拿 ResourceBudget | `cc.resource_budget()` |
| 切换 dev/prod | `cc.switch_environment("prod")` |
| 序列化为 JSON | `cc.snapshot()` |
| 写快照 | `cc.save_run_snapshot()` / `cc.save_run_snapshot("/path/x.json")` |
| 重放历史 | `cc.load_run_snapshot("path/config_snapshot.json")` |
| 看加载了哪些文件 | `cc.loaded_files` |
| 覆盖 Project 目录 | env `TORCHAVERSE_CONFIG_DIR` |
| 覆盖 User 目录 | env `TORCHAVERSE_USER_CONFIG_DIR` |
