# Placeholder Registry

> 集中视图:TorchaVerse 仓库内**所有** `pass` 与 `NotImplementedError` 出现位置、分类、说明。
>
> **本文档是占位条目的单一来源**(single source of truth)。任何新增的 `pass` /
> `NotImplementedError` 必须同步登记到此处,否则会被
> `scripts/check_placeholders.py` 在 CI 中拦截。
>
> 最近一次更新:2026-06-25 · 合计:**47 处**(7 协议/抽象 + 2 TP/PP placeholder + 35 try/except 兜底 + 3 if-branch noop)

---

## 1. 分类规约

| 类别 | 简称 | 说明 | 处理策略 |
|---|---|---|---|
| `protocol` | 协议/抽象方法 | 基类定义的抽象方法,子类必须实现 | 不改,文档化由所在模块负责 |
| `tp_pp` | TP/PP placeholder | `infrastructure/device_manager` 中尚未实现的并行原语 | 已走 `safe_call` 包装,单卡环境不抛;重启见 `DEFERRED_TASKS D3` |
| `protocol_stub` | Protocol stub | Protocol 类的方法占位(无 body) | 视调用方而定,目前 D2 审计下为空 |
| `degrade_try_except` | try/except 兜底 | 资源清理 / 降级路径中的 `pass` | 资源/外部依赖失败时的 best-effort 兜底,文档化理由 |
| `degrade_noop` | if-branch noop | 条件分支中的 `pass`(无操作占位) | 显式说明"此处无操作"意图 |

---

## 2. 注册表(共 43 条)

### 2.1 协议/抽象方法(`protocol`,7 条)

子类必须实现,基类抛 `NotImplementedError` 显式表达契约。

| # | 文件:行 | 类/函数 | 方法 | 说明 |
|---:|---|---|---|---|
| 1 | `models/source/huggingface.py:114` | `HttpTransport` | `get_json` | HTTP transport 抽象;`UrllibTransport` 实现 |
| 2 | `models/source/huggingface.py:120` | `HttpTransport` | `get_bytes` | HTTP transport 抽象;`UrllibTransport` 实现 |
| 3 | `training/dataset.py:147` | `BaseTorchDataset` | `__getitem__` | 子类按数据格式覆盖 |
| 4 | `training/dataset.py:235` | `BaseTorchDataset` | `_load` | 子类按文件格式覆盖 |
| 5 | `papers/adapter.py:88` | `PaperAdapter` | `load_model` | `@abc.abstractmethod` 装饰 |
| 6 | `papers/adapter.py:101` | `PaperAdapter` | `infer` | `@abc.abstractmethod` 装饰 |
| 7 | `nodes/base.py:401` | `BaseNode` | `execute` | `@abc.abstractmethod` 装饰;30 节点均已实现 |

### 2.2 TP/PP placeholder(`tp_pp`,2 条)

走 `safe_call` 包装后单卡环境不抛错;具体重启条件见 `DEFERRED_TASKS D3`。

| # | 文件:行 | 函数 | 说明 |
|---:|---|---|---|
| 8 | `infrastructure/device_manager.py:42` | `_tensor_parallel_impl` | 张量并行未实现,`DeviceManager.tensor_parallel` 走 `safe_call` fallback |
| 9 | `infrastructure/device_manager.py:58` | `_pipeline_parallel_impl` | 流水线并行未实现,`DeviceManager.pipeline_parallel` 走 `safe_call` fallback |

### 2.3 Protocol stub(`protocol_stub`,0 条)

D2 审计中提到 `infrastructure/resource_budget.py` 有 2 处,经核验实际是
doctest 中的 `>>> ... pass` 文本片段,**不算**可执行占位,故本类别为空。
`ColdStorageProtocol`(`assets/store.py:75`)、`CheckpointBackend`
(`infrastructure/checkpoint_manager.py:55`)有方法签名但无 `pass` /
`NotImplementedError`,靠子类化约束。

### 2.4 try/except 兜底(`degrade_try_except`,35 条)

资源清理或外部依赖失败时的 best-effort 兜底,均为"失败时静默跳过的合理性路径"。

#### 资源清理(原子写 / 临时文件 / 锁)

| # | 文件:行 | 上下文 | 理由 |
|---:|---|---|---|
| 10 | `models/providers/tiny_transformer.py:445` | `save_tiny_transformer` 失败后 `os.unlink(tmp_name)` | `os.replace` 失败时清理残留 `.tmp` |
| 11 | `models/source/cache.py:401` | `ModelCache.atomic_write_*` 失败后清理 `.tmp` | 同上,原子写兜底 |
| 12 | `models/source/cache.py:470` | `ModelCache` 递归 `rmdir` | 个别空目录非空时容错 |
| 13 | `models/source/cache.py:474` | `ModelCache` 删根 `target` 目录 | 兜底 |
| 14 | `plugins/manager.py:922` | `PluginManager` 持久化失败清理 tmp | best-effort |
| 15 | `infrastructure/config_center.py:817` | `ConfigCenter.reset` 中 `_lock.release()` | 重置单例时容忍锁已释放 |
| 16 | `infrastructure/device_manager.py:608` | `DeviceManager.reset` 中 `cleanup_ddp()` | 测试期重置容忍 DDP 清理失败 |
| 17 | `tools/python_executor.py:351` | `PythonExecutorTool._run` 删 tmp 脚本 | `finally` 块 |
| 18 | `tools/python_executor.py:390` | `_set_resource_limits` `setrlimit(RLIMIT_AS, ...)` | 平台不支持时降级 |
| 19 | `tools/python_executor.py:393` | `_set_resource_limits` 外层 | best-effort 总兜底 |
| 20 | `security/sandbox.py:666` | `SandboxExecutor._restore_resource_limits` | 恢复旧 rlimit 失败不抛 |
| 21 | `assets/store.py:714` | `AssetStore._cleanup_staging` 删 staging | 文件不存在时忽略 |
| 22 | `assets/store.py:716` | `AssetStore._cleanup_staging` | 其他 OS 错误也忽略 |

#### bus / 模块解析降级

| # | 文件:行 | 上下文 | 理由 |
|---:|---|---|---|
| 23 | `nodes/_helpers.py:305` | `_resolve_via_bus_or_default` `bus.resolve(...)` | bus 缺失/未注册时退回默认 factory |

#### 外部可选依赖(降级到 stub)

| # | 文件:行 | 上下文 | 理由 |
|---:|---|---|---|
| 24 | `training/sft_trainer.py:670` | `save_checkpoint` 调 LoRA merge | LoRA 是 optional dep,缺失则跳过 |
| 25 | `rag/loaders/document_loader.py:195` | `PDFLoader.load` import PyPDF2 | PyPDF2 不可用则降级到 pdfplumber |
| 26 | `rag/loaders/document_loader.py:203` | `PDFLoader.load` import pdfplumber | 都不行则最后 `raise ImportError` |
| 27 | `infrastructure/checkpoint_manager.py:581` | `_capture_rng_states` import numpy | numpy optional,缺失则跳过 |
| 28 | `infrastructure/checkpoint_manager.py:600` | `_restore_rng_states` np.random.set_state | 同上 |
| 29 | `consistency/scene.py:134` | `_to_tensor` import PIL | 缺失则降级到 numpy / tensor 路径 |
| 30 | `consistency/score.py:172` | `_to_tensor` import PIL | 同上 |
| 31 | `consistency/score.py:316` | `_try_load_real_extractors` open_clip | 加载失败静默,返回 result 不含 `clip` |
| 32 | `consistency/score.py:335` | `_try_load_real_extractors` DINOv2 | 同上 |

#### 类型转换降级(ConfigCenter 字段类型容错)

| # | 文件:行 | 上下文 | 理由 |
|---:|---|---|---|
| 33 | `infrastructure/config_center.py:710` | `_get_float` 类型转换 | 字段类型不匹配则用 default |
| 34 | `infrastructure/config_center.py:719` | `_get_int` 类型转换 | 同上 |

#### 子进程沙箱代码生成

| # | 文件:行 | 上下文 | 理由 |
|---:|---|---|---|
| 35 | `tools/python_executor.py:167` | `_make_safe_import` 生成的子进程源码 `del globals()[_name] except KeyError: pass` | 子进程沙箱预清理,删除内部名字防被用户代码重新导入 |

#### 学习率调度器降级

| # | 文件:行 | 上下文 | 理由 |
|---:|---|---|---|
| 36 | `training/sft_trainer.py:743` | `SFTTrainer._get_lr` 取 `lr_scheduler.get_last_lr()` | scheduler 不可用时退回 optimizer 默认 lr |

#### 节点 export 编码器降级

| # | 文件:行 | 上下文 | 理由 |
|---:|---|---|---|
| 37 | `nodes/export.py:565` | `_encode_image` 真实编码失败 | 退回 32 字节 STUB 占位 |
| 38 | `nodes/export.py:612` | `_encode_video` writer 异常 | 退回 STUB |
| 39 | `nodes/export.py:614` | `_encode_video` 外层总兜底 | OpenCV/imageio 不可用 |
| 40 | `nodes/export.py:644` | `_encode_audio` scipy.wavfile.write | 退回手工 RIFF header |
| 41 | `nodes/export.py:652` | `_encode_audio` 外层总兜底 | 最终 STUB bytes |
| 42 | `serving/app.py:263` | filter block `pass` | filter errors should not block the stream |
| 43 | `serving/app.py:398` | filter block `pass` | 同上 |
| 44 | `serving/app.py:916` | filter block `pass` | 同上 |

### 2.5 if-branch noop(`degrade_noop`,3 条)

| # | 文件:行 | 上下文 | 理由 |
|---:|---|---|---|
| 45 | `models/components/rope.py:140` | `RotaryPositionEmbedding.__init__` `if scaling_type == "linear": pass` | 线性缩放在 forward 时处理 cos/sin,init 无操作 |
| 46 | `models/components/rope.py:148` | `RotaryPositionEmbedding.__init__` `elif scaling_type == "dynamic": pass` | 动态 NTK 在 forward 时按 seq_len 重算频率 |
| 47 | `scripts/check_hardcoding.py:526` | `pass  # booleans / None are never numeric violations` | `if-elif` 分支:bool / None 永不视为 numeric violation |

---

## 3. 维护规则

1. **新增占位** — 在提交前必须在本文件登记一条,包含:文件:行、上下文、分类(`protocol` / `tp_pp` / `protocol_stub` / `degrade_try_except` / `degrade_noop`)、理由。
2. **删除占位** — 实现后从本文件移除对应条目,并在 `CHANGELOG` 中说明。
3. **CI 校验** — `scripts/check_placeholders.py` 扫描所有源文件中的 `pass` 与
   `NotImplementedError`,与本表求差集;差集非空则 fail(协议/抽象方法类别
   用 `# placeholder:protocol` 等行内标记豁免)。
4. **复审节奏** — 每月扫一次 D2 / D3 涉及的 28 + 2 = 30 处降级路径,
   若有用户报错的调用方应改显式 raise(参考 `DEFERRED_TASKS D2` 复审触发条件)。

---

## 4. 与 `DEFERRED_TASKS.md` 的关系

| DEFERRED 条目 | 关联的占位类别 | 何时复审 |
|---|---|---|
| D2 (pass/NotImplementedError 审计) | 全部 43 条 | 每次 release 切换 |
| D3 (device_manager TP/PP + safe_call) | `tp_pp` (8, 9) + 部分 `degrade_try_except` (15, 16) | 分布式 backend 选定后 |

`docs/DEFERRED_TASKS.md` 关注"为什么延后 / 何时重启",本文件关注
"占位在哪儿 / 为什么这样写",两者互补不冲突。
