# Placeholder Registry

仓库内**所有** `pass` 与 `NotImplementedError` 的单一来源(Single Source of Truth)。

> **最近更新**: 2026-06-26 · 合计 **35 个**已注册占位 (协议/抽象 + try/except 兜底 + if-branch noop + 其它)
>
> 完整清单按文件行号分布于各子包:`models/source/` 4 个、`nodes/` 5 个、
> `assets/` 7 个、`infrastructure/` 5 个、`serving/` 4 个、`papers/` 1 个、
> `consistency/` 4 个、`plugins/` 1 个、`tools/` 3 个、`security/` 1 个。

**CI 守卫**: `python scripts/check/hardcoding/_cli.py --path src/ --format json`
之外的 placeholder 由 `scripts/check/placeholders.py` 扫描。
新增 `pass` / `NotImplementedError` 必须在此登记,否则 CI fail。

---

## 1. 分类规约

| 类别 | 说明 | 处理策略 |
|---|---|---|
| `protocol` | 基类抽象方法 (子类必须实现) | 不改,文档化由所在模块负责 |
| `tp_pp` | TP/PP 并行原语占位 | 走 `safe_call`,单卡不抛;分布式 backend 选定后重启 |
| `degrade_try_except` | try/except 兜底 | 资源清理/外部依赖失败的 best-effort 路径 |
| `degrade_noop` | if-branch 无操作 | 显式说明"此处无操作"意图 |

---

## 2. 注册表(共 95 条)

### 2.1 协议/抽象方法(`protocol`,12 条)

| # | 文件:行 | 上下文 | 状态 |
|---:|---|---|---| |
| 1 | `models/source/huggingface.py:176` | `HttpTransport.get_json` | 3 个 transport 子类已实现 | |
| 2 | `models/source/huggingface.py:182` | `HttpTransport.get_bytes` | 同上 | |
| 3 | `training/dataset.py:147` | `BaseTorchDataset.__getitem__` | 5 个 Dataset 子类已实现 | |
| 4 | `training/dataset.py:235` | `BaseTorchDataset._load` | 同上 | |
| 5 | `papers/adapter.py:88` | `PaperAdapter.load_model` | `@abc.abstractmethod` | |
| 6 | `papers/adapter.py:101` | `PaperAdapter.infer` | 同上 | |
| 7 | `nodes/base/_node.py:90` | `BaseNode.execute` | 39 节点均已实现 | |
| 57 | `models/source/huggingface.py:173` | `HttpTransport.get_json` (v0.6 重构) | 协议方法 | |
| 58 | `models/source/huggingface.py:179` | `HttpTransport.get_bytes` | 同上 | |
| 63 | `scripts/check/hardcoding_rules/_protocol.py:131` | `Rule.check` | 9 个 Rule 子类已实现 | |
| 70 | `training/dataset/_base.py:135` | `BaseDataset.__getitem__` | 5 个子类已实现 | |
| 71 | `training/dataset/_base.py:227` | `BaseDataset._load` | 同上 | |
| 73 | `models/source/huggingface/_transport.py:58` | `HttpTransport.get_json` | 3 个 transport 子类已实现 | |
| 74 | `models/source/huggingface/_transport.py:77` | `HttpTransport.get_bytes` | 同上 | |

### 2.2 TP/PP placeholder(`tp_pp`,2 条)

走 `safe_call` 包装后单卡环境不抛;重启条件见 `DEFERRED_TASKS D3`。

| # | 文件:行 | 函数 |
|---:|---|---|
| 8 | `infrastructure/device_manager.py:42` | `_tensor_parallel_impl` |
| 9 | `infrastructure/device_manager.py:58` | `_pipeline_parallel_impl` |

### 2.3 try/except 兜底(`degrade_try_except`,35 条)

**资源清理** (#10-22):

| # | 文件:行 | 上下文 |
|---:|---|---|
| 10 | `models/providers/tiny_transformer.py:445` | save 失败后清残留 `.tmp` |
| 11-13 | `models/source/cache.py:401/470/474` | atomic write / rmdir 兜底 |
| 14 | `plugins/manager.py:922` | 持久化失败清 tmp |
| 15-16 | `infrastructure/{config_center,device_manager}.py:817/608` | reset 路径容忍锁已释放 |
| 17-19 | `tools/python_executor.py:351/390/393` | sandbox / rlimit 兜底 |
| 20 | `security/sandbox.py:666` | restore rlimit 失败不抛 |
| 21-22 | `assets/store.py:714/716` | staging 清理 (v0.4.x 旧行号) |
| 54-56 | `models/source/cache.py:551/620/624` | atomic write / rmdir (v0.6 重构后) |

**bus 解析降级** (#23):

| # | 文件:行 | 上下文 |
|---:|---|---|
| 23 | `nodes/_helpers/_backends.py:263` | bus 缺失退回默认 factory |

**外部可选依赖** (#24-32, 50-53, 92-94):

| # | 文件:行 | 上下文 |
|---:|---|---|
| 24 | `training/sft_trainer.py:670` | LoRA merge 可选 |
| 25-26 | `rag/loaders/document_loader.py:195/203` | PyPDF2 → pdfplumber 降级 |
| 27-28 | `infrastructure/checkpoint_manager.py:581/600` | numpy optional |
| 29-32 | `consistency/{scene,score}.py:134/172/316/335` | PIL / open_clip / DINOv2 可选 |
| 50-53 | `nodes/_helpers/_backends.py:115/124/133/142` | local_*_factory 失败 → echo fallback |
| 92 | `assets/store/_cold.py:45` | `_log_warning` `except Exception: pass` | logger.warning() 失败兜底,不该因 logger 抛错让冷层路由挂掉 | |
| 93 | `assets/store/_cold.py:54` | `_log_error` `except Exception: pass` | logger.error() 失败兜底,同上 | |
| 94 | `assets/store/_cold.py:152` | `evict_to_cold` `except OSError: pass` | 删空 shard 目录失败兜底,沿用 v0.4.x assets/store.py:316 行为 | |

**类型转换降级** (#33-34):

| # | 文件:行 | 上下文 |
|---:|---|---|
| 33-34 | `infrastructure/config_center.py:710/719` | `_get_float` / `_get_int` 字段类型容错 |

**其它** (#35-44, 59-62, 64-69, 72, 95):

| # | 文件:行 | 上下文 |
|---:|---|---|
| 35 | `tools/python_executor.py:167` | sandbox 子进程预清理 globals |
| 36 | `training/sft_trainer.py:743` | scheduler 不可用退回 optimizer lr |
| 37-41 | `nodes/export.py:565/612/614/644/652` | encode 失败 → STUB bytes |
| 42-44 | `serving/app.py:263/398/916` | filter block 不阻塞流 |
| 59-62 | `models/source/huggingface.py:625/668/705/727` | progress 回调 robustness |
| 64 | `assets/cold_storage.py:195` | local single-part 无需分块 |
| 65 | `assets/store.py:317` | cold mirror 失败不影响 warm write |
| 66 | `training/dataset.py:1044` | parquet 无 numpy 字段跳过 |
| 67 | `papers/adapters/stable_diffusion_3.py:235` | 文档化占位 |
| 68 | `infrastructure/config_center/_schema.py:245` | schema re-seed `pass` | ConfigCenter 构造时回填默认值,失败不抛 | |
| 69 | `infrastructure/config_center/_center.py:172` | config re-seed `except Exception: pass` | 重新初始化默认值失败,允许后续尝试 | |
| 72 | `training/dataset/_readers.py:91` | pyarrow → pandas 切换 |
| 95 | `serving/cli/_image.py:87` | PIL ImageDraw 不可用时回退纯色 |
| 101 | `nodes/_helpers/_backends.py:759` | call_frame_interpolation_backend bus 路径失败后回退 FrameInterpolator |
| 102 | `nodes/_helpers/_backends.py:909` | call_motion_module_backend bus 路径失败后回退 MotionModule |
| 103 | `nodes/_helpers/_backends.py:1091` | call_music_backend bus 路径失败后回退 MusicDiT+HiFiGAN |
| 104 | `nodes/_helpers/_backends.py:1186` | call_video_stitch_backend bus 路径失败后回退 ffmpeg |
| 105 | `nodes/_helpers/_backends.py:1206` | call_video_stitch_backend ffmpeg 失败后回退 torch |
| 106 | `nodes/_helpers/_backends.py:1375` | call_diffusion_scheduler_backend bus 路径失败后回退 DiffusionScheduler |
| 107 | `nodes/_helpers/_backends.py:1456` | call_depth_backend bus 路径失败后回退 SceneEngine |
| 108 | `nodes/_helpers/_backends.py:1476` | call_depth_backend SceneEngine 路径失败后回退元数据字典 |
| 109 | `nodes/_helpers/_backends.py:1536` | call_consistency_score_backend bus 路径失败后回退 ScoreCalculator |
| 110 | `nodes/_helpers/_backends.py:1578` | call_diffusion_loop_backend bus 路径失败后回退 torch 循环 |
| 111 | `nodes/_helpers/_backends.py:1598` | call_diffusion_loop_backend torch 循环失败后回退元数据字典 |
| 112 | `nodes/_helpers/_backends.py:1658` | call_diffusion_loop_backend fall-through 标记 (双 except) |
| 113 | `nodes/image.py:343` | image_txt2img 真实循环失败后回退 F-10 路径 |

### 2.4 if-branch noop(`degrade_noop`,5 条)

| # | 文件:行 | 上下文 |
|---:|---|---|
| 45 | `models/components/rope.py:140` | RoPE linear scaling 在 forward 处理 |
| 46 | `models/components/rope.py:148` | RoPE dynamic NTK 在 forward 重算 |
| 47 | `scripts/check/hardcoding/_visitor.py:146` | bool / None 不视为 numeric violation |
| 114 | `models/runtime/device_planner.py:113` | pick_default_device CUDA 探测异常 → 继续尝试 MPS |
| 115 | `models/runtime/device_planner.py:118` | pick_default_device MPS 探测异常 → 继续回退 CPU |

### 2.5 其它兜底(36 条,含 v0.6 重构新行号)

| # | 文件:行 | 上下文 |
|---:|---|---|
| 48-49 | `nodes/_helpers/_backends.py:388/392` | call_image_backend TypeError retry |
| 75-78 | `assets/store/_warm.py:51/56/66/72` | atomic copy / hash / rmdir |
| 79-83 | `assets/store/_hot.py:39/44/55/77/85` | HotCache cleanup |
| 84-90 | `assets/store/_cold.py:36/45/54/64/89/124/145/152` | logger / rmdir / fetch |
| 91 | `scripts/check_hardcoding.py:57` | (shim 已撤,历史条目) |
| 96-100+ | (新增,待 v0.6.1 扫) | — |

---

## 3. 维护规则

1. **新增占位** — 提交前必须登记一条,包含:文件:行、上下文、分类、理由。
2. **删除占位** — 实现后从本文件移除,并在 `CHANGELOG` 说明。
3. **CI 校验** — `scripts/check/placeholders.py` 扫 `pass` + `NotImplementedError`,
   与本表求差集;差集非空则 fail。协议/抽象方法用行内标记豁免
   (`# placeholder:protocol`)。
4. **复审节奏** — 每月扫一次 D2 / D3 涉及的 30+ 降级路径,用户报错时
   改显式 raise(参考 `DEFERRED_TASKS D2` 触发条件)。

---

## 4. 与 `DEFERRED_TASKS.md` 的关系

| DEFERRED 条目 | 关联占位类别 | 何时复审 |
|---|---|---|
| D2 (pass/NotImplementedError 审计) | 全部 95 条 | 每次 release 切换 |
| D3 (device_manager TP/PP + safe_call) | `tp_pp` (8, 9) + 部分 `degrade_try_except` (15, 16) | 分布式 backend 选定后 |

`DEFERRED_TASKS.md` 关注"为什么延后 / 何时重启",本文件关注
"占位在哪儿 / 为什么这样写",两者互补不冲突。
