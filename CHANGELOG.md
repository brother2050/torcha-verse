# Changelog

> 项目变更记录。架构简洁、节点能跑、测试可过。

## [Unreleased]

### F-* — 真实实现填充 (F-0 ~ F-14)

把 39 个 capability 节点中仍为「骨架 + 简单处理」的 24 个节点的 `execute()`
替换为真实模型 + 真实算法实现,测试数 **1086 → 1118 (+32)**。

**F-1 — 6 个数字人节点 + 11 PaperAdapter**:

- 新文件 `papers/adapters/digital_human.py` (11 个 class):
  MuseTalk / VideoReTalking (lip-sync), SadTalker / EchoMimic (talking-head),
  EchoMimicV2 (full-body), LivePortrait (portrait-anim),
  GFPGAN / CodeFormer (face-enhance), CosyVoice / F5TTS / ChatTTS (voice-clone)
  共享 `_AudioFeatureEncoder` / `_FaceLandmarkNet` / `_DMMRegressor` /
  `_UNetRestoration` / `_SpeakerEncoder` 5 个真 nn.Module
- `papers/__init__.py` 增 11 条 `_ADAPTER_NAME_TO_MODULE` lazy entry
- `nodes/_helpers/_backends.py` 增 7 个新 `call_*_backend` helper
  (lipsync / talking_head / portrait_anim / full_body / face_enhance /
  digital_human / tts)
- `nodes/digital_human.py` 6 个节点 `execute` 改用真 helper
- F-0: 修 `dh_full_body` 的 `image` / `reference_image` 别名 bug

**F-2 ~ F-5 — 4 个字幕节点全链路真实化**:

- 新 `nodes/_subtitle_codec.py` (480 行):
  - `Cue` / `SubtitleTrack` dataclass
  - `read_audio_waveform` (stdlib wave + scipy fallback)
  - `asr_transcribe` (能量法 25 ms / 10 ms hop + 自适应阈值,
    无 whisper 依赖)
  - `batch_translate_cues` (window=8 滑窗 LLM + 字符长度比
    自适应 end-timestamp)
  - `serialize_srt` / `serialize_vtt` / `serialize_ass` 真序列化
  - `burn_subtitles` (cv2 VideoCapture/Writer + PIL ImageDraw CJK)
- `nodes/subtitle.py` 4 个节点 `execute` 全部接入 codec

**F-6 — depth_condition → SceneEngine._DepthEstimator**:

- 新 `call_depth_backend` helper (bus 优先 + SceneEngine fallback)
- `nodes/consistency.py::DepthConditionNode.execute` 改用真深度估计

**F-7 — character_five_view → ScoreCalculator.clip_i_distance**:

- 新 `call_consistency_score_backend` helper
- 5 view 每张跑 CLIP-I → 返回均值 `consistency_score ∈ [0, 1]`

**F-8 — video_interpolate → FrameInterpolator**:

- 新 `call_frame_interpolation_backend` helper 真插帧
- cv2 → tensor → FrameInterpolator.pair forward → tensor
  返回 `target_fps / source_fps - 1` 个中间帧

**F-9 — video_txt2vid → MotionModule**:

- 新 `call_motion_module_backend` helper 真 motion injection
- `[B, C, T, H, W]` 张量 → MotionModule → 含 motion 调制结果

**F-10 — image_txt2img/img2img → DiffusionScheduler**:

- 新 `call_diffusion_scheduler_backend` helper 真调度器
- `DiffusionScheduler(sampler_name=...).set_timesteps()` → 真实
  timestep 列表透传到下游

**F-11 — image_upscale/inpaint → 真 SR/Inpaint UNet**:

- 新 `models/image/restoration.py`:
  `SuperResolutionUNet` (PixelShuffle 头部) + `InpaintUNet`
  (RGB + mask → RGB) + `to_image_tensor` 强转
- 新 `call_super_resolution_backend` / `call_inpaint_backend` helper
- 2 个节点 `execute` 优先用真 UNet,失败时回退 image backend

**F-12 — audio_music → MusicDiT + HiFiGAN**:

- 新 `models/audio/music.py`:
  `MusicTransformer` (4-layer Transformer) + `MusicDiT`
  (AdaLN-Zero style 步调制)
- 新 `call_music_backend` helper:MusicDiT 生成 mel →
  HiFiGAN vocoder → 真 waveform
- `audio_music` 节点挂上真后端

**F-13 — video_stitch → ffmpeg xfade / torch cross-fade**:

- 新 `call_video_stitch_backend` helper
- 全路径 → `ffmpeg -filter_complex xfade=...` 优先
- 张量输入 → torch linear cross-fade fallback

**F-14 — 测试与文档**:

- 新 `tests/test_real_implementations.py` (32 个 test):
  F-1 ~ F-13 每个新代码路径都有 ≥ 1 个 test 覆盖
- 修正 `docs/placeholder_registry.md` 9 条新 `pass` 行(总条目 35 → 44)
- `models/image/restoration.py` UNet 上采样 bug 修复
  (dec2 / up3 / up2 三段 reshape 正确)

## v0.8.5 — HunyuanDiT-Tiny 接入 + 端到端 Latent 验证

第 3-4 周目标(参 `docs/V0.8_UPGRADE_PLAN.md` §4)。本节新增 **29 个
test**,总测试数 **1128 → 1157 (+29)**,第一次达到 ≥ 1150 目标。

**模型侧**:

- 新 `models/image.dit.HunyuanDiT` (tiny preset,96-dim / 2-block / GQA):
  - 完整的 adaLN-Zero 调制 + 联合 QKV self-attention + 交叉 attention
  - 上采样 patch 化 → 12 个 patch(8x8 → 2x2) → unpatch
  - `time_embed.{0,2}` / `pooled_embed.proj` / `style_embed` / `size_embed` /
    `rope_freqs` 完整暴露
  - **local layout** 参数命名与
    `core.checkpoint_loader.HUNYUAN_DIT_KEY_MAP` target values 一一对应,
    `load_hunyuan_dit()` 可直接重写 Tencent 上游 checkpoint
- `HunyuanDiTConfig.tiny()` 工厂方法 + dict config 自动 coerce

**端到端 Latent 验证 (LatentValidator)**:

- 新 `nodes._helpers._latent` 子模块(纯 Python,无额外依赖):
  - `LatentStats` dataclass(JSON 可序列化)+ `LatentValidationError`
  - `LatentValidator` 9 项检查:shape / dtype / finite / NaN 计数 / Inf 计数
    / std band (default `0.05 <= std <= 10.0`) / `abs_max` band / `allow_nan` /
    `allow_inf`
  - `quick_validate` / `validate_range` / `validate_shape` 一行封装
- 集成到 `call_diffusion_loop_backend`:
  - 新增 `latent_validator` / `validate_latent` kwarg
  - 响应新增 `latent_valid` (bool) / `latent_stats` (dict) /
    `latent_validation` (完整 report)三个键
  - `model=None` 与循环异常路径也返回结构化 `latent_valid=False` 报告

**修复与改进**:

- 修 `core.checkpoint_loader`:
  `load_state_dict_with_renames` 现在同时处理参数与 persistent buffer
  (`rope_freqs` 等)
- 修 `core.schedulers.schedules`:
  补 `LinearSchedule = NormalSchedule` alias(原 import 路径在
  `__init__` 中引用但类未定义,导致每个 e2e diffusion loop 静默回落到
  `placeholder` 分支)
- `models/base.py` 加载器增强:buffer 与 parameter 走同一条 copy_ 路径,
  dtype / device 自动匹配

**测试**:

- 新 `tests/test_v085_hunyuan_dit.py` (29 个 test, 4 个 Section):
  1. **HunyuanDiTConfig**:tiny preset / 默认构造 / 显式 config / dict config
  2. **Forward + Sample**:shape / dtype / text context / CFG 开关
  3. **save_pretrained / from_pretrained round-trip**:
     - local layout round-trip
     - config sidecar 写入
     - subfolder 加载
     - variant=fp16 加载
     - `load_hunyuan_dit` local helper
     - `load_hunyuan_dit` upstream-style checkpoint(模拟 Tencent 上游 ckpt)
  4. **LatentValidator** (12 个):default pass / zero / flat / NaN / Inf /
     shape mismatch / strict 抛错 / strict 返回 / Stats dataclass /
     allow_nan override / validate_range helper
  5. **E2E Latent 验证** (4 个):HunyuanDiT-Tiny 走 `call_diffusion_loop_backend` 完整
     流程,验证 `latent_valid=True` / `latent_stats` 合理 / 缺失 model 报错 /
     `validate_latent=False` 禁用 / 自定义 validator
- 4 个 pre-existing `rich` ModuleNotFoundError 与 3 个 pre-existing
  warnings test 不计入本节新增,均与本节无关

## [v0.6.0] - 2026-06-26

### R-* 重构 (R-3 ~ R-15 + R-19) — 65+ 聚焦子模块

13 个 PR 把 9,419 行的单文件拆成 65+ 子模块,**0 回归**:

| PR | 范围 | 行数 | 子文件数 |
|---|---|---:|---:|
| R-3 | `checkpoint_manager` 拆 | 631 | 7 |
| R-4 | `agent` 拆 | 692 | 4 |
| R-5 | `training/dataset` 拆 | 1063 | 6 |
| R-6 | `huggingface` 拆 | 983 | 9 |
| R-7 | `models/source/fetch` 拆 | 837 | 6 |
| R-8 | `serving/app` 拆 | 1208 | 8 |
| R-9 | `serving/cli` 拆 | 962 | 9 |
| R-10 | `serving/service` 拆 + PipelineService | 848 | 8 |
| R-11 | `scripts/check_*` 合并 | 2634 | 11 (子包) |
| R-12 | `assets/store` 拆 + 协议分层 | 861 | 8 |
| R-13 | `assets/model_asset` 拆 | 473 | 6 |
| R-14 | `nodes/base` 拆 | 850 | 7 |
| R-15 | `nodes/_helpers` 拆 | 710 | 5 |
| R-19 | 撤 shim + MD 重写 | - | - |

## [v0.6.1] - 2026-06-26

### R-18 — papers 懒化

`import papers` 不再触发 1000+ 行 torch 模块加载:
- **Cold import**: 1288 ms → **47 ms (-96%)**
- **import 后 `sys.modules` 中 0 个 `papers.adapters.*` 模块**

变更:
- `papers/__init__.py` — 移除 `from .adapters import ...` 的 eager 加载;
  - 加 `_ADAPTER_NAME_TO_MODULE` 字典 + `_loaded_adapters` 模块级 cache
  - `AdapterRegistry.get` / `has` 加 lazy fallback (monkey-patch): bundled name 未注册时,先 import 模块再 register
  - `__getattr__` (PEP 562) 懒导出 `StableDiffusion3Adapter` / `HunyuanDiTAdapter` / `cli` (用 module-level cache,不写 `globals()`,支持测试 purge)
  - `__dir__` 暴露懒导出给 IDE
  - `TYPE_CHECKING` 守卫 import 给静态分析器
- `papers/adapters/__init__.py` — 改为 lazy: `PaperAdapter` 仍 eager (无 torch),`StableDiffusion3Adapter` / `HunyuanDiTAdapter` 改 `__getattr__` 懒加载子模块
- `tests/test_r18_lazy.py` — 11 个新测试: import 不加载 torch / `has` 不触发 import / `get` 触发 / cache hit / 未知名字抛 `AdapterNotFoundError` / PEP 562 懒导出 / 子包懒 / `PaperAdapter` 仍 eager

### R-16 — 性能优化

优化 `NodeContext` / `NodeRegistry` hot path,微基准:
- `get_output` (100k): **85 → 66 ms (-23%)** — 拆 fast path 无锁
- `resolve_executor` cache hit (100k): **255 → 22 ms (-91%)** — LRU 1024
- `resolve_executor` mixed 5 types (100k): **255 → 66 ms (-74%)**
- `registry.list` (1k): **39 → 0.4 ms (-99%)** — 缓存到 register/unregister
- `set_output` (100k): **87 → 84 ms** — `RLock` 改 `Lock`

变更:
- `nodes/base/_context.py` — `RLock` 改双 `Lock` (`_outputs_lock` / `_executors_lock`),
  保留 `_lock` 兼容 alias,`get_output` / `has_output` 走 GIL 保护无锁 fast path,
  `resolve_executor` 加 FIFO 1024 LRU,`register_executor` 失效缓存,
  负结果也缓存 (unregistered type 不再每次打 bus)
- `nodes/base/_registry.py` — `list()` 缓存到 `register` / `unregister`
- `tests/test_r16_perf.py` — 11 个新测试 (并发不破 + 缓存 + 失效 + 负缓存 + 驱逐)

### R-19 — 撤 shim + MD 重写

**撤 shim** (5 个):
- `scripts/check_hardcoding.py` / `check_hardcoding_rules.py` / `check_placeholders.py` / `check_degrade_logging.py` / `check_ci_gates.py` 全删
- 测试 3 处 `from scripts.check_*` → `from scripts.check.<subpkg>` 子包路径
- `_cli.py` 加 `if __name__ == "__main__": sys.exit(main())` 入口
- `hardcoding/__init__.py` 把 `main` 改成 PEP 562 `__getattr__` 懒导出,避免循环

**重写 MD** (13 个):
- `README.md` / `docs/architecture.md` / `docs/operations.md` / `docs/ROADMAP.md` / `docs/DEFERRED_TASKS.md` / `docs/open_items.md` / `docs/examples_catalog.md` / `docs/hardcoding_convention.md` / `docs/placeholder_registry.md` / `docs/config_access.md` / `docs/docker.md` / `examples/README.md` / `CHANGELOG.md`

**Bug 顺手修** (3 个,非重构):
- CWD config 兜底: `_paths.py` 加 sentinel files 验证
- `torcha models` rich 渲染: `_info.py` 改用 dict 字段
- `image_txt2img` PIL 转换: `_image.py` 加 `_to_pil()` 支持 4 种类型

## [v0.5.2] - 2026-06-25

### D-补丁: 1 个 fp16 matmul 测试在 CPU 缺 kernel 时自动 skip

`tests/test_performance_quantization.py` 里 `test_fp16_changes_dtype`
在沙箱 CPU 环境下 fail (`addmm_impl_cpu_` not implemented for 'Half') —
PyTorch 公开 CPU wheel 故意不实现 fp16 matmul kernel。

- 加 `_has_fp16_matmul()` 探针 + `@requires_fp16_matmul` skipif
- `bf16_changes_dtype` 不装饰 (CPU bf16 mkl+oneDNN wheel 跑得通)

## [v0.5.1] - 2026-06-25

### D-补丁: 撤掉 prometheus_client swap-in (回退 v0.4.3 之前的纯 stdlib 路径)

swap-in 后引入 4 个新 `pass` 兜底 (TCP socket / file descriptors),
与 v0.4.x D1 关闭的 silent-degrade 战略冲突。`StdoutHandler` 即可
对接 ELK / Loki。C3 metrics 阶段会重新评估。

## [v0.5.0] - 2026-06-25

### Added (v0.5.x 功能扩展)

- **L2 资产子系统** — `Asset` / `AssetRef` / `AssetStore` + 5 子类
  (Model / Character / Outfit / Scene / Depth) + 三级存储 (hot/warm/cold)
- **L6 一致性** — `CharacterEngine` / `OutfitEngine` / `SceneEngine`
  + `ScoreCalculator`
- **L5 Pipeline 模板** — `TemplateRegistry` (12 个开箱即用模板)
- **L5 Prompt Studio** — 提示词工程与版本管理
- **L5 Canvas** — 可视化画布 + `AutoDirector` 自动编排
- **安全** — `OutputFilter` (毒性 / NSFW)
- **v0.5 feature demo** — `examples/v05_feature_demo.py`

测试: 986 → 1053 (+67)。

## [v0.4.3] - 2026-06-25

### C1b-C7b 加深

详见 git log `b032082`:
- C1b `BudgetTracker` 排队 + 超时 stub
- C2b `RuntimeScheduler` 抽象 + stub
- C3b 监控端点 stub
- C4b Dockerfile + compose stub
- C5b 多租户 stub
- C6b leaderboard stub
- C7b Docker / k8s 配置

测试: 747 → 830。

## [v0.4.2] - 2026-06-25

### C1-C7 v1.0 骨架

`de35b14`。详见 ROADMAP 历史段。

## [v0.4.1] - 2026-06-25

### B1 silent-degrade 清零

`cec3e5b`。所有 `pass` 改为 `logging.WARN` + 登记到
`docs/placeholder_registry.md`,CI 扫描器强制登记。

## [v0.4.0] - 2026-06-25

### P0 真模型 + P0 多模态 + P1 评估 + P2 模型源

详见 ROADMAP 历史段。测试: 369 → 559。

## [v0.3.0] - 2026-06-24

### 架构骨架 (L1-L5)

6 层分层、ModuleBus、Pipeline DAG、NodeSpec、ConfigCenter、
ResourceBudget、AuditLogger、Security 中游件。
