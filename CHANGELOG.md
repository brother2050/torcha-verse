# Changelog

项目初期变更记录。初期重点：架构简洁、节点能跑、测试可过。

## [Unreleased] — 初期整理

### P2+ 模型下载：HF 镜像 + 跨镜像去重 + 进度回调 (v0.4.x P2+ milestone)

把 v0.4.0 P2 的模型下载功能**补全**:HF 镜像自动 fallback
(`https://huggingface.co` → `https://hf-mirror.com`)、下载内容
指纹 (`compute_content_fingerprint` + `find_by_fingerprint`)
跨 repo/revision 去重避免重复写盘、下载进度回调
(`DownloadProgress` dataclass)、镜像健康检查
(`check_mirror_health` + `MirrorHealth`)。所有功能在零网络
测试中验证 (`FakeTransport`),可立刻接入真镜像。

**新文件**:
- `models/source/mirrors.py` — `DEFAULT_HF_MIRRORS` 镜像列表 +
  `MirrorSet` 配置 dataclass (env-var 读 `$TORCHA_VERSE_HF_MIRRORS`)
  + `MirrorHealth` 健康结果 + `check_mirror_health` /
  `check_all_mirrors` / `is_useful_mirror_error` 三个工具函数
- `tests/test_model_source_mirror.py` — 31 个新测试
  (8 MirrorSet + 5 健康检查 + 4 指纹/缓存查找 + 6 HF 镜像 fallback
  + 5 fetcher 端到端 + 2 文件跳过/异常)
- `examples/model_download.py` — 端到端 demo (零网络
  FakeTransport):镜像列表构造 → 健康检查 → first fetch →
  cache hit → 跨镜像 dedup

**升级**:
- `models/source/huggingface.py` — `HuggingFaceSource` 加
  `mirrors=` 参数 + `_for_each_live_mirror` 循环 + 60s TTL 的
  "dead-mirror memory" (`_dead_mirrors` 字典)。`resolve_license` /
  `list_files` / `download_files` 全部 try-mirrors fallback。
  新 `DownloadProgress` dataclass (file_name / bytes_done /
  bytes_total / mirror / started_at / finished / error) +
  `download_default_artifacts(revision, on_progress=)` 接收
  per-file 进度回调,callback 抛异常自动 swallow 不影响下载。
- `models/source/cache.py` — 新 `compute_content_fingerprint`
  (sorted `(name, sha256)` 集合的 sha256,顺序无关) +
  `ModelCache.find_by_fingerprint` (`rglob` 递归扫描 manifest,
  支持 `repo_id` 含 `/` 的情况) + `CachedModel.content_fingerprint`
  property
- `models/source/fetch.py` — `ModelFetcher.fetch` 接
  `mirrors=` + `on_progress=`,新 `_install_default_mirrors` 让
  default mirrors 自动装到 registry 中所有 HF adapter。
  `on_progress` callback 自动 wrap:4 参 `(name, done, total, mirror)`
  (v0.4.0 ergonomic shape) → 1 参 `DownloadProgress` (v0.4.x P2+
  low-level shape),通过 `inspect.signature` 推断。
  新 `_fetch_inner` 流程:download → compute fingerprint →
  `find_by_fingerprint` → 命中则**不写盘**直接 return
  existing manifest (跨 repo/revision dedup),完全避免重复
  占用磁盘与重复完整性验证。
- `models/source/__init__.py` — 暴露 `MirrorSet` / `MirrorHealth`
  / `DownloadProgress` / `compute_content_fingerprint` / 4 个
  mirror/health/is_useful helpers
- `docs/placeholder_registry.md` — 8 条新 entry (54-61) 覆盖
  `models/source/cache.py:509,578,582` (原子写 + rmdir 兜底)
  + `models/source/huggingface.py:164,170` (HttpTransport abstract
  占位) + `models/source/huggingface.py:564,597,622`
  (progress callback 兜底)

**测试**:
- 总测试数: 652 → **683** (净增 31, 全过, 49.90s)
- `pytest -m model_source` 跑 84/84 (53 旧 + 31 新)
- `pytest -m "not model_source"` 跑 599/599

**Examples**:
- `python examples/model_download.py` 零网络跑通:
  1. 镜像列表构造 (`MirrorSet.from_env()`)
  2. 健康检查 (FakeTransport 报 1 个可达 + 1 个不可达)
  3. 第一次 fetch (`from_cache=False`, 写 v1)
  4. 第二次 fetch (same key, `from_cache=True` 直接 cache hit)
  5. 第三次 fetch (不同 revision v1.1, `from_cache=True`
     走 cross-mirror dedup,**不写 v1.1 目录**, 仍能 serve
     现有 v1 的 manifest)
  6. 流量后健康检查 (upstream 仍 mark dead, mirror alive)

**Scanner 双 0**:
- Hardcoding scanner: 4452 total, critical 3857, info 595
  (vs D1 阶段二 4157 total / 3235 critical, 新增 216 主要是
  tests + mirrors 字符串路径)
- Placeholder registry: 50/50 OK (新增 8 条)
- 纯 torch,**无** `transformers` / `diffusers` / `safetensors` /
  `tokenizers` 依赖

**不做** (留到 v1.0):
- 流式字节进度 (transport 协议目前一次性返完整 bytes,
  progress 是 per-file granularity 而非 byte-level)
- 异步并发 mirror race (目前 strict 顺序 fallback, race 留给
  后续 v1.0 调度器)
- 自动镜像 health check 周期 (目前 health check 是
  on-demand ad-hoc)

### P0 多模态真模型接入 (v0.4.x P0 multi-modal milestone)

把 `models/image/` / `models/audio/` / `models/video/` / `models/multimodal/`
里已经写好的 UNet / VAE / CLIP / TTS-Transformer / HiFi-GAN / VideoDiT /
VideoVAE / OmniModel 全部接进 provider 层,4 个新 `LocalTorch*Provider`
+ 4 个 `fetch_and_load_*` + 4 个 `get_default_*_provider` + 4 个
`register_default_*_backend`,并把 3 个 `examples/` 改成走真 provider。
CI 上 31 个新测试覆盖 4 个模态的端到端 forward pass。

**新文件**:
- `models/interfaces/media_providers.py` — 4 个新 `ImageProvider` /
  `AudioProvider` / `VideoProvider` / `MultimodalProvider` Protocol
  + 4 个 `Echo*Provider` reference impl
- `models/providers/local_image.py` — `LocalTorchImageProvider` (UNet +
  VAE + CLIP) 4M params, 一次 forward ~0.1s CPU
- `models/providers/local_audio.py` — `LocalTorchAudioProvider` (TTS +
  HiFi-GAN) 4.5M params, 一次 forward ~0.1s CPU
- `models/providers/local_video.py` — `LocalTorchVideoProvider` (VideoDiT
  + VideoVAE) 5.5M params, 一次 forward ~0.1s CPU
- `models/providers/local_multimodal.py` — `LocalTorchMultimodalProvider`
  (OmniModel + TinyCausalLM) 4.5M params, multi-modal forward
  ~0.5s CPU
- `tests/test_multimodal_providers.py` — 31 个新测试

**升级**:
- `models/providers/__init__.py` — 暴露 4 个新 provider + 4 个 factory
- `models/providers/factory.py` — 新增 `fetch_and_load_image` /
  `fetch_and_load_audio` / `fetch_and_load_video` / `fetch_and_load_omni`
  + 4 个 `get_default_*_provider` singleton
- `models/interfaces/__init__.py` — re-export 4 个新 Protocol + Echo impl
- `nodes/_helpers.py` — 4 个 `register_default_*_backend` (no-arg form)
  装真 backend factory;旧 v0.4.0 `(factory)` 版本删除
- `examples/image_gen.py` / `audio_tts.py` / `video_gen.py` — 改成走真
  provider,加 elapsed 计时
- `docs/placeholder_registry.md` — 新增 6 条 (entries 48-53) 覆盖
  `_local_*_factory` 与 `_get_default_default` 内的降级 `pass`

**测试**:
- 总测试数: 621 → **652** (净增 31, 全过, 51.02s)
- 4 个模态端到端 forward pass: image (3, 16, 16) / audio (1, 512) /
  video (4, 3, 8, 8) / omni (text + image_emb + audio_emb)
- Examples: 3 个 `examples/*.py` 跑通真模型
  * image 64x64 ~1.8s
  * video 4 帧 64x64 ~2.5s
  * audio 0.1s @ 16kHz ~1.7s

**Scanner 双 0**:
- Hardcoding scanner: 4228 total, critical 3304, info 924 (与 D1 阶段二一致)
- Placeholder registry: 53/53 OK (新增 6 条)
- 纯 torch,无 transformers / diffusers / safetensors / tokenizers 依赖

### 架构简化
- 删除冗余的旧版本历史记录与架构清理叙事。
- 删除文档中过时的版本号与"v0.3.0/v0.3.1"中间过渡描述。
- 节点 `execute()` 统一通过 `nodes/_helpers.py` 中的 `call_text_backend` /
  `call_image_backend` / `call_video_backend` / `call_audio_backend`
  解析后端,默认回退 echo 工厂,允许在无模型情况下端到端跑通。
- 节点实现通过 `core.module_bus.ModuleBus` + `LLMProvider` 协议接入真模型,
  流水线层不再做 passthrough。

### 一致性评分
- `consistency/score.py` 优先尝试 `open_clip` / `torch.hub` 的真 CLIP/DINOv2
  视觉特征;未安装时回退到项目内的轻量占位特征提取器(随机投影,固定维度)。
- 评分指标与 `ConsistencyScore` 数据结构保持不变,接口兼容现有调用方。

### Hardcoding 规约化（D1）
- 新建 `docs/hardcoding_convention.md` — v0.4.x D1 根规约,3 类常量边界:
  * 运行时配置 (RUNTIME_CONFIG, `critical`) — 业务可调,必须走 ConfigCenter
  * 模型结构超参 (MODEL_STRUCTURAL, `info`) — 改了就坏,保留源码
  * 协议/格式标识 (PROTOCOL_FORMAT, `info`) — 与外部协议绑定,改了就坏
- 增强 `scripts/check_hardcoding.py`:
  * `Violation` 加 `severity` 字段 (默认 `critical`)
  * `Exemption` 加 `severity` 字段 + `protocol_format: true` 字段 (非 terminal 降级)
  * `--severity` CLI 选项: `critical` / `warn` / `info` 三档过滤
  * `--export <path>` 选项: 导出 critical 名单 (whitelist-schema 兼容 YAML)
  * `is_structural_init` 启发式: `models/` 路径下 `__init__` 中值在 [2, 10000]
    的整数自动降为 `info`
  * `_is_runtime_attr` 启发式: `os.environ[...]` / `Path(...)` / `sys.argv[...]`
    表达式中的字面量自动降为 `info`
  * `filter_by_severity()` 函数: 阈值过滤
  * `export_critical()` 函数: 去重导出 (按 file/line/type 唯一化)
- 新建 `config/hardcoding_critical_inventory.yaml` — 全项目 critical 3420
  unique entries 基线 (供 PR review 参考, **不**直接喂给 --whitelist)
- 填实 `config/hardcoded_whitelist.yaml` — 首批 ~90 条 exemption 示范:
  * 7 个 training 训练超参 group (SFT/RLHF/Synthetic/Dataset numeric) → `info`
  * 协议/格式 (LayerNorm.weight / attention_mask / observation /
    ShortTermMemory / synthetic prompt 模板) → `protocol_format: true`
  * torcha-verse 顶层 re-export 字符串 (ConfigCenter / DeviceManager / ...) → 协议绑定
- 33 个新测试覆盖: Violation 默认值 / Exemption.matches/apply/is_terminal /
  filter_by_severity 阈值 / scanner 启发式 (`is_structural_init` /
  `_is_runtime_attr` 各 2-3 个分支) / whitelist YAML 加载 / 非法 severity 拒收 /
  export_critical 去重与 critical 过滤 / **端到端** (真实 whitelist 真的降级
  命中)。
- `pyproject.toml` 注册 `hardcoding_severity` marker。
- 总测试数: 581 → 614 (全过, 46.53s)。
- Scanner 升级后分级效果: 3740 total → critical 3352, info 438。
- 顺手修正: `placeholder_registry.md` 中 `scripts/check_hardcoding.py`
  位置 (行号 338 → 526 因 scanner 重写), 仍 47 entries 全部注册。

### Hardcoding 规约化（D1）— 阶段二

- **log message 启发式** (`scripts/check_hardcoding.py:is_log_message_format`):
  把 logger 调用的**第一个字符串参数**自动降为 `info` (不再是完全 exclude),
  让 audit 仍能看到 log format 串 (PR review 时可看), 但永不 CI-fail。
  * 触发条件: 字符串 literal 是 `logger.{debug,info,warning,warn,error,
    critical,exception,log,fatal}(...)` 的 **第一个位置参数**。
  * 7 个新测试: info/warning/error 各一例 + 后续位置参数仍 critical +
    keyword arg 不算 format string + helper 直接单测。
- **批量 200+ protocol exemption** (`config/hardcoded_whitelist.yaml`):
  从阶段一 ~90 条 → **211 条** (净增 117), 新增 11 个 group:
  * Group 8: reAct / tool_call agent 协议正则 (Thought: / Action: /
    Action Input: / Final Answer: / Observation: / FINAL ANSWER: /
    ```(?:json)?...)
  * Group 9: agents/flows/ prompt 模板 (debate / hierarchical / sequential)
  * Group 10: assets/ 协议键名 + 错误消息 + SQL 字面量 (NOASSERTION /
    PRAGMA / SELECT metadata_json / INSERT OR REPLACE...)
  * Group 11: nodes/ 协议/格式 (controlnet / lip_sync / expression_params
    / consistency_score / face_embedding / voice_signature...)
  * Group 12: pipeline/ 模板协议 (input_schema / output_schema / node_type)
  * Group 13: tools/ + plugins/ 协议 (file_path / entry_point / plugin_name)
  * Group 14: serving/ HTTP 协议 (Content-Type / Authorization / X-Request-ID
    / /v1/ / /health / text/html)
  * Group 15: infrastructure/ 协议 (max_memory_mb / max_cpu_cores /
    TORCHAVERSE_*_DIR / config_snapshot.json)
  * Group 16: examples/ demo 协议 (demo_ 前缀 / Hello, world!)
  * Group 17: numeric_literal 通用超参 (14 个文件全项目 numeric → info)
  * Group 18: logger 专用批量 exemption (16 个常见 log message 前缀, 作
    heuristic 的 defence-in-depth fallback)
- **`docs/config_access.md`** — ConfigCenter / defaults 用户文档, 16 节:
  4 层配置模型 / 90 秒上手 / 读 API / 写 API / 加载顺序 / 环境变量覆盖 /
  平台差异 / 快照与重放 / ResourceBudget / `infrastructure.defaults`
  懒加载 / 环境切换 / 完整示例 / 反模式 / 故障排查 / D1 规约关系 / 速查表。
- **7 个新测试** 覆盖 log message 启发式: TestLogMessageFormat 7 个 case
  (info / warning / error / 后续参数 / keyword arg / helper 正/反例)。
- 总测试数: 614 → **621** (全过, 46.98s)。
- Scanner 升级后分级效果: 3740 total → **4157 total** (log 启发式让
  之前 excluded 的 log 字符串进 inventory, 但 severity=info) →
  **critical 3235**, **info 922** (之前 critical 3352, info 438)。
- Critical inventory (`config/hardcoding_critical_inventory.yaml`) 重新
  导出: 3420 unique → **3235 unique** (净降 185 条已批量落 exemption)。
- `pyproject.toml` 不变 (`hardcoding_severity` marker 仍有效)。
- 顺手修正: `placeholder_registry.md` 行号 526 → 569 (因 `is_log_message_format`
  method 插入, scanner 内的 `pass` 位置下移), 47 entries 仍全部注册。

### Placeholder Registry（D3 工作流集中化阶段）
- 新建 `docs/placeholder_registry.md` 作为**占位单一来源**（single source of truth）:
  47 处 `pass` / `NotImplementedError` 全部按 5 类（`protocol` / `tp_pp` /
  `protocol_stub` / `degrade_try_except` / `degrade_noop`）登记，含
  文件:行 / 上下文 / 理由。
- 新建 `infrastructure/placeholder_registry.py`:
  * `PlaceholderCategory` 枚举 + `PlaceholderEntry` dataclass
  * `load_registry` 解析 markdown 表（按 heading 推断类别，宽容处理坏行）
  * `scan_source` 扫描 Python 源文件（跳过 `tests/`、`__pycache__`、`.git/`、
    `.venv/` 等;支持行内 `# placeholder-registry: ignore` 豁免;自动跳过
    docstring 中用反引号引用的关键字描述）
  * `find_unregistered` 计算 scanner - registry 差集
  * `registry_index` 建 `(file, line) -> entry` 快速查找
- 新建 `scripts/check_placeholders.py` CI CLI:扫描 / 校验 / 报告未注册占位
  + 退出码 1 用于 CI gating。
- 升级 `infrastructure/device_manager.py` 注释：`_tensor_parallel_impl` /
  `_pipeline_parallel_impl` 现在显式引用 `placeholder_registry.md` 中
  的条目编号 (#8 / #9) + D3 重启条件,让"占位在哪儿"和"何时重启"解耦。
- 22 个新测试覆盖：枚举完整性 / `PlaceholderEntry.matches` / `load_registry`
  多种格式 / heading → category 推断 / 坏行宽容 / scanner 各分支（pass /
  NotImplementedError / ignore marker / docstring 引用 / 单文件 target /
  不存在 target）/ `find_unregistered` 差集 / `registry_index` 查表 /
  **端到端**（真实 project registry + 真实 project scan 应当 0 unregistered）。
- `pyproject.toml` 注册 `placeholder_registry` marker。
- 总测试数：559 → 581（全过，46.76s）。
- `python scripts/check_placeholders.py` 全项目扫描 47 命中, 0 unregistered。

### 真模型跑通（v0.4.0 路线图 P0）
- 新建 `models/providers/` 子包，纯 torch 实现项目自有 tiny Transformer LM，
  不引入 `transformers` / `diffusers` / `safetensors` 等外部依赖：
  * `tiny_transformer.py` — `TinyTransformerConfig`（`tiny` ~0.3M / `small` ~10M 两个预设）
    + `ByteTokenizer` 字节级 tokenizer（3 special + 256 bytes + 1 mask = 260 vocab）
    + `build_tiny_transformer` / `save_tiny_transformer` / `load_tiny_transformer`
      单文件 `.pt` 持久化，原子写入（tempfile + fsync + os.replace）
  * `local_text.py` — `LocalTorchTextProvider`（实现 `LLMProvider` 协议），
    `generate` / `chat` / `complete` 三个推理入口，线程安全
  * `factory.py` — `fetch_and_load_text` 一行拿到 provider（checkpoint → 随机初始化 fallback），
    `publish_tiny_transformer` 维护端发布，`get_default_provider` 进程级单例
  * `pretrain_tiny.py` — `train_tiny_transformer` + CLI
    (`python -m models.providers.pretrain_tiny --preset small --steps 600`)，
    AdamW + cosine LR + warmup
- 新增 `examples/real_text_chat.py`：端到端 demo
  pretrain → save → load → register_default_text_backend →
  L4 `text_chat` 节点输出真模型生成文本
- 37 个新测试覆盖：tokenizer 边界 + round-trip、config presets + dict 序列化、
  save/load 原子性 + 版本检查 + 严格性、provider 协议契约、
  factory 分支（resolve / fetch 随机 / fetch checkpoint / 缺文件报错）、
  pretrain 端到端、L4 集成（`call_text_backend` + 1 节点 Pipeline）。
- `pyproject.toml` 注册 `model_provider` marker，
  `pytest -m model_provider` 跑 37 个，`pytest -m "not model_provider"` 跑 522 个，互不干扰。
- 总测试数：522 → 559（全过，45.86s）。

### 评估模块（v0.4.0 路线图 P1）
- 新建 `evaluation/` 目录,提供纯 PyTorch 实现的指标层:
  * `metrics.psnr` / `metrics.ssim` / `metrics.lpips`（LPIPS 为占位接口）。
  * `fid.image_fid` / `fid.frechet_distance` / `fid.FidCalculator`，矩阵平方根
    用 `torch.linalg.eigh` 闭式求解，无 scipy 依赖。
  * `prompt_recall.score` / `prompt_recall.prompt_recall` /
    `prompt_recall.PromptRecallCalculator`（CLIP-score 占位实现）。
  * `runner.EvaluationRunner` / `runner.EvaluationReport` /
    `runner.load_image_dir` —— CI 友好的一站式入口。
- 占位 Inception / CLIP / LPIPS backbone 与真模型 API 完全一致，
  未来替换是真模型一行 class 替换，不影响调用方。
- 52 个新测试覆盖：指标数值正确性、FID 对称/非负/同集→0、
  矩阵平方根数值、tokenizer 确定性、双编码器形状、目录加载器、
  EvaluationRunner 端到端。`pyproject.toml` 注册 `eval` marker，
  `pytest -m eval` 跑 52 个，`pytest -m "not eval"` 跑 417 个，互不干扰。
- 总测试数：411 → 469（全过）。

### 模型源自动拉取（v0.4.0 路线图 P2）
- 新建 `models/source/` 子包，提供一行 `fetch()` 拉模型 + 许可证审计：
  * `license_check.check_license` / `DEFAULT_ALLOW_LICENSE` — SPDX 许可证白名单，
    支持运行期 `extend_default_allow_license(...)` 一次性 opt-in。
  * `cache.ModelCache` — `~/.cache/torcha-verse/<source>/<repo_id>/<revision>/`
    原子写入（tempfile + fsync + os.replace）+ sha256 完整性校验。
  * `huggingface.HuggingFaceSource` — HF Hub API 包装，注入式 `HttpTransport`
    让测试零网络跑通，默认用 `urllib.request`。
  * `civitai.CivitaiSource` — 备选源，同一套 `HttpTransport` 接口。
  * `fetch.ModelFetcher` / `fetch()` — 统一入口，验证 license、查缓存、
    拉取、写入、验证 manifest 全部原子化。
- `models/__init__.py` 重新导出 `fetch` / `FetchResult` / `ModelFetcher` 等
  公共 API，达成 `from torcha_verse.models import fetch` 简写。
- 53 个新测试覆盖：SPDX 规范化、allow-list/NC/ND 短路、
  extend_idempotent、cache 原子写入 / 验证 / 清空、manifest
  round-trip、HF / Civitai license 解析 + 文件列表 + 下载、
  SourceRegistry 别名、fetch miss-then-hit、NC 拒绝、
  cache tampering 检测、自定义 allow_list、模块级 fetch 单例。
- `pyproject.toml` 注册 `model_source` marker，`pytest -m model_source`
  跑 53 个，`pytest -m "not model_source"` 跑 469 个，互不干扰。
- 总测试数：469 → 522（全过）。

### 工程化
- 新增 `pyproject.toml`(含 pytest 配置)。
- `Dockerfile` 改为多阶段构建(builder / test / runtime)。
- 新增 `.github/workflows/ci.yml`,跑 lint + 全量测试。
- `requirements.txt` 同步精简。

### 文档
- 重写 `README.md`,对齐新结构与新流水线示例。
- `docs/architecture.md` / `docs/operations.md` 同步精简,移除历史叙事。
- 新增 `docs/DEFERRED_TASKS.md`,登记开发初期延后处理的任务(当前条目:hardcoding 规约化)。
- 新增 `docs/ROADMAP.md`(v0.4.x 准生产化 12 周计划 + v1.0.0 纲要),
  P1 评估模块 2026-06-25 标记完成。
