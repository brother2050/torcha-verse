# Changelog

项目初期变更记录。初期重点：架构简洁、节点能跑、测试可过。

## [Unreleased] — 初期整理

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
