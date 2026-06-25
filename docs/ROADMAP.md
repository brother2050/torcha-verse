# TorchaVerse 路线图

> 日期:2026-06-25 · 状态:初期(early-stage)→ 准生产(quasi-production)
> 当前 HEAD: `fdfdb1f` · 目标 next release: **v0.4.0**

## 定位

初期项目已完成架构骨架(6 层 · 30 节点 · ModuleBus · 369 测试),**没有任何真模型实际跑通**。路线图分两段:

- **v0.4.x 准生产化**(当前在做):补齐工程质量、跑通 1-2 个真模型、补薄弱的包
- **v1.0.0 生产化**(后续):分布式 / 多租户 / 监控 / 完整评估

---

## v0.4.x — 准生产化(本期,约 12 周)

| 优先级 | 范围 | 状态 | 估时 |
|---|---|---|---:|
| **P0** | 真模型跑通(Qwen2.5-0.5B + SDXL-Turbo)| **延后**(用户决定) | — |
| **P1** | 评估模块最小版(FID + prompt 还原率 + CI 集成) | **完成 2026-06-25** | 1-2 周 |
| **P2** | 模型源自动拉取(HuggingFace + 许可证审计) | **完成 2026-06-25** | 1 周 |
| **P3** | pass/NotImplementedError 审计 | **完成 2026-06-25** | 0 |
| **P4** | performance / training 补基础测试 | **完成 2026-06-25** | 1 周 |
| **P5** | examples 重写(对齐 30 节点) | **完成 2026-06-25** | 1 周 |
| **P5** | ROADMAP + DEFERRED_TASKS 维护 | 进行中 | 持续 |

> P0(真模型)被用户标记为"延后到初期过后",不在本路线图。完整讨论见 `docs/DEFERRED_TASKS.md`。

---

## P1 — 评估模块最小版 ✅ 完成 2026-06-25

**目标**:能用 `pytest` 跑过,生成质量有量化指标

**目录**:`evaluation/`(从无到新建)
```
evaluation/
├── __init__.py
├── metrics.py        # PSNR / SSIM / LPIPS 占位
├── fid.py            # Inception 特征 + Frechet 距离
├── prompt_recall.py  # CLIP 文本-图像相似度
└── runner.py         # 统一入口
```

**最小可用特性**:
- `eval.image_fid(real_dir, gen_dir)` → 标量 FID
- `eval.prompt_recall(images, prompts)` → 平均 CLIP score
- CI 集成: `pytest -m eval` 跑通(用 fixture 生成的小数据集)

**实现要点**:
- 纯 PyTorch + 标准库(无 scipy / torchmetrics / pytorch-fid)。
- 矩阵平方根用 `torch.linalg.eigh` 闭式求解,Frechet 距离数值精确(已用单元测试验证 `||mu1-mu2||^2 + Tr(S1+S2-2*sqrt(S1*S2))`)。
- Inception / CLIP / LPIPS 三个 backbone 均为占位实现(随机初始化 + 随机投影)。API 与真模型保持一致,未来替换是真模型一行 class 替换。
- 52 个新测试覆盖:PSNR 单调性、SSIM 边界、FID 对称/非负/同集→0、矩阵平方根数值、tokenizer 确定性、双编码器形状、EvaluationRunner 端到端、目录加载器。
- `pyproject.toml` 注册 `eval` marker,`pytest -m eval` 跑 52 个,`pytest -m "not eval"` 跑 417 个,互不干扰。
- 总测试:411 → 469(全过)。

**不做**(留到 v1.0):
- 大规模数据集评测
- 人工评估 UI
- 模型排行榜

---

## P2 — 模型源自动拉取 + 许可证审计 ✅ 完成 2026-06-25

**目标**:`from torcha_verse.models import fetch("Qwen/Qwen2.5-0.5B-Instruct")` 一行拉模型

**目录**:`models/source/`(从无到新建)
```
models/source/
├── __init__.py        # 公共 API 重导出
├── huggingface.py     # HF Hub API 包装 (注入式 HttpTransport)
├── civitai.py         # 备选源 (同样的 HttpTransport 接口)
├── license_check.py   # SPDX 许可证白名单 (DEFAULT_ALLOW_LICENSE)
├── cache.py           # ~/.cache/torcha-verse 原子写入 + sha256 校验
└── fetch.py           # ModelFetcher + fetch() 统一入口
```

**最小可用特性**:
- `fetch(repo_id, allow_license=["apache-2.0", "mit"])` 一行拉模型
- 自动落 `~/.cache/torcha-verse/<source>/<repo_id>/<revision>/`
- 默认白名单:apache-2.0 / mit / bsd-3-clause / cc-by-4.0
- 拒绝:non-commercial / 没有 license / 未知
- 原子写入:tempfile + fsync + os.replace,失败不留半文件
- 完整性:sha256 校验,manifest 与实际文件一致
- 缓存命中:二次 fetch 不发网络请求

**实现要点**:
- 纯标准库实现 HTTP transport (`urllib.request`),可选 `huggingface_hub` 集成
  留作未来 opt-in。
- 注入式 `HttpTransport` 接口让所有 53 个新测试在**零网络**环境下跑通
  (用 `FakeTransport` 路由 URL 子串到预设响应,按最长子串优先匹配避免重叠)。
- 默认白名单集中于 `license_check.DEFAULT_ALLOW_LICENSE`,
  `extend_default_allow_license(...)` 支持运行期一次性 opt-in
  (e.g. GPL-3.0)。
- License check 优先级:caller-显式-白名单 > NC 短路 > ND 短路 >
  默认白名单 > known-OK SPDX 提示 > unknown 拒绝。
- 53 个新测试覆盖:SPDX 规范化、allow-list/NC/ND 短路、
  extend_idempotent、cache 原子写入 / 验证 / 清空、manifest
  round-trip、HF license 解析 / 文件列表 / 下载、
  Civitai license 解析 / 下载、SourceRegistry 别名、
  fetch miss-then-hit、NC 拒绝、cache tampering 检测、
  自定义 allow_list、模块级 fetch 单例。
- `pyproject.toml` 注册 `model_source` marker,`pytest -m model_source`
  跑 53 个,`pytest -m "not model_source"` 跑 469 个,互不干扰。
- 总测试:469 → 522(全过)。

**不做**:自动转换格式 / 量化 / 性能分析

---

## P4 — performance / training 补基础测试

**目标**:两个 0 测试包(共 4700 行)有烟雾测试

**新增文件**:
```
tests/test_performance_quantization.py   # ~80 行
tests/test_performance_optimizer.py     # ~80 行
tests/test_performance_benchmark.py      # ~80 行
tests/test_training_sft_smoke.py         # ~100 行
tests/test_training_rlhf_smoke.py        # ~100 行
tests/test_training_synthetic_data.py    # ~80 行
```

**最小覆盖**:
- `Quantizer.fp16(model).weight.dtype == torch.float16`
- `PerformanceOptimizer.optimize_model(Linear(4,4))(randn(1,4)).shape == (1,4)`
- `BenchmarkSuite.run_text_benchmark("dummy", "hi", max_tokens=4).latency_ms >= 0`
- `SFTTrainer.fit_step(...)` 单步梯度下降,loss 下降
- `RLHFTrainer.compute_dpo_loss(...)` 输出标量 loss
- `SyntheticDataGenerator.filter_quality(...)` 过滤掉空串

---

## P5 — examples 重写

**目标**:6 个 examples 与 30 节点对齐,跑得通(用 echo 后端)

**重写**:
- `examples/basic_text_gen.py` → `text_chat` 节点
- `examples/image_gen.py` → `image_txt2img` 节点
- `examples/audio_tts.py` → `audio_tts` 节点
- `examples/video_gen.py` → `video_txt2vid` 节点
- **新增** `examples/consistency_character.py` → `character_apply` + `character_five_view` 链
- **新增** `examples/dh_lipsync.py` → `dh_lip_sync` 节点

**风格**:每个 example < 50 行,顶部注释"如何跑",输出可读 summary。

---

## v1.0.0 — 生产化(后续,2026 Q4 之后)

> 这部分**不**在本期路线图,只列纲要,避免忘记。

| 主题 | 内容 |
|---|---|
| ResourceBudget | 真实接调度,GPU 满了排队 |
| RuntimeScheduler | 统一线程/异步/GPU 流 |
| 分布式训练 | FSDP / DeepSpeed |
| 监控 | Prometheus metrics + Grafana 面板 |
| 多租户 | 用户隔离、配额、审计 |
| 完整评估 | 大规模数据集、模型排行榜 |
| 真实部署 | K8s operator、Helm chart |

---

## 进度跟踪

每完成一个 PR,更新本文件"状态"列。每月初扫一次 `DEFERRED_TASKS.md`,评估是否要重新启动任何延后项。
