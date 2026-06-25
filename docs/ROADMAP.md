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
| **P1** | 评估模块最小版(FID + prompt 还原率 + CI 集成) | 准备中 | 1-2 周 |
| **P2** | 模型源自动拉取(HuggingFace + 许可证审计) | 待开始 | 1 周 |
| **P3** | pass/NotImplementedError 审计 | **完成 2026-06-25** | 0 |
| **P4** | performance / training 补基础测试 | 待开始 | 1 周 |
| **P5** | examples 重写(对齐 30 节点) | 待开始 | 1 周 |
| **P5** | ROADMAP + DEFERRED_TASKS 维护 | 进行中 | 持续 |

> P0(真模型)被用户标记为"延后到初期过后",不在本路线图。完整讨论见 `docs/DEFERRED_TASKS.md`。

---

## P1 — 评估模块最小版

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

**不做**(留到 v1.0):
- 大规模数据集评测
- 人工评估 UI
- 模型排行榜

---

## P2 — 模型源自动拉取 + 许可证审计

**目标**:`from torcha_verse.models import fetch("Qwen/Qwen2.5-0.5B-Instruct")` 一行拉模型

**目录**:`models/source/`(新建)
```
models/source/
├── __init__.py
├── huggingface.py     # HF Hub API 包装
├── civitai.py         # 备选源
├── license_check.py   # SPDX 许可证白名单
└── cache.py           # 已有 safetensors 缓存
```

**最小可用特性**:
- `fetch(repo_id, allow_license=["apache-2.0", "mit"])` 拉模型
- 自动落 `~/.cache/torcha-verse/`
- 默认白名单:apache-2.0 / mit / bsd-3-clause / cc-by-4.0
- 拒绝:non-commercial / 没有 license / 未知

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
