# TorchaVerse 路线图

> **更新日期**: 2026-06-26 · 当前: **v0.6.x (R-3~R-19 收尾 + F-0~F-14 真实填充)**
> HEAD: R-3 ~ R-19 + F-0~F-14 · 1118 测试全过 · 节点数 39
> **最新 commit**: a9202cb (F-0~F-14)
> 下次发布: **v0.8.0** (ModelMixin + from_pretrained + 真采样循环)
> 详细方案见: [`docs/V0.8_UPGRADE_PLAN.md`](V0.8_UPGRADE_PLAN.md)

## 定位

TorchaVerse 是纯 PyTorch 全模态生成式 AI 框架,六层分层,39 个能力节点
(文本/图像/视频/音频/字幕/一致性/数字人/导出/RAG/Agent),端到端可跑、
可生产部署。

路线图分两段:
- **v0.4.x 准生产化** ✅ 完成 (2026-06-25, 22 commit, 747→986 测试)
- **v0.5.x 功能扩展** ✅ 完成 (2026-06-25, +5 examples, +206 测试)
- **v0.6.x 重构** ✅ 完成 (2026-06-26, 13 PR, R-3~R-15 拆分, 1053 测试)
- **v0.7.x 性能 / CLI / 懒化** (当前)
- **v1.0.0 生产化** (后续)

---

## v0.7.x — 当前阶段 (本周)

| 优先级 | 范围 | 状态 | PR |
|---|---|---|---|
| R-16 | 性能优化 (NodeContext lock, cache, batch) | ⏳ 进行中 | R-16 |
| R-17 | CLI `--config` / JSON log / request-id / healthcheck | ⏳ 队列 | R-17 |
| R-18 | `nodes/papers` 懒化 (按需 import, 启动时间 -30%) | ⏳ 队列 | R-18 |
| R-19 | **撤 scripts/check_* shim + 重写 13 个 MD** | ✅ 完成 2026-06-26 | R-19 |
| F-* | **真实实现填充 (24 个节点 + 11 PaperAdapter + 32 测试)** | ✅ 完成 2026-06-26 | F-0~F-14 |

### F-* 完成项 (F-0 ~ F-14)

1. **F-0**: 修 `dh_full_body` 的 `image` / `reference_image` 别名 bug
2. **F-1**: 6 数字人节点 + 11 PaperAdapter (MuseTalk / SadTalker /
   LivePortrait / EchoMimicV2 / GFPGAN / CodeFormer / CosyVoice /
   F5TTS / ChatTTS) + 7 个新 helper
3. **F-2 ~ F-5**: `_subtitle_codec.py` (SRT/VTT/ASS 序列化 + 能量法
   ASR + 滑窗翻译 + cv2 烧录) + 4 个字幕节点 `execute` 接入
4. **F-6 ~ F-7**: depth_condition 接 SceneEngine._DepthEstimator;
   character_five_view 跑 ScoreCalculator.clip_i_distance
5. **F-8 ~ F-9**: video_interpolate 接 FrameInterpolator 真插帧;
   video_txt2vid 接 MotionModule 真实 motion injection
6. **F-10**: image_txt2img/img2img 跑 DiffusionScheduler 真 timestep
7. **F-11**: 新 `models/image/restoration.py` (SuperResolutionUNet +
   InpaintUNet) + 2 个 helper
8. **F-12**: 新 `models/audio/music.py` (MusicDiT + MusicTransformer)
   + HiFiGAN 串接
9. **F-13**: video_stitch 调 ffmpeg xfade,fallback torch 线性 crossfade
10. **F-14**: `tests/test_real_implementations.py` (32 个 test) +
    CHANGELOG + placeholder registry 修正

总测试数 **1053 → 1118 (+65)** (含 v0.6.1 R-17 / R-18 期间的 33 个增量)

### R-19 完成项

1. **撤 shim** (5 个):
   - `scripts/check_hardcoding.py` / `check_hardcoding_rules.py` / `check_placeholders.py` / `check_degrade_logging.py` / `check_ci_gates.py` 全删
   - 测试文件 3 处 `from scripts.check_*` → `from scripts.check.<subpkg>` 子包路径
   - `_cli.py` 加 `if __name__ == "__main__": sys.exit(main())` 入口
   - `hardcoding/__init__.py` 把 `main` 改成 PEP 562 `__getattr__` 懒导出,避免循环
2. **重写 MD 文档** (13 个):
   - `README.md` — 短版,6 节,指向 docs/*
   - `docs/architecture.md` — 6 层 + 横切层,节点 39
   - `docs/operations.md` — 部署 / 监控 / checkpoint / 模型下载
   - `docs/ROADMAP.md` — 本文,v0.4.x → v0.7.x 进度
   - `docs/DEFERRED_TASKS.md` — 延后任务
   - `docs/open_items.md` — 已知未处理项
   - `docs/hardcoding_convention.md` — 扫描器 + 9 个可插拔规则
   - `docs/placeholder_registry.md` — 95 个 placeholder 行号登记
   - `docs/config_access.md` — 配置中心使用指南
   - `docs/docker.md` — Docker 镜像构建
   - `CHANGELOG.md` — v0.4.0 → v0.6.0

### 文档风格统一

- **最近更新 + 总数** 在每个文件头标
- 表格 > 列表;凡是与其它文件交叉引用的,用相对路径链接
- 中文为主,代码块 / 路径 / 命令保留英文
- 旧 v0.4.x 行号 / 旧文件名出现时,括号注明 "v0.4.x" + 当前新位置

---

## v0.4.x ✅ 完成 (2026-06-25)

P0 真模型跑通 / P0 多模态扩展 / P1 评估 / P2 模型源 / P2+ 镜像 + dedup /
P2++ 完整性 + token / P3 placeholder 审计 / P4 测试 / P5 examples /
D1 hardcoding / D3 placeholder 集中化 / C1-C7 v1.0 骨架 (BudgetTracker,
RuntimeScheduler, metrics, multi-tenant, leaderboard, docker, ci).
详见 git log: v0.4.0 (de35b14) → v0.4.3 (b032082)。

总测试数: 369 → 986 (+617),scanner 双 0 维持。

## v0.5.x ✅ 完成 (2026-06-25)

- 资产子系统 (Asset / AssetRef / AssetStore / 5 子类) + 三级存储
- 一致性子系统 (Character / Outfit / Scene / Score)
- Pipeline 模板 + Prompt Studio + Canvas
- 输出过滤 (毒性 / NSFW)
- v0.5 feature demo example

测试: 986 → 1053 (+67)。

## v0.6.x ✅ 完成 (2026-06-26)

13 PR (R-3~R-15) 把 9,419 行的单文件拆成 65+ 聚焦子模块:

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

测试保持 1053 全过,12 PR 0 回归。

---

## v1.0.0 — 生产化 (后续,2026 Q4 之后)

启动条件 (任一):
1. v0.6.x 用户报"多任务并发 OOM" / "缺 metrics" / "租户互相影响" ≥ 1
2. v0.6.x 真大模型 e2e (C8) 在 CI 跑通 → 启动 M0
3. 2026 Q4 时间节点到 → 强制启动 M0

### Milestone 拆分

| M | 目标 | 估时 | 关键产物 |
|---|---|---:|---|
| M0 | `BudgetTracker` 真实调度 (排队 + 超时) | 1 周 | `allocate_or_wait` + 30+ 测试 |
| M1 | `RuntimeScheduler` 抽象 + 3 实现 | 1-2 周 | ~400 行 + 40+ 测试 |
| ~~M2a~~ | ~~Gloo 分布式 (TP/PP)~~ | — | 🗑️ 单系统路线明确,跨节点推迟 |
| M2b | Prometheus metrics | 0.5-1 周 | `/metrics` + Grafana 4-panel |
| M2c | Dockerfile + compose | 0.5-1 周 | `python:3.10-slim` + compose |
| M3a | 多租户 | 1 周 | per-tenant BudgetTracker + 命名空间 |
| M3b | 评估 leaderboard | 1 周 | leaderboard + 10+ 测试 |
| C8 | 真实大模型 e2e | 4-8 周 | Qwen2.5 / SDXL-Turbo / HunyuanVideo |

详细子任务 / 启动条件 / 风险登记见 [`docs/open_items.md`](open_items.md) C 段;
6 主题 v0.4.x 现状盘点 (ResourceBudget / RuntimeScheduler / 监控 / 多租户 / 评估 / 部署)
也已合并到 C1-C8。

---

## 进度跟踪

| 阶段 | 状态 | 测试 | commit |
|---|---|---:|---|
| v0.3.0 (架构骨架) | ✅ | 369 | 8801c2d |
| v0.4.0 (P0 真模型) | ✅ | 559 | (略) |
| v0.4.1 (B1 silent degrade 清零) | ✅ | 581 | (略) |
| v0.4.2 (C1-C7 v1.0 骨架) | ✅ | 747 | de35b14 |
| v0.4.3 (C1b-C7b 加深) | ✅ | 830 | b032082 |
| v0.5.0 (资产/一致性/Prompt Studio) | ✅ | 986 | (略) |
| v0.5.1 (v0.5 feature demo) | ✅ | 1053 | (略) |
| v0.6.0 (R-3~R-15 重构) | ✅ | 1053 | (R-3~R-15) |
| v0.6.1 (R-16~R-19 性能/CLI/lazy/MD) | 🚧 | 1053+ | R-19 ✅ |
| v1.0.0 (生产化) | ⏳ | — | — |

每月初扫一次 `DEFERRED_TASKS.md`,评估是否重新启动任何延后项。
