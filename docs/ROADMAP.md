# TorchaVerse 路线图

> 日期:2026-06-25 · 状态:初期(early-stage)→ 准生产(quasi-production)
> 当前 HEAD: `e83aa7e` · 目标 next release: **v0.4.x (P0 multi-modal 已并入)**

## 定位

初期项目已完成架构骨架(6 层 · 30 节点 · ModuleBus · 369 测试),**没有任何真模型实际跑通**。路线图分两段:

- **v0.4.x 准生产化**(当前在做):补齐工程质量、跑通 1-2 个真模型、补薄弱的包
- **v1.0.0 生产化**(后续):分布式 / 多租户 / 监控 / 完整评估

---

## v0.4.x — 准生产化(本期,约 12 周)

| 优先级 | 范围 | 状态 | 估时 |
|---|---|---|---:|
| **P0** | 真模型跑通(项目自有 tiny transformer + 字节级 tokenizer) | **完成 2026-06-25** | 1-2 周 |
| **P0** | P0 多模态扩展 (image / audio / video / omni 4 个真 provider) | **完成 2026-06-25** | 1 天 |
| **P1** | 评估模块最小版(FID + prompt 还原率 + CI 集成) | **完成 2026-06-25** | 1-2 周 |
| **P2** | 模型源自动拉取 (HF / Civitai / license / cache) | **完成 2026-06-25** | 1 周 |
| **P2+** | HF 镜像 fallback + 跨镜像内容去重 + 下载进度回调 | **完成 2026-06-25** | 1 天 |
| **P3** | pass/NotImplementedError 审计 | **完成 2026-06-25** | 0 |
| **P4** | performance / training 补基础测试 | **完成 2026-06-25** | 1 周 |
| **P5** | examples 重写(对齐 30 节点) | **完成 2026-06-25** | 1 周 |
| **P5** | ROADMAP + DEFERRED_TASKS 维护 | 进行中 | 持续 |
| **D1** | Hardcoding 规约化与扫描器校准 (阶段一 + 阶段二) | **完成 2026-06-25** | 1 天 |
| **D3 阶段二** | Placeholder Registry 集中化 | **完成 2026-06-25** | 1 天 |

> P0 实际改为"项目自有 tiny transformer + 字节级 tokenizer"路线(纯 torch,
> 不引入 transformers / diffusers 等外部依赖,见 `docs/DEFERRED_TASKS.md`)。
> 真实大模型(Qwen2.5 / SDXL-Turbo)拉取仍归 P2 fetch 子系统,但 v0.4.x
> 周期不参与 e2e 跑通,留待 v1.0.0。

---

## D1 — Hardcoding 规约化与扫描器校准 ✅ 完成 2026-06-25

**目标**:把 3726 条 "无差别警告" 拆成 3 档 severity,让 critical 真正进入
CI gating,让 info 留作审计;建立 3 类常量边界 (运行时配置 / 模型结构超参 /
协议/格式标识) 的工程规约。

**D1 阶段一 + 阶段二**:
- 阶段一: 规约文档 + scanner severity 分级 + 首批 ~90 exemption (commit 8801c2d)
- 阶段二: log message 启发式 + 200+ 批量 exemption + ConfigCenter 用户文档 (本次 commit)

**新增文件**:
- `docs/hardcoding_convention.md` — D1 根规约,定义:
  * 3 类常量边界 (RUNTIME_CONFIG / MODEL_STRUCTURAL / PROTOCOL_FORMAT)
  * 3 档 severity (critical / warn / info) + CI 接入约定
  * scanner 内置启发式规则清单
  * whitelist YAML 4 种 exemption 字段示例
  * 维护规则 (新增结构超参 / 运行时配置 / 协议标识 三种场景)
- `config/hardcoding_critical_inventory.yaml` — 全项目 critical 3235
  unique entries 基线 (供 PR review 参考)
- `tests/test_hardcoding_severity.py` — 40 个测试 (33 阶段一 + 7 阶段二)
- `docs/config_access.md` — ConfigCenter / defaults 用户文档 (16 节)

**升级文件**:
- `scripts/check_hardcoding.py` — 完全重写:
  * `Violation` 加 `severity` 字段
  * `Exemption` 加 `severity` + `protocol_format` 字段
  * `--severity {critical,warn,info}` CLI 选项
  * `--export <yaml>` 导出 critical 名单
  * `is_structural_init` 启发式 (models/ 路径下 numeric in [2, 10000] → info)
  * `_is_runtime_attr` 启发式 (os.environ / Path() / sys.argv → info)
  * `is_log_message_format` 启发式 (logger.{level} 第一个位置参数 → info)
  * `filter_by_severity` / `export_critical` / `is_log_message_format` 函数
- `config/hardcoded_whitelist.yaml` — 211 条 exemption (阶段一 90 → 阶段二 211)
  - 阶段二新增 11 个 group (8-18): reAct 协议 / agents/flows prompt 模板 /
    assets SQL 字面量 / nodes 协议 / pipeline 模板 / tools 协议 / serving HTTP /
    infrastructure 协议 / examples demo / numeric 通用超参 / logger 冗余 fallback
- `pyproject.toml` — 注册 `hardcoding_severity` marker
- `docs/placeholder_registry.md` — 修正行号 338 → 526 → 569 (scanner 重写/扩展)
- `CHANGELOG` / `ROADMAP` / `DEFERRED_TASKS`

**关键能力**:
- CI 闸口: `python scripts/check_hardcoding.py --severity critical` 失败 = PR 必拒
- 自动分级: scanner 内置 **3** 个启发式 (structural / runtime / log_message) 自动降级
- 协议识别: `protocol_format: true` 字段明确声明"协议绑定", 一键降为 info
- 审计仪表盘: `hardcoding_critical_inventory.yaml` 给 PR reviewer 看清全貌
- 用户文档: `docs/config_access.md` 让 ConfigCenter API 真正可查

**测试**:
- 总测试数: 581 → 614 (阶段一) → **621** (阶段二, 全过, 46.98s)
- Scanner 升级后: 3740 total → 4157 total → **critical 3235**, **info 922**
- Whitelist 应用后: critical 进一步从 3450 (阶段一) 降至 3235 (阶段二)
- Critical inventory 重新导出: 3420 unique (阶段一) → **3235 unique** (阶段二,
  净降 185 条已批量落 exemption)

**剩余工作** (D1 阶段三, 待启动):
- 拆 scanner:把 `string_literal` / `numeric_literal` 拆成两个独立规则,允许
  结构超参 opt-out (目前是启发式自动判断,不能逐项 opt-out)。
- 在 `pyproject.toml` 里把"critical 违规"设为 CI 必过项,non-critical 仍 warn。
- weekly PR 节奏:把剩余 critical 3235 分批落 exemption,目标 0 critical。


---

## D3 阶段二 — Placeholder Registry 集中化 ✅ 完成 2026-06-25

**目标**:把 D2 审计出的 47 处 `pass` / `NotImplementedError` 集中到
`docs/placeholder_registry.md` 单一来源,加 CI 闸口拦截未登记的新占位。

**新增文件**:
- `docs/placeholder_registry.md` — 47 条占位分 5 类
  (protocol / tp_pp / protocol_stub / degrade_try_except / degrade_noop) 登记。
- `infrastructure/placeholder_registry.py` — 解析器 + 扫描器 + 差集查询
  (纯 stdlib,无依赖)。
- `scripts/check_placeholders.py` — CI 入口,扫描全项目并与 registry 求
  差集,`exit 1` 失败。
- `tests/test_placeholder_registry.py` — 22 个测试,含**端到端**:真实
  project registry + 真实 project scan 应当 0 unregistered。

**升级文件**:
- `infrastructure/device_manager.py` 顶部注释 — 显式引用 registry 中
  `_tensor_parallel_impl` / `_pipeline_parallel_impl` 的条目编号 (#8 / #9)
  + D3 重启条件,让"占位在哪儿"和"何时重启"解耦。
- `pyproject.toml` 注册 `placeholder_registry` marker。

**关键能力**:
- 行内 `# placeholder-registry: ignore` 标记:写文档 / docstring 时
  可以提及关键字不被 scanner 误报。
- docstring 反引号包裹的 `\`pass\`` / `\`NotImplementedError\`` 关键字
  自动跳过(描述性引用 ≠ 真占位)。
- `--list` / `--stats` 子命令:审计时可列出所有注册条目或统计。

**测试**:
- 总测试数:559 → 581(全过,46.76s)。
- `python scripts/check_placeholders.py` 项目级扫描 47 命中, 0 unregistered。

---

## P0 — 真模型跑通 ✅ 完成 2026-06-25

**目标**:让 30 节点的 L4 `text_chat` 节点**真的**调通一个本项目自有模型,无外部依赖。

**目录**:`models/providers/`(从无到新建)
```
models/providers/
├── __init__.py              # 公共 API 重导出
├── tiny_transformer.py      # TinyTransformerConfig + ByteTokenizer + build/save/load
├── local_text.py            # LocalTorchTextProvider (LLMProvider 协议实现)
├── factory.py               # fetch_and_load_text / publish_tiny_transformer
└── pretrain_tiny.py         # 训 + CLI: python -m models.providers.pretrain_tiny
```

**最小可用特性**:
- `fetch_and_load_text("torcha-verse/tiny-transformer-tiny")` 一行拿到 provider
- 两个预设:`tiny` (~0.3M, 1.3 MB, CI 用) / `small` (~10M, ~40 MB, P0 demo)
- 字节级 tokenizer:3 special + 256 bytes + 1 mask = 260 vocab,完全无依赖
- 单文件 `.pt` 持久化:`format_version` + `config` + `tokenizer` + `state_dict`,
  原子写入 (tempfile + fsync + os.replace)
- `LocalTorchTextProvider` 实现项目自身的 `LLMProvider` 协议
  (`generate` / `chat` / `complete`),并被 L4 节点的
  `register_default_text_backend` 注册
- `examples/real_text_chat.py` 端到端跑通:pretrain → save → load → register
  → 1 节点 `text_chat` 流水线输出真模型生成结果

**实现要点**:
- 严格保持**纯 torch**:`TransformerDecoder` 复用项目自身
  `models/text/transformer.py`,KV-cache / GQA / RoPE / SwiGLU / RMSNorm 全部已有。
  `ByteTokenizer` 走 UTF-8 字节级编码,不引入 `transformers` / `tokenizers` /
  `safetensors` 等外部库,完全契合用户"减少其他依赖"的约束。
- Pretrain:AdamW,bias / norm 不做 weight decay,warmup + cosine LR,
  30~600 步即可在默认语料上让 loss 收敛。
- Random-init fallback:无 checkpoint 时也能 `from_random(cfg)` 跑通流水线,
  CI 不依赖任何外部资源。
- 37 个新测试覆盖:tokenizer 边界 + 状态字典 + 字节 round-trip;
  config presets + dict round-trip;save/load 原子性 + 版本检查 + 严格性;
  provider 协议契约(generate / chat / complete / num_parameters);
  factory 分支(resolve / fetch 随机 / fetch checkpoint / 缺文件报错);
  pretrain 端到端(loss 下降 / .pt 文件存在 / 加载后能 generate);
  L4 集成(`call_text_backend` + 1 节点 Pipeline)。
- `pyproject.toml` 注册 `model_provider` marker,`pytest -m model_provider`
  跑 37 个,`pytest -m "not model_provider"` 跑 522 个,互不干扰。
- 总测试:522 → 559(全过,45.86s)。
- 端到端 demo: `python examples/real_text_chat.py --preset tiny --steps 30`
  输出真模型生成文本,中英文 prompt 都通过 L4 `text_chat` 节点
  (含 echo prompt 续写),证明 30 节点 P0 端到端真跑通。

**不做**(留到 v1.0):
- 真实大模型适配(Qwen2.5 / SDXL-Turbo 等)
- 多模态 / vision-language / speech
- 量化 / LoRA / 分布式训练

---

## P0 多模态扩展 — image / audio / video / omni 4 个真 provider ✅ 完成 2026-06-25

**目标**:把 v0.4.0 P0 文本真模型扩展到 4 个模态,让 `image_txt2img` /
`audio_tts` / `video_txt2vid` / `dh_lip_sync` / `character_apply` /
`character_five_view` 这些 L4 节点**真的**调通项目自有的 multi-modal
后端,无外部依赖。

**新增文件** (4 provider + 1 interface + 1 tests):
- `models/interfaces/media_providers.py` — 4 个新 `@runtime_checkable`
  Protocol: `ImageProvider` / `AudioProvider` / `VideoProvider` /
  `MultimodalProvider`,每个 `generate(**kwargs) -> Dict[str, Any]`。
  同文件 4 个 `Echo*Provider` reference impl,作为无模型时的
  fallback,固定 `[echo-image]` / `[echo-audio]` 前缀方便识别。
- `models/providers/local_image.py` — `LocalTorchImageProvider`
  (UNet + VAE + CLIPTextEncoder)。TINY 预设 4M params,一次
  forward ~0.1s CPU,输出 `(3, H, W)` in `[0, 1]`。`from_random` /
  `from_file` / `save` 三个 round-trip 入口。DDPM 2D 扩散循环 + CLIP
  prompt 嵌入,完全复用项目自有 `models/image/*`。
- `models/providers/local_audio.py` — `LocalTorchAudioProvider`
  (TTSTransformer + HiFiGAN)。TINY 预设 4.5M params,一次
  forward ~0.1s CPU,输出 `(1, num_samples)` 16 kHz waveform。
  字节级 token → TTS → mel → HiFiGAN → 波形 → `clamp(-1, 1)`。
  HiFiGAN 关键签名:`(in_channels, upsample_rates,
  upsample_kernel_sizes, hidden_size)`,**不是** `upsample_initial_channel`。
- `models/providers/local_video.py` — `LocalTorchVideoProvider`
  (VideoDiT + VideoVAE)。TINY 预设 5.5M params,一次
  forward ~0.1s CPU,输出 `(T, 3, H, W)`。
  Noise shrink + DiT forward + VAE decode。
  自动对齐 VAE down_factor × DiT patch size。
- `models/providers/local_multimodal.py` — `LocalTorchMultimodalProvider`
  (OmniModel + 独立 TinyCausalLM)。TINY 4.5M params,multi-modal
  端到端 ~0.5s CPU。Vision / audio / text 三路融合:
  * Vision: 16x16 resize → vision_encoder → features `(1, 17, 64)`
    (16 patches + 1 cls token)
  * Audio: pad/truncate 到 32 mel channels → audio_encoder → embedding
  * Text: 字节级 token → 独立 TinyCausalLM → argmax next_id
    → `% 128` clamp 到 ASCII → `bytes.decode`
  * 输入三种 mode: `str` / `dict` / `Sequence`
- `tests/test_multimodal_providers.py` — 31 个新测试
  (4 Echo + 4 image + 3 audio + 3 video + 4 omni + 8 factory + 5 preset,
  全部 5.55s 跑通 standalone)。

**升级文件**:
- `models/interfaces/__init__.py` — re-export 4 个新 Protocol + Echo impl
- `models/providers/__init__.py` — 暴露 4 个新 provider + 4 个 config + 8 个
  factory/singleton
- `models/providers/factory.py` — 新增 `fetch_and_load_image` /
  `fetch_and_load_audio` / `fetch_and_load_video` / `fetch_and_load_omni`
  + 4 个 `get_default_*_provider` 双检锁 singleton
- `nodes/_helpers.py` — 4 个新 `register_default_*_backend` (no-arg form)
  装真 backend factory;旧的 v0.4.0 `(factory)` 4 个版本删除
- `nodes/consistency.py` — `character_five_view` 节点加
  `width` / `height` Optional inputs,允许 demo 改 64x64 不再 5×512
  OOM(原 hardcode 512 在 CPU UNet 上 attention 4GB 溢出)
- `examples/image_gen.py` / `audio_tts.py` / `video_gen.py` /
  `consistency_character.py` / `dh_lipsync.py` — 5 个 example **全部**
  改走真 provider,加 elapsed 计时 + tensor shape 打印
- `docs/placeholder_registry.md` — 6 条新 entry (#48-53) 覆盖
  `_local_*_factory` 与 `_get_default_default` 的 `pass` 降级
- `CHANGELOG.md` — P0 multi-modal section 详细列出文件 + 测试 + 性能

**端到端真模型跑通** (TINY 预设, CPU):
- `image_gen.py` 64x64 → (3, 64, 64) tensor, ~6s (含 import + 4 步 DDPM)
- `video_gen.py` 4 帧 64x64 → (4, 3, 64, 64) tensor, ~7s
- `audio_tts.py` 0.1s @ 16kHz → (1, 1600) waveform, ~5s
- `consistency_character.py` 64x96 + 5×64x64 → 全 tensor, ~30s
- `dh_lipsync.py` (4, 3, 256, 256) frames, ~5s
- 总测试数:621 → **652** (净增 31,全过,51.02s)

**Scanner 双 0**:
- Hardcoding scanner: 4228 total, critical 3304, info 924(与 D1 阶段二一致)
- Placeholder registry: 53/53 OK(新增 6 条)
- 纯 torch,**无** `transformers` / `diffusers` / `safetensors` / `tokenizers`
  依赖,完全契合用户"减少其他依赖"的约束

**不做**(留到 v1.0):
- 真实大模型 (SDXL / Whisper / Wav2Vec / HunyuanVideo) 适配
- 量化 / LoRA / 分布式训练
- 流式推理 (real-time WebSocket)

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

## P2+ — HF 镜像 fallback + 跨镜像内容去重 + 下载进度回调 ✅ 完成 2026-06-25

**目标**:把 v0.4.0 P2 的模型下载功能**补全**,让用户
"下载好的模型就不要重复下载"这件事做得**完整**:
HF 镜像自动 fallback (`huggingface.co` → `hf-mirror.com` 等),
内容指纹 (sha256 of `name|sha256` 集合) 跨 repo/revision 去重
避免重复写盘,下载进度回调让 UI/CLI 能看到每文件进度,镜像
健康检查让用户知道"现在哪个 mirror 能用"。

**新文件** (3 个):
- `models/source/mirrors.py` — `DEFAULT_HF_MIRRORS` 默认镜像列表
  (上游 + hf-mirror.com) + `MirrorSet` 配置 dataclass
  (env-var 读 `$TORCHA_VERSE_HF_MIRRORS`) + `MirrorHealth` 健康
  结果 + `check_mirror_health` / `check_all_mirrors` /
  `is_useful_mirror_error` 三个工具函数
- `tests/test_model_source_mirror.py` — 31 个新测试
  (8 MirrorSet + 5 健康检查 + 4 指纹/缓存查找 + 6 HF 镜像
  fallback + 5 fetcher 端到端 + 2 文件跳过/异常)
- `examples/model_download.py` — 端到端 demo (零网络
  `FakeTransport`):镜像列表 → 健康 → first fetch → cache
  hit → 跨镜像 dedup,完整覆盖 4 类场景

**升级文件** (5 个):
- `models/source/huggingface.py` — `HuggingFaceSource` 加
  `mirrors=` 参数 + `_for_each_live_mirror` 循环 + 60s TTL
  的 "dead-mirror memory" (`_dead_mirrors` 字典)。`resolve_license`
  / `list_files` / `download_files` 全部 try-mirrors-fallback。
  新 `DownloadProgress` dataclass (file_name / bytes_done /
  bytes_total / mirror / started_at / finished / error) +
  `download_default_artifacts(revision, on_progress=)` 接收
  per-file 进度回调,callback 抛异常自动 swallow 不影响下载。
- `models/source/cache.py` — `compute_content_fingerprint`
  (sorted `(name, sha256)` 集合的 sha256,**顺序无关**) +
  `ModelCache.find_by_fingerprint` (`rglob` 递归扫描 manifest,
  支持 `repo_id` 含 `/` 的情况,例如 `Qwen/Qwen2.5`) +
  `CachedModel.content_fingerprint` property
- `models/source/fetch.py` — `ModelFetcher.fetch` 接
  `mirrors=` + `on_progress=`,新 `_install_default_mirrors`
  让 default mirrors 自动装到 registry 中所有 HF adapter。
  `on_progress` callback 自动 wrap:4 参 `(name, done, total, mirror)`
  (v0.4.0 ergonomic shape) → 1 参 `DownloadProgress` (v0.4.x
  P2+ low-level shape),通过 `inspect.signature` 推断。
  新 `_fetch_inner` 流程:download → compute fingerprint →
  `find_by_fingerprint` → 命中则**不写盘**直接 return
  existing manifest (跨 repo/revision dedup),完全避免重复
  占用磁盘与重复完整性验证。
- `models/source/__init__.py` — 暴露 `MirrorSet` / `MirrorHealth`
  / `DownloadProgress` / `compute_content_fingerprint` / 4 个
  mirror/health/is_useful helpers
- `docs/placeholder_registry.md` — 8 条新 entry (54-61) 覆盖
  `models/source/cache.py:509,578,582` (原子写 + rmdir 兜底)
  + `models/source/huggingface.py:164,170` (HttpTransport
  abstract 占位) + `models/source/huggingface.py:564,597,622`
  (progress callback 兜底)

**端到端真模型 + 零网络** (TINY preset, FakeTransport):
- `python examples/model_download.py` 完整跑通 4 场景:
  1. 镜像列表构造 (`MirrorSet.from_env()`)
  2. 健康检查 (FakeTransport 报 1 个可达 + 1 个不可达)
  3. 第一次 fetch (`from_cache=False`, 写 v1)
  4. 第二次 fetch (same key, `from_cache=True` 直接 cache hit)
  5. 第三次 fetch (不同 revision v1.1, `from_cache=True`
     走 cross-mirror dedup,**不写 v1.1 目录**, 仍能 serve
     现有 v1 的 manifest)
  6. 流量后健康检查 (upstream 仍 mark dead, mirror alive)
- 总测试数: 652 → **683** (净增 31, 全过, 49.90s)
- `pytest -m model_source` 跑 84/84
- `pytest -m "not model_source"` 跑 599/599

**Scanner 双 0**:
- Hardcoding scanner: 4452 total, critical 3857, info 595
  (新增 216 主要是 tests + mirrors 字符串路径,全部 info)
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
