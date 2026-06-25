# TorchaVerse Examples

> 11 个可运行示例,覆盖 v0.4.x 准生产化阶段的核心节点能力、模型源
> 子系统、RAG 链路与 agent 编排。每个示例都设计为单一 `python
> examples/<name>.py` 一行命令跑通,默认走项目自带的 echo / fake
> backend,**不需要网络、不需要 GPU、不需要预训练权重**。
>
> 详解(节点映射、依赖、真实模型路径、失败排查)见
> [`docs/examples_catalog.md`](../docs/examples_catalog.md)。

---

## 索引

| Example | 一句话 | 真实模型? | 节点层 | 关键依赖 |
|---|---|---|---|---|
| [`basic_text_gen.py`](basic_text_gen.py) | 1 节点 `text_chat` echo 演示 | 否(echo) | L4 | `nodes._helpers`, `pipeline.composer` |
| [`real_text_chat.py`](real_text_chat.py) | **真跑**项目自有 tiny Transformer LM | 是(`--preset tiny` ~2s) | L4 | `models.providers.local_text`, `pretrain_tiny` |
| [`image_gen.py`](image_gen.py) | `image_txt2img` + `image_upscale` 链路 | 是(UNet + VAE) | L4 | `models.providers.local_image` |
| [`video_gen.py`](video_gen.py) | `video_txt2vid` 视频生成 | 是(VideoDiT + VideoVAE) | L4 | `models.providers.local_video` |
| [`audio_tts.py`](audio_tts.py) | `audio_tts` 文本转语音 | 是(TTS-Transformer + HiFi-GAN) | L4 | `models.providers.local_audio` |
| [`consistency_character.py`](consistency_character.py) | `character_apply` + `character_five_view` 角色一致性 | 是(UNet + VAE) | L4 + L6 | `models.providers.local_image` |
| [`dh_lipsync.py`](dh_lipsync.py) | `dh_lip_sync` 数字人口型同步 | 是(VideoDiT + VideoVAE) | L4 + L6 | `models.providers.local_video` |
| [`agent_demo.py`](agent_demo.py) | ReAct agent + 工具调用 + 多 agent 编排 | 否 | L4 + agents | `tools.{calculator,python_executor,file_ops}` |
| [`rag_demo.py`](rag_demo.py) | 文档摄取 + chunking + 向量存储 + 检索 | 否 | rag/ | `rag.{loaders,chunkers,vectorstore,retrievers}` |
| [`model_download.py`](model_download.py) | 镜像 fallback + 跨镜像去重 + 进度回调 + 401/403 处理 | fake transport | models.source | `models.source.{cache,huggingface,civitai}`, `models.source.auth` |

---

## 三步跑通

```bash
# 1. 安装项目(开发模式)
pip install -e .

# 2. 跑任意一个 echo / fake 示例(不需要网络 / GPU)
python examples/basic_text_gen.py

# 3. 跑真模型示例(需要 ~30s CPU 训一个 tiny Transformer)
python examples/real_text_chat.py --preset tiny --skip-pretrain
```

> `real_text_chat.py` 默认会调 `train_tiny_transformer` 训一个 tiny
> preset;加 `--skip-pretrain` 跳过训练,直接用项目仓库内
> `assets/checkpoints/tiny-transformer-tiny.pt` 跑推理。

---

## 按"看哪部分代码"挑 example

| 你想了解... | 看这个 |
|---|---|
| PipelineBuilder / 单节点 | `basic_text_gen.py` |
| 真实模型端到端 + 字节级 tokenizer | `real_text_chat.py` |
| 多模态 provider 注册 / fallback | `image_gen.py` / `video_gen.py` / `audio_tts.py` |
| L6 一致性 + L4 节点联动 | `consistency_character.py` / `dh_lipsync.py` |
| Agent + 工具 + ReAct | `agent_demo.py` |
| RAG 4 步链路 | `rag_demo.py` |
| 模型源子系统(镜像 / 完整性 / token) | `model_download.py` |

---

## 离线 / 在线 / GPU

所有示例都设计为**默认离线**。它们的共同特征:

- 默认 backend 走 `nodes/_helpers.py` 的 echo / fake factory, **不**
  调任何真实模型;
- 真模型示例(`real_text_chat.py` / `image_gen.py` / `video_gen.py` /
  `audio_tts.py` / `consistency_character.py` / `dh_lipsync.py`) 走
  项目自带的 `LocalTorch{Text,Image,Video,Audio}Provider`, **CPU
  可跑** (tiny preset 在 CPU 上 < 5s);
- 唯一需要**网络**的示例是 `model_download.py`, 默认走 fake
  transport, 离线可跑; 加 `--real` 才会真发 HTTP 请求。

---

## 故障排查

| 现象 | 检查 |
|---|---|
| `ModuleNotFoundError: examples` | 在仓库根目录跑,不要 `cd examples/` |
| `Real model preset X not found` | 跑 `python examples/real_text_chat.py --list-presets` 查可用 preset |
| 跑真模型卡住 | 检查 `assets/checkpoints/` 是否已生成,加 `--skip-pretrain` 跳过训练 |
| `model_download.py --real` 报 401 | 设 `$HF_TOKEN` / `$HUGGING_FACE_HUB_TOKEN`;见 `docs/config_access.md` |

---

## CI smoke test

[`tests/test_examples_import.py`](../tests/test_examples_import.py) 对
所有 11 个示例做 import-clean + main() 存在 + 文档约定检查;每天
PR 都会跑。
