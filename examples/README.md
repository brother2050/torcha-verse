# TorchaVerse Examples

> **最近更新**: 2026-06-26 · 11 个示例,全部默认离线 / echo
>
> 详解(节点映射、依赖、真实模型路径、失败排查)见
> [`docs/examples_catalog.md`](../docs/examples_catalog.md)。

## 索引

| Example | 一句话 | 真模型? | 节点层 | 关键依赖 |
|---|---|---|---|---|
| [`basic_text_gen.py`](basic_text_gen.py) | 1 节点 `text_chat` echo | 否 | L4 | `nodes._helpers` / `pipeline.composer` |
| [`real_text_chat.py`](real_text_chat.py) | **真跑**项目自有 tiny Transformer LM | 是(`--preset tiny` ~2s) | L4 | `models.providers.local_text` / `pretrain_tiny` |
| [`image_gen.py`](image_gen.py) | `image_txt2img` + `image_upscale` 链路 | 是 (UNet + VAE) | L4 | `models.providers.local_image` |
| [`video_gen.py`](video_gen.py) | `video_txt2vid` 视频生成 | 是 (VideoDiT + VideoVAE) | L4 | `models.providers.local_video` |
| [`audio_tts.py`](audio_tts.py) | `audio_tts` 文本转语音 | 是 (TTS-Transformer + HiFi-GAN) | L4 | `models.providers.local_audio` |
| [`consistency_character.py`](consistency_character.py) | `character_apply` + `character_five_view` 角色一致性 | 是 | L4 + L6 | `models.providers.local_image` |
| [`dh_lipsync.py`](dh_lipsync.py) | `dh_lip_sync` 数字人口型同步 | 是 | L4 + L6 | `models.providers.local_video` |
| [`agent_demo.py`](agent_demo.py) | ReAct agent + 工具调用 + 多 agent 编排 | 否 | L4 + agents | `tools.{calculator,python_executor,file_ops}` |
| [`rag_demo.py`](rag_demo.py) | 文档摄取 + chunking + 向量存储 → 检索 | 否 | rag/ | `rag.{loaders,chunkers,vectorstore,retrievers}` |
| [`model_download.py`](model_download.py) | 镜像 fallback / 去重 / 进度 / 401/403 | fake transport | models.source | `models.source.{cache,huggingface,civitai}` |
| [`v05_feature_demo.py`](v05_feature_demo.py) | v0.5.x feature surface (资产 / 一致性 / 模板 / 过滤) | 否 | L2-L6 | 全栈 |

## 三步跑通

```bash
git clone <repo> torcha-verse
cd torcha-verse
pip install -e .
python examples/basic_text_gen.py        # 跑通即 OK
```

## 跑真模型

```bash
# 真实模型前向 (CPU OK, ~2-30s)
python examples/real_text_chat.py --preset tiny --skip-pretrain
python examples/image_gen.py
python examples/video_gen.py
python examples/audio_tts.py
python examples/consistency_character.py
python examples/dh_lipsync.py

# 网络 (真镜像 + 真下载)
python examples/model_download.py --real
```

## 全局约定

- **sys.path 注入**: 每个 example 顶部都有 `sys.path.insert(0, ...)`,任意 cwd 都能跑
- **`if __name__ == "__main__":` 守卫**: 全部 example 都包,`import examples.<name>` 不会跑 demo
- **默认离线 / 默认 echo**: 真模型 / 网络入口都通过显式 flag 打开
