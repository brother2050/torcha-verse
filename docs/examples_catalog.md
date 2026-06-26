# Examples Catalog

TorchaVerse 提供 11 个示例脚本,默认全部**离线 + echo/fake backend**;
真实模型 / 网络入口通过显式 flag 打开。

> **更新日期**: 2026-06-26 · 11 个示例,全部 import-clean
>
> 总览见 [`examples/README.md`](../examples/README.md),本文件按
> 1 个 example 1 个小节,信息深度更大。

---

## 1. `basic_text_gen.py`

**L4 节点 + PipelineBuilder 最小演示**。

```bash
python examples/basic_text_gen.py
```

- 节点: `text_chat`
- Backend: 默认 echo factory,零依赖
- 输出: `{node_id: {"text": ..., "usage": ...}}`

## 2. `real_text_chat.py` — v0.4.0 P0 headline

**真跑项目自有 tiny Transformer LM** (纯 torch,零外部依赖)。

```bash
python examples/real_text_chat.py --preset tiny --skip-pretrain
python examples/real_text_chat.py --preset small
```

- 节点: `text_chat`
- Backend: `LocalTorchTextProvider` (字节级 tokenizer)
- Preset: `tiny` (0.3M 参数, ~2s) / `small` (10M, ~30s CPU)
- 双语 prompt: English + Chinese

## 3. `image_gen.py` — UNet + VAE

**L4 `image_txt2img` 节点真模型前向** (64×64, ~3-5s CPU)。

```bash
python examples/image_gen.py
```

- 节点: `image_txt2img`
- Backend: `LocalTorchImageProvider` (UNet + VAE,纯 torch)
- 输出: `(3, H, W)` tensor + 完整 metadata
- 已知限制: `image_upscale` 链式路径(P0 类型检查)未完全适配,当前禁用

## 4. `video_gen.py` — VideoDiT + VideoVAE

**L4 `video_txt2vid` 真模型前向** (2-4 帧短视频)。

```bash
python examples/video_gen.py
```

- 节点: `video_txt2vid`
- Backend: `LocalTorchVideoProvider`
- 输出: `(T, 3, H, W)` tensor

## 5. `audio_tts.py` — TTS-Transformer + HiFi-GAN

**L4 `audio_tts` 真模型前向** (短 mel → waveform)。

```bash
python examples/audio_tts.py
```

- 节点: `audio_tts`
- Backend: `LocalTorchAudioProvider`
- 输出: `(samples,)` waveform tensor

## 6. `consistency_character.py` — L6 一致性

**角色化生图 + 五视图扩展**。

```bash
python examples/consistency_character.py
```

- 节点: `image_character_apply` / `image_five_view` / `image_txt2img`
- L6: `consistency.character` (profile + asset refs)
- Backend: `LocalTorchImageProvider`

## 7. `dh_lipsync.py` — 数字人口型同步

**driving audio 口型 reanimate 到目标 video**。

```bash
python examples/dh_lipsync.py
```

- 节点: `dh_lip_sync`
- Backend: `LocalTorchVideoProvider`

## 8. `agent_demo.py` — ReAct agent

**工具注册 → 执行 → ReAct 循环 → 多 agent 编排**。

```bash
python examples/agent_demo.py
```

- 工具: `CalculatorTool` / `PythonExecutorTool` / `FileOpsTool`
- 完全离线,无真实模型
- 模拟 ReAct `10! → sqrt(10!)` + Researcher → Writer → Reviewer 串接

## 9. `rag_demo.py` — RAG 4 步链路

**摄取 → chunking → 向量存储 → 检索 → context 组装**。

```bash
python examples/rag_demo.py
```

- 4 docs (项目自描述) → 4-6 chunks → 3 results
- Embeddings: `torch.randn` (非真模型)
- 零外部依赖

## 10. `model_download.py` — 模型源子系统 surface

**镜像 / 健康检查 / 进度 / 去重 / token / sha256 / 401/403**。

```bash
python examples/model_download.py             # 零网络 (FakeTransport)
python examples/model_download.py --real      # 真镜像 (需网络)
```

- 5 场景: 镜像构造 / 健康 / 进度 / 跨镜像去重 / 异常
- 已知 fail 模式: `--real` + 401 → 设 `$HF_TOKEN`;ChecksumMismatch → 镜像污染

## 11. `v05_feature_demo.py` — v0.5.x feature surface

**资产 / 一致性 / Pipeline 模板 / 输出过滤 等 v0.5 新功能的端到端**。

```bash
python examples/v05_feature_demo.py
```

- 覆盖: 资产 CRUD / 模板构建器 / Prompt Studio / 输出过滤 / RAG 摄取
- 默认 echo + 内存向量库

---

## 全局约定

### sys.path 注入

每个 example 顶部都有一行:
```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```
让 `python examples/<name>.py` 在任意 cwd 都能 import `nodes.*` /
`models.*`。CI 跑测试时是 `pip install -e .`,不依赖这行。

### `if __name__ == "__main__":` 守卫

所有 example 的入口都包了 `if __name__ == "__main__":`,保证
`import examples.<name>` 不会真跑 demo。
CI 守卫: `tests/test_examples_import.py::TestExamplesImport::test_module_has_main_guard`。

### 默认离线 / 默认 echo

所有 example 设计为**默认离线 + 默认 echo / fake backend**。
真模型 / 真网络入口都通过显式 flag 打开
(`--real` / `--preset small` / 不传 `--skip-pretrain` 等)。

### `Run with::` vs `Usage::`

`tests/test_examples_import.py` 两种 docstring 头都认。
新代码用 `Run with::`,老代码(例如 `real_text_chat.py`)沿用 `Usage::`。
