# TorchaVerse

纯 PyTorch 全模态生成式 AI 框架。39 个能力节点,九大域
(文本 / 图像 / 视频 / 音频 / 字幕 / 一致性 / 数字人 / 导出 / RAG),
端到端可跑、可生产部署。

## 安装

```bash
git clone <repo> torcha-verse
cd torcha-verse
pip install -e .
```

启动 CLI:

```bash
torcha --version        # 0.3.0
torcha info             # 框架元信息
torcha models           # 列出 39 个节点
```

## 快速开始

```python
from pipeline.composer import PipelineBuilder
from nodes.base import NodeContext

p = (PipelineBuilder("demo")
    .node("image_txt2img", id="img",
          prompt="a cat playing piano", width=512, height=512)
    .node("image_upscale", id="up", scale=2)
    .connect("img", "up", output_key="image", input_key="image")
    .build())

result = p.run(NodeContext())
```

启动 HTTP 服务 (FastAPI / OpenAI-兼容):

```bash
python -m uvicorn serving.app:create_app --factory --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models | jq .
```

无模型注册时,所有节点回退到内置的 echo 工厂,流水线仍可端到端跑通;
生产环境通过 `ModuleBus.register("model.text"|"model.image"|..., name, factory)`
注入真实模型即可,无需改业务代码。

## 架构

六层分层,高层依赖低层:

| 层 | 名称 | 关键模块 |
|----|------|---------|
| L1 | Infrastructure | 配置中心、设备、限流、检查点、审计、日志 |
| L2 | Assets | 三级存储 (hot/warm/cold) + 5 种资产类型 |
| L3 | Core | 模块总线、扩散调度器、工具注册、Pipeline 引擎 |
| L4 | Nodes | 39 个能力节点,统一 `BaseNode` + `NodeSpec` 契约 |
| L5 | Pipeline | DAG、构建器、模板、画布、Prompt 工作室 |
| L6 | Consistency | 角色/服装/场景引擎 + 评分计算器 |

横切层:Security (输入消毒、AST 沙箱、输出过滤) 与 Plugins。

## 节点清单 (39)

| 域 | 节点 |
|------|------|
| 文本 | `text_chat` / `text_complete` |
| 图像 | `image_txt2img` / `image_img2img` / `image_upscale` / `image_inpaint` |
| 视频 | `video_txt2vid` / `video_interpolate` / `video_stitch` |
| 音频 | `audio_tts` / `audio_music` |
| 字幕 | `subtitle_generate` / `subtitle_translate` / `subtitle_burn` / `subtitle_export` |
| 一致性 | `character_apply` / `outfit_apply` / `scene_apply` / `depth_condition` / `character_five_view` |
| 数字人 | `dh_lip_sync` / `dh_talking_head` / `dh_portrait_animate` / `dh_full_body` / `dh_face_enhance` / `dh_voice_clone` |
| 导出 | `export_image` / `export_video` / `export_audio` |
| RAG | `rag_ingest` / `rag_query` / `rag_delete` / `rag_list_indices` |
| Agent | `agent_react` / `agent_plan_solve` |

## 模型注册

```python
from core.module_bus import ModuleBus
from models.interfaces.llm_provider import LLMProvider, LLMResponse, LLMMessage

class MyLLM(LLMProvider):
    def chat(self, messages, **kw):
        return LLMResponse(text="hi", usage=...)

bus = ModuleBus()
bus.register("model.text", "my-llm", MyLLM())
```

更完整的模型下载/缓存/校验见 [`docs/operations.md`](docs/operations.md)。

## 测试

```bash
python -m pytest tests/ -q            # 1053 tests
python -m pytest tests/test_v03_asset_store.py -q  # 子集
```

## 文档

| 文档 | 内容 |
|------|------|
| [docs/architecture.md](docs/architecture.md) | 六层架构、模块依赖图、扩展点 |
| [docs/operations.md](docs/operations.md) | 部署、监控、checkpoint、模型下载 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | v0.4 ~ v0.6 路线图与当前进度 |
| [docs/DEFERRED_TASKS.md](docs/DEFERRED_TASKS.md) | 延后任务与原因 |
| [docs/open_items.md](docs/open_items.md) | 已知未处理项 |
| [docs/hardcoding_convention.md](docs/hardcoding_convention.md) | 硬编码扫描器 + 9 个可插拔规则 |
| [docs/placeholder_registry.md](docs/placeholder_registry.md) | 95 个 placeholder 行号登记 |
| [docs/config_access.md](docs/config_access.md) | 配置中心使用指南 |
| [docs/docker.md](docs/docker.md) | Docker 镜像构建 |

## 贡献流程

1. `git checkout -b feat/xxx`
2. 改代码 + 在 [`docs/placeholder_registry.md`](docs/placeholder_registry.md)
   登记新增的 `pass` / `except: pass`
3. `python -m pytest` 通过
4. 提交并推 PR

## 许可证

Apache-2.0
