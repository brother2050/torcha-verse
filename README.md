# TorchaVerse

纯 PyTorch 全模态生成式 AI 框架。初期项目,关注**简洁架构 + 端到端可跑**。

| 能力 | 节点 |
|------|------|
| 文本 | `text_chat` / `text_complete` |
| 图像 | `image_txt2img` / `image_img2img` / `image_upscale` / `image_inpaint` |
| 视频 | `video_txt2vid` / `video_interpolate` / `video_stitch` |
| 音频 | `audio_tts` / `audio_music` |
| 字幕 | `subtitle_generate` / `subtitle_translate` / `subtitle_burn` / `subtitle_export` |
| 一致性 | `character_apply` / `outfit_apply` / `scene_apply` / `depth_condition` / `character_five_view` |
| 数字人 | `dh_lip_sync` / `dh_talking_head` / `dh_portrait_animate` / `dh_full_body` / `dh_face_enhance` / `dh_voice_clone` |
| 导出 | `export_image` / `export_video` / `export_audio` |

## 安装

```bash
pip install -e .
```

## 快速开始

```python
from pipeline.composer import PipelineBuilder
from nodes.base import NodeContext

p = (PipelineBuilder("demo")
    .node("image_txt2img", id="img", prompt="a cat playing piano", width=512, height=512)
    .node("image_upscale", id="up", scale=2)
    .connect("img", "up", output_key="image", input_key="image")
    .build())

result = p.run(NodeContext())
```

## 架构

六层分层,高层依赖低层:

| 层 | 名称 | 内容 |
|----|------|------|
| L1 | Infrastructure | 配置中心、设备、日志、审计、限流、检查点 |
| L2 | Assets | 资产模型与版本化存储 |
| L3 | Core | 模块总线、扩散调度器、工具注册 |
| L4 | Nodes | 29 个能力节点(文本/图像/视频/音频/字幕/一致性/数字人/导出) |
| L5 | Pipeline | DAG、构建器、模板、画布、Prompt 工作室 |
| L6 | Consistency | 角色/服装/场景引擎 + 评分计算器 |

横切层:Security(输入消毒、AST 沙箱、输出过滤)与 Plugins。

## 模型后端

节点 `execute()` 统一通过 `nodes/_helpers.py` 中的 `call_*_backend` 解析
后端。注册模型示例:

```python
from core.module_bus import ModuleBus
from models.interfaces.llm_provider import LLMProvider, LLMMessage

class MyLLM(LLMProvider):
    def chat(self, messages, **kw):
        return LLMResponse(text="hi", usage=...)

bus = ModuleBus()
bus.register("model.text", "my-llm", MyLLM())
```

无模型注册时,节点会回退到内置的 echo 工厂,流水线仍可端到端跑通。

## 测试

```bash
python -m pytest tests/ -q
```

## 文档

- [架构设计](docs/architecture.md)
- [操作指南](docs/operations.md)

## 许可证

Apache-2.0
