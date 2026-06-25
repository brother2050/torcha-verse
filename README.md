# TorchaVerse

TorchaVerse 是一个纯 PyTorch 实现的全模态生成式 AI 框架，覆盖文本、图像、视频、音频、数字人与多模态理解。它采用六层分层架构，通过可组合的节点与流水线编排复杂生成任务，并内置一致性框架保证跨镜头的角色/场景身份稳定。

**版本:0.3.1**

## 核心特性

| 能力 | 说明 |
|------|------|
| 文本生成 | 文本补全、多轮对话、流式输出,支持 Llama / Qwen / Mixtral 等模型 |
| 图像生成 | 文生图、图生图、放大、修复,支持 SD1.5 / SDXL / SD3 |
| 视频生成 | 文生视频、帧插值、视频拼接,支持 Wan2.2 等视频 DiT 模型 |
| 音频合成 | 语音合成 (TTS)、音乐生成,支持 CosyVoice / HiFi-GAN |
| 一致性生成 | 角色/服装/场景三引擎 + Depth 条件节点,保证跨镜头身份一致 |
| 多模态理解 | 文本、图像、音频统一输入,跨模态问答与理解 |

## 快速开始

```bash
pip install -e .
```

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

## 架构概览

TorchaVerse 采用六层分层架构,高层依赖低层,禁止反向依赖:

| 层级 | 名称 | 职责 |
|------|------|------|
| L1 | Infrastructure | 配置中心、设备管理、日志、审计、资源预算、缓存、限流、检查点、资源获取 |
| L2 | Assets | 统一资产模型与版本化存储(模型/角色/服装/场景/深度图) |
| L3 | Core | 模块装配总线、采样器、扩散调度器、内存池、分页 KV 缓存、运行时调度、工具注册 |
| L4 | Nodes | 29 个可组合能力节点(文本/图像/视频/音频/字幕/一致性/数字人/导出) |
| L5 | Pipeline | DAG、流水线构建器、连接校验、模板注册表、Prompt 工作室、可视化画布 |
| L6 | Consistency | 角色引擎、服装引擎、场景引擎、评分计算器、一致性流水线 |

## 目录结构

```
torcha-verse/
├── config/            # 配置文件(YAML)与系统默认值
├── infrastructure/   # L1: 配置、设备、日志、审计、资源预算等基础设施
├── assets/           # L2: 资产模型与版本化存储
├── core/             # L3: 模块总线、采样器、调度器、内存与缓存
├── nodes/            # L4: 29 个能力节点与类型系统
├── pipeline/         # L5: DAG、构建器、模板、校验器、Prompt 工作室
├── canvas/           # L5: 可视化画布、版本管理、分享、AutoDirector
├── consistency/      # L6: 一致性引擎与评分流水线
├── security/         # 横切层: 输入消毒、沙箱、输出过滤、供应链审计
├── plugins/          # 横切层: 插件发现与加载
├── models/           # 纯 PyTorch 模型实现(文本/图像/视频/音频/多模态)
├── rag/              # RAG 子系统(分块、加载、检索、向量存储)
├── agents/           # Agent 子系统(ReAct、工具调用、多智能体流)
├── tools/            # 内置工具(计算器、文件、Python 执行器、搜索)
├── training/         # 训练与微调(SFT、RLHF、合成数据)
├── serving/          # 应用层(API 服务、CLI、Web UI)
├── papers/           # 论文集成(注册表、适配器、YAML 配置)
├── performance/      # 性能优化(量化、编译、基准测试)
├── evaluation/       # 评估(文本/图像基准)
├── examples/         # 示例代码
├── tests/            # 测试套件
└── scripts/          # 工具脚本
```

## 配置

所有配置位于 `config/` 目录,由 `ConfigCenter` 按四级合并加载(System < Project < User < Run):

| 文件 | 用途 |
|------|------|
| `inference_config.yaml` | 推理参数:批处理、采样、KV 缓存、扩散、流式、内存 |
| `model_config.yaml` | 模型注册:默认设置与文本/图像/音频/视频模型清单 |
| `training_config.yaml` | 训练参数:优化器、学习率调度、训练循环、检查点、LoRA、日志 |
| `prompt_templates.yaml` | 提示词模板:系统提示、对话、RAG、Agent、图像 |

## 测试

```bash
python -m pytest tests/ -q
```

共 368 项测试,覆盖基础设施、节点、流水线、一致性、安全、插件与端到端集成。

## 文档

- [架构设计](docs/architecture.md) — 六层分层架构、节点系统、配置系统与安全设计
- [操作指南](docs/operations.md) — 安装、API 服务、CLI、Web UI、配置详解与故障排查

## Docker 部署

```bash
docker build -t torcha-verse .
docker run -p 8000:8000 torcha-verse
```

容器基于 `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`,默认以非 root 用户启动 API 服务,监听 8000 端口。

## 许可证

Apache-2.0
