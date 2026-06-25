# 操作指南

本指南面向新用户,介绍 TorchaVerse 的安装、启动 API 服务、使用 CLI、使用 Web UI、配置文件详解、Docker 部署、开发指南与故障排查。

## 安装

### 基本安装

```bash
git clone <仓库地址> torcha-verse
cd torcha-verse
pip install -e .
```

`pip install -e .` 以可编辑模式安装,安装后会注册 `torcha` 命令行工具。

### 依赖说明

核心依赖(随安装自动拉取):

| 依赖 | 用途 |
|------|------|
| `torch` / `torchvision` / `torchaudio` | 深度学习核心 |
| `PyYAML` | 配置文件解析 |
| `numpy` / `Pillow` | 数值计算与图像处理 |
| `fastapi` / `uvicorn` / `pydantic` | API 服务 |
| `click` / `rich` | CLI 命令与美化输出 |
| `gradio` | Web UI |
| `faiss-cpu` | RAG 向量检索 |
| `safetensors` | 模型权重安全存储 |

可选依赖:

```bash
# 量化推理(INT4/INT8/NF4)
pip install -e ".[quantization]"

# 开发与测试
pip install -e ".[dev]"
```

### 环境要求

- Python >= 3.9
- PyTorch >= 2.1.0
- 支持 CPU / CUDA / MPS 设备(自动检测)

---

## 启动 API 服务

### 启动命令

```bash
python -m serving.app
```

默认监听 `127.0.0.1:8000`。可通过参数自定义:

```bash
python -m serving.app --host 0.0.0.0 --port 8000 --reload
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | 绑定地址 |
| `--port` | `8000` | 监听端口 |
| `--reload` | 关闭 | 开发模式自动重载 |

启动后访问 `http://127.0.0.1:8000/docs` 查看 Swagger 文档。

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查,返回状态、版本、设备、节点数 |
| GET | `/metrics` | Prometheus 格式指标 |
| GET | `/v1/models` | 列出所有已注册节点类型 |
| POST | `/v1/text/completions` | 文本补全(支持流式) |
| POST | `/v1/text/chat` | 多轮对话(支持流式) |
| POST | `/v1/images/generate` | 文生图 |
| POST | `/v1/audio/synthesize` | 语音合成 |
| POST | `/v1/videos/generate` | 文生视频 |
| POST | `/v1/multimodal/understand` | 多模态理解 |
| POST | `/v1/rag/query` | RAG 检索增强问答 |
| POST | `/v1/agent/run` | Agent 执行(支持流式) |

### 调用示例

```bash
curl -X POST http://127.0.0.1:8000/v1/text/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 128}'
```

流式输出:

```bash
curl -X POST http://127.0.0.1:8000/v1/text/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "stream": true}'
```

---

## 使用 CLI

安装后可直接使用 `torcha` 命令:

```bash
torcha --help
```

### 子命令列表

| 子命令 | 说明 |
|--------|------|
| `torcha text generate` | 文本补全 |
| `torcha text chat` | 交互式多轮对话 |
| `torcha image txt2img` | 文生图 |
| `torcha image img2img` | 图生图 |
| `torcha audio tts` | 语音合成 |
| `torcha video txt2vid` | 文生视频 |
| `torcha rag ingest` | RAG 文档导入 |
| `torcha rag query` | RAG 问答 |
| `torcha agent run` | Agent 执行 |
| `torcha plugin ...` | 插件管理 |
| `torcha info` | 显示框架与设备信息 |
| `torcha models` | 列出已注册节点类型 |

### 使用示例

文本生成:

```bash
torcha text generate --model llama-8b --prompt "写一首关于海洋的诗" --stream
```

图像生成:

```bash
torcha image txt2img --model sd15 --prompt "a cat playing piano" --width 512 --height 512 --output out.png
```

图生图:

```bash
torcha image img2img --input photo.png --prompt "cinematic lighting" --strength 0.3 --output out.png
```

语音合成:

```bash
torcha audio tts --model cosyvoice --text "你好,世界" --output out.wav
```

视频生成:

```bash
torcha video txt2vid --model wan2.2 --prompt "sunset over the ocean" --num-frames 16 --fps 8 --output out.gif
```

交互式对话:

```bash
torcha text chat --model llama-8b --system "你是一个有用的助手"
```

查看框架信息:

```bash
torcha info
```

---

## 使用 Web UI

### 启动命令

```bash
python -m serving.web_ui
```

默认监听 `0.0.0.0:7860`。启动后浏览器访问 `http://127.0.0.1:7860`。

### 界面说明

Web UI 基于 Gradio 构建,提供六个标签页:

| 标签页 | 功能 |
|--------|------|
| Multimodal Chat | 多模态统一对话,支持文本、图像、音频输入 |
| Image Studio | 图像生成工作室,可调节宽高、步数、引导强度、种子 |
| Video Studio | 视频生成工作室,可调节帧数、帧率、步数 |
| RAG Manager | RAG 文档管理,上传文档、构建知识库、查询 |
| Agent Playground | Agent 演练场,运行单/多 Agent 并可视化推理链 |
| Workflow Orchestrator | 工作流编排器,以 JSON 定义节点图并执行 |

Image Studio 参数控制:

- **Width / Height**:256-1024 像素,步进 64
- **Steps**:1-100 步
- **Guidance Scale**:1.0-20.0
- **Seed**:-1 表示随机

---

## 配置文件详解

所有配置位于 `config/` 目录,由 `ConfigCenter` 按四级合并加载(System < Project < User < Run)。

### inference_config.yaml

推理参数配置:

```yaml
# 批处理
batch:
  max_batch_size: 32              # 最大批大小
  enable_continuous_batching: true # 连续批处理
  max_waiting_time: 0.1           # 部分批次刷新等待秒数

# 采样策略
sampling:
  default:                        # 默认策略
    temperature: 0.7
    top_k: 50
    top_p: 0.9
    repetition_penalty: 1.1
  creative:                       # 创意策略(高温度)
    temperature: 0.9
    top_k: 100
    top_p: 0.95
    repetition_penalty: 1.05
  precise:                        # 精确策略(低温度)
    temperature: 0.1
    top_k: 10
    top_p: 0.5
    repetition_penalty: 1.2

# KV 缓存
kv_cache:
  enabled: true
  strategy: "paged"               # static | paged
  page_size: 16
  max_pages: 1024
  cpu_offload: true
  offload_threshold: 0.85        # GPU 内存使用阈值触发卸载

# 扩散
diffusion:
  default_steps: 30
  default_guidance_scale: 7.5
  scheduler: "dpm_solver"         # ddpm | ddim | euler | dpm_solver | consistency
  eta: 0.0

# 流式
streaming:
  enabled: true
  chunk_size: 4                  # 文本每块 token 数
  frame_batch: 1                 # 视频每块帧数

# 内存管理
memory:
  auto_offload: true
  gpu_memory_fraction: 0.9
  enable_peak_prediction: true
```

| 配置项 | 说明 |
|--------|------|
| `batch` | 批处理大小、连续批处理、等待时间 |
| `sampling` | 三种采样策略:default / creative / precise |
| `kv_cache` | KV 缓存策略、页大小、CPU 卸载阈值 |
| `diffusion` | 扩散步数、引导强度、调度器类型 |
| `streaming` | 流式输出的块大小 |
| `memory` | 自动卸载、GPU 内存占比、峰值预测 |

### model_config.yaml

模型注册配置:

```yaml
# 默认设置
default:
  dtype: "bf16"                  # bf16 | fp16 | fp32
  device: "auto"                 # auto | cuda | cpu | mps
  quantization: null             # null | int8 | int4 | gptq | awq
  max_seq_len: 4096
  trust_remote_code: true

# 文本模型
text_models:
  llama-8b:                     # Llama-3-8B,decoder_only,GQA
    ...
  qwen2-7b:                     # Qwen2-7B,32K 上下文
    ...
  mixtral-8x7b:                 # Mixtral-8x7B,MoE 架构
    ...

# 图像模型
image_models:
  sd15:                         # Stable-Diffusion-1.5,512x512
    ...
  sdxl:                         # Stable-Diffusion-XL,1024x1024
    ...
  sd3:                          # Stable-Diffusion-3,DiT 架构
    ...

# 音频模型
audio_models:
  cosyvoice:                    # CosyVoice TTS,24kHz
    ...
  hifi-gan:                     # HiFi-GAN 声码器,22kHz
    ...

# 视频模型
video_models:
  wan2-2:                       # Wan2.2,视频 DiT,81 帧
    ...
```

| 配置项 | 说明 |
|--------|------|
| `default` | 全局默认:精度、设备、量化、最大序列长度 |
| `text_models` | 文本模型清单(Llama / Qwen / Mixtral),含层数、头数、隐藏维度 |
| `image_models` | 图像模型清单(SD1.5 / SDXL / SD3),含潜在通道、图像尺寸、VAE |
| `audio_models` | 音频模型清单(CosyVoice / HiFi-GAN),含采样率 |
| `video_models` | 视频模型清单(Wan2.2),含潜在通道、帧数、帧率 |

### training_config.yaml

训练参数配置:

```yaml
# 优化器
optimizer:
  type: "adamw"                  # adamw | sgd | lion
  lr: 2.0e-5
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1.0e-8

# 学习率调度
lr_scheduler:
  type: "cosine"                 # cosine | linear | constant | cosine_with_restarts
  warmup_steps: 100
  warmup_ratio: 0.03
  min_lr: 1.0e-6

# 训练循环
training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  mixed_precision: "bf16"       # no | fp16 | bf16
  seed: 42
  num_workers: 4
  pin_memory: true

# 检查点
checkpoint:
  save_dir: "data/checkpoints"
  save_steps: 500
  save_total_limit: 3
  save_only_weights: false       # false = 同时保存优化器状态
  resume_from: null

# LoRA / QLoRA
lora:
  enabled: false
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  qlora: false                  # 4-bit 量化

# RLHF
rlhf:
  method: "dpo"                 # ppo | dpo | grpo
  beta: 0.1
  clip_range: 0.2
  reward_model: null

# 日志
logging:
  log_steps: 10
  eval_strategy: "steps"
  save_strategy: "steps"
  report_to: ["console"]        # console | wandb | tensorboard
```

| 配置项 | 说明 |
|--------|------|
| `optimizer` | 优化器类型、学习率、权重衰减、beta |
| `lr_scheduler` | 调度器类型、预热步数/比例、最小学习率 |
| `training` | 训练轮数、批大小、梯度累积、混合精度、随机种子 |
| `checkpoint` | 保存目录、保存步数、保留数量、恢复路径 |
| `lora` | LoRA 秩、alpha、dropout、目标模块、QLoRA 开关 |
| `rlhf` | RLHF 方法(PPO/DPO/GRPO)、beta、clip 范围 |
| `logging` | 日志步数、评估策略、报告后端 |

### prompt_templates.yaml

提示词模板配置:

```yaml
# 系统提示
system:
  default: |                     # 通用助手
    You are a helpful, harmless, and honest AI assistant...
  coding: |                      # 编程专家
    You are an expert software engineer...
  creative: |                    # 创意写作
    You are a creative writing assistant...
  analysis: |                    # 分析助手
    You are an analytical assistant...

# 对话模板
chat:
  default: |                     # 通用格式
  llama3: |                       # Llama3 格式
  qwen: |                         # Qwen 格式

# RAG 模板
rag:
  query: |                       # 检索问答
  follow_up: |                    # 追问

# Agent 模板
agent:
  react: |                        # ReAct 框架
  planning: |                     # 任务分解

# 图像提示
image:
  enhance: |                      # 质量增强后缀
  negative: |                     # 负面提示
```

| 配置项 | 说明 |
|--------|------|
| `system` | 系统提示词:default / coding / creative / analysis |
| `chat` | 对话模板:default / llama3 / qwen 格式 |
| `rag` | RAG 模板:检索问答与追问 |
| `agent` | Agent 模板:ReAct 推理与任务规划 |
| `image` | 图像提示:质量增强后缀与负面提示 |

---

## Docker 部署

### 构建镜像

```bash
docker build -t torcha-verse .
```

镜像基于 `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`,预装 ffmpeg、libsndfile 等系统依赖。

### 运行容器

```bash
docker run -d \
  --name torcha-verse \
  -p 8000:8000 \
  --gpus all \
  torcha-verse
```

| 参数 | 说明 |
|------|------|
| `-p 8000:8000` | 映射 API 端口 |
| `--gpus all` | 使用所有 GPU(可选) |
| `-d` | 后台运行 |

容器以非 root 用户 `appuser` 运行,默认启动 API 服务监听 8000 端口。

### 环境变量

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `TORCHA_CORS_ORIGINS` | `*` | CORS 允许的源(逗号分隔),生产环境应配置具体域名 |
| `TORCHAVERSE_CONFIG_DIR` | 自动检测 | Project 层配置目录覆盖 |
| `TORCHAVERSE_SYSTEM_CONFIG_DIR` | `config/_defaults/` | System 层默认值目录覆盖 |
| `TORCHAVERSE_USER_CONFIG_DIR` | `~/.config/torcha-verse/` | User 层配置目录覆盖 |
| `TORCHAVERSE_RUN_DIR` | `~/.local/share/torcha-verse/` | 运行快照目录覆盖 |

生产环境示例:

```bash
docker run -d \
  -p 8000:8000 \
  -e TORCHA_CORS_ORIGINS="https://app.example.com" \
  torcha-verse
```

---

## 开发指南

### 项目结构

```
torcha-verse/
├── config/            # 配置文件
├── infrastructure/   # L1 基基础设施
├── assets/           # L2 资产层
├── core/             # L3 核心层
├── nodes/            # L4 节点层
├── pipeline/         # L5 流水线层
├── canvas/           # L5 画布层
├── consistency/      # L6 一致性层
├── security/         # 安全横切层
├── plugins/          # 插件横切层
├── models/           # PyTorch 模型实现
├── rag/              # RAG 子系统
├── agents/           # Agent 子系统
├── tools/            # 内置工具
├── training/         # 训练与微调
├── serving/          # 应用层(API/CLI/Web UI)
├── papers/           # 论文集成
├── performance/      # 性能优化
├── evaluation/       # 评估
├── examples/         # 示例
├── tests/            # 测试
└── scripts/          # 工具脚本
```

分层依赖规则:高层依赖低层,禁止反向依赖。详见 [架构设计](architecture.md)。

### 添加新节点

1. 在 `nodes/` 对应类别的模块中定义节点类:

```python
# nodes/text.py
from nodes.base import BaseNode, NodeSpec, register_node

@register_node("text_summarize")
class TextSummarizeNode(BaseNode):
    spec = NodeSpec(
        type="text_summarize",
        name="Text Summarize",
        description="Summarize the input text.",
        inputs={"text": "TEXT", "max_length": "Optional[INT]"},
        outputs={"summary": "TEXT"},
        tags=["text", "summarize"],
    )

    def execute(self, ctx, **inputs):
        text = inputs["text"]
        # 实现摘要逻辑
        return {"summary": summarized_text}
```

2. 在 `nodes/__init__.py` 中导入新类(若为新模块)。

3. 导入 `nodes` 包后,节点自动注册到 ModuleBus,可通过 `NodeRegistry` 发现。

### 添加新模板

1. 创建 YAML 模板文件,或直接在 `pipeline/templates.py` 的 `BUILTIN_TEMPLATES` 中添加:

```python
TemplateRegistry().register(PipelineTemplate(
    name="my_workflow",
    description="My custom workflow",
    category="image",
    dag_dict={
        "nodes": [
            {"id": "n1", "type": "image_txt2img", "params": {...}},
            {"id": "n2", "type": "image_upscale", "params": {...}},
        ],
        "edges": [
            {"from": "n1", "to": "n2", "output_key": "image", "input_key": "image"},
        ],
    },
    default_params={},
    tags=["image", "custom"],
))
```

2. 也可将 YAML 文件放入模板目录,通过 `TemplateRegistry.load_directory()` 加载。

### 运行测试

```bash
# 运行全部测试
python -m pytest tests/ -q

# 运行特定测试文件
python -m pytest tests/test_v03_nodes.py -v

# 运行端到端测试
python -m pytest tests/test_e2e_*.py -v

# 运行并显示简短回溯
python -m pytest tests/ -q --tb=short
```

共 368 项测试,覆盖基础设施、节点、流水线、一致性、安全、插件与端到端集成。

---

## 故障排查

### 常见问题

**Q: 安装后 `torcha` 命令找不到?**

确认以可编辑模式安装并激活了对应 Python 环境:

```bash
pip install -e .
which torcha
```

**Q: 启动 API 服务报 `ModuleNotFoundError: No module named 'fastapi'`?**

安装 serving 依赖:

```bash
pip install fastapi uvicorn pydantic
```

或重新执行 `pip install -e .`。

**Q: 启动 Web UI 报 `ImportError: gradio is required`?**

安装 Web UI 依赖:

```bash
pip install gradio
```

**Q: GPU 不可用,只能用 CPU?**

检查 PyTorch CUDA 安装:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

若输出 `False`,需安装对应 CUDA 版本的 PyTorch。`DeviceManager` 会自动回退到 CPU。

**Q: 图像/视频节点返回占位数据,未生成真实文件?**

节点系统在未配置真实模型后端时返回占位数据。这是预期行为,用于在不下载模型权重的情况下演练流水线编排。要生成真实输出,需在 `model_config.yaml` 中配置模型路径并下载对应权重。

**Q: 配置修改后未生效?**

`ConfigCenter` 是单例,进程内缓存。修改 YAML 后需重启进程。也可通过环境变量覆盖配置目录:

```bash
export TORCHAVERSE_CONFIG_DIR=/path/to/config
```

**Q: 提示词被拒绝,返回 `Input rejected` 或 `Prompt injection detected`?**

输入消毒关卡拦截了潜在恶意输入。检查输入是否包含控制字符、路径遍历模式或提示词注入语句。如为误报,可调整 `InputSanitizer` 的检测规则。

**Q: 输出被过滤,返回 `Output filtered`?**

输出过滤关卡拦截了不合规内容。这是安全设计的一部分。如需调整,可配置 `OutputFilter` 的过滤策略。

**Q: Docker 容器无法访问 GPU?**

确保安装了 NVIDIA Container Toolkit,并使用 `--gpus all` 参数:

```bash
docker run --gpus all -p 8000:8000 torcha-verse
```

**Q: 如何查看已注册的节点类型?**

```bash
torcha models          # CLI 方式
curl http://127.0.0.1:8000/v1/models  # API 方式
```

**Q: 如何重放之前的运行配置?**

`ConfigCenter` 每次运行会生成 `config_snapshot.json`。通过以下方式重放:

```python
from infrastructure.config_center import ConfigCenter

cc = ConfigCenter()
cc.load_run_snapshot("/path/to/config_snapshot.json")
```
