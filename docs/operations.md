# Operations

部署 / 监控 / checkpoint / 模型下载 操作指南。

> **最近更新**: 2026-06-26 · 1053 测试全过

## 目录

1. [启动 HTTP 服务](#启动-http-服务)
2. [配置中心与默认值](#配置中心与默认值)
3. [Checkpoint 备份与恢复](#checkpoint-备份与恢复)
4. [模型下载](#模型下载)
5. [运行端到端 example](#运行端到端-example)
6. [CLI 速查](#cli-速查)
7. [健康检查与日志](#健康检查与日志)

---

## 启动 HTTP 服务

```bash
# CPU 单机开发
python -m uvicorn serving.app:create_app --factory --host 0.0.0.0 --port 8000
```

| 端点 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查,返回 `{status, node_types, version}` |
| `/v1/models` | GET | 列出 39 个节点 |
| `/v1/text/completions` | POST | OpenAI 兼容 text completions |
| `/v1/text/chat` | POST | OpenAI 兼容 chat completions |
| `/v1/images/generate` | POST | 图像生成 |
| `/v1/videos/generate` | POST | 视频生成 |
| `/v1/audio/synthesize` | POST | TTS |
| `/v1/rag/ingest` | POST | RAG 摄取 |
| `/v1/rag/query` | POST | RAG 查询 |
| `/v1/agent/run` | POST | ReAct agent |
| `/docs` | GET | Swagger UI |

集成的中游件 (v0.4.x 加): Security (4 道关卡)、RateLimiter、AuditLogger、ResourceBudget。

---

## 配置中心与默认值

详见 [`config_access.md`](config_access.md)。要点:
- `infrastructure/defaults.py` 是推理默认值的**唯一数据源**。
- `ConfigCenter` 四级合并: System → Project → User → Run snapshot。
- 修改默认值要改 `inference_config.yaml` + `defaults.py` 同步。

```python
from infrastructure.defaults import (
    DIFFUSION_STEPS, DIFFUSION_GUIDANCE_SCALE,
    SAMPLING_TEMPERATURE, SAMPLING_TOP_K, SAMPLING_TOP_P,
)
```

---

## Checkpoint 备份与恢复

```python
from infrastructure.checkpoint_manager import CheckpointManager
cm = CheckpointManager(root="~/.local/share/torcha-verse/checkpoints")
cm.save("model-1", step=1000, model_state_dict=sd, optimizer_state_dict=osd)
ckpt = cm.load("model-1", step=1000)  # 返回完整 snapshot dict
```

`CheckpointManager` 子包 (R-3 拆): `_snapshot.py` (数据类) +
`_storage.py` (路径 / 索引) + `_atomic.py` (atomic write) +
`_manager.py` (主类) + `_retention.py` (k-of-n 保留)。

---

## 模型下载

```python
from models.source import fetch
result = fetch("Qwen/Qwen2.5-0.5B-Instruct")
print(result.cache_path)  # ~/.cache/torcha-verse/huggingface/Qwen/...
```

镜像加速:

```bash
export TORCHA_VERSE_HF_MIRRORS="https://hf-mirror.com"
export HF_TOKEN="hf_xxx"
```

Token 解析顺序: `token=` 参数 > `$HF_TOKEN` > `$HUGGING_FACE_HUB_TOKEN` > `~/.cache/huggingface/token`。

完整 API 见 [`models_source.md`](models_source.md)。

---

## CLI 速查

```bash
torcha --version
torcha info
torcha models                   # 39 个节点表
torcha text generate "hello" --max-tokens 10
torcha text chat "what is 2+2?" --system "you are a math tutor"
torcha image txt2img "a cat" --width 64 --height 64 --steps 1 --output /tmp/cat.png
torcha image img2img /tmp/cat.png "a red cat" --steps 1 --output /tmp/cat2.png
torcha video txt2vid "a cat" --frames 4 --output /tmp/v.mp4
torcha audio tts "hello" --output /tmp/a.wav
torcha agent run "what is 2+2?" --agent-type react
```

**所有命令必须从项目根目录执行** (ConfigCenter 按 CWD 找 `config/`);
v0.6.x 已加 CWD + sentinel files 兜底 (R-19 修),
从 `/tmp` / `~` 任意目录都跑通。

---

## 健康检查与日志

```bash
curl http://127.0.0.1:8000/health | jq .
```

```json
{
  "status": "healthy",
  "node_types": 39,
  "version": "0.3.0"
}
```

日志: `infrastructure.logger.get_logger(name)`;生产环境对接
`StdoutHandler` (默认) + 自由扩展 (文件 / 远程)。`R-17` 之后会
加 JSON 格式 + request-id (类似 OpenTelemetry trace id)。

---

## 故障排查

| 症状 | 解决 |
|---|---|
| `FileNotFoundError: /workspace/config/model_config.yaml` | v0.6.x 已加 CWD 兜底(R-19 修),从项目根跑 |
| `torcha models` 渲染 dict 报错 | v0.6.x 已改 `_info.py`,显示四列 |
| `image_txt2img` 报 `int() argument ... not 'dict'` | v0.6.x 已加 `_to_pil()`,支持 torch.Tensor / PIL / ndarray / dict |
| `subprocess tests` 报 `ModuleNotFoundError` | `_cli.py` 已在 v0.6.x 加 sys.path 注入,直接跑子进程 |
| 1053 测试出现 1 个 fail | 先看 `git log` 最近有没有 R-* commit,跑 `pytest --tb=long` |
