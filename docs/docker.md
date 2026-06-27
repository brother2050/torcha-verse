# Docker

> **最近更新**: 2026-06-27

## 镜像目标

`Dockerfile` 多 stage build,3 个 target:

| Target | Base | 用途 |
|---|---|---|
| `cpu` | `python:3.10-slim` + CPU PyTorch | CI / dev / smoke tests |
| `gpu` | `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` + GPU PyTorch | v1.0.0 production |
| `serving` | `cpu` + `python -m serving.app` | API on :8000 |

## Quick start (CPU)

```bash
docker build --target serving -t torcha-verse:0.10.2-serving .
docker compose up torcha-verse
curl -fsS http://localhost:8000/health
```

非 `serving` target 默认 `CMD []`,可作 `python -m ...` 的底包。

## GPU

```bash
docker build --target gpu -t torcha-verse:0.10.2-gpu .
docker compose --profile gpu up gpu
```

## 数据卷

| 路径 | 用途 |
|---|---|
| `/data` | 资产 / checkpoint (宿主机持久化) |
| `~/.cache/torcha-verse/` | 模型缓存 |
| `/tmp` | scratch |

## compose

```bash
docker compose up torcha-verse    # cpu serving
docker compose --profile gpu up gpu
```

## 健康检查

```dockerfile
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1
```

返回 `{status, node_types, version}`,生产环境配 LB 用。

## 镜像大小

- `cpu` stage: ~2.5 GB (python:3.10-slim + CPU torch)
- `serving` stage: ~2.6 GB (cpu + serving dep)
- `gpu` stage: ~6 GB (CUDA base + GPU torch)

CI 镜像固定到 sha256,不浮动升级。
