# Docker

TorchaVerse ships a multi-stage `Dockerfile` and a `docker-compose.yml`
for both CPU and GPU deployments.

## Quick start (CPU)

```bash
docker build --target serving -t torcha-verse:0.4.1-serving .
docker compose up torcha-verse
curl -fsS http://localhost:8000/healthz
```

The image is multi-stage:

| Target | Base | Use case |
| --- | --- | --- |
| `cpu` | `python:3.10-slim` + CPU PyTorch | CI / dev / smoke tests |
| `gpu` | `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` | v1.0.0 production |
| `serving` | `cpu` + `python -m serving.app` | Long-running API on :8000 |

All non-`serving` targets default to a no-op CMD so they can be used
as a base for ad-hoc `python -m ...` invocations.

## GPU

```bash
docker build --target gpu -t torcha-verse:0.4.1-gpu .
docker compose --profile gpu up gpu
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
The container pins CUDA 12.1 and PyTorch 2.1.0+cu121 to match
`pyproject.toml`.

## Dev mode

```bash
docker compose --profile dev up dev
```

Bind-mounts the working directory into the container and runs
`serving.app --reload` for live reload. The cache volumes survive
restarts.

## Healthcheck

The `serving` image defines a `HEALTHCHECK` against `/healthz` (30s
interval, 5s timeout, 3 retries, 10s start period).  Make sure your
reverse proxy respects it.

## Production checklist

* Pin image by digest (`image: torcha-verse@sha256:...`).
* Mount the on-disk config tree read-only.
* Set `LOG_LEVEL=INFO` (or `WARNING`) and ship logs to a collector
  that understands JSON.
* Put the API behind a reverse proxy that terminates TLS.
* Set a real Prometheus scrape target (see `infrastructure/metrics.py`
  once the v1.0.0 M2b milestone is in).

## Troubleshooting

* **`torch.cuda.is_available()` returns False in the GPU image**: you
  are missing the NVIDIA runtime; install
  `nvidia-container-toolkit` and restart the Docker daemon.
* **Out-of-memory in dev mode**: the bind-mount does not change
  available RAM; pass `--shm-size=2g` to `docker run` if you rely
  on shared memory.
* **`/healthz` 404s**: the serving image requires the
  v0.4.x API server (`python -m serving.app`); older tags used
  `python -m serving.http_server` and may not expose that route.
