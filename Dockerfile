# TorchaVerse Dockerfile
# Multi-stage build for v0.4.x — production image for the v0.4.1 release.
#
# Stages:
#   1. ``base``     - system deps + Python + PyTorch CPU runtime.
#   2. ``runtime``  - project install; reused by both CPU and GPU targets.
#   3. ``cpu``      - default CPU-only image (small, suitable for CI / dev).
#   4. ``gpu``      - CUDA-enabled image for v1.0.0 production.
#   5. ``serving``  - API server entry-point (consumes either cpu / gpu).
#
# Build examples:
#   docker build --target cpu     -t torcha-verse:0.4.1-cpu     .
#   docker build --target gpu     -t torcha-verse:0.4.1-gpu     .
#   docker build --target serving -t torcha-verse:0.4.1-serving .
#
# All build targets are runnable; ``serving`` is the only one that
# starts a long-running process by default (the API server on
# :8000).  The other targets are designed to be used as a base for
# ad-hoc ``python -m ...`` invocations.

# ---------------------------------------------------------------------------
# Stage 1: base (system + Python + PyTorch CPU)
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/var/cache/torch \
    HF_HOME=/var/cache/huggingface

# System deps: git (for plugins), curl (for healthcheck), ffmpeg (for
# video examples), and a couple of build essentials that some optional
# dependencies need (numpy / pillow already prebuilt for py310 wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for the runtime images.
RUN groupadd --system torcha && useradd --system --gid torcha --uid 10001 torcha \
    && mkdir -p /app /var/cache/torch /var/cache/huggingface \
    && chown -R torcha:torcha /app /var/cache/torch /var/cache/huggingface

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 2: runtime (project install)
# ---------------------------------------------------------------------------
FROM base AS runtime

# Install Python deps first so we can cache the layer when only the
# project source has changed.
COPY requirements.txt pyproject.toml README.md ./
COPY agents ./agents
COPY assets ./assets
COPY canvas ./canvas
# Copy *source* directories so the editable install works in CI.  The
# pyproject ``[tool.setuptools]`` packages are picked up at install
# time; tests are intentionally excluded from the wheel.
COPY config ./config
COPY consistency ./consistency
COPY core ./core
COPY infrastructure ./infrastructure
COPY models ./models
COPY nodes ./nodes
COPY pipeline ./pipeline
COPY plugins ./plugins
COPY rag ./rag
COPY security ./security
COPY serving ./serving
COPY tools ./tools
COPY training ./training
COPY evaluation ./evaluation

USER torcha

# ``pip install --no-deps`` first, then a separate deps install.  This
# way the second step can re-use the build cache even if the project
# source is touched.
RUN pip install --upgrade pip wheel \
    && pip install -e . \
    && pip install rich fastapi uvicorn[standard] pydantic

# ---------------------------------------------------------------------------
# Stage 3: cpu (default, small, no CUDA)
# ---------------------------------------------------------------------------
FROM runtime AS cpu

USER torcha

# The default container CMD is a no-op shell so users can run
# arbitrary ``python -m ...`` commands.  Use ``--target serving`` to
# get a long-running API.
CMD ["python", "-c", "import torcha_verse; print(torcha_verse.__version__)"]

# ---------------------------------------------------------------------------
# Stage 4: gpu (CUDA, for v1.0.0 production; pinned CUDA 12.1)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/var/cache/torch \
    HF_HOME=/var/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git curl ffmpeg libsndfile1 ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY agents assets canvas config consistency core infrastructure \
     models nodes pipeline plugins rag security serving tools training evaluation ./

RUN groupadd --system torcha && useradd --system --gid torcha --uid 10001 torcha \
    && mkdir -p /var/cache/torch /var/cache/huggingface \
    && chown -R torcha:torcha /app /var/cache/torch /var/cache/huggingface

USER torcha

# CUDA wheels from the PyTorch index; pinned to match ``pyproject.toml``.
RUN pip install --upgrade pip wheel \
    && pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -e . \
    && pip install rich fastapi uvicorn[standard] pydantic

CMD ["python", "-c", "import torcha_verse; print(torcha_verse.__version__)"]

# ---------------------------------------------------------------------------
# Stage 5: serving (long-running API on :8000)
# ---------------------------------------------------------------------------
FROM cpu AS serving

USER torcha

EXPOSE 8000

# ``--reload`` is intentionally off in the container; the entrypoint
# wires through ``python -m serving.app`` so the same image can be
# reused for dev (override command) or prod (default).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["python", "-m", "serving.app", "--host", "0.0.0.0", "--port", "8000"]
