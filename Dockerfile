# =============================================================================
# TorchaVerse — multi-stage Dockerfile
#   builder : install build deps + project (-e)
#   test    : run pytest on a copy of the source (CI use)
#   runtime : slim image, non-root, default command = serve
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: builder
# -----------------------------------------------------------------------------
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .


# -----------------------------------------------------------------------------
# Stage 2: test (uses builder artefacts but runs as a verification step)
# -----------------------------------------------------------------------------
FROM builder AS test

RUN pip install --no-cache-dir pytest pytest-cov

# Default to a no-network / offline test run; CI overrides the marker filter.
RUN python -m pytest tests/ -q --no-header -m "not gpu"


# -----------------------------------------------------------------------------
# Stage 3: runtime
# -----------------------------------------------------------------------------
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY --from=builder /app /app
RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

CMD ["python", "-m", "serving.app", "--host", "0.0.0.0", "--port", "8000"]
