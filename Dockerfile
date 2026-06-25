FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for runtime security
RUN useradd -m appuser

# Copy source code
COPY . .

# Fix ownership so appuser can run pip install -e .
RUN chown -R appuser:appuser /app

# Switch to non-root user and install package
USER appuser
RUN pip install -e .

# Expose API port
EXPOSE 8000

# Default command
CMD ["python", "-m", "serving.api_server", "--host", "0.0.0.0", "--port", "8000"]
