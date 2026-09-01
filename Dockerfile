# syntax=docker/dockerfile:1
# CPU image for cv-car-counter. Bind-mount videos, outputs, configs, and
# weights at the paths below (see docker-compose.yml).
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs /configs

# Install CPU torch first so the detector extra does not pull a CUDA wheel.
# Override at build time: --build-arg EXTRAS=rfdtr
ARG TORCH_BACKEND=cpu
ARG EXTRAS=yolo
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$TORCH_BACKEND" = "cpu" ]; then \
        pip install --no-cache-dir torch torchvision \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi && \
    pip install --no-cache-dir ".[${EXTRAS}]"

# Persist Hugging Face / torch / Ultralytics caches via the /models mount.
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    YOLO_CONFIG_DIR=/models/ultralytics

WORKDIR /work

ENTRYPOINT ["cv-car-counter"]
CMD ["--help"]
