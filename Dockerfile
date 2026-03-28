# ---- GPU base image ----
# Swap this line depending on your GPU vendor:
#   NVIDIA:  nvidia/cuda:12.6.3-runtime-ubuntu24.04
#   AMD:     rocm/pytorch:latest
#   Intel:   intel/intel-extension-for-pytorch:latest
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    IMAGEGEN_CHECKPOINT=black-forest-labs/FLUX.1-schnell \
    IMAGEGEN_DEFAULT_STEPS=4 \
    IMAGEGEN_GPU_MEM=0.9

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.12 python3.12-venv curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY giraf_imagegen/ giraf_imagegen/

RUN uv venv && uv sync --no-dev --extra serve

EXPOSE 8300

CMD ["uv", "run", "imagegen-serve", "--port", "8300"]
