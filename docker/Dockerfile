# Match CUDA / TensorRT versions to your deployment target.
# Example base with PyTorch preinstalled; swap for a TensorRT SDK image if needed.
FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip && pip install --no-cache-dir .

# Default image command: lightweight health-style check (override in compose/k8s).
ENTRYPOINT ["inference-engine"]
CMD ["info"]
