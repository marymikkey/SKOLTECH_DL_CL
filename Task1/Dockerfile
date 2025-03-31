ARG BASE_IMAGE=nvidia/cuda:12.1.1-devel-ubuntu20.04
FROM ${BASE_IMAGE} AS base

ARG USE_CPU=false
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    tzdata \
    python3.9 python3.9-venv python3.9-dev python3-pip git && \
    ln -s /usr/bin/python3.9 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* || true

RUN apt-get update && apt-get install -y --fix-missing \
    tzdata \
    python3.9 python3.9-venv python3.9-dev python3-pip git && \
    ln -s /usr/bin/python3.9 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* || true

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN if [ "$USE_CPU" = "true" ]; then \
        echo "Устанавливаем CPU-версию PyTorch"; \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    elif [ "$(uname -m)" = "arm64" ]; then \
        echo "Apple Silicon M1/M2/M3 – ставим CPU-версию"; \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    else \
        echo "Используем GPU (CUDA)"; \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
    fi

CMD ["python3", "Task1_Roads_YOLO_Mary.py"]