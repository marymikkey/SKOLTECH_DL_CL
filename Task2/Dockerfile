ARG BASE_IMAGE=nvidia/cuda:12.1.1-devel-ubuntu20.04
FROM ${BASE_IMAGE} AS base

ARG USE_CPU=false
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    tzdata \
    python3.9 python3.9-venv python3.9-dev python3-pip git \
    libglib2.0-0 \
    libgl1-mesa-glx \
 && ln -sf /usr/bin/python3.9 /usr/bin/python \
 && ln -sf /usr/bin/pip3 /usr/bin/pip \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app/Task2

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN if [ "$USE_CPU" = "true" ]; then \
        echo "Installing CPU version of PyTorch"; \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    elif [ "$(uname -m)" = "arm64" ]; then \
        echo "Apple Silicon detected – installing CPU version of PyTorch"; \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    else \
        echo "Using GPU (CUDA) version of PyTorch"; \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
    fi

CMD ["python3", "run.py"]