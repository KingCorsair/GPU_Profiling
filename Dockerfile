FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    git wget build-essential openssh-server cmake pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN pip install --no-cache-dir uv

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache -r /tmp/requirements.txt

RUN git clone https://github.com/Theia-4869/VisPruner.git /opt/VisPruner && \
    uv pip install --system --no-cache -e /opt/VisPruner --no-deps

WORKDIR /workspace