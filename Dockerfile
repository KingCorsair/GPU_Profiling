FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip git wget curl build-essential \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python

RUN pip install --no-cache-dir --break-system-packages uv

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache -r /tmp/requirements.txt

RUN git clone https://github.com/Theia-4869/VisPruner.git /opt/VisPruner && \
    uv pip install --system --no-cache -e /opt/VisPruner --no-deps

WORKDIR /workspace