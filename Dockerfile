FROM runpod/pytorch:2.2.2-py3.12-cuda12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf_cache

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache -r /tmp/requirements.txt

RUN git clone https://github.com/Theia-4869/VisPruner.git /opt/VisPruner && \
    uv pip install --system --no-cache -e /opt/VisPruner --no-deps

CMD ["/start.sh"]
