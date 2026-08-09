FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# System packages. deadsnakes provides Python 3.12 on Ubuntu 22.04,
# which only ships 3.10 by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    git wget unzip build-essential openssh-server cmake pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN pip install --no-cache-dir uv

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache -r /tmp/requirements.txt

RUN git clone https://github.com/Theia-4869/VisPruner.git /opt/VisPruner && \
    git -C /opt/VisPruner checkout aefa01adc7c7ce6334e880c88225e90cede760d1 && \
    uv pip install --system --no-cache -e /opt/VisPruner --no-deps

# SSH setup. RunPod injects your public key as $PUBLIC_KEY at runtime.
RUN mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    sed -i 's/#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

WORKDIR /workspace

# Write the injected key, start sshd, then stay alive.
CMD ["/bin/bash", "-c", "\
    if [ -n \"$PUBLIC_KEY\" ]; then \
        echo \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys; \
        chmod 600 /root/.ssh/authorized_keys; \
    fi; \
    service ssh start; \
    sleep infinity"]
