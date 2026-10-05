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

# Node.js 20 for the measurement service and TypeScript load generator.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    node --version && npm --version && \
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

# Write the injected key, stage the persistent git deploy key (if present on the
# /workspace volume) into /root with correct permissions, start sshd, then stay alive.
#
# /root is wiped on every pod restart, so it's rebuilt here each boot. The git deploy
# key itself lives on /workspace so it survives restarts, but /workspace is a MooseFS
# network mount that does not honor chmod -- ssh refuses a key left there directly, so
# it has to be copied into /root and locked down to 600 here instead.

CMD ["/bin/bash", "-c", "\
if [ -n \"$PUBLIC_KEY\" ]; then \
    echo \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys; \
    chmod 600 /root/.ssh/authorized_keys; \
fi; \
if [ -f /workspace/.ssh_git/id_ed25519 ]; then \
    cp /workspace/.ssh_git/id_ed25519 /root/.ssh/id_ed25519_git; \
    chmod 600 /root/.ssh/id_ed25519_git; \
    echo 'Host github.com' > /root/.ssh/config; \
    echo '    IdentityFile /root/.ssh/id_ed25519_git' >> /root/.ssh/config; \
    echo '    IdentitiesOnly yes' >> /root/.ssh/config; \
    chmod 600 /root/.ssh/config; \
fi; \
if [ ! -d /workspace/GPU_Profiling/.git ]; then \
    echo 'Cloning GPU_Profiling...'; \
    git clone https://github.com/KingCorsair/GPU_Profiling.git /workspace/GPU_Profiling; \
fi; \
service ssh start; \
sleep infinity"]
