#!/usr/bin/env bash
# Explicitly approved on-pod development setup; production recipe: Dockerfile.gemma.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gemma_env="$repo_root/venvs/gemma"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv-gemma}"
export TMPDIR="${TMPDIR:-/workspace/.cache/gemma-tmp}"
export UV_LINK_MODE=copy
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
mkdir -p "$repo_root/venvs"
if [[ ! -e "$gemma_env/bin/python" ]]; then
  uv venv --python /usr/bin/python3.12 "$gemma_env"
fi
uv pip install --python "$gemma_env/bin/python" \
  --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.8.0+cu126 torchvision==0.23.0+cu126
uv pip install --python "$gemma_env/bin/python" -r "$repo_root/csnbs/requirements-gemma.txt"
"$gemma_env/bin/python" -c 'from transformers import AutoProcessor, Gemma3ForConditionalGeneration; import torch; print("Gemma runtime:", torch.__version__, "CUDA", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))'
node_root="$repo_root/venvs/node24"
if [[ ! -x "$node_root/bin/node" ]]; then
  node_archive="$TMPDIR/node-v24.13.0-linux-x64.tar.xz"
  curl -fSL --retry 3 https://nodejs.org/dist/v24.13.0/node-v24.13.0-linux-x64.tar.xz -o "$node_archive"
  echo "e798599612f4bb71333a3397ab0d095fd62214e115aea45aa858a145fc72d67e  $node_archive" | sha256sum --check -
  mkdir -p "$node_root"
  tar -xJf "$node_archive" -C "$node_root" --strip-components=1 --no-same-owner --no-same-permissions
fi
"$node_root/bin/node" --version
echo "Ready: GEMMA_PYTHON=$gemma_env/bin/python bash csnbs/start_gemma_server.sh"
