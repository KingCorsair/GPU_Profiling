#!/bin/bash
# Downloads the LLaVA-1.5-7B checkpoint into checkpoints/llava-v1.5-7b.
# Weights are never committed to git (see .gitignore) — run this instead after every clone.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-$REPO_ROOT/venvs/vispruner/bin/python}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints}"
CKPT="llava-v1.5-7b"
HF_REPO="liuhaotian/llava-v1.5-7b"

if [ -f "$CKPT_DIR/$CKPT/pytorch_model-00001-of-00002.bin" ] && [ -f "$CKPT_DIR/$CKPT/pytorch_model-00002-of-00002.bin" ]; then
    echo "Checkpoint already present at $CKPT_DIR/$CKPT — skipping download."
    exit 0
fi

"$VENV_PYTHON" -m pip install -q -U huggingface_hub

"$VENV_PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$HF_REPO', local_dir='$CKPT_DIR/$CKPT')
"

echo "Downloaded $HF_REPO to $CKPT_DIR/$CKPT"
