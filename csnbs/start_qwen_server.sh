#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
qwen_python="${QWEN_PYTHON:-/opt/qwen/bin/python}"

if [[ ! -x "$qwen_python" ]]; then
  echo "Qwen runtime missing: $qwen_python" >&2
  echo "Qwen is optional: build/deploy csnbs/Dockerfile.qwen or set QWEN_PYTHON to an existing compatible runtime. See csnbs/QWEN_SETUP.md." >&2
  exit 1
fi

export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-$repo_root/checkpoints/Qwen3-VL-8B-Instruct}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
# Avoid imports from the LLaVA environment or vendored source.
unset PYTHONPATH PYTHONHOME
cd "$repo_root"
exec "$qwen_python" -m uvicorn csnbs.qwen_server:app \
  --host "${QWEN_HOST:-127.0.0.1}" --port "${QWEN_PORT:-8001}" --workers 1
