#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gemma_python="${GEMMA_PYTHON:-/opt/gemma/bin/python}"
if [[ ! -x "$gemma_python" ]]; then
  echo "Gemma runtime missing: $gemma_python. Deploy the optional Gemma image; see csnbs/GEMMA_SETUP.md." >&2
  exit 1
fi
export GEMMA_MODEL_PATH="${GEMMA_MODEL_PATH:-$repo_root/checkpoints/gemma-3-4b-it}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
unset PYTHONPATH PYTHONHOME
cd "$repo_root"
exec "$gemma_python" -m uvicorn csnbs.gemma_server:app \
  --host "${GEMMA_HOST:-127.0.0.1}" --port "${GEMMA_PORT:-8002}" --workers 1
