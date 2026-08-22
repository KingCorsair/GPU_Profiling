#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="$repo_root/vis_pruner_copy${PYTHONPATH:+:$PYTHONPATH}"
export SERVER_MODE="model"
export MODEL_PATH="${MODEL_PATH:-$repo_root/vis_pruner_copy/checkpoints/llava-v1.5-7b}"

cd "$repo_root"
exec python -m uvicorn csnbs.server:app \
  --host "${MODEL_SERVER_HOST:-0.0.0.0}" \
  --port "${MODEL_SERVER_PORT:-8000}"
