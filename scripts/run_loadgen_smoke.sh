#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$repo_root/csnbs/measure"
exec npm run dev -- \
  --endpoint "${LOADGEN_ENDPOINT:-http://127.0.0.1:8000/infer}" \
  --run-kind smoke \
  --warmup 0 \
  --rps "${LOADGEN_RPS:-0.1}" \
  --duration "${LOADGEN_DURATION_SECONDS:-10}" \
  --timeout "${LOADGEN_TIMEOUT_MS:-120000}" \
  --dataset "${LOADGEN_DATASET_PATH:-$repo_root/vis_pruner_copy/vispruner_eval_dataset/dev.json}"
