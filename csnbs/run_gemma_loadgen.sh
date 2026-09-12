#!/usr/bin/env bash
# Convenience entry point for the existing TypeScript load generator.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$repo_root/venvs/node24/bin/node" ]]; then
  export PATH="$repo_root/venvs/node24/bin:$PATH"
fi
cd "$repo_root/csnbs/measure"
exec npm run dev -- --endpoint http://127.0.0.1:8002/infer "$@"
