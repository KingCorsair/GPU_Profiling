#!/usr/bin/env bash
# Downloads checkpoints only on request. No package installation or model loading.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
download_python="${MODEL_DOWNLOAD_PYTHON:-python}"

exec "$download_python" - "$repo_root" "$@" <<'PY'
import argparse
from pathlib import Path
import sys

repo_root = Path(sys.argv[1])
parser = argparse.ArgumentParser(
    prog="bash scripts/download_models.sh",
    description="Download LLaVA, Qwen, or both into ignored checkpoint folders.",
)
parser.add_argument("model", choices=("llava", "qwen", "all"))
parser.add_argument("--check", action="store_true",
                    help="compare local file sizes with the pinned Hub revision; download nothing")
args = parser.parse_args(sys.argv[2:])

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    parser.exit(1, "huggingface_hub is missing from this Python. Use the project's RunPod image "
                   "or set MODEL_DOWNLOAD_PYTHON to its Python executable.\n")

models = {
    "llava": ("liuhaotian/llava-v1.5-7b", "4481d270cc22fd5c4d1bb5df129622006ccd9234",
              repo_root / "vis_pruner_copy/checkpoints/llava-v1.5-7b"),
    "qwen": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
             repo_root / "checkpoints/Qwen3-VL-8B-Instruct"),
}
selected = models if args.model == "all" else {args.model: models[args.model]}
api = HfApi()
for name, (repo_id, revision, destination) in selected.items():
    print(f"\n{name}: {repo_id}@{revision}\nDestination: {destination}", flush=True)
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    missing = []
    for entry in info.siblings:
        local = destination / entry.rfilename
        if not local.is_file() or entry.size is None or local.stat().st_size != entry.size:
            missing.append(entry)
    total = sum(entry.size or 0 for entry in info.siblings)
    needed = sum(entry.size or 0 for entry in missing)
    print(f"Checkpoint: {total / 2**30:.2f} GiB. "
          f"Missing/different-size files: {len(missing)} ({needed / 2**30:.2f} GiB).", flush=True)
    if args.check:
        print("Check only; no files downloaded. Size checks are not checksum verification.", flush=True)
        continue
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=destination,
                      max_workers=4)
    print(f"Ready: {destination}", flush=True)

if not args.check and args.model in ("llava", "all"):
    print("LLaVA's CLIP vision encoder is fetched separately on first model startup.")
PY
