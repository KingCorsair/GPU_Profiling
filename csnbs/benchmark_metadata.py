"""Read-only service provenance collected once, outside measured requests."""
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_provenance():
    commit = command(["git", "rev-parse", "HEAD"])
    dirty = command(["git", "status", "--porcelain", "--untracked-files=normal"])
    gpu = command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"])
    devices = []
    for line in (gpu or "").splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 4:
            devices.append(dict(zip(("name", "uuid", "driver_version", "memory_total_mib"), fields)))
    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "fastapi", "uvicorn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "source": {"gitCommit": commit, "gitDirty": None if dirty is None else bool(dirty),
                   "containerImage": os.environ.get("BENCHMARK_IMAGE_DIGEST")},
        "hardware": {"gpuModels": [device["name"] for device in devices], "devices": devices, "source": "server-nvidia-smi", "error": None if gpu else "nvidia-smi unavailable"},
        "runtime": versions,
    }


def checkpoint_provenance(path):
    try:
        data = json.loads((Path(path) / "download-provenance.json").read_text())
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None
