"""Download the pinned Gemma checkpoint; installs no packages and accepts no terms."""

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


MODEL_ID = "unsloth/gemma-3-4b-it"
REVISION = "bf46152c47f5dd20b896357cb51abc4c03b8ee8c"
# Public Hub LFS hashes match Google's pinned checkpoint weight shards.
WEIGHT_HASHES = {
    "model-00001-of-00002.safetensors": "eb5fd5e97ddd07b56778733e9653c07312529cb00980a318fc3e1c4e3b5a8f1f",
    "model-00002-of-00002.safetensors": "fdde0e5aa5ced0fa203b3d50f4ab78168b7e3a3e08c6349f5cc9326666e1bb13",
}
destination = Path(os.environ.get(
    "GEMMA_MODEL_PATH", str(Path(__file__).resolve().parents[1] / "checkpoints/gemma-3-4b-it"),
))

hf_hub_download(MODEL_ID, "config.json", revision=REVISION, local_dir=destination, token=False)
snapshot_download(
    MODEL_ID, revision=REVISION, local_dir=destination, max_workers=4, token=False,
    allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "README.md", "LICENSE*"],
)
for filename, expected in WEIGHT_HASHES.items():
    print(f"Verifying {filename}", flush=True)
    with (destination / filename).open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch: {filename}")

(destination / "download-provenance.json").write_text(json.dumps({
    "model_id": MODEL_ID, "revision": REVISION,
    "original_model_id": "google/gemma-3-4b-it",
    "original_revision": "093f9f388b31de276ce2de164bdc2081324b9767",
    "verified_weight_sha256": WEIGHT_HASHES,
}, indent=2) + "\n")
print(f"Ready: {MODEL_ID}@{REVISION} in {destination}")
