# Run Qwen3-VL through the measurement harness

The checkpoint is `Qwen/Qwen3-VL-8B-Instruct`. Downloaded weights alone do not
provide a running server. The Qwen wrapper accepts the same request as LLaVA:
`POST /infer {"image_b64": "...", "question": "..."}` and returns `{"answer": "..."}`.

## 1. Download models only when needed

The standard Docker image does not include Qwen's extra runtime or any model
weights. The download script is tracked in Git; checkpoint directories are ignored.
Run these commands inside the RunPod repository:

```bash
cd /workspace/GPU_Profiling
bash scripts/download_models.sh all --check
```

`--check` queries the pinned Hugging Face revisions and compares local file sizes;
it downloads no model files. It is not a checksum verification. Select the weights
you actually need:

```bash
bash scripts/download_models.sh llava
bash scripts/download_models.sh qwen
```

Or request both with `bash scripts/download_models.sh all`. The script uses the
existing Python downloader, installs no packages, and reuses completed downloads.
Interrupted downloads can be resumed by rerunning the same command. Full checkpoints
occupy about 12.6 GiB for LLaVA and 16.3 GiB for Qwen. LLaVA's CLIP vision encoder
is fetched separately on first startup if it is not already cached.

The defaults are:

| Model | Revision | Destination |
|---|---|---|
| LLaVA | `4481d270cc22fd5c4d1bb5df129622006ccd9234` | `vis_pruner_copy/checkpoints/llava-v1.5-7b` |
| Qwen | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | `checkpoints/Qwen3-VL-8B-Instruct` |

Downloading Qwen's weights does not install the software needed to execute it.
The base LLaVA environment uses Transformers 4.37.2, which cannot load Qwen3-VL.

### Optional Qwen runtime

`csnbs/Dockerfile.qwen` is a separate, opt-in image recipe. The normal GitHub Actions
workflow does not build it. It adds Python 3.12, PyTorch 2.8.0 (CUDA 12.6), and
Transformers 4.57.6 under `/opt/qwen`, preserving LLaVA's system Python packages.
Build it only when preparing to serve Qwen, on a machine/runner with Docker:

```bash
docker build --platform linux/amd64 -f csnbs/Dockerfile.qwen \
  -t kingcorsair/gpu_profiling_project:qwen .
```

The default base is the standard `:latest` image. For a reportable experiment, pass
`--build-arg BASE_IMAGE=kingcorsair/gpu_profiling_project:<published-commit>` and
record the resulting image digest. Publish the optional image and deploy it on
RunPod with the existing persistent storage during an agreed downtime window.
This optional build/deployment is not performed by the download script. No packages
are installed into a running pod. If a compatible Qwen runtime already exists,
set `QWEN_PYTHON` to its Python executable instead.

## 2. Start Qwen on the pod

Ensure the GPU is available and other model servers or benchmark jobs have
finished. This wrapper loads the whole model in BF16 onto GPU 0.

```bash
cd /workspace/GPU_Profiling
bash csnbs/start_qwen_server.sh
```

The launcher uses `/opt/qwen/bin/python`, one Uvicorn worker, and
`127.0.0.1:8001`. It fails with an explanation if the Qwen runtime is missing.
It does not install packages or download missing weights during startup.

From a second SSH terminal, after startup finishes:

```bash
curl --fail http://127.0.0.1:8001/health
```

Expect `service: csnbs-qwen-server`, `model_loaded: true`, and the effective
model/runtime configuration. A successful health response establishes model
loading; also check a real image request before running a campaign:

```bash
/opt/qwen/bin/python - <<'PY'
import base64
from io import BytesIO
import json
from urllib.request import Request, urlopen
from PIL import Image

buffer = BytesIO()
Image.new("RGB", (64, 64), "red").save(buffer, "PNG")
payload = {"image_b64": base64.b64encode(buffer.getvalue()).decode(),
           "question": "What color is this image? Answer with one word."}
request = Request("http://127.0.0.1:8001/infer",
                  data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=120) as response:
    print(response.read().decode())
PY
```

This is a functionality check, not an accuracy evaluation or benchmark.

## 3. Point the harness at Qwen

Use the existing TypeScript load generator with
`--endpoint http://127.0.0.1:8001/infer` and the same recorded workload/arrival
conditions used for the corresponding LLaVA trial. Keep its required `--rps`,
`--duration`, `--timeout`, and `--dataset` arguments. No timing or load-generation
code changes are needed to send requests to this endpoint.

The existing LLaVA campaign launchers start and verify a LLaVA server; they do
not automatically manage Qwen. Use the standalone generator while this Qwen
server is running. Save `/health` with the run: the current generator does not
automatically persist Qwen's effective model configuration. The image still
uses the existing Node 20 configuration, while the measurement package requests
Node 24; resolve that pre-existing mismatch before claiming a supported campaign.

Defaults: greedy generation, natural EOS, at most 64 new tokens, cache enabled,
SDPA attention, BF16, synchronous batch-one inference. Set `QWEN_MAX_NEW_TOKENS`,
`QWEN_MIN_PIXELS`, and `QWEN_MAX_PIXELS` before startup to change them; record
the effective values. The initial image pixel range is 4,096–1,048,576 pixels,
corresponding to roughly 4–1,024 merged image tokens after resizing. This is a
bounded initial serving configuration, not an accuracy-tuned setting.

Qwen uses its native preprocessing and chat template. Different models,
precisions, tokenizers, image policies, and library versions make this a
whole-system comparison. It does not isolate VisPruner's effect. For that
experiment retain the 576-versus-128-token comparison within LLaVA.

## Validation

CPU HTTP contract tests (FastAPI, Pillow, and HTTPX required):

```bash
python -m unittest csnbs.test_qwen_server
```

These checks inject a stub backend. They do not establish CUDA compatibility,
model accuracy, or throughput. The optional Qwen image includes import checks for Qwen and
checks that LLaVA's original system package versions remain installed. Verify
the real GPU startup and image request above after deploying the image.

References: [Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
[Transformers Qwen3-VL documentation](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/qwen3_vl),
[official PyTorch CUDA wheels](https://pytorch.org/get-started/previous-versions/).
