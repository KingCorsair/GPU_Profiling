# Gemma 3 4B with the existing load generator

Model: Gemma 3 4B Instruct, downloaded without authentication from the public
[`unsloth/gemma-3-4b-it`](https://huggingface.co/unsloth/gemma-3-4b-it) mirror,
pinned revision `bf46152c47f5dd20b896357cb51abc4c03b8ee8c`.
Both weight shards match the published SHA-256 hashes of Google's original
`093f9f388b31de276ce2de164bdc2081324b9767` revision. The downloader verifies those
hashes after downloading. Config and tokenizer settings are from the pinned
Unsloth mirror and can differ from Google's files; record the mirror identity.
The wrapper exposes the existing `POST /infer` request `{image_b64, question}`
and response `{answer}`. It needs no changes to loadgen scheduling or timing.

## Runtime and model access

Build `csnbs/Dockerfile.gemma` from the repository root, or dispatch the existing
Docker workflow with `runtime=gemma`. It publishes the separate `:gemma` and
`:gemma-<commit>` tags; deploy the immutable commit tag on RunPod with the existing
`/workspace` volume. It includes Python 3.12, PyTorch 2.8.0/CUDA 12.6,
Transformers 4.57.6 under `/opt/gemma`, and Node 24.13.0 under `/opt/node24`.
The LLaVA system Python packages remain separate. No packages are installed into
the running pod by these scripts.

No Hugging Face account or token is needed for this public mirror. Gemma's license
still applies. Once weights are downloaded under `/workspace`, inference is
local-only. The download script can run in the base image
before switching images; it only requires the existing `huggingface_hub` package.

```bash
cd /workspace/GPU_Profiling
python csnbs/download_gemma.py
```

If running the Docker build locally, pin its base for repeatable environments:

```bash
docker build --platform linux/amd64 -f csnbs/Dockerfile.gemma \
  --build-arg BASE_IMAGE=kingcorsair/gpu_profiling_project:<base-commit> \
  -t kingcorsair/gpu_profiling_project:gemma .
```

## Serve and connect

After other GPU jobs finish, start the server on the pod:

```bash
cd /workspace/GPU_Profiling
bash csnbs/start_gemma_server.sh
```

From a second pod terminal:

```bash
curl --fail http://127.0.0.1:8002/health
cd /workspace/GPU_Profiling/csnbs/measure
npm ci
npm run dev -- --endpoint http://127.0.0.1:8002/infer \
  --rps 1 --duration 20 --timeout 120000 --dataset /absolute/path/to/dev.json
```

Replace the dataset path with the existing locked development workload. The short
run above checks integration only; it is not a reportable performance result.
Save the `/health` response as `server.json` beside that run's `run.json` and
`requests.jsonl`. The existing generator does not automatically fetch server
metadata. Health includes the checkpoint download revision, generation settings,
GPU, precision, and library versions; an unmarked custom checkpoint has no asserted
download revision.

Default behavior: BF16, SDPA, greedy generation with natural EOS, at most 64 new
tokens, caching enabled, pan-and-scan disabled, no pruning, synchronous batch one,
one Uvicorn worker. `GEMMA_MAX_NEW_TOKENS`, `GEMMA_MODEL_PATH`, `GEMMA_HOST`,
`GEMMA_PORT`, and `GEMMA_PYTHON` override the corresponding values.
Gemma/LLaVA/Qwen comparisons measure whole serving systems; use the same model
with and without pruning to isolate pruning's effect.

## Checks

`python -m unittest csnbs.test_gemma_server` checks HTTP compatibility and malformed
input rejection using a stub. It does not establish real GPU inference. Verify
both a real image answer and a saved loadgen run after deployment.
