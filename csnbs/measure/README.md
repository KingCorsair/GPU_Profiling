# Rithvik's measurement pipeline

The canonical client is TypeScript in this directory. It records full HTTP latency with a monotonic clock, separate completed warmup, fixed seeded workload replay, durable raw outcomes, declared tail gates and explicit failure records. The Python load generator and September historical report remain readable reference implementations.

Node 20.19+ is supported; Node 20 is the deployment baseline and CI also checks Node 24. Run `npm ci`, `npm run typecheck`, `npm test`, and `npm run build` here. The local test suite uses loopback HTTP. GPU runtime dependencies come from the existing Docker image; do not install Python packages on a running pod.

## End-to-end CPU validation

From the repository root:

```sh
npm --prefix csnbs/measure run campaign -- campaigns/cpu-integration.json --output /tmp/measurement-campaign
npm --prefix csnbs/measure run report -- /tmp/measurement-campaign/campaign.json
python csnbs/plot_campaign.py --report /tmp/measurement-campaign/report/report.json
DATABASE_URL=postgresql://... npm --prefix csnbs/measure run ingest -- /tmp/measurement-campaign/campaign.json
npm --prefix csnbs/frontend run build
```

Synthetic results verify the pipeline; they do not establish GPU performance. See `db/README.md` for storage setup. The dashboard discovers finalized runs under both `results/loadgen` and `results/campaigns` and preserves the old study.

## GPU deployment

Use a separate clean checkout under `/workspace` on the existing pod. Keep logs/results outside that checkout during collection so subsequent trials retain clean source provenance. Check that no other compute processes are running, reserve the GPU, and set `gpuExclusive:true` only for that reservation. The campaign checks for existing GPU processes before every model trial and refuses any occupied HTTP endpoint. Set `LLAVA_TIMING_FILE` outside both the measurement checkout and the shared checkout; the supplied GPU specifications already do this. It only terminates its own child server; it never stops the pod.

1. Use the immutable commit of the measurement branch; run `npm ci --prefix csnbs/measure` once. Keep the installed Python/CUDA environment from the pod image.
2. Download the pinned LLaVA checkpoint using `scripts/download_models.sh llava` (the pinned, no-install downloader) or the already-installed `huggingface_hub.snapshot_download` at revision `4481d270cc22fd5c4d1bb5df129622006ccd9234` into an ignored/external model directory.
3. Verify the downloaded bytes with `python csnbs/verify_checkpoint.py /path/to/checkpoint`. This records a checkpoint provenance file. Set `BENCHMARK_IMAGE_DIGEST` to the actual immutable image digest when available; absence remains null.
4. Edit a copy of `campaigns/llava-baseline-pilot.json` for the verified model path. First run the six identical baseline trials and inspect spread, failures and dispatch fidelity. This pilot has 90 measured requests per trial, so neither p95 nor p99 is reportable.
5. The supplied `llava-token-ab.json` is a separately declared five-pair, four-rate protocol using 270 requests per trial (p95 eligible after sufficient successes, p99 unavailable). It is a substantial GPU run; examine the pilot before executing it. Do not keep adding favorable trials or silently replace failed trials.
6. Report each offered rate separately. To study long outputs, declare a separate workload/output policy, inspect actual generated lengths when supplied by the serving owner, and repeat matched pairs. Raising `max_new_tokens` alone does not ensure longer responses.

```sh
npm --prefix csnbs/measure run campaign -- campaigns/llava-baseline-pilot.json --plan
npm --prefix csnbs/measure run campaign -- campaigns/llava-baseline-pilot.json --output /workspace/rithvik-results/llava-baseline-pilot
```

A campaign stores an immutable schedule and mutable trial statuses. Re-running the identical specification skips completed trials and puts retries in new attempt directories. A failed trial stops the campaign; retain and diagnose it. After a hard crash, the runner recovers only a stale same-host lock whose recorded process is dead. Inspect and stop any surviving owned server first; an occupied endpoint prevents reuse. Do not remove a lock belonging to an active runner.

## Interpretation

`successfulThroughputWithinWindowRps` counts successful completions within the predeclared measurement window. `successfulThroughputIncludingDrainRps` uses final drain as part of its denominator. Neither alone proves sustainable capacity. Capacity screens check client delivery, failures, and outstanding-work growth; repeated longer boundary trials are still required.

Nearest-rank p50 is descriptive; p95 is withheld below 200 successful requests and p99 below 1,000. Requests repeated inside one run are not independent trial replication. Paired comparisons bootstrap whole matched trial blocks, require at least five pairs, reject mismatched controls, and return improvement, regression or inconclusive. A confidence interval spanning zero is not equivalence.

The current LLaVA HTTP server preserves the serving owner's execution path. It exposes handler-to-answer wall time and request correlation, but cannot observe arrival before its blocked event loop. Queue time, GPU prefill/decode timings and verified actual token counts remain null. Sampled device memory is not exact KV-cache memory. Accuracy/scorer results, random-scoring controls and batching implementations remain owner inputs; this pipeline does not manufacture them.
