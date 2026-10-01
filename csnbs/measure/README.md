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
5. The selected [October 1 protocol](OCT01_PROTOCOL.md) uses five pairs at each of 1 and 3 RPS, followed by isolated HTTP trials. The broader four-rate `llava-token-ab.json` remains an unexecuted example. Both use 270 requests per load trial (p95 eligible after sufficient successes, p99 unavailable). Examine the pilot before executing a substantial GPU study. Do not keep adding favorable trials or silently replace failed trials.
6. Report each offered rate separately. The [observation follow-up](OCT01_OBSERVATION_PROTOCOL.md) declares separate instrumentation and short/long/mixed output-policy studies. Inspect actual generated lengths and repeat matched pairs. Raising `max_new_tokens` alone does not ensure longer responses.

```sh
npm --prefix csnbs/measure run campaign -- campaigns/llava-baseline-pilot.json --plan
npm --prefix csnbs/measure run campaign -- campaigns/llava-baseline-pilot.json --output /workspace/rithvik-results/llava-baseline-pilot
```

A campaign stores an immutable schedule and mutable trial statuses. Re-running the identical specification skips completed trials and puts retries in new attempt directories. A failed trial stops the campaign; retain and diagnose it. After a hard crash, the runner recovers only a stale same-host lock whose recorded process is dead. Inspect and stop any surviving owned server first; an occupied endpoint prevents reuse. Do not remove a lock belonging to an active runner.

## Portable artifact export

After collection stops, copy the whole campaign directory from the pod, including every attempt and raw artifact. Export that local copy into a new destination. If the saved trial paths are absolute pod paths, supply their exact original campaign root with `--source-root`; the exporter maps only that root to the copied tree.

```sh
python csnbs/export_campaign.py /tmp/copied-pod-campaign results/campaigns/2026-10-01/llava-baseline-pilot \
  --source-root /workspace/rithvik-results/llava-baseline-pilot
npm --prefix csnbs/measure run report -- ../../results/campaigns/2026-10-01/llava-baseline-pilot/campaign.json
python csnbs/plot_campaign.py --report results/campaigns/2026-10-01/llava-baseline-pilot/report/report.json
DATABASE_URL=postgresql://... npm --prefix csnbs/measure run ingest -- ../../results/campaigns/2026-10-01/llava-baseline-pilot/campaign.json
```

Choose an unused destination; existing directories are never overwritten. `campaign.source.json` preserves the original manifest bytes. The portable `campaign.json` changes only trial and attempt artifact paths to relative paths; specification, hash, schedule, statuses and all raw run/request/resource bytes stay unchanged. `export-provenance.json` records the original/exported hashes, file inventory and path mappings. Rebuild the report from the portable manifest to verify the moved evidence and refresh report links before importing that finalized snapshot. Importing the identical snapshot twice is a no-op; importing a different snapshot under an existing campaign ID is rejected.

Active locks, running trials/attempts, escaping paths, symlinks and missing completed-run artifacts are rejected. A stopped failed campaign can be exported with its partial attempt journals and `campaignComplete:false`; export alone does not establish successful completion or reportability.

## Interpretation

`successfulThroughputWithinWindowRps` counts successful completions within the predeclared measurement window. `successfulThroughputIncludingDrainRps` uses final drain as part of its denominator. Neither alone proves sustainable capacity. Capacity screens check client delivery, failures, and outstanding-work growth; repeated longer boundary trials are still required.

Nearest-rank p50 is descriptive; p95 is withheld below 200 successful requests and p99 below 1,000. Requests repeated inside one run are not independent trial replication. Paired comparisons bootstrap whole matched trial blocks, require at least five pairs, reject mismatched controls, and return improvement, regression or inconclusive. A confidence interval spanning zero is not equivalence.

The current LLaVA HTTP server preserves the serving owner's execution path. It exposes handler-to-answer wall time and request correlation, but cannot observe arrival before its blocked event loop. Queue time and GPU prefill/decode timings remain unavailable. Actual token counts are null in default mode and in the frozen primary study; [extended observations](OCT01_OBSERVATION_PROTOCOL.md) can record counts when the pinned wrapper contract validates. Sampled device memory is not exact KV-cache memory. Accuracy/scorer results, random-scoring controls and batching implementations remain owner inputs; this pipeline does not manufacture them.
