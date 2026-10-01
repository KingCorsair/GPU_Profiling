# Explain the measurement pipeline aloud

“My component creates reproducible traffic, records what actually happened, and checks whether repeated matched trials support a claim. I keep intended arrivals, completed work, and unavailable measurements separate.”

Rehearse, then close this and explain one saved run. Details: [contracts](contracts.md), [methodology](methodology.md), [study status](../MEASUREMENT_STATUS.md).

## 1. Three client timestamps

In [`runLoad()` and `sendOne()`](src/loadgen.ts), Node `performance.now()` is the monotonic duration clock. UTC timestamps identify a run; they do not measure latency.

Illustrative example: at 2 requests/second, requests are scheduled at 0, 500, and 1,000 ms. Suppose the second is sent at 520 ms and finishes at 1,520 ms:

| Quantity | Calculation | Meaning |
|---|---|---|
| Dispatch lateness | 520 − 500 = 20 ms | How late the client sent it |
| HTTP latency | 1,520 − 520 = 1,000 ms | Send through response parsing/validation |
| Scheduled to completion | 1,520 − 500 = 1,020 ms | Both delays together |

The third request is still due at 1,000 ms while the second is outstanding. `isolated` mode deliberately awaits completion; its configured rate is not offered load. A timeout ends client waiting, not necessarily GPU work.

In the [saved CPU demonstration](../../results/calibration/2026-10-01/coordinated-omission-fixed-10000/report.md), at a 750-RPS target, open loop delivered 10,000 requests in the 13.333-second reference window; the completion-paced control delivered 4,442. Their p95s were 9,768.202 ms and 3.359 ms. The control hid intended arrivals before dispatch. These synthetic observations do not establish model/GPU performance.

## 2. Warmup and replay are part of the experiment

[`runLoad()`](src/loadgen.ts) completes sequential warmups and optional settling before a fresh measured clock. Warmup failure aborts measurement because server drain is unverified. Warmup has a separate phase, not an after-the-fact subtraction.

[`buildWorkloadOrder()` and `describeWorkload()`](src/loadgen.ts) save a seeded, category-balanced order and its identities/hash. Fixed budgets and complete cycles preserve question mix across rates; A/B legs replay the same order. Images load before client timing; server decoding/preprocessing remain inside HTTP latency.

[`planCampaign()` and `executeCampaign()`](src/campaign.ts) save randomized rate blocks and adjacent A/B or B/A legs, restart the owned server, verify effective settings, and preserve failures/retries. Change one declared factor per comparison.

## 3. Requests and experimental replications are different

[`summarizePercentiles()`](src/loadgen.ts) uses nearest rank: sorted index `ceil(p × n) − 1`. Each distribution has its own observation count. p95 is withheld below 200 and p99 below 1,000; passing a gate does not guarantee precision. Successful-only latency must be read beside failures and the prescribed request denominator.

[`comparePairs()`](src/analyze.ts) estimates the median paired percentage effect using 10,000 seeded bootstrap draws of whole blocks. Five pairs are required for an interval; unmatched controls prevent a conclusive comparison. Ten thousand correlated requests are not ten thousand replications. Positive means higher throughput or lower latency; crossing zero means inconclusive, not equivalent.

[`baselineRepeatability()`](src/analyze.ts) describes trial-to-trial spread. A/A estimates noise, not improvement. Incomplete budgets, failed checks, and retries cannot be discarded to improve a conclusion.

## 4. Throughput is not automatically capacity

[`summarizeRun()`](src/loadgen.ts) separates successes inside the fixed window divided by its duration from all successes divided by elapsed time including drain. Low-load throughput may just match arrivals. Eventually draining a backlog does not establish sustainable capacity.

[`classifyCapacity()` and `summarizeCapacityBracket()`](src/capacity.ts) screen delivery, failures, outstanding-work growth, and repeated rates. Passing requires at least 30 seconds and 200 requests. Sustainable capacity still needs longer boundary trials and observable queues. Client outstanding work includes transport/service, not just queue depth. No latency SLO is assumed.

## 5. Follow the evidence from request to chart

[`createJournal()` and `main()`](src/loadgen.ts) write a partial declaration, append terminal outcomes to `requests.jsonl` in completion order, and finalize `run.json` with an exact-byte hash. Sequence identifies replay order. Interrupted or missing records are not successes.

[`readVerifiedRun()`](src/analyze.ts) verifies hashes, phases/counts, timing algebra, schedules, workload identity, and recomputed summaries. Commits, effective settings, checkpoint identity, and server-host GPU metadata provide attribution. `quality.reportable` means run checks passed, not significance or completed research.

[`export_campaign.py`](../export_campaign.py) preserves bytes and maps paths. [`readImportBundle()` and `ingestBundle()`](src/ingest.ts) verify finalized evidence before transactional PostgreSQL import: identical imports are no-ops; conflicts fail. Request identity includes phase. Storage does not introduce another percentile definition.

[`buildReport()`](src/report.ts), the [plotter](../plot_campaign.py), and the [dashboard](../frontend/README.md) preserve missing values, limits, and raw links/hashes. Downloads retain original bytes. Validated accuracy remains a separate owner handoff.

## 6. Optional HTTP observations have narrower boundaries

Extended mode in [`infer()`, `_infer_model()`, and `_token_observations()`](../server.py) adds host-wall base64, preprocessing, generation, postprocessing, and observation durations to default service time. These use `perf_counter()`. Generation includes existing synchronization/diagnostic writes; it is not GPU prefill/decode timing. Separate stage medians are not additive.

Actual IDs/visual-feature counts require the pinned contract. Generated steps include EOS; text tokens and Unicode characters are separate. Output caps are not observed lengths. [`summarizeHostObservations()`](src/report.ts) preserves missing/invalid values, reasons, and coverage.

The [lifecycle contract](lifecycle-contract.md) defines boundaries. The [off/on study](OCT01_OBSERVATION_PROTOCOL.md) measures incremental HTTP observations, not all model instrumentation. Queue admission, GPU stages, batching/padding, exact KV allocation, and validated accuracy require owner inputs.

## Close the laptop

Explain why the third request in the example still launches; why a timeout does not prove drain; why one large run cannot replace five paired blocks; why finishing a backlog does not establish capacity; and which file/function would reveal a missing or fabricated measurement. Then trace one real saved request through its run, report, and database identity.
