# Measurement contract, version 2

`src/loadgen.ts` produces `loadgen-run` version 2 manifests and request records version 3. Version 1 historical artifacts remain immutable and use `csnbs/plot_load_comparison.py`. The V2 reader rejects unknown versions instead of silently interpreting incompatible fields. Existing summary names remain available to callers; V1's warmup subtraction and unconditional numeric tail fields do not carry into V2.

## Invocation and phases

Use `--endpoint URL --rps RATE --requests COUNT --dataset PATH`. `--duration SECONDS` remains supported: its measured budget is `floor(RATE * SECONDS)`. If both count and duration are provided they must agree. For matched comparisons prefer a constant `--requests` budget at every RPS; count alone sets duration to count/rate. Also supported:

- `--warmup N` (default 10), `--seed N` (default 90), `--settle-ms N` (default 0).
- `--timeout MS` (default 120000) is a per-request client timeout.
- `--run-deadline-ms MS` optionally bounds warmup, settling, measurement and drain together, after setup. A reached deadline aborts the run and preserves outcomes.
- `--run-kind open-loop|isolated|smoke` and `--output-dir PATH`.

Images and questions are loaded before any timing. Server image decoding and preprocessing remain inside HTTP latency. Warmup requests run sequentially, each completing before the next. After successful warmup and optional settling, a fresh measured clock begins. Warmup failure aborts measurement because client failure does not prove server work has drained. Every attempted warmup and every predeclared measured request still receives a terminal record on graceful interruption; undispatched records are cancelled with null request durations.

Open-loop arrivals use absolute deadlines and never await prior measured completions. Late dispatch catches up and records the lateness; it does not reduce offered load to match completion. Measurement continues for its full declared window, followed by drain. `isolated` deliberately waits for each response and measures a full HTTP request baseline; it cannot establish capacity. `smoke` defaults to zero warmups, accepts one measured request, and is always non-reportable.

Each complete workload cycle visits every record once. A seeded shuffle within categories and category round-robin makes prefixes balanced when categories have equal size (as in the 90-record dev set). Unequal categories retain their actual proportions over full cycles; exhausted categories are skipped. The paired trial's seed, ordered question IDs, sequence hash and category counts are saved. Fixed count and seed yield the same measured sequence at every RPS. Warmup has its own sequence and seed and cannot shift measured request identity.

## Raw records and lifecycle

`requestId` is `runId:phase:sequence`. Sequence is **local to the phase**: warmup sequence 0 and measurement sequence 0 are distinct records. Database uniqueness must use requestId or `(run_id, phase, sequence)`. Each row includes phase, sequence, source dataset, category, question ID, dataset index, and one terminal outcome:

`success`, `http-error`, `invalid-response`, `timeout`, `network-error`, or `cancelled`.

Node `performance.now()` supplies scheduled, sent and completed timestamps. For dispatched requests:

- `latencyMs = completedAtMs - sentAtMs` includes request body serialization, transport, service, response transfer and response parsing/validation.
- `dispatchLatenessMs = sentAtMs - scheduledAtMs`.
- `plannedToCompleteMs = latencyMs + dispatchLatenessMs`.

Undispatched cancellation has null sent time and all three durations null. Its terminal timestamp can precede its future scheduled deadline. Timeout/cancellation stops client waiting; it does **not** establish that GPU work was cancelled. Owned campaign servers must be stopped before the next trial after unresolved work.

The request body and `x-request-id` header carry the correlation ID. A returned `request_id`, when present, must match. A server may return `metrics` containing component-local `service_ms`, `queue_ms`, `preprocess_ms`, `generation_wall_ms`, `postprocess_ms`, actual `generated_text_tokens`, `prompt_text_tokens`, and `visual_tokens`. Missing observations stay null/absent. Numeric observations must be finite and nonnegative.

Server durations remain in their own clock domain. `unattributedClientMs = latencyMs - service_ms` is a residual, **not queue time**: it includes network, client work, and server waiting outside observed boundaries. Negative residuals are retained for diagnosis rather than clamped. The current blocking model service cannot observe pre-handler waiting and reports queue time null with a reason. Wall generation time is not CUDA kernel time. Token counts must be actual counts from the server; configured maximum output length is not a generated-token count.

## Persistence and provenance

Every run has a unique directory under the requested output root, containing:

- `run.partial.json`, written before requests, declares the budget/configuration/workload/provenance.
- `requests.jsonl`, created exclusively and appended asynchronously in **completion order**, outside dispatch scheduling. Sequence defines replay order.
- `run.json`, atomically renamed only after journal writes and filesystem synchronization complete, records the SHA-256 of exact raw bytes.

SIGINT, SIGTERM and an optional run deadline abort pending client waits and finalize an aborted manifest with cancelled records for undispatched measured requests. Abrupt process or machine death can leave only a partial journal; missing terminal records are unknown, never successes. This journal aids recovery; it does not claim immunity to power loss. Raw file write failure prevents a final manifest.

Record separate harness/server commits and dirty states, server-confirmed configuration/checkpoint, and **server-host** GPU identity. A laptop GPU lookup is not model-server provenance. Unknown metadata stays unknown. `quality.reportable` describes individual run checks only; it does not establish a statistically resolved optimization or complete experiment. Fake/smoke runs, interrupted runs, dirty/unknown source, missing model/checkpoint/configuration/GPU identity, incomplete dataset cycles, insufficient warmup or measured failures prevent this flag. Successful-only latency remains conditional on success and always has failure counts beside it.

## Summary and analysis

Warmup is never subtracted from measured rows; phases already separate it. Nearest-rank percentiles use `sorted[ceil(fraction * n) - 1]`. Every distribution carries its own sample count. p50 is descriptive at any positive count; p95 is null below 200 observations; p99 is null below 1000. Null reasons are recorded. These project gates are necessary sample checks, not promises of precision or independence.

Throughput names distinguish boundaries:

- `successfulThroughputWithinWindowRps` counts successful measured completions by the prescribed end divided by prescribed duration.
- `successfulThroughputIncludingDrainRps` uses all successful measured requests divided by the later of the prescribed end and last terminal completion, minus measured start.
- `successfulThroughputRps` remains an explicit compatibility alias for the including-drain metric.
- Actual client dispatch rate, dispatched/request/failure counts, end-of-window outstanding client requests and drain duration are separate fields. Outstanding client requests include transport and service, not just the server queue.

`readVerifiedRun()` verifies final schema, raw hash/count, phase identities, complete planned outcome coverage, order/category metadata, timing algebra, schedules, summaries and quality flags. It cannot prove the honesty of an externally fabricated server or signed provenance; it checks internal evidence consistency.

`comparePairs()` compares matched, independent paired blocks at one offered rate. Token-count comparisons allow only visual token count to differ in effective configuration; implementation comparisons allow the declared implementation identity/server revision to differ while holding token count fixed. Workload/order, GPU identity, harness revision, decoding controls, runtime, measurement policy and all other known controls must match. It reports median paired percentage effect and a 95% percentile-bootstrap interval from 10,000 resamples of **paired blocks**, never individual requests. Positive means improved. At least five independent blocks are required for an interval; any failed run checks or comparability issue keeps the conclusion inconclusive. An interval crossing zero is inconclusive, not equivalence. Baseline repeatability reports observed trial range and median absolute deviation; neither predicts a guaranteed detectable gain.

`classifyCapacity()` is a conservative finite-window screen of delivered load, failures and client outstanding-work growth. It never declares sustainable capacity from a single run. Queue boundaries, longer confirmation windows and repeated boundary trials remain necessary. Very short/small screens stay inconclusive. Failures fail the operational criterion but do not establish their cause as overload without diagnosis.

Freeze campaign budgets before collecting results. More samples inside one run are not more independent trials, and repeatedly adding trials until the interval becomes favorable invalidates the uncertainty interpretation. No server batching, token-removal, KV-cache, scoring or accuracy implementation belongs to this measurement contract.
