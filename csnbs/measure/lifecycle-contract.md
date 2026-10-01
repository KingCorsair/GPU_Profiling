# Observable request lifecycle

This contract describes the TypeScript client and the default observation mode in `csnbs/server.py`. The exact stored fields, version rules and optional extended host-wall/token observations are in [contracts.md](contracts.md#optional-http-observations). Instrumentation preserves the existing blocking model-serving path; it does not establish queue, prefill, decode, or KV-cache measurements.

## Boundaries and clocks

| Observation | Current boundary | Includes / excludes |
|---|---|---|
| `scheduledAtMs` | Planned client dispatch on Node `performance.now()` | Absolute open-loop schedule; independent of prior responses. This is not server arrival. |
| `sentAtMs` | Immediately before client body serialization and `fetch()` | Images are already read and base64 encoded during setup. HTTP body serialization is inside request latency. |
| `completedAtMs` | After response body reading, JSON parsing, correlation/metric validation, or terminal error | Client completion; an error or timeout can end waiting while server work continues. |
| `latencyMs` | `completedAtMs - sentAtMs` | Client serialization, network, unobserved server waiting, observed service, response transport and client parsing. |
| `dispatchLatenessMs` | `sentAtMs - scheduledAtMs` | Client schedule fidelity. A late dispatch is recorded; the schedule does not back off. |
| `plannedToCompleteMs` | `completedAtMs - scheduledAtMs` | Dispatch lateness plus client HTTP latency. |
| `metrics.service_ms` | Python `perf_counter()` from entry into `infer()` until the answer is available, just before response construction | Includes base64 decoding, image/prompt preparation, model generation and output decoding. Excludes request JSON parsing/validation before handler entry, earlier waiting, response-model validation/serialization, and transport after the handler. |
| `unattributedClientMs` | Client HTTP duration minus returned service duration | Residual covering everything outside the observed service boundary. It is not queue time and can be negative when boundaries or instrumentation disagree. Preserve the value for diagnosis. |

Do not subtract Node timestamps from Python timestamps. Only durations measured within one clock domain may be compared. UTC timestamps align logs approximately; they are not a substitute for a shared monotonic clock. The client currently measures full-response latency, not streaming time to first token.

Warmup requests finish sequentially before optional settling and a fresh measured schedule. Workload loading, source discovery, `/health`, and model startup are outside measured HTTP requests. `server.json` records startup separately. Warmup remains in raw records under its own phase and is never silently mixed into measured distributions.

## Correlation and outcomes

The client sends `request_id` in JSON and `x-request-id` in the header. The model server echoes the JSON identifier. A returned identifier must match; missing correlation remains missing. `requestId = runId:phase:sequence`; warmup and measurement both start their phase-local sequence at zero.

Each predeclared measured request receives one terminal record on graceful completion/interruption: `success`, `http-error`, `invalid-response`, `timeout`, `network-error`, or `cancelled`. Undispatched cancellations retain identity/schedule but have null send time and durations. HTTP errors can lack server metrics. Abrupt process loss can leave a partial journal; absence of a terminal record does not prove success, failure, or server cancellation.

A timeout only bounds client waiting. The campaign stops its owned server and waits for process termination before starting the next model trial. Health responsiveness alone does not prove an empty queue. Failed/interrupted attempts remain in the campaign history, including when a later attempt succeeds.

## Default-mode limits and optional observations

The default service returns `metrics.schema_version = 1`. Setting `EXTENDED_HTTP_OBSERVATIONS=1` enables schema 2, with an explicit changed instrumentation identity. The client preserves the returned object and rejects negative/nonfinite known numeric metrics; accepting a metric is not independent validation of its claimed boundary. Historical schema-1 records retain their original unavailable values.

| Field or question | Current evidence | Required owner work before interpretation |
|---|---|---|
| `queue_ms` | Null, with `queue_unavailable_reason` | Amay must expose responsive intake, payload eligibility, and service admission on a defined clock. The blocking event loop can wait before `infer()` begins. |
| `preprocess_ms`, `generation_wall_ms`, `postprocess_ms` | Null by default; observed host-wall stages in extended mode | Interpret the boundaries in contracts.md. Generation wall time includes existing model-wrapper synchronization and logging; it is not CUDA event time. |
| GPU prefill versus decode | Unavailable in the canonical per-request HTTP records | Amay supplies correlated CUDA-event boundaries without a per-request pipeline-draining synchronization. Existing diagnostic trace files do not automatically satisfy this contract. |
| `generated_text_tokens`, `prompt_text_tokens`, `visual_tokens` | Null by default; extended mode checks actual IDs/returned feature count against the pinned wrapper contract | Version, BOS/EOS, shape, budget and truncation checks must pass for the affected count. Neither configured `max_new_tokens` nor `visual_token_num` is an observed count. Contract failures remain null with reasons. |
| Batch occupancy, batch-formation wait, padded/packed work | Unavailable; current declared batch size is one | Serving-owner implementation and actual per-batch/request observations. |
| Exact KV-cache allocation and peak | Unavailable | Allocator/cache instrumentation with a precise peak definition; sampled device memory cannot isolate KV cache. |

Changing intake/worker behavior creates a new serving variant and needs a fresh baseline. It must not be hidden in an instrumentation-only comparison. Real-versus-masked removal, raster ordering, attention/position IDs, and cache correctness belong to the serving/method owners.

## Resource observations and overhead

The campaign samples `nvidia-smi` outside request handling, approximately once per second: GPU UUID/name, utilization, used device memory, temperature, power and SM clock. Collection spans startup through cleanup, not just measurement. The copied `resources.jsonl` descriptor records count/hash and scope. Retain sample timestamps, units and collection errors; do not interpret missing samples as zero load. CPU-load sampling is not currently implemented.

The largest observed memory sample is **peak sampled whole-device memory**. Short peaks between samples can be missed, and the value includes weights, workspaces, caches and other device allocations. Resource monitoring, service instrumentation and per-request event markers can introduce overhead. A matched enabled/disabled experiment on the same serving implementation is still required to quantify that overhead.

`csnbs/gpu_calibration.py` separately checks CUDA timing behavior using synthetic operations. Its warmup/event/host-enqueue records, script hash and GPU identity do not measure the model's prefill/decode split or certify serving-instrumentation overhead. Calibration artifacts use their own schema and are explicitly non-serving results.

## Acceptance before adding a lifecycle metric

Record its name, units, start/end events, clock domain, inclusion/exclusion rules, and unavailable reason. Use a controlled serial fake service to check queue growth and fixed service duration, then verify request correlation and boundary reconciliation on the real service. Preserve failures and negative residuals. Save both the implementation identity and an instrumentation-overhead comparison before treating the new stage decomposition as reportable.
