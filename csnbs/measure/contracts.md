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

### Optional HTTP observations

`csnbs/server.py` defaults to `EXTENDED_HTTP_OBSERVATIONS=0`. This preserves metrics schema 1, the original service wall boundary and unavailable stage/token observations. Health `configuration.instrumentation` remains `handler-service-wall-v1`. Set the flag to `1` for metrics schema 2 and `handler-stage-token-observations-v2`; other effective configuration fields are unchanged. Missing new fields in historical records remain unknown. This flag does not turn the model wrapper's existing CUDA events, synchronizations or diagnostic file writes off. Both modes execute the same `model.generate` and `batch_decode` calls; no new explicit CUDA synchronization is added.

In extended model mode the wall durations are local `perf_counter()` observations:

- `base64_decode_ms`: handler entry through validated base64 decoding.
- `preprocess_ms`: image opening/RGB conversion, conversation formatting, prompt tokenization, image preprocessing and existing device transfers. This is host-observed wall time, not CPU compute alone.
- `generation_wall_ms`: the unchanged complete `model.generate` call, including the existing wrapper's vision/token preparation, generation, boundary synchronization and diagnostic writes. It does not isolate prefill, decode or GPU kernel time.
- `postprocess_ms`: existing `batch_decode(..., skip_special_tokens=True)[0].strip()`.
- `observation_wall_ms`: inspecting the already completed prompt/output IDs, validating contracts and obtaining the answer character count. The extra post-decode output tensor host copy is included here; the original decode is preserved. It is part of the incremental overhead the off/on study measures.
- `service_ms`: handler entry through observation construction; excludes subsequent response model/JSON serialization and socket transfer. Imports, transitions and dictionary work outside the named stages remain in service wall time. The stages do not establish queue time.

`output_characters` is `len(answer)` after the existing strip, measured in Unicode code points rather than bytes, graphemes or tokens. It is available even when a token contract fails, and in extended fake mode; fake mode never fabricates token counts or model generation time. Reports summarize only explicit nonnegative integer observations from successful measured rows and retain coverage counts.

Token observations use the actual IDs and the vendored wrapper return, never re-tokenized answer text or requested limits:

- `prompt_text_tokens_pretruncation` counts the formatted CPU prompt IDs excluding the image sentinel. It includes BOS, conversation roles/separators and other prompt special IDs; it is not just the question length. `image_placeholder_tokens` records the sentinel count.
- `visual_tokens_pretruncation` is the actual retained visual feature count returned by the wrapper, before its combined-sequence truncation. `prompt_text_tokens`, `visual_tokens` and `multimodal_prompt_tokens` are available only for one image placeholder, a valid returned feature count, and a known nonbinding `tokenizer_model_max_length`. If truncation could remove text or visual positions, those post-truncation fields stay null and the pre-truncation observations remain labeled.
- `returned_output_ids` is the number of IDs in the single returned sequence. For the verified decoder-only, greedy, batch-one **Transformers 4.37.2** `inputs_embeds` path, it must start with the declared synthetic BOS. `output_seed_tokens=1`; the remaining count is `generated_token_steps`, including terminal EOS. Do not subtract the formatted prompt length from this output.
- `generated_text_tokens` counts those generated IDs excluding every tokenizer special ID. `generated_eos_tokens` and `generation_ended_with_eos` distinguish natural EOS from exhausting `max_new_tokens`. An immediate EOS is one generated step and zero text tokens. This is tokenizer-ID accounting, not words or a guarantee of semantic content.
- Version, shape, BOS, special-ID, greedy-mode, EOS-position and budget checks fail to null for affected derived fields, with `prompt_token_unavailable_reason`, `output_token_unavailable_reason` and combined `token_unavailable_reason`. A mismatch does not change an already decoded answer or fail an otherwise successful request. `token_contract` identifies the interpretation. The narrow version guard intentionally requires review after dependency/wrapper changes.

The BOS interpretation is checked against the tagged [Transformers 4.37.2 generation implementation](https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/generation/utils.py), particularly `_prepare_model_inputs`, `_maybe_initialize_input_ids_for_generation` and `greedy_search`, together with the existing vendored `llava_llama.py` call that supplies embeddings without input IDs.

To measure incremental observation overhead, freeze one new clean source revision and declare `comparisonKind: "instrumentation"` for matched off/on legs. This allows **only** `configuration.instrumentation` to differ; server/harness revisions, token count, implementation, checkpoint, runtime, hardware, workload and decoding controls must still match. Compare only those separately collected legs, not an old default-off revision against a new extended revision. The result describes the extra HTTP observations, not the cost of all existing model instrumentation.

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

The configured budget must equal `floor(rate * duration + 1e-9)`, preserving standalone duration-based invocation. Open-loop/smoke window end must equal start plus configured duration; a completed run cannot finish before that end or before any terminal record. Isolated windows end at actual completion instead. Aborted runs may finish before the prescribed future end and retain undispatched cancellation timestamps before their scheduled deadlines. Campaigns further require the predeclared count/rate duration exactly, so a rewritten but internally consistent summary cannot change the throughput denominator. Monotonic clock comparisons allow one microsecond for floating-point noise; the serialized UTC projections allow one millisecond for `Date` truncation.

`comparePairs()` compares matched, independent paired blocks at one offered rate. Token-count comparisons allow only visual token count to differ in effective configuration; implementation comparisons allow the declared implementation identity/server revision to differ while holding token count fixed; instrumentation comparisons allow only the instrumentation field to differ on the same source revision. Workload/order, GPU identity, harness revision, decoding controls, runtime, measurement policy and all other known controls must match. It reports median paired percentage effect and a 95% percentile-bootstrap interval from 10,000 resamples of **paired blocks**, never individual requests. Positive means improved. At least five independent blocks are required for an interval; any failed run checks or comparability issue keeps the conclusion inconclusive. An interval crossing zero is inconclusive, not equivalence. Baseline repeatability reports observed trial range and median absolute deviation; neither predicts a guaranteed detectable gain.

`classifyCapacity()` is a conservative finite-window screen of delivered load, failures and client outstanding-work growth. It never declares sustainable capacity from a single run. Queue boundaries, longer confirmation windows and repeated boundary trials remain necessary. Very short/small screens stay inconclusive. Failures fail the operational criterion but do not establish their cause as overload without diagnosis.

Freeze campaign budgets before collecting results. More samples inside one run are not more independent trials, and repeatedly adding trials until the interval becomes favorable invalidates the uncertainty interpretation. No server batching, token-removal, KV-cache, scoring or accuracy implementation belongs to this measurement contract.
