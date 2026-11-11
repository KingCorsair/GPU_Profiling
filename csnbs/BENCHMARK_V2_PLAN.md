# Rithvik — Benchmark Harness V2 implementation roadmap

Implementation and experiment progress is tracked in [MEASUREMENT_STATUS.md](MEASUREMENT_STATUS.md). The roadmap below preserves the original planning context.

This plan turns the Google Doc **Rithvik’s Plan** into work grounded in the local repository at commit `ecc205ea751d936e28e7165fa909311c95693c21`. It is a plan, not an implementation or a new benchmark result. The four weeks below are an effort estimate, not calendar deadlines.

## Outcome

Given a baseline and an optimization from Amay, produce a reproducible answer to: **Did it improve request latency or sustainable throughput, by how much, under which workload, and with what uncertainty?**

The core deliverable is one measurement pipeline: experiment specification → verified server configuration → controlled trials → immutable request records → validated analysis → human-readable report. Continue in `csnbs/measure`; moving to a new top-level service is unnecessary for this month. Keep the TypeScript measurement stack and existing Python reporting tools.

Rithvik writes and understands the arrival scheduling, timing, percentile, and comparison logic. AI assistance can scaffold configuration, schemas, reporting, CI, and tests, and review the core logic. Amay owns changes to model execution, batching, KV cache, kernels, and pruning. Sribhav owns accuracy scoring and evaluation design. This roadmap does not transfer those responsibilities.

## What already exists, and what actually needs work

| Area | Verified current implementation | Required addition |
|---|---|---|
| Open-loop traffic | `csnbs/measure/src/loadgen.ts`: `runLoad()` schedules absolute arrivals and drains promises after launching. `sendOne()` records HTTP/transport failures. | Behavioral tests, explicit phases, workload identity per request, bounded experiment termination, and calibration of client dispatch/connection behavior. |
| Client timing | `makeRequestResult()` uses `performance.now()` for send-to-completion, dispatch lateness, and scheduled-to-completion durations. | Preserve these separate meanings; add server-side measurements without mixing clock origins. |
| Persistence | `persistRun()` writes unique `run.json` and `requests.jsonl`; source commit, dirty state, workload hash, and local GPU names are recorded. | Schema versioning, complete effective server configuration, campaign manifests, incremental crash-safe records, and separate client/server provenance. |
| Warmup | `summarizeRun()` removes the first ten requests by sequence. | Dedicated configurable warmup followed by a clean measurement boundary. Excluding records does not remove warmup-created backlog. |
| Sweep | `csnbs/run_load_sweep.py` randomizes RPS for one configuration. `scripts/run_ab_load_sweep.py` runs one token configuration and reuses an earlier baseline. | Adjacent paired A/B trials, balanced configuration order, repeated trials, exact run-ID capture, and explicit recovery between runs. |
| Serving | `csnbs/server.py`: `_infer_model()` synchronously preprocesses and calls `model.generate()` inside the event loop. `/health` reports identity/readiness, not effective model settings. | Timing/configuration contract; responsive arrival observation requires a coordinated serving change. |
| Tail latency | TypeScript computes nearest-rank p50/p95/p99 for any nonempty successful sample. | Sample-size and uncertainty policy, failure disclosure, and consistent percentile definitions. Python's older `load_gen.percentile()` uses a different rank convention. |
| Reports | `csnbs/plot_load_comparison.py` verifies stored summaries and matching controls; preserves missing measurements. Sixteen tests pass. | Repeated trials, optional metrics, uncertainty, capacity brackets, per-request verification, and a versioned reader. |

### Findings that change the order of work

1. **Workload composition changed with offered RPS.** The current dev file has 90 questions: 15 each of OCR, counting, spatial reasoning, object presence, coarse description, and fine attributes. `runLoad()` starts at record zero on every run. In the saved 30-second campaign, after excluding ten warmups, 0.5 RPS measures five OCR questions; 1 RPS measures five OCR plus fifteen counting questions; 3 RPS measures eighty questions across all categories. This is reconstructed from the saved request sequences and the current matching workload. A/B at a given rate shares a subset, but the load curve changes both traffic intensity and workload composition. Fix this before interpreting its knee.
2. **The 3.83% result is exploratory finite-run throughput.** The September 1 campaign has one trial per setting, 5–80 measured successful requests, dirty source provenance, and A/B legs collected at different times. Its throughput denominator includes final draining. It establishes neither a precise sustainable-capacity ceiling nor a statistically demonstrated improvement.
3. **Handler-entry time is not socket-arrival time.** A blocked event loop may not start processing an arriving request until the preceding generation finishes. Timestamps placed only inside `infer()` can miss that wait. This follows from the source and Python's documented [event-loop behavior](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code).
4. **Producer and report schemas have drifted.** Current `RunManifest` and `main()` no longer emit a `benchmark` object, although an unused `BenchmarkMetadata` type remains. Historical runs contain it with several null fields; `hydrate_and_verify_raw_runs()` expects parts of it. New runs need a tested end-to-end reader contract.
5. **The default smoke invocation cannot yield a summary.** `scripts/run_loadgen_smoke.sh` defaults to 0.1 RPS for ten seconds: one request. Validation accepts that, but summarization requires more than ten.
6. **Runtime declarations disagree.** Docker and project guidance say Node 20; the measurement package requires Node 24; this Mac runs Node 25.7.0. Local typechecking passes, which does not prove deployment compatibility. `npm test` is still a placeholder; Python tests exercise the older Python generator.
7. **The historical no-load number is not a clean HTTP baseline.** `scripts/run_noload_ab_comparison.py:run_one()` starts timing after `build_request()`. The report's 13.4% correction is a derived estimate. Replace it with a measured full HTTP request baseline. Do not attribute the difference between this estimate and the load result solely to serialization: workload, output length, measurement boundaries, and trial conditions also differ.

## Dependency order and effort

| Task | Depends on | Focused effort | Suggested placement |
|---|---|---:|---|
| 1. Lock the measurement contract and runtime | None | 3–4 hours | Week 1 |
| 2. Make warmup, traffic, and workload reproducible | 1 | 6–9 hours | Week 1 |
| 3. Build the paired campaign runner | 1–2; effective-config support from 4 | 6–9 hours | Week 1–2 |
| 4. Add honest lifecycle measurement | 1; serving coordination with Amay | 6–10 hours of Rithvik's work | Week 2 |
| 5. Establish tail and noise-floor policy | 2–3; 4 for server metrics | 5–8 hours | Start Week 1; finish Week 3 |
| 6. Estimate sustainable capacity | 3–5 | 6–9 hours | Week 2–3 |
| 7. Measure workload sensitivity | 2, 4–6 | 4–6 hours | Week 3 |
| 8. Standardize optimization reports and CI | 1–7 | 6–9 hours | Week 4 |
| 9. Add queryable result storage | 1, 8 | 3–5 hours | Week 4 |
| 10. Evaluate batching when available | 4–8; Amay's implementation | 4–8 hours, additional | Optional extension |

Core estimate: **45–69 focused hours**, approximately 11–17 hours per week. GPU runtime, coordination delays, and learning/debugging buffer are additional. If available time is lower, extend the schedule or defer the database and expanded workload study; preserve measurement correctness. Task 10 is not required to finish the core month.

## Task 1 — Lock the measurement contract and runtime

**Files/functions:** `csnbs/measure/src/loadgen.ts` (`LoadConfig`, `InferResponse`, `RequestResult`, `RunManifest`, `validateConfig`, `parseLoadConfig`, `main`); `csnbs/measure/package.json`; `csnbs/measure/tsconfig.json`; `Dockerfile`; `.github/workflows/docker.yml`; the reader in `csnbs/plot_load_comparison.py`.

**Implementation steps:**

- Define version 2 of the run/request/campaign contracts in a proposed `csnbs/measure/contracts.md`. Keep historical version 1 inputs readable without modifying raw artifacts. Record unavailable metrics as null plus a reason.
- Use actual controls: `visual_token_num`, `important_ratio`, method/implementation identifier, checkpoint identity, prompt template, dtype/backend, batching configuration, decoding parameters, and EOS policy. Distinguish requested settings from server-confirmed effective settings. A path alone does not uniquely identify model weights.
- Record separate load-generator and server commits/dirty states, container image digest or immutable tag, runtime/library versions, GPU model/UUID/driver, workload hash, request-order seed, phases, and experiment purpose. Obtain server hardware from the server host, not a remote laptop's `nvidia-smi`.
- Resolve Node 20 versus 24 as an explicit project decision and make package, container, CI, and guidance agree. Node 24 is the current package target; adopting it requires a coordinated update to the documented Node 20 stack. Do not infer compatibility from this Mac's Node 25 check.
- Replace the package's placeholder test entry and repair the smoke contract: either a clearly labeled smoke mode without tail claims or a valid warmup-plus-measurement invocation.

**Experiment/correctness:** Run synthetic version-2 records through the producer and reader. Missing effective configuration must prevent a reportable comparison. Unknown schema versions must fail clearly; historical missing fields must remain unknown. Validate positive finite settings and the measured-request budget before model work starts.

**Artifacts/definition of done:** Contract document, valid/invalid fixtures, runtime decision, and a smoke run that reaches a readable summary on the chosen supported runtime.

## Task 2 — Make warmup, traffic, and workload reproducible

**Files/functions:** `loadgen.ts` (`DatasetRecord`, `describeWorkload`, `runLoad`, `sendOne`, `makeRequestResult`, `summarizeRun`, `persistRun`); proposed `csnbs/measure/test/loadgen.test.ts` and `csnbs/measure/test/fixtures/`. Keep reference cases in `csnbs/test_load_harness.py` as behavioral inspiration, not proof of TypeScript correctness.

**Implementation steps:**

- Separate setup, warmup, optional fixed settling period, measurement, and drain. Start with ten completed warmup requests, configurable and validated by pilot observations. Finish warmup and clear outstanding server work before starting the measured arrival clock. Save phase labels.
- Preserve `question_id`, category, source dataset, and a stable request ID. Create a seeded, category-balanced sequence for each trial; paired A/B trials replay the same sequence. Rotate or reshuffle by a recorded seed across repetitions. Use complete balanced cycles or a predefined balanced subset at every RPS.
- Keep offered arrivals independent of response completion. Record scheduling lateness separately from HTTP latency. Calibrate actual dispatch and server receipts so client connection limits or CPU work cannot silently throttle offered traffic.
- Make timeout, run deadline, overload termination, and drain policy explicit. If a predeclared safety limit stops a run, save its partial records and mark it aborted/overloaded; never silently convert it into completion-paced traffic. A client timeout does not prove server work stopped.
- Incrementally persist request outcomes, with exactly one terminal client outcome per scheduled request. Record cancellations, failures, and incomplete runs. Buffer writes to avoid putting synchronous disk work in the timing path.

**Experiment/correctness:** Use deterministic fake service delays over real HTTP as well as small unit fixtures. Prove requests overlap when service exceeds the interval; test a deliberately closed-loop reference; inject failures, late dispatch, timeouts, process interruption, and warmup backlog. Verify counts, ordering, phases, and recovery. Ensure `plannedToCompleteMs = dispatchLatenessMs + latencyMs` within numerical tolerance. Freeze image-loading boundaries: today's client preloads images outside request timing; server decode/preprocessing remains inside end-to-end latency.

**Artifacts/definition of done:** Passing TypeScript behavioral tests; fake open-versus-closed-loop demonstration; replayable workload manifest; durable raw records; a full balanced dev pass that is identical across paired conditions.

## Task 3 — Build the paired campaign runner

**Files/functions:** `csnbs/run_load_sweep.py` (`main`, `check_model_server`, `require_model_server`, `wait_for_server`) supplies lifecycle checks to preserve. Put the durable orchestrator in proposed `csnbs/measure/src/campaign.ts`; use `scripts/start_model_server.sh` as the existing launch contract. Read `scripts/run_ab_load_sweep.py` for lessons, but do not continue treating that diagnostic script as the canonical experiment runner.

**Implementation steps:**

- Accept a campaign specification containing configurations A/B, rates, repetition count, duration/sample budget, workload, seed, timeouts, and quality limits.
- For each rate and repetition, run an adjacent A/B or B/A pair, balancing order across repetitions. Randomize the order of rate blocks too, and persist the planned schedule before execution. Interleave trials, not requests to two simultaneously resident model copies.
- Start with A = stock 576 tokens and B = stock 128 tokens, fixed `important_ratio=0.5`, natural EOS, `max_new_tokens=64`, greedy decoding, cache enabled, batch size one. Verify the effective settings before each trial.
- Because token count is currently selected during server construction, restart the server between configurations; record startup separately and repeat the same warmup procedure. Do not silently mutate model state to avoid startup costs.
- Capture the exact run ID/output path directly from the child process. Eliminate “newest file since timestamp” matching. Use paths derived from the repository or explicit configuration instead of fixed `/workspace` assumptions.
- Require draining or termination of owned server work before the next trial. A responsive health endpoint alone does not prove an empty queue. Resume interrupted campaigns from durable trial state without overwriting prior trials.

**Experiment/correctness:** Dry-run the full schedule against fake mode. Inject wrong server identity/configuration, failed startup, child failure, and interrupted trials. Then perform five identical baseline trials and an A/A sham comparison on the reserved GPU before interpreting A/B gains.

**Artifacts/definition of done:** `campaign.json`, an immutable trial schedule, logs, and exact links to every run. One command executes a balanced repeated comparison and either produces verified trials or explicit failure records.

## Task 4 — Add honest lifecycle measurement

**Files/functions:** `csnbs/server.py` (`HealthResponse`, `health`, `InferRequest`, `InferResponse`, `infer`, `_infer_model`); `loadgen.ts` (`InferResponse`, `RequestResult`, `sendOne`); proposed `csnbs/measure/lifecycle-contract.md` and `csnbs/test_server_metrics.py`.

**Implementation steps:**

- Add request correlation and backward-compatible metrics/configuration fields. Export effective settings early so Task 3 can use them.
- Define the boundaries precisely: earliest observable request receipt, full payload ready/eligible, service admission, preprocessing start/end, generation start/end, postprocessing, and response handoff. “Handed to the server transport” is not “last byte received by the client.”
- On the current blocking server, measure handler/service intervals and retain an explicitly unattributed client-minus-server residual. This residual includes transport and unobserved waiting; it is not a queue-time measurement.
- Agree with Amay on responsive intake plus exactly one model execution worker, or an equivalent serving-owned arrangement that exposes admission and queue boundaries. Amay owns the execution change. Multiple unsynchronized threads calling the shared `generate()` are not the solution; the repository records a prior crash.
- Assign that serving arrangement a new variant ID and re-establish its baseline. Changing event-loop behavior can change capacity; it must not be hidden inside “instrumentation.” Compare instrumentation enabled/disabled on the same arrangement to estimate observer overhead.
- Compute durations within their own monotonic clock domains. Do not subtract Node `performance.now()` values from Python `perf_counter()` values. Where events cross processes, document and verify a common clock or preserve component-local durations.
- Record actual text tokens generated and actual visual tokens retained. Validate against the generation wrapper's BOS/EOS behavior; neither `max_new_tokens` nor raw output tensor length is automatically the number of generated text tokens. Keep CPU service wall time distinct from GPU event time. Amay supplies optional GPU-stage timings, collected without per-request pipeline-draining synchronization.

**Experiment/correctness:** First use a single-worker fake server with known service delays. Under overload, observed queue waiting must grow while fixed service time remains stable. Verify nonnegative ordered lifecycle events, correlation, error/cancellation outcomes, and reconciliation within the declared boundary. Then repeat on the model server and measure instrumentation overhead.

**Artifacts/definition of done:** Versioned lifecycle contract, correlated server events, per-request queue/service/total records, and an overhead check. If Amay's serving change is unavailable, complete client/service reporting and mark queue metrics unavailable; do not call the queue-measurement task complete.

## Task 5 — Establish tail and noise-floor policy

**Files/functions:** `loadgen.ts` (`percentile`, `summarizePercentiles`, `summarizeRun`); proposed `csnbs/measure/src/analyze.ts`, `csnbs/measure/methodology.md`, and tests; reporting reader for quality labels.

**Implementation steps:**

- Keep one documented percentile convention and add sample counts for every metric. Never average per-run p95 values and label the result the pooled p95.
- Proposed initial reporting gates: fewer than 200 successful measured observations → descriptive p50, p95 flagged as exploratory; 200–999 → p95 eligible with uncertainty; at least 1,000 → p99 eligible with uncertainty. These are project gates, not guarantees of precision. Correlation, workload coverage, failures, and observed stability still matter.
- Choose duration from offered rate and a predeclared measurement budget. At 0.5 RPS, 1,000 scheduled measured requests require 2,000 seconds (33m20s), before warmup/drain and assuming success. Round up when using full 90-question cycles. Do not silently extend a run until failures disappear from the quota.
- Use early baseline repetitions to estimate variability before the full sweep. Later use balanced A/B pairs and summarize effect sizes across independent trials/blocks. Requests within one run and repeated visits to the same image are not independent replication.
- Predefine primary metrics, desired resolution (3%, 5%, or 10%), stopping budget, and uncertainty procedure from pilot data. A pilot cannot promise that a fixed number of trials will detect 3%. Use paired trial-level resampling or another justified paired method; avoid treating thousands of correlated requests as thousands of independent experimental runs.
- Report improvement, regression, or inconclusive. An interval crossing zero is inconclusive, not equivalence. If the goal is to rule out a practically meaningful gain, use a predeclared equivalence margin and enough precision to support it. All latency summaries remain accompanied by failure/timeout rates; successful-only tails are conditional on success.

The general design principle—allocate repetitions based on sources of variation and report effect-size uncertainty—is supported by Kalibera and Jones, [Rigorous Benchmarking in Reasonable Time](https://kar.kent.ac.uk/33611/). The thresholds and implementation choices above are proposals for this project.

**Experiment/correctness:** Validate on synthetic unchanged, faster, slower, noisy, and correlated trial sets. An A/A campaign must not be narrated as a discovered optimization. Do not repeatedly add trials until a favorable significance threshold appears.

**Artifacts/definition of done:** Tail policy, baseline repeatability report, predeclared comparison protocol, and justified repetition budget for each target effect. Inconclusive results are valid outputs.

## Task 6 — Estimate sustainable capacity

**Files/functions:** proposed `csnbs/measure/src/capacity.ts`; `campaign.ts`; `loadgen.ts:summarizeRun`; server queue events from Task 4; resource-monitor integration based on `scripts/run_ab_load_sweep.py:ResourceMonitor`.

**Implementation steps:**

- Separate target offered RPS, actual client dispatch rate, server receipt rate when observable, successful completions during the measurement window, failures, outstanding work, and post-window drain time. Keep the historical completion-after-drain statistic under an explicit name; it is not sufficient to establish sustainable capacity.
- Define success before sweeping: delivery fidelity, acceptable failures, bounded/non-growing queue over a fixed observation horizon, and an optional user-facing latency SLO. No SLO has been supplied, so first report operational capacity; add SLO-constrained capacity only after the SLO is specified.
- Use a coarse pilot range suggested by existing evidence, such as 0.5–3 RPS, expanding if needed. Locate a bracket of sustainable versus overloaded rates, then test finer rates inside it. Randomize trials within each preselected stage; the adaptive stage order and all decisions stay recorded.
- Repeat boundary points. Return the highest verified sustainable rate and lowest verified failing rate, plus uncertainty/repetition evidence. If the bracket cannot resolve 3%, say so. Contradictory or nonmonotonic classifications trigger more evidence or an inconclusive bracket, not blind binary search.
- Collect a server-host resource time series outside the request path: sampled GPU utilization, used memory, available clocks/temperature/power and CPU load, with interval and GPU UUID. Label peak sampled device memory accurately; it is not exact peak KV-cache memory.
- Treat overload as a time-varying process: report queue growth and latency over time. A fixed overloaded-window p95 is not a steady-state tail.

**Experiment/correctness:** Start with a controlled serial fake service whose known processing rate lets you test the bracket and overload classification. Use longer confirmation windows than coarse screening windows. Repeat with the model under the same workload and isolated GPU conditions. A client that cannot deliver its target rate invalidates the capacity trial.

**Artifacts/definition of done:** Capacity bracket/report, time series, offered-versus-achieved plot, queue-growth plot, and documented pass/fail reasons. At least one sustainable and one overloaded boundary are verified, or the result explicitly remains unbounded/inconclusive.

## Task 7 — Measure workload sensitivity

**Files/functions:** `loadgen.ts` (`DatasetRecord`, `describeWorkload`, `main`, `runLoad`); dev metadata in `vis_pruner_copy/vispruner_eval_dataset/dev.json` (read-only input); proposed `csnbs/measure/workloads/` manifests and `analyze.ts` workload summaries.

**Implementation steps:**

- Preserve the existing six categories and their source/question IDs. The file contains TEXT_VQA, MME, and GQA examples; do not characterize every run as single-letter ScienceQA. Normalize label formatting in measurement metadata without changing the dataset's definitions.
- Record actual model-token prompt length, generated text length, and visual-token count. Character count is a useful preliminary descriptor, not token count. Distinguish text-only length from the formatted multimodal prompt.
- Run a balanced short-answer workload first. Add a separately labeled longer-output diagnostic with a documented prompt/output policy, then a fixed known mixture. Increased `max_new_tokens` alone does not guarantee a longer response.
- Replay the same prescribed workload in A/B. Natural-EOS trials measure real response behavior and may differ in generated length; use a separate controlled-length diagnostic to isolate output-length effects, with the necessary model-side support from Amay. Do not relabel forced-length traffic as realistic user traffic.
- Hold `important_ratio` fixed; vary one workload or configuration factor at a time. Keep the locked test set and scorer decisions with Sribhav. Repeated measurements of 90 dev examples do not establish broad population coverage.

**Experiment/correctness:** Compare A/B at a stable load, near the established boundary, and in one explicitly overloaded condition for each selected workload. Reuse trial seeds within pairs. Verify workload proportions and actual output-length distributions before interpreting the curves.

**Artifacts/definition of done:** Versioned workload manifests, token/category characterization, and a report stating exactly which workloads support each performance conclusion and where it fails or remains uncertain.

## Task 8 — Standardize optimization reports and CI

**Files/functions:** `csnbs/plot_load_comparison.py` (`hydrate_and_verify_raw_runs`, `verify_pair`, `story_rows`, `build_evidence`, `provenance_lines`); `csnbs/test_plot_load_comparison.py`; `csnbs/plotting.md`; proposed `csnbs/measure/src/report.ts` and `.github/workflows/measure.yml`.

**Implementation steps:**

- Define Amay's candidate handoff: immutable server commit/image, implementation/configuration ID, expected changed behavior, required flags, and correctness/accuracy evidence. `AMAY_TRACE_PLAN.md` explicitly supersedes the older speed-plan ordering; consume variants as available rather than requiring Triton first.
- Include the project's random-scoring control when evaluating whether a pruning method works, using the variant and validated accuracy results supplied by its owners. Rithvik measures it through the same harness; he does not implement the scoring method. Missing control or accuracy evidence remains pending and limits method-level conclusions.
- Support repeated trials and quality/null fields through a versioned report reader. The existing reader rejects duplicate RPS points and requires numeric p95/p99; it cannot just consume the new policy unchanged.
- Distinguish two comparisons: token reduction within the same implementation (576 versus 128), and engineering improvement at the same token count across versions. Keep the harness version fixed in an engineering comparison and allow the declared server revision to differ. Reject unrelated workload/hardware/policy differences.
- Produce a standard report: settings/provenance, request counts and failures, latency distributions, queue/service decomposition, offered/achieved throughput, capacity bracket, uncertainty, resource samples, output lengths, and limitations. Include the clean isolated HTTP baseline as a distinct run kind, never relabel a low-RPS load run as no-load.
- Verify selected or all raw request records against their summaries and hashes. Preserve absent stock/engineered/isolated conditions as unmeasured. Historical artifacts remain unchanged.
- Add CPU-only CI for the TypeScript runner, fake-server integration, schemas, reporting, and typechecking on the declared runtime. GPU campaigns run separately on the controlled GPU, with deliberate environment rebuilds through Docker.

**Experiment/correctness:** Run one end-to-end fake campaign through raw storage, analysis, and reporting. Reject tampered summaries, missing trials, mismatched hardware/workload, and fabricated metadata. Reproduce the historical plot through its version-1 reader. Test a genuine regression fixture so the report does not assume positive gains.

**Artifacts/definition of done:** A reusable report command producing `report.md`, tables, PNG/SVG plots, and a provenance manifest from raw files; a verified baseline report; passing CI. Every numerical conclusion links to trial/run IDs and its interpretation limits.

## Task 9 — Add queryable result storage

**Files/functions:** `loadgen.ts:persistRun` remains the raw-file source; proposed `csnbs/measure/db/001_initial.sql`, `csnbs/measure/src/ingest.ts`, and `csnbs/measure/compose.yml`.

**Implementation steps:** Add campaigns, trials, runs, per-request outcomes, and resource samples in Postgres, with run/request identity and time indexes. Ingest after measurement so database delays do not enter request timing. Make imports idempotent and preserve raw-file hashes. Store missing metrics as null. Use the shared Postgres target; a lightweight local alternative is optional, not an additional required implementation.

**Experiment/correctness:** Import a campaign twice without duplicates, reject conflicting records, recover from a failed import, and compare a stored A/B result with the same raw-file analysis. Do not duplicate percentile conventions in a database query using a different default definition.

**Artifacts/definition of done:** Schema, importer, documented comparison query, and matching file/database results. Storage provides history and retrieval; it does not replace raw evidence or gate GPU collection.

## Task 10 — Evaluate batching when Amay supplies it

**Files/functions:** candidate entries for `campaign.ts`, lifecycle contract, workload manifests, `capacity.ts`, and report dimensions. Any implementation of model batching stays with Amay.

**Implementation steps:** Verify the supplied implementation first. Measure supported batch sizes 1, 2, 4, and 8 at fixed token count/workload; sweep batch wait separately. Record requested maximum versus actual batch occupancy, queue wait, batch-formation wait if exposed, throughput, valid tails, sampled memory, and failures/OOMs. Record scheduling policy and actual padded/packed sequence work if Amay exposes it.

**Experiment/correctness:** Begin with a new batch-one control from the same implementation. Compare at equal offered load and separately compare each variant's capacity; do not combine those questions. Do not force different token counts within one run unless that serving variant supports it and the mixed policy is explicitly defined.

**Artifacts/definition of done:** Throughput-versus-waiting tradeoff report with measured batch occupancy and supported operating range. If batching is unavailable, retain a ready experiment specification and mark this task pending.

## Four-week checkpoints

- **Week 1:** Contracts/runtime reconciled; deterministic TypeScript tests; warmup and workload fixes; campaign dry-run; first baseline repetitions. Write down the provisional noise floor before budgeting the broad GPU campaign.
- **Week 2:** Effective configuration and request correlation; measured lifecycle boundaries where serving support exists; isolated HTTP baseline; first repeatable capacity bracket.
- **Week 3:** Boundary confirmation, meaningful tail sample budgets, paired uncertainty, and selected workload sensitivity. Freeze methodology for the final baseline campaign.
- **Week 4:** Final campaign/report, standardized candidate handoff, CI, and result ingestion. Produce accuracy-versus-throughput data only when Sribhav provides matching, validated accuracy results for the same configurations; otherwise label accuracy pending.

Every GPU campaign needs one operator on the GPU, recorded effective settings, warmup, workload replay, balanced order, raw artifacts, and a predeclared stopping budget. Use a clean attributable source revision for reportable runs; the current checkout's unrelated untracked directories should be preserved, not deleted to manufacture a clean state.

## First working session

Complete Task 1's contract draft and write failing behavioral tests for three concrete cases: the default one-request smoke run, warmup spilling into measurement, and the category mix changing with RPS. Then Rithvik implements the warmup/workload changes and gets them reviewed. This creates a defensible foundation before another expensive model sweep.

## Inspection and verification for this plan

Read the full exported Google Doc; the current TypeScript generator, server, sweep scripts, lifecycle wrappers, tests, plotting reader, dev metadata, and report limitations. Checked all ten September 1 run files against their request-record counts and unique sequences. Confirmed the workload-composition issue from the sequence-to-dataset mapping. Local TypeScript typecheck passed on Node 25.7.0; all sixteen existing plot tests passed. No new GPU experiment was run, no remote deployment was verified, and no implementation code or historical result was changed.
