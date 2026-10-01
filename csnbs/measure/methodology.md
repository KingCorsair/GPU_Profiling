# Measurement and interpretation policy

Use [contracts.md](contracts.md) for file/clock semantics and [lifecycle-contract.md](lifecycle-contract.md) for what the server can currently observe. The implementation and a successful test suite establish measurement capabilities. Only finalized, verified experimental artifacts establish measured findings. The current GPU pilot must be interpreted from its completed artifacts; this document does not mark the full roadmap complete.

## Freeze the experiment before collection

Save an immutable specification with the question, baseline/candidate identities, intended changed control, primary metric, workload/order seed, offered rates, measured budget per trial, repetitions, warmup/settling, timeouts, stopping limits, and failure/retry policy. Declare desired effect resolution and GPU-time budget in the accompanying protocol; neither is inferred from a favorable pilot. Keep that protocol with the campaign.

Use an exclusive GPU, immutable harness/server revisions, verified checkpoint identity, and server-confirmed effective settings. Preserve runtime, GPU UUID/driver, container identity when available, decode/EOS policy and batching configuration. Unknown provenance remains unknown. `quality.reportable` is an individual-run gate, not proof that all desired controls were observed or that a gain exceeds noise.

For LLaVA, 576 visual tokens is the unpruned reference. The token-count experiment changes 576 to 128 while holding `important_ratio=0.5`, implementation, prompts, output policy and all other controls fixed. An engineering experiment holds token count fixed and declares the specific serving revision/implementation change. Do not change token count and the importance ratio together.

The runner randomizes rate-block order and alternates A/B versus B/A order across paired repetitions. Both legs use the same measured question order/seed; repetition advances the seed. It restarts the server and repeats warmup for every leg. A balanced schedule reduces order confounding but does not remove thermal drift or prove independent blocks. Inspect the saved order, resource traces and run conditions.

## Sample budgets and tail policy

Use a fixed measured request count at every rate, preferably complete workload cycles. Do not let higher RPS silently select different question categories. With the current 90-question workload:

| Purpose | Measured requests per trial | Duration at 0.5 RPS | Duration at 1 RPS | Interpretation |
|---|---:|---:|---:|---|
| Baseline pilot, one cycle | 90 | 180 s | 90 s | Descriptive p50; p95/p99 unavailable |
| p95-oriented comparison, three cycles | 270 | 540 s | 270 s | p95 becomes eligible at 200 successful observations; p99 unavailable |
| p99-oriented experiment, twelve cycles | 1080 | 2160 s | 1080 s | p99 becomes eligible at 1000 successful observations; precision still requires assessment |

These are per-trial measured windows; startup, warmup, settling, drain and paired repetitions add time. A count budget is scheduled observations, not a promise of that many successes. Never extend collection until failures disappear from a successful-request quota.

The committed baseline pilot specification declares three pairs of the same 576-token configuration: six 90-request trials. It can expose repeatability problems but supplies fewer than five paired blocks and no eligible tails. The token A/B specification declares five pairs at each of four rates, with 270 measured requests per trial. That file is a protocol, not evidence it has run. The 1080-request row above is an optional future budget, not an executed campaign.

All canonical client summaries use nearest-rank `sorted[ceil(q*n)-1]`, with a separate count per metric. p50 is descriptive whenever a sample exists; p95 is withheld below 200 observations and p99 below 1000. These gates do not guarantee stable tails, representativeness, or independence. Successful-request latency excludes failures; always present scheduled, dispatched, successful, failed, timed-out and cancelled counts alongside it. Do not average per-run percentiles and call the result a pooled percentile.

## Repeatability and paired effects

Run identical baseline trials and an A/A sham comparison before interpreting an optimization. `baselineRepeatability()` reports trial median, observed range, relative range and median absolute deviation. At least five baseline repetitions are a starting check; the observed spread is not a guaranteed detection threshold. The report currently summarizes variant-0 baseline legs, so a three-pair pilot's repeatability section contains three observations even though six identical runs exist; reviewing all six requires an explicit combined baseline analysis.

`comparePairs()` works at one offered rate and uses independent paired blocks as the resampling unit. More requests inside a run are not more trial replications. Each block contributes a percentage effect relative to its baseline: `(B-A)/A` for throughput and `(A-B)/A` for latency, so positive means better. The result is the median paired effect with a 95% percentile-bootstrap interval from 10,000 whole-block resamples using the recorded seed. At least five matched pairs are required for an interval; repeated run IDs, mismatched controls, failed run gates, unavailable metrics, incomplete prescribed trials, or retried failures keep the conclusion inconclusive.

A wholly positive interval supports improvement under the measured controls; a wholly negative interval supports regression. Crossing zero means inconclusive, not equivalence. With very few blocks, the bootstrap interval is only a limited description of uncertainty. Do not repeatedly add trials or select rates until the interval becomes favorable. Diagnose a failed fixed-budget experiment, retain it, and preregister a separate follow-up.

The analysis API distinguishes `token-count` from `implementation` comparisons. The report command defaults to token-count comparisons; set `comparisonKind:"implementation"` in the campaign specification for an engineering candidate. The report passes that choice to the comparison checks. Intentionally different server revisions must not be disguised as identical to bypass matching checks.

## Throughput, capacity and workload sensitivity

Prefer `successfulThroughputWithinWindowRps` as the current report's primary throughput metric. It counts completions inside the predeclared window. Also show `successfulThroughputIncludingDrainRps`, drain duration, delivered arrival rate and outstanding requests at the window end. At low offered load, equal delivered throughput is expected even when latency changes; it does not prove equal maximum capacity.

`classifyCapacity()` is a finite-window screen. It rejects materially late or incorrect client delivery, identifies failed operational criteria, and checks growth in client outstanding work. A passing screen requires at least 30 seconds and 200 measured requests; it still returns `sustainableCapacityEstablished:false`. The `overloaded` label can follow any request failure, so diagnose whether the cause was actual overload, transport, invalid output, or a service defect. Client outstanding work includes transport and service; it is not measured server queue depth.

Declare coarse rates, then separate longer repeated boundary trials to bracket the highest passing and lowest failing rate under a workload, latency/failure policy and observation duration. The current screen does not execute this adaptive search or establish a sustainable-capacity ceiling. A different workload or output policy needs its own bracket.

Keep isolated full-HTTP runs separate from open-loop load runs. To study output length, use a declared short/long/mixed workload and verify actual generated token counts when the server exposes them. Merely raising `max_new_tokens` does not force longer answers. Single-letter ScienceQA-style answers emphasize prefill and can overstate benefits for longer responses. Prefill/decode attribution, batching/padding effects, max concurrency and exact KV-cache savings remain pending appropriate serving measurements.

## Candidate handoff from Amay

Use [candidate-handoff.example.json](campaigns/candidate-handoff.example.json) as an incomplete handoff template, not a runnable campaign. Supply:

- Immutable baseline/candidate commits or image digests, separate checkout/launch commands, implementation IDs, expected effective configuration, and a narrow explanation of the changed execution behavior.
- Matching model/checkpoint hashes, tokenizer/prompt, dtype/backend, visual-token count/importance ratio, decoding/EOS settings and batching policy; identify the one intentionally changed dimension.
- Evidence of real removal versus masking, kept-token spatial order, downstream position/mask/cache correctness and regression tests, as applicable to the change. Measurement must not silently repair the candidate.
- The observable request/resource fields, their timing boundaries, actual token-count semantics, and all unavailable metrics. Any new intake/batching policy has its own variant identity and fresh baseline.
- Correctness and accuracy evidence identifiers from the appropriate owner, supported workloads/settings, and known failure limits. A random-scoring control is a separate owner-supplied method variant, measured through the same harness when assessing whether importance scoring helps.

Before spending the final campaign budget, verify the handoff, run the smallest useful correctness/integration check, and confirm effective configuration. Missing inputs keep the corresponding claim pending; they do not make synthetic or older unrelated evidence a replacement.

## Matching accuracy input from Sribhav

An accuracy-throughput point needs validated evaluation results for the exact implementation/model/checkpoint, token count, importance ratio, prompt/decoding policy and relevant preprocessing controls used by the speed run. Keep the evaluation workload/split identity and serving workload identity distinct and visible. Supply the locked split hash, eval code/scorer revision, validation/agreement evidence, per-category counts/scores including OCR/counting/spatial reasoning, and baseline/candidate/random-control identities where applicable. Use dev for tuning; do not revise the method after examining the locked test results.

The measurement pipeline neither invents scores nor implements Sribhav's scorer. Until matching validated inputs arrive, publish throughput/latency evidence with **accuracy pending**, and leave the joint accuracy-versus-throughput chart incomplete. Overall accuracy alone must not conceal category regressions.

## Evidence retained for every conclusion

Retain the frozen campaign/specification and all attempt statuses, exact run IDs, raw request bytes/hashes, effective configuration and source/runtime identity, resource observations, report verification output and limitations. Database import follows collection; its stored summaries reproduce file evidence rather than introducing another percentile implementation. Historical exploratory runs and standalone CUDA calibration remain separately labeled. Instrumentation overhead, long-output studies and final capacity/accuracy claims require their own completed evidence; a working pipeline does not mark them measured.
