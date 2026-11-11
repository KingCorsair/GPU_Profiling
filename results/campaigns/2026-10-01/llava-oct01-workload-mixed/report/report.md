# Measurement report: llava-oct01-workload-mixed

Purpose: **ab**. 4/4 trials completed and passed raw-artifact verification.

Interpret these results only within the measured workload and supplied server configuration.

Measured requests per trial: 90; warmup: 10; seed: 465.

| Trial | Variant | Offered RPS | Successes / offered | Window RPS | RPS including drain | p50 ms | p95 ms | p99 ms | Drain ms | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| r1-p1-v0 | baseline-A | 1 | 90 / 90 | 1.00 | 1.00 | 276.24 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p1-v0-1e42c711/runs/2026-10-01T21-44-40.037Z_rps-1_423d00f6/run.json) |
| r1-p1-v1 | pruned-128 | 1 | 90 / 90 | 1.00 | 1.00 | 253.71 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p1-v1-950858e6/runs/2026-10-01T21-46-26.542Z_rps-1_5636064e/run.json) |
| r1-p2-v1 | pruned-128 | 1 | 90 / 90 | 0.99 | 0.99 | 239.73 | unavailable | unavailable | 1055.92 | [raw manifest](../r1-p2-v1-40296e5a/runs/2026-10-01T21-48-12.323Z_rps-1_3ec476f9/run.json) |
| r1-p2-v0 | baseline-A | 1 | 90 / 90 | 1.00 | 1.00 | 268.85 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p2-v0-fbae2630/runs/2026-10-01T21-49-59.032Z_rps-1_aa723538/run.json) |

## Observed host-wall stages

Only successful measured requests contribute. Each cell is a separate p50 in milliseconds with its observed count; absent observations remain unavailable. Per-stage medians are not additive. These host-wall durations do not isolate GPU prefill, decode or kernel time. The JSON retains independently gated p95/p99 (200/1000 observations).

Boundaries: base64 is handler entry through validated decoding; preprocessing covers image/prompt preparation and existing device transfers; generation covers the entire unchanged model.generate call including its existing synchronizations and diagnostic writes; postprocessing covers batch_decode and strip; observation covers token-contract checks, the extra post-decode output host copy and answer character counting. Service time also contains work outside these named stages.

| Trial | Base64 p50 ms (n) | Preprocess p50 ms (n) | Generation wall p50 ms (n) | Postprocess p50 ms (n) | Observation p50 ms (n) | Successful measured |
|---|---:|---:|---:|---:|---:|---:|
| r1-p1-v0 | 0.46 (90) | 63.42 (90) | 195.95 (90) | 0.28 (90) | 0.09 (90) | 90 |
| r1-p1-v1 | 0.46 (90) | 67.64 (90) | 173.22 (90) | 0.41 (90) | 0.10 (90) | 90 |
| r1-p2-v1 | 0.51 (90) | 60.45 (90) | 172.12 (90) | 0.41 (90) | 0.10 (90) | 90 |
| r1-p2-v0 | 0.45 (90) | 58.41 (90) | 196.71 (90) | 0.29 (90) | 0.09 (90) | 90 |

## Observed generation lengths and stopping

Counts require the validated output-ID contract; a configured maximum is never substituted for an observed length. Generated steps include terminal EOS and other special IDs. Reached-budget counts include EOS at the budget; cap without EOS distinguishes budget termination. Missing/invalid observations stay unavailable, including all counts when coverage is zero.

| Trial | Configured max | Observed / successful | Steps min / p50 / max | EOS terminated | Reached budget | EOS at budget | Cap without EOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1-p1-v0 | 64 | 90 / 90 | 2 / 2 / 53 | 90 | 0 | 0 | 0 |
| r1-p1-v1 | 64 | 90 / 90 | 2 / 3 / 64 | 89 | 1 | 0 | 1 |
| r1-p2-v1 | 64 | 90 / 90 | 2 / 3 / 64 | 89 | 1 | 0 | 1 |
| r1-p2-v0 | 64 | 90 / 90 | 2 / 2 / 53 | 90 | 0 | 0 | 0 |

## Paired comparisons

- 1 RPS, window throughput: inconclusive; 2 pairs; effect -1.11% (positive means improved); 95% interval unavailable. Fewer than five independent paired blocks; uncertainty is not resolved
- 1 RPS, p50 latency: inconclusive; 2 pairs; effect 8.16% (positive means improved); 95% interval unavailable. Fewer than five independent paired blocks; uncertainty is not resolved

## Baseline repeatability

- 1 RPS: 2 baseline trial observations, 2 paired blocks. Baseline leg only. Trial p50 latency median 268.85 ms; relative range 2.75%; median absolute deviation 0.00%. Descriptive pilot spread, not a guaranteed detectable effect. Fewer than five baseline repetitions; provisional spread only

## Repeated fixed-window capacity screens

- baseline-A: inconclusive; highest all-pass tested rate unavailable RPS, lowest all-fail tested rate unavailable RPS. One or more tested rates remain inconclusive; Bounds describe repeated fixed-window client screens only; longer boundary trials and observable queue behavior are still required for sustainable capacity
  - 1 RPS: inconclusive, 2/2 prescribed trials. At least one trial was too small/short or otherwise inconclusive
- pruned-128: inconclusive; highest all-pass tested rate unavailable RPS, lowest all-fail tested rate unavailable RPS. One or more tested rates remain inconclusive; Bounds describe repeated fixed-window client screens only; longer boundary trials and observable queue behavior are still required for sustainable capacity
  - 1 RPS: inconclusive, 2/2 prescribed trials. At least one trial was too small/short or otherwise inconclusive

## Run quality and capacity screens

- r1-p1-v0: inconclusive. Screen is too short or small: require at least 30 seconds and 200 measured requests; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p1-v1: inconclusive. Screen is too short or small: require at least 30 seconds and 200 measured requests; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p2-v1: inconclusive. Screen is too short or small: require at least 30 seconds and 200 measured requests; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p2-v0: inconclusive. Screen is too short or small: require at least 30 seconds and 200 measured requests; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim

## Limits and outstanding owner inputs

- Each capacity classification describes a fixed observation window. Sustainable capacity requires repeated longer boundary trials.
- Tails are conditional on successful requests. p95 requires 200 successes and p99 requires 1000; sample size does not guarantee precision.
- Paired uncertainty resamples independent trial blocks, not individual requests. Fewer than five pairs remains inconclusive.
- No validated matching accuracy scores are supplied. Accuracy versus throughput and random-scoring controls remain pending with the accuracy owner.
- Model batching, exact KV-cache allocation and any new prefill/decode CUDA event boundaries require the serving owner. Optional HTTP token observations remain unavailable when their wrapper contract checks fail.
- Service wall time is not CUDA event time. Client minus service is an unattributed residual, not measured server queue time.
- Results apply to the declared workload, observed output lengths and recorded instrumentation. Claims about longer outputs, mixed traffic or other instrumentation settings require corresponding completed matched studies.

The machine-readable report preserves run IDs, hashes, server identity, workload order and comparison reasons. Historical evidence is unchanged.
