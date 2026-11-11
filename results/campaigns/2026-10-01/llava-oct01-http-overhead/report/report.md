# Measurement report: llava-oct01-http-overhead

Purpose: **ab**. 10/10 trials completed and passed raw-artifact verification.

Interpret these results only within the measured workload and supplied server configuration.

These isolated HTTP requests are completion-paced: each response finishes before the next request. Offered RPS is unavailable. The configured rate is only a campaign placeholder; completion rate does not establish serving capacity.

Measured requests per trial: 90; warmup: 10; seed: 465.

| Trial | Variant | Offered RPS | Successes / offered | Window RPS | RPS including drain | p50 ms | p95 ms | p99 ms | Drain ms | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| r1-p1-v0 | observations-off | unavailable (isolated) | 90 / 90 | 2.40 | 2.40 | 476.28 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p1-v0-a76aa0fa/runs/2026-10-01T21-27-08.701Z_rps-1_c8bbb446/run.json) |
| r1-p1-v1 | observations-on | unavailable (isolated) | 90 / 90 | 2.43 | 2.43 | 468.20 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p1-v1-53853771/runs/2026-10-01T21-28-03.259Z_rps-1_0375b2e8/run.json) |
| r1-p2-v1 | observations-on | unavailable (isolated) | 90 / 90 | 2.43 | 2.43 | 466.54 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p2-v1-73802a94/runs/2026-10-01T21-28-57.004Z_rps-1_757a463b/run.json) |
| r1-p2-v0 | observations-off | unavailable (isolated) | 90 / 90 | 2.37 | 2.37 | 483.79 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p2-v0-bcb36e5a/runs/2026-10-01T21-29-50.942Z_rps-1_9de3a774/run.json) |
| r1-p3-v0 | observations-off | unavailable (isolated) | 90 / 90 | 2.41 | 2.41 | 471.34 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p3-v0-541d9ffa/runs/2026-10-01T21-30-46.224Z_rps-1_5193b512/run.json) |
| r1-p3-v1 | observations-on | unavailable (isolated) | 90 / 90 | 2.40 | 2.40 | 472.01 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p3-v1-3f6b45f4/runs/2026-10-01T21-31-40.754Z_rps-1_afa109e5/run.json) |
| r1-p4-v1 | observations-on | unavailable (isolated) | 90 / 90 | 2.38 | 2.38 | 473.62 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p4-v1-4871d9e4/runs/2026-10-01T21-32-35.212Z_rps-1_d646f819/run.json) |
| r1-p4-v0 | observations-off | unavailable (isolated) | 90 / 90 | 2.38 | 2.38 | 469.69 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p4-v0-bd163dda/runs/2026-10-01T21-33-29.815Z_rps-1_65e081fe/run.json) |
| r1-p5-v0 | observations-off | unavailable (isolated) | 90 / 90 | 2.42 | 2.42 | 474.02 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p5-v0-f6c2e25c/runs/2026-10-01T21-34-23.850Z_rps-1_2724c8a2/run.json) |
| r1-p5-v1 | observations-on | unavailable (isolated) | 90 / 90 | 2.39 | 2.39 | 471.90 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p5-v1-5449e4fb/runs/2026-10-01T21-35-18.466Z_rps-1_b225da6d/run.json) |

## Observed host-wall stages

Only successful measured requests contribute. Each cell is a separate p50 in milliseconds with its observed count; absent observations remain unavailable. Per-stage medians are not additive. These host-wall durations do not isolate GPU prefill, decode or kernel time. The JSON retains independently gated p95/p99 (200/1000 observations).

Boundaries: base64 is handler entry through validated decoding; preprocessing covers image/prompt preparation and existing device transfers; generation covers the entire unchanged model.generate call including its existing synchronizations and diagnostic writes; postprocessing covers batch_decode and strip; observation covers token-contract checks, the extra post-decode output host copy and answer character counting. Service time also contains work outside these named stages.

| Trial | Base64 p50 ms (n) | Preprocess p50 ms (n) | Generation wall p50 ms (n) | Postprocess p50 ms (n) | Observation p50 ms (n) | Successful measured |
|---|---:|---:|---:|---:|---:|---:|
| r1-p1-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 90 |
| r1-p1-v1 | 0.14 (90) | 22.57 (90) | 439.07 (90) | 0.36 (90) | 0.08 (90) | 90 |
| r1-p2-v1 | 0.14 (90) | 23.40 (90) | 439.16 (90) | 0.39 (90) | 0.08 (90) | 90 |
| r1-p2-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 90 |
| r1-p3-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 90 |
| r1-p3-v1 | 0.14 (90) | 28.18 (90) | 444.21 (90) | 0.41 (90) | 0.08 (90) | 90 |
| r1-p4-v1 | 0.16 (90) | 27.85 (90) | 441.58 (90) | 0.44 (90) | 0.09 (90) | 90 |
| r1-p4-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 90 |
| r1-p5-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 90 |
| r1-p5-v1 | 0.15 (90) | 25.42 (90) | 443.60 (90) | 0.38 (90) | 0.08 (90) | 90 |

## Observed generation lengths and stopping

Counts require the validated output-ID contract; a configured maximum is never substituted for an observed length. Generated steps include terminal EOS and other special IDs. Reached-budget counts include EOS at the budget; cap without EOS distinguishes budget termination. Missing/invalid observations stay unavailable, including all counts when coverage is zero.

| Trial | Configured max | Observed / successful | Steps min / p50 / max | EOS terminated | Reached budget | EOS at budget | Cap without EOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1-p1-v0 | 64 | 0 / 90 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p1-v1 | 64 | 90 / 90 | 2 / 11 / 53 | 90 | 0 | 0 | 0 |
| r1-p2-v1 | 64 | 90 / 90 | 2 / 11 / 53 | 90 | 0 | 0 | 0 |
| r1-p2-v0 | 64 | 0 / 90 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p3-v0 | 64 | 0 / 90 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p3-v1 | 64 | 90 / 90 | 2 / 11 / 53 | 90 | 0 | 0 | 0 |
| r1-p4-v1 | 64 | 90 / 90 | 2 / 11 / 53 | 90 | 0 | 0 | 0 |
| r1-p4-v0 | 64 | 0 / 90 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p5-v0 | 64 | 0 / 90 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p5-v1 | 64 | 90 / 90 | 2 / 11 / 53 | 90 | 0 | 0 | 0 |

## Paired comparisons

- Isolated completion-paced requests, serial completion rate: inconclusive; 5 pairs; effect 0.21% (positive means improved); 95% interval -1.23 to 2.77%. 
- Isolated completion-paced requests, p50 latency: inconclusive; 5 pairs; effect 0.45% (positive means improved); 95% interval -0.84 to 3.56%. 

## Baseline repeatability

- Isolated completion-paced requests: 5 baseline trial observations, 5 paired blocks. Baseline leg only. Trial p50 latency median 474.02 ms; relative range 2.97%; median absolute deviation 0.57%. Descriptive pilot spread, not a guaranteed detectable effect. 

## Repeated fixed-window capacity screens

Not applicable: isolated completion-paced requests have no offered-load capacity bracket.

## Run quality and capacity screens

- r1-p1-v0: capacity not applicable. Capacity requires open-loop arrivals; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p1-v1: capacity not applicable. Capacity requires open-loop arrivals; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p2-v1: capacity not applicable. Capacity requires open-loop arrivals; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p2-v0: capacity not applicable. Capacity requires open-loop arrivals; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p3-v0: capacity not applicable. Capacity requires open-loop arrivals; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p3-v1: capacity not applicable. Capacity requires open-loop arrivals; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p4-v1: capacity not applicable. Capacity requires open-loop arrivals; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p4-v0: capacity not applicable. Capacity requires open-loop arrivals; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p5-v0: capacity not applicable. Capacity requires open-loop arrivals; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p5-v1: capacity not applicable. Capacity requires open-loop arrivals; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim

## Limits and outstanding owner inputs

- Isolated HTTP requests are completion-paced. Their configured rate is a placeholder, not offered load; serial completion rate is not serving capacity.
- Tails are conditional on successful requests. p95 requires 200 successes and p99 requires 1000; sample size does not guarantee precision.
- Paired uncertainty resamples independent trial blocks, not individual requests. Fewer than five pairs remains inconclusive.
- No validated matching accuracy scores are supplied. Accuracy versus throughput and random-scoring controls remain pending with the accuracy owner.
- Model batching, exact KV-cache allocation and any new prefill/decode CUDA event boundaries require the serving owner. Optional HTTP token observations remain unavailable when their wrapper contract checks fail.
- Service wall time is not CUDA event time. Client minus service is an unattributed residual, not measured server queue time.
- Results apply to the declared workload, observed output lengths and recorded instrumentation. Claims about longer outputs, mixed traffic or other instrumentation settings require corresponding completed matched studies.
- This comparison changes only extended HTTP observations. Existing model-internal CUDA events, synchronizations and diagnostic writes remain enabled in both legs; it does not estimate total instrumentation overhead.

The machine-readable report preserves run IDs, hashes, server identity, workload order and comparison reasons. Historical evidence is unchanged.
