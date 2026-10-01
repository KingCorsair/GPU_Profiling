# Measurement report: llava-oct01-isolated

Purpose: **ab**. 10/10 trials completed and passed raw-artifact verification.

Interpret these results only within the measured workload and supplied server configuration.

These isolated HTTP requests are completion-paced: each response finishes before the next request. Offered RPS is unavailable. The configured rate is only a campaign placeholder; completion rate does not establish serving capacity.

Measured requests per trial: 90; warmup: 10; seed: 465.

| Trial | Variant | Offered RPS | Successes / offered | Window RPS | RPS including drain | p50 ms | p95 ms | p99 ms | Drain ms | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| r1-p1-v0 | baseline-A | unavailable (isolated) | 90 / 90 | 2.44 | 2.44 | 471.82 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p1-v0-ca225736/runs/2026-10-01T21-16-11.745Z_rps-1_65e0259f/run.json) |
| r1-p1-v1 | pruned-128 | unavailable (isolated) | 90 / 90 | 2.74 | 2.74 | 417.26 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p1-v1-b26f98e5/runs/2026-10-01T21-17-06.090Z_rps-1_aaa57c84/run.json) |
| r1-p2-v1 | pruned-128 | unavailable (isolated) | 90 / 90 | 2.77 | 2.77 | 405.84 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p2-v1-43c10d72/runs/2026-10-01T21-17-55.142Z_rps-1_07a4a876/run.json) |
| r1-p2-v0 | baseline-A | unavailable (isolated) | 90 / 90 | 2.38 | 2.38 | 478.68 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p2-v0-a7bdcc9f/runs/2026-10-01T21-18-43.579Z_rps-1_97f63972/run.json) |
| r1-p3-v0 | baseline-A | unavailable (isolated) | 90 / 90 | 2.43 | 2.43 | 471.42 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p3-v0-37ff6866/runs/2026-10-01T21-19-38.495Z_rps-1_2df07e3b/run.json) |
| r1-p3-v1 | pruned-128 | unavailable (isolated) | 90 / 90 | 2.76 | 2.76 | 407.88 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p3-v1-0ddeb045/runs/2026-10-01T21-20-32.473Z_rps-1_c57c520d/run.json) |
| r1-p4-v1 | pruned-128 | unavailable (isolated) | 90 / 90 | 2.77 | 2.77 | 398.15 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p4-v1-7c65fe7a/runs/2026-10-01T21-21-20.925Z_rps-1_f20a4891/run.json) |
| r1-p4-v0 | baseline-A | unavailable (isolated) | 90 / 90 | 2.40 | 2.40 | 479.36 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p4-v0-fbf7f59c/runs/2026-10-01T21-22-09.773Z_rps-1_2358dcf8/run.json) |
| r1-p5-v0 | baseline-A | unavailable (isolated) | 90 / 90 | 2.41 | 2.41 | 467.98 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p5-v0-3afe0fb9/runs/2026-10-01T21-23-02.492Z_rps-1_cde847ef/run.json) |
| r1-p5-v1 | pruned-128 | unavailable (isolated) | 90 / 90 | 2.72 | 2.72 | 419.70 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p5-v1-f1a814e5/runs/2026-10-01T21-23-57.098Z_rps-1_3d08daa6/run.json) |

## Observed host-wall stages

Only successful measured requests contribute. Each cell is a separate p50 in milliseconds with its observed count; absent observations remain unavailable. Per-stage medians are not additive. These host-wall durations do not isolate GPU prefill, decode or kernel time. The JSON retains independently gated p95/p99 (200/1000 observations).

Boundaries: base64 is handler entry through validated decoding; preprocessing covers image/prompt preparation and existing device transfers; generation covers the entire unchanged model.generate call including its existing synchronizations and diagnostic writes; postprocessing covers batch_decode and strip; observation covers token-contract checks, the extra post-decode output host copy and answer character counting. Service time also contains work outside these named stages.

| Trial | Base64 p50 ms (n) | Preprocess p50 ms (n) | Generation wall p50 ms (n) | Postprocess p50 ms (n) | Observation p50 ms (n) | Successful measured |
|---|---:|---:|---:|---:|---:|---:|
| r1-p1-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 90 |
| r1-p1-v1 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 90 |
| r1-p2-v1 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 90 |
| r1-p2-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 90 |
| r1-p3-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 90 |
| r1-p3-v1 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 90 |
| r1-p4-v1 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 90 |
| r1-p4-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 90 |
| r1-p5-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 90 |
| r1-p5-v1 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 90 |

## Observed generation lengths and stopping

Counts require the validated output-ID contract; a configured maximum is never substituted for an observed length. Generated steps include terminal EOS and other special IDs. Reached-budget counts include EOS at the budget; cap without EOS distinguishes budget termination. Missing/invalid observations stay unavailable, including all counts when coverage is zero.

| Trial | Configured max | Observed / successful | Steps min / p50 / max | EOS terminated | Reached budget | EOS at budget | Cap without EOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1-p1-v0 | 64 | 0 / 90 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p1-v1 | 64 | 0 / 90 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p2-v1 | 64 | 0 / 90 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p2-v0 | 64 | 0 / 90 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p3-v0 | 64 | 0 / 90 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p3-v1 | 64 | 0 / 90 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p4-v1 | 64 | 0 / 90 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p4-v0 | 64 | 0 / 90 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p5-v0 | 64 | 0 / 90 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p5-v1 | 64 | 0 / 90 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |

## Paired comparisons

- Isolated completion-paced requests, serial completion rate: improvement; 5 pairs; effect 13.75% (positive means improved); 95% interval 12.44 to 16.47%. 
- Isolated completion-paced requests, p50 latency: improvement; 5 pairs; effect 13.48% (positive means improved); 95% interval 10.32 to 16.94%. 

## Baseline repeatability

- Isolated completion-paced requests: 5 baseline trial observations, 5 paired blocks. Baseline leg only. Trial p50 latency median 471.82 ms; relative range 2.41%; median absolute deviation 0.81%. Descriptive pilot spread, not a guaranteed detectable effect. 

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

The machine-readable report preserves run IDs, hashes, server identity, workload order and comparison reasons. Historical evidence is unchanged.
