# Measurement report: llava-oct01-primary

Purpose: **ab**. 20/20 trials completed and passed raw-artifact verification.

Interpret these results only within the measured workload and supplied server configuration.

Measured requests per trial: 270; warmup: 10; seed: 465.

| Trial | Variant | Offered RPS | Successes / offered | Window RPS | RPS including drain | p50 ms | p95 ms | p99 ms | Drain ms | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| r3-p1-v0 | baseline-A | 3 | 270 / 270 | 2.46 | 2.47 | 10588.18 | 26164.39 | unavailable | 19258.66 | [raw manifest](../r3-p1-v0-69033c9e/runs/2026-10-01T20-05-32.074Z_rps-3_d0559558/run.json) |
| r3-p1-v1 | pruned-128 | 3 | 270 / 270 | 2.81 | 2.82 | 2860.29 | 6072.49 | unavailable | 5808.62 | [raw manifest](../r3-p1-v1-5cffcac1/runs/2026-10-01T20-07-38.731Z_rps-3_1acc2d63/run.json) |
| r3-p2-v1 | pruned-128 | 3 | 270 / 270 | 2.77 | 2.80 | 4963.53 | 9565.70 | unavailable | 6466.02 | [raw manifest](../r3-p2-v1-88836c9a/runs/2026-10-01T20-09-30.765Z_rps-3_e2673e4d/run.json) |
| r3-p2-v0 | baseline-A | 3 | 270 / 270 | 2.43 | 2.46 | 9574.22 | 22827.76 | unavailable | 19861.13 | [raw manifest](../r3-p2-v0-bd59a363/runs/2026-10-01T20-11-23.570Z_rps-3_2abc7c6d/run.json) |
| r3-p3-v0 | baseline-A | 3 | 270 / 270 | 2.49 | 2.47 | 8414.72 | 23198.66 | unavailable | 19520.67 | [raw manifest](../r3-p3-v0-decdbd99/runs/2026-10-01T20-13-30.777Z_rps-3_fd32738c/run.json) |
| r3-p3-v1 | pruned-128 | 3 | 270 / 270 | 2.83 | 2.82 | 3426.01 | 7936.23 | unavailable | 5667.07 | [raw manifest](../r3-p3-v1-66be1539/runs/2026-10-01T20-15-37.003Z_rps-3_118d79ce/run.json) |
| r3-p4-v1 | pruned-128 | 3 | 270 / 270 | 2.86 | 2.83 | 3924.82 | 5694.34 | unavailable | 5361.60 | [raw manifest](../r3-p4-v1-6bd778e3/runs/2026-10-01T20-17-29.137Z_rps-3_669b1fa3/run.json) |
| r3-p4-v0 | baseline-A | 3 | 270 / 270 | 2.49 | 2.45 | 9825.64 | 23648.51 | unavailable | 20002.14 | [raw manifest](../r3-p4-v0-5036dc1b/runs/2026-10-01T20-19-21.391Z_rps-3_d85e80d2/run.json) |
| r3-p5-v0 | baseline-A | 3 | 270 / 270 | 2.39 | 2.41 | 10637.04 | 23325.03 | unavailable | 22055.93 | [raw manifest](../r3-p5-v0-6d255e08/runs/2026-10-01T20-21-26.825Z_rps-3_f8d37663/run.json) |
| r3-p5-v1 | pruned-128 | 3 | 270 / 270 | 2.72 | 2.76 | 5240.45 | 9485.81 | unavailable | 7901.07 | [raw manifest](../r3-p5-v1-9e0c6c2d/runs/2026-10-01T20-23-37.296Z_rps-3_d16b49bd/run.json) |
| r1-p1-v0 | baseline-A | 1 | 270 / 270 | 1.00 | 1.00 | 531.90 | 910.95 | unavailable | 0.00 | [raw manifest](../r1-p1-v0-01fceecb/runs/2026-10-01T20-25-32.358Z_rps-1_d93ce8df/run.json) |
| r1-p1-v1 | pruned-128 | 1 | 270 / 270 | 1.00 | 1.00 | 467.73 | 882.02 | unavailable | 0.00 | [raw manifest](../r1-p1-v1-560f271d/runs/2026-10-01T20-30-21.000Z_rps-1_05faed24/run.json) |
| r1-p2-v1 | pruned-128 | 1 | 270 / 270 | 1.00 | 1.00 | 478.95 | 897.85 | unavailable | 0.00 | [raw manifest](../r1-p2-v1-da74cfaf/runs/2026-10-01T20-35-08.008Z_rps-1_6ca38fc7/run.json) |
| r1-p2-v0 | baseline-A | 1 | 270 / 270 | 1.00 | 1.00 | 522.87 | 894.39 | unavailable | 0.00 | [raw manifest](../r1-p2-v0-6e031844/runs/2026-10-01T20-39-54.151Z_rps-1_1b1c6521/run.json) |
| r1-p3-v0 | baseline-A | 1 | 270 / 270 | 1.00 | 1.00 | 528.01 | 899.19 | unavailable | 0.00 | [raw manifest](../r1-p3-v0-56c05bf7/runs/2026-10-01T20-44-40.859Z_rps-1_edd29e1e/run.json) |
| r1-p3-v1 | pruned-128 | 1 | 270 / 270 | 1.00 | 1.00 | 480.43 | 876.28 | unavailable | 0.00 | [raw manifest](../r1-p3-v1-67cde2af/runs/2026-10-01T20-49-27.948Z_rps-1_ebb1bf5e/run.json) |
| r1-p4-v1 | pruned-128 | 1 | 270 / 270 | 1.00 | 1.00 | 467.45 | 1006.00 | unavailable | 0.00 | [raw manifest](../r1-p4-v1-09f1b303/runs/2026-10-01T20-54-16.714Z_rps-1_89177da0/run.json) |
| r1-p4-v0 | baseline-A | 1 | 270 / 270 | 1.00 | 1.00 | 521.50 | 896.07 | unavailable | 0.00 | [raw manifest](../r1-p4-v0-38471bbf/runs/2026-10-01T20-59-04.551Z_rps-1_43f62605/run.json) |
| r1-p5-v0 | baseline-A | 1 | 270 / 270 | 1.00 | 1.00 | 524.23 | 897.36 | unavailable | 0.00 | [raw manifest](../r1-p5-v0-8d94dc6b/runs/2026-10-01T21-03-50.476Z_rps-1_6ad0480f/run.json) |
| r1-p5-v1 | pruned-128 | 1 | 270 / 270 | 1.00 | 1.00 | 472.08 | 933.46 | unavailable | 0.00 | [raw manifest](../r1-p5-v1-016f4475/runs/2026-10-01T21-09-19.369Z_rps-1_686fc506/run.json) |

## Observed host-wall stages

Only successful measured requests contribute. Each cell is a separate p50 in milliseconds with its observed count; absent observations remain unavailable. Per-stage medians are not additive. These host-wall durations do not isolate GPU prefill, decode or kernel time. The JSON retains independently gated p95/p99 (200/1000 observations).

Boundaries: base64 is handler entry through validated decoding; preprocessing covers image/prompt preparation and existing device transfers; generation covers the entire unchanged model.generate call including its existing synchronizations and diagnostic writes; postprocessing covers batch_decode and strip; observation covers token-contract checks, the extra post-decode output host copy and answer character counting. Service time also contains work outside these named stages.

| Trial | Base64 p50 ms (n) | Preprocess p50 ms (n) | Generation wall p50 ms (n) | Postprocess p50 ms (n) | Observation p50 ms (n) | Successful measured |
|---|---:|---:|---:|---:|---:|---:|
| r3-p1-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r3-p1-v1 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r3-p2-v1 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r3-p2-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r3-p3-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r3-p3-v1 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r3-p4-v1 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r3-p4-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r3-p5-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r3-p5-v1 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r1-p1-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r1-p1-v1 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r1-p2-v1 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r1-p2-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r1-p3-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r1-p3-v1 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r1-p4-v1 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r1-p4-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r1-p5-v0 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |
| r1-p5-v1 | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | unavailable (0) | 270 |

## Observed generation lengths and stopping

Counts require the validated output-ID contract; a configured maximum is never substituted for an observed length. Generated steps include terminal EOS and other special IDs. Reached-budget counts include EOS at the budget; cap without EOS distinguishes budget termination. Missing/invalid observations stay unavailable, including all counts when coverage is zero.

| Trial | Configured max | Observed / successful | Steps min / p50 / max | EOS terminated | Reached budget | EOS at budget | Cap without EOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| r3-p1-v0 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r3-p1-v1 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r3-p2-v1 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r3-p2-v0 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r3-p3-v0 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r3-p3-v1 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r3-p4-v1 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r3-p4-v0 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r3-p5-v0 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r3-p5-v1 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p1-v0 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p1-v1 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p2-v1 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p2-v0 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p3-v0 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p3-v1 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p4-v1 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p4-v0 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p5-v0 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |
| r1-p5-v1 | 64 | 0 / 270 | unavailable / unavailable / unavailable | unavailable | unavailable | unavailable | unavailable |

## Paired comparisons

- 1 RPS, window throughput: inconclusive; 5 pairs; effect 0.00% (positive means improved); 95% interval 0.00 to 0.00%. 
- 1 RPS, p50 latency: improvement; 5 pairs; effect 9.95% (positive means improved); 95% interval 8.40 to 12.06%. 
- 3 RPS, window throughput: improvement; 5 pairs; effect 13.95% (positive means improved); 95% interval 13.70 to 14.73%. 
- 3 RPS, p50 latency: improvement; 5 pairs; effect 59.29% (positive means improved); 95% interval 48.16 to 72.99%. 

## Baseline repeatability

- 1 RPS: 5 baseline trial observations, 5 paired blocks. Baseline leg only. Trial p50 latency median 524.23 ms; relative range 1.98%; median absolute deviation 0.52%. Descriptive pilot spread, not a guaranteed detectable effect. 
- 3 RPS: 5 baseline trial observations, 5 paired blocks. Baseline leg only. Trial p50 latency median 9825.64 ms; relative range 22.62%; median absolute deviation 7.76%. Descriptive pilot spread, not a guaranteed detectable effect. 

## Repeated fixed-window capacity screens

- baseline-A: screen-bracketed; highest all-pass tested rate 1.00 RPS, lowest all-fail tested rate 3.00 RPS. Bounds describe repeated fixed-window client screens only; longer boundary trials and observable queue behavior are still required for sustainable capacity
  - 1 RPS: all-pass, 5/5 prescribed trials. 
  - 3 RPS: all-fail, 5/5 prescribed trials. 
- pruned-128: inconclusive; highest all-pass tested rate 1.00 RPS, lowest all-fail tested rate unavailable RPS. Mixed repeated outcomes need confirmation before choosing a bracket; Bounds describe repeated fixed-window client screens only; longer boundary trials and observable queue behavior are still required for sustainable capacity
  - 1 RPS: all-pass, 5/5 prescribed trials. 
  - 3 RPS: mixed, 5/5 prescribed trials. Repeated trials disagree between pass and fail

## Run quality and capacity screens

- r3-p1-v0: overloaded. Client outstanding work grows through the observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r3-p1-v1: finite-window-pass. Client delivery, failures and outstanding-work screen passed for this fixed observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r3-p2-v1: overloaded. Client outstanding work grows through the observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r3-p2-v0: overloaded. Client outstanding work grows through the observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r3-p3-v0: overloaded. Client outstanding work grows through the observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r3-p3-v1: finite-window-pass. Client delivery, failures and outstanding-work screen passed for this fixed observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r3-p4-v1: finite-window-pass. Client delivery, failures and outstanding-work screen passed for this fixed observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r3-p4-v0: overloaded. Client outstanding work grows through the observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r3-p5-v0: overloaded. Client outstanding work grows through the observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r3-p5-v1: overloaded. Client outstanding work grows through the observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p1-v0: finite-window-pass. Client delivery, failures and outstanding-work screen passed for this fixed observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p1-v1: finite-window-pass. Client delivery, failures and outstanding-work screen passed for this fixed observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p2-v1: finite-window-pass. Client delivery, failures and outstanding-work screen passed for this fixed observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p2-v0: finite-window-pass. Client delivery, failures and outstanding-work screen passed for this fixed observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p3-v0: finite-window-pass. Client delivery, failures and outstanding-work screen passed for this fixed observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p3-v1: finite-window-pass. Client delivery, failures and outstanding-work screen passed for this fixed observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p4-v1: finite-window-pass. Client delivery, failures and outstanding-work screen passed for this fixed observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p4-v0: finite-window-pass. Client delivery, failures and outstanding-work screen passed for this fixed observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p5-v0: finite-window-pass. Client delivery, failures and outstanding-work screen passed for this fixed observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p5-v1: finite-window-pass. Client delivery, failures and outstanding-work screen passed for this fixed observation window; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim

## Limits and outstanding owner inputs

- Each capacity classification describes a fixed observation window. Sustainable capacity requires repeated longer boundary trials.
- Tails are conditional on successful requests. p95 requires 200 successes and p99 requires 1000; sample size does not guarantee precision.
- Paired uncertainty resamples independent trial blocks, not individual requests. Fewer than five pairs remains inconclusive.
- No validated matching accuracy scores are supplied. Accuracy versus throughput and random-scoring controls remain pending with the accuracy owner.
- Model batching, exact KV-cache allocation and any new prefill/decode CUDA event boundaries require the serving owner. Optional HTTP token observations remain unavailable when their wrapper contract checks fail.
- Service wall time is not CUDA event time. Client minus service is an unattributed residual, not measured server queue time.
- Results apply to the declared workload, observed output lengths and recorded instrumentation. Claims about longer outputs, mixed traffic or other instrumentation settings require corresponding completed matched studies.

The machine-readable report preserves run IDs, hashes, server identity, workload order and comparison reasons. Historical evidence is unchanged.
