# Measurement report: llava-baseline-pilot

Purpose: **baseline**. 6/6 trials completed and passed raw-artifact verification.

Interpret these results only within the measured workload and supplied server configuration.

Measured requests per trial: 90; warmup: 10; seed: 465.

| Trial | Variant | Offered RPS | Successes / offered | Window RPS | RPS including drain | p50 ms | p95 ms | p99 ms | Drain ms | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| r1-p1-v0 | baseline-A | 1 | 90 / 90 | 1.00 | 1.00 | 523.16 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p1-v0-d9ed18a5/runs/2026-10-01T19-53-48.853Z_rps-1_c77c2c8c/run.json) |
| r1-p1-v1 | baseline-B | 1 | 90 / 90 | 1.00 | 1.00 | 526.07 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p1-v1-000a2469/runs/2026-10-01T19-55-36.562Z_rps-1_9d105bb1/run.json) |
| r1-p2-v1 | baseline-B | 1 | 90 / 90 | 1.00 | 1.00 | 558.52 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p2-v1-e4ef119d/runs/2026-10-01T19-57-23.626Z_rps-1_76fab203/run.json) |
| r1-p2-v0 | baseline-A | 1 | 90 / 90 | 1.00 | 1.00 | 526.37 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p2-v0-6e0b24bf/runs/2026-10-01T19-59-38.390Z_rps-1_4765f54a/run.json) |
| r1-p3-v0 | baseline-A | 1 | 90 / 90 | 1.00 | 1.00 | 515.57 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p3-v0-c2a60081/runs/2026-10-01T20-01-24.962Z_rps-1_8e7d86e4/run.json) |
| r1-p3-v1 | baseline-B | 1 | 90 / 90 | 1.00 | 1.00 | 516.97 | unavailable | unavailable | 0.00 | [raw manifest](../r1-p3-v1-a5b12604/runs/2026-10-01T20-03-12.637Z_rps-1_d9545367/run.json) |

## Paired comparisons

- 1 RPS, window throughput: inconclusive; 3 pairs; effect 0.00% (positive means improved); 95% interval unavailable. Fewer than five independent paired blocks; uncertainty is not resolved; A/A sham comparison estimates noise; it cannot establish an optimization
- 1 RPS, p50 latency: inconclusive; 3 pairs; effect -0.56% (positive means improved); 95% interval unavailable. Fewer than five independent paired blocks; uncertainty is not resolved; A/A sham comparison estimates noise; it cannot establish an optimization

## Baseline repeatability

- 1 RPS: 6 baseline trial observations, 3 paired blocks. Both legs when their effective configurations match. Trial p50 latency median 523.16 ms; relative range 8.21%; median absolute deviation 0.61%. Descriptive pilot spread, not a guaranteed detectable effect. 

## Repeated fixed-window capacity screens

- baseline-A: inconclusive; highest all-pass tested rate unavailable RPS, lowest all-fail tested rate unavailable RPS. One or more tested rates remain inconclusive; Bounds describe repeated fixed-window client screens only; longer boundary trials and observable queue behavior are still required for sustainable capacity
  - 1 RPS: inconclusive, 3/3 prescribed trials. At least one trial was too small/short or otherwise inconclusive
- baseline-B: inconclusive; highest all-pass tested rate unavailable RPS, lowest all-fail tested rate unavailable RPS. One or more tested rates remain inconclusive; Bounds describe repeated fixed-window client screens only; longer boundary trials and observable queue behavior are still required for sustainable capacity
  - 1 RPS: inconclusive, 3/3 prescribed trials. At least one trial was too small/short or otherwise inconclusive

## Run quality and capacity screens

- r1-p1-v0: inconclusive. Screen is too short or small: require at least 30 seconds and 200 measured requests; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p1-v1: inconclusive. Screen is too short or small: require at least 30 seconds and 200 measured requests; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p2-v1: inconclusive. Screen is too short or small: require at least 30 seconds and 200 measured requests; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p2-v0: inconclusive. Screen is too short or small: require at least 30 seconds and 200 measured requests; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p3-v0: inconclusive. Screen is too short or small: require at least 30 seconds and 200 measured requests; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim
- r1-p3-v1: inconclusive. Screen is too short or small: require at least 30 seconds and 200 measured requests; Server queue boundaries unavailable; client outstanding work includes transport and service; Repeat boundary rates over longer windows before a sustainable capacity claim

## Limits and outstanding owner inputs

- Each capacity classification describes a fixed observation window. Sustainable capacity requires repeated longer boundary trials.
- Tails are conditional on successful requests. p95 requires 200 successes and p99 requires 1000; sample size does not guarantee precision.
- Paired uncertainty resamples independent trial blocks, not individual requests. Fewer than five pairs remains inconclusive.
- No validated matching accuracy scores are supplied. Accuracy versus throughput and random-scoring controls remain pending with the accuracy owner.
- Model batching, exact KV-cache allocation and any new prefill/decode CUDA event boundaries require the serving owner. Optional HTTP token observations remain unavailable when their wrapper contract checks fail.
- Service wall time is not CUDA event time. Client minus service is an unattributed residual, not measured server queue time.
- Results apply to the declared workload, observed output lengths and recorded instrumentation. Claims about longer outputs, mixed traffic or other instrumentation settings require corresponding completed matched studies.

The machine-readable report preserves run IDs, hashes, server identity, workload order and comparison reasons. Historical evidence is unchanged.
