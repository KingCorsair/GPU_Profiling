# Rithvik's measurement work

The TypeScript measurement pipeline, PostgreSQL storage, reports, benchmark UI and CI are implemented. The October 1 GPU studies use fixed saved protocols; the collection table below distinguishes completed evidence from work still running. Read the [results and interpretation](measure/OCT01_RESULTS.md) for the verified findings. This document covers Rithvik's scope. The shared serving and accuracy implementations remain with Amay and Sribhav.

## Implemented and verified

| Area | Delivered behavior and evidence |
|---|---|
| Reproducible traffic | Absolute open-loop arrivals, completed warmup/settling, seeded category-balanced question order, fixed measured budgets, request IDs and separate latency/lateness clocks. Behavioral tests include overlap, failures, timeout, interruption and isolated HTTP boundaries. |
| Controlled campaigns | Saved randomized schedules, adjacent paired trials, verified effective settings, owned process cleanup, GPU-idle checks, durable attempt history and bounded startup/request waiting. |
| Statistical analysis | Nearest-rank p50/p95/p99 with sample gates; paired-block bootstrap uncertainty; A/A sham labeling; descriptive noise estimates; repeated finite-window capacity screens with missing/mixed/nonmonotonic evidence kept inconclusive. |
| Integrity | Raw request/resource hashes and counts, timing algebra, configured observation-window checks, phase coverage, recomputed summaries and matching workload/hardware/source controls. |
| HTTP observations | Default service-wall measurements and optional request stages/actual token-ID counts with strict runtime-contract guards. Queue, GPU-stage and KV metrics remain separately identified. |
| Storage | PostgreSQL schema, transactional/idempotent imports, exact raw evidence retention, conflict rollback, finalized-snapshot checks, comparison query and local Compose configuration. Real PostgreSQL integration tests pass. |
| Reports and UI | Portable campaign export preserving source bytes, Markdown/JSON/CSV reports, PNG/SVG plots, searchable React run archive and raw downloads. Isolated requests are labeled sequential, without an offered-rate claim. |
| CI/deployment | CPU integration campaigns through raw storage and reporting on Node 20/24, Python checks, frontend checks/build and isolated RunPod deployment instructions. |
| Accuracy handoff | Strict aggregate-input adapter and documented identity/validation contract. Existing evaluation files are retained as unmatched evidence. No numeric accuracy is invented. |

## Collected evidence

| Study | Status | Saved evidence |
|---|---|---|
| A40 clock/warmup/cache diagnostic | Complete; synthetic operations, not serving results | [CUDA calibration](../results/calibration/2026-10-01/2026-10-01T19-48-34Z_5c1e8d96/calibration.json) |
| Coordinated omission and latency distributions | Complete; 40,000 synthetic measured requests, zero failures | [Demonstration and plots](../results/calibration/2026-10-01/coordinated-omission-fixed-10000/report.md) |
| Repeated 576-token model baseline | Complete; 6 trials, 540 measured requests plus 60 warmups, zero failures | [Pilot report](../results/campaigns/2026-10-01/llava-baseline-pilot/report/report.md) and [PostgreSQL verification](../results/campaigns/2026-10-01/llava-baseline-pilot/report/storage-verification.json) |
| Paired 576/128 tokens at 1 and 3 RPS | Complete; 20 trials, 5,400 measured requests plus 200 warmups, zero failures | [Verified report](../results/campaigns/2026-10-01/llava-oct01-primary/report/report.md), [PostgreSQL verification](../results/campaigns/2026-10-01/llava-oct01-primary/report/storage-verification.json), [fixed protocol](measure/OCT01_PROTOCOL.md) |
| Isolated full-HTTP comparison | Complete; 10 trials, 900 measured requests plus 100 warmups, zero failures | [Verified report](../results/campaigns/2026-10-01/llava-oct01-isolated/report/report.md), [fixed protocol](measure/OCT01_PROTOCOL.md) |
| Incremental HTTP observation overhead | Complete; 10 trials, 900 measured requests plus 100 warmups, zero failures; effect inconclusive | [Verified report](../results/campaigns/2026-10-01/llava-oct01-http-overhead/report/report.md), [observation protocol](measure/OCT01_OBSERVATION_PROTOCOL.md) |
| Short/long/mixed natural output policies | Running in declared long/mixed/short order; 12 prescribed descriptive trials | [Observation protocol](measure/OCT01_OBSERVATION_PROTOCOL.md) and [workload manifest](measure/workloads/oct01/manifest.json) |

The six identical baseline trials had per-trial p50 latencies from 515.57 to 558.52 ms: an 8.21% observed range and 0.61% median absolute deviation relative to the nearest-rank median. This is descriptive pilot spread, not a guaranteed detection threshold. Their 90 measured observations per trial do not support p95/p99 under this project's policy.

The primary comparison completed all five pairs at each rate. At 3 RPS, the median paired improvement in within-window completions was **13.95%** (95% paired-block bootstrap interval **13.70% to 14.73%**). At 1 RPS, the median paired p50-latency reduction was **9.95%** (**8.40% to 12.06%**), while both variants completed the offered 1 RPS. At 3 RPS, paired p50 fell **59.29%** (**48.16% to 72.99%**); these are overloaded-window latencies, not steady-state tails. Five paired blocks give limited uncertainty resolution. Both variants passed the 1-RPS finite-window screen in every trial; the 576-token variant failed every 3-RPS screen, while the 128-token variant had mixed screen outcomes and outstanding work in every 3-RPS trial. No precise sustainable-capacity ceiling is established. Actual output lengths were not recorded by the frozen primary revision, and matching validated accuracy remains pending.

The synthetic 750 RPS experiment illustrates coordinated omission: open-loop traffic delivered the prescribed 10,000 requests in its 13.333-second arrival window and observed 9.77-second p95 latency. The completion-paced control sent only 4,442 requests during that window and appeared to have 3.36-ms p95 latency. It eventually sent all 10,000 over 29.907 seconds. These numbers describe the synthetic CPU service and do not estimate GPU performance.

## Remaining interpretation boundaries

The final accuracy-versus-throughput chart requires a matching validated aggregate from Sribhav, including per-category results, scorer validation, a locked split and random-control provenance. [The adapter contract](measure/accuracy-handoff.md) specifies the exact handoff. Existing scorer outputs lack the necessary identity/validation fields for these serving trials.

Queue admission, batching/padding, exact KV-cache allocation, new GPU prefill/decode boundaries and alternative pruning implementations are Amay's inputs. The current serving path already includes model-internal diagnostic events, synchronization and file writes. The HTTP overhead experiment isolates only the additional request observations. These measurements do not complete Amay's pending correctness/profiler work or validate accuracy.

Capacity screens describe fixed windows. They do not establish a precise sustainable ceiling or a latency SLO; no SLO was supplied. The output-policy follow-up is descriptive and has two paired blocks per workload, so its optimization comparisons remain inconclusive. GPU p99 is not claimed. A broader workload/rate study or finer boundary resolution is a new explicitly budgeted experiment.

## Run and inspect

Start with [the measurement README](measure/README.md), [clock/record contracts](measure/contracts.md), [methodology](measure/methodology.md) and [database instructions](measure/db/README.md). The dashboard runs with `npm --prefix csnbs/frontend run dev`; production data/build uses `npm --prefix csnbs/frontend run build`.

GPU source for the primary and isolated experiments is clean commit `356df7d66f7cf4c7524c93d7c9d373e2b920fbfc`. Optional observation follow-ups use clean commit `e90148c647c62eb8dbc1a856f5cd96e44bd6f202`. Both are separate checkouts under `/workspace`; results stay outside them during collection. Only owned model processes are terminated. The shared pod and shared checkout are preserved.

To explain the core component aloud: distinguish scheduled, sent and completed times; explain why warmup must drain before measurement; show why a completion-paced client loses intended arrivals; explain why requests within one trial are not independent replications; and state what a confidence interval crossing zero does and does not mean. The raw examples and corresponding source are retained for that review.
