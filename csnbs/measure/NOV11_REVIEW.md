# November 11: upstream load-test review and local fixes

Reviewed GitHub `main` and `amay/wp3-wp5-serving` at
`3f7c488befc28541476ac761078ec8c722f6edb0`, against this checkout's
`measurement-v2` at `31c92e89606908002decffdd89a41d68c76891f5` and its existing
uncommitted work. There are 28 measurement-only and 14 main-only commits.
[Measurement PR #4](https://github.com/KingCorsair/GPU_Profiling/pull/4) remains
an unmerged draft at review time.

## Integration branch

`codex/measurement-serving-integration` combines both committed histories and
these fixes in a separate checkout. It retains the latest main dashboard and
adds the arrival metadata changes. Amay's serving/model files and other owners'
files match `3f7c488`; neither existing shared branch is rewritten. Unrelated
uncommitted files in the original checkout were left there.

The current `csnbs/measure` campaign runner is the supported path for this
measurement version. The old `scripts/wp6_load_matrix.py` still searches the old
date/run artifact layout, and `wp6_analyze.py` removes warmup by sorting and
slicing rather than the V2 phase field. Do not run those scripts against V2
artifacts or treat the merge as a new validated S0/S2 experiment. Existing WP6
evidence remains tied to its original source commit.

The S0 wrapper and batched server also expose older health metadata. Current
model campaigns correctly refuse missing source/checkpoint/GPU provenance;
standalone collection does not make it reportable. The wrapper's hard-coded
output-cap metadata must not be used for a changed `MAX_NEW_TOKENS` setting.
A serving-owner handoff or a separately reviewed adapter is required before
running a new S0/S2 comparison. No validation checks were relaxed to bypass it.

## What the server-load report refers to

[Amay's handoff](https://github.com/KingCorsair/GPU_Profiling/blob/3f7c488befc28541476ac761078ec8c722f6edb0/kingcorsair/RITHVIK_LOADGEN_REQUESTS.md)
identifies excessive PyTorch CPU threads and requests more varied traffic.
His [WP3 diagnostic](https://github.com/KingCorsair/GPU_Profiling/blob/3f7c488befc28541476ac761078ec8c722f6edb0/STATUS.md#wp3--server-diagnostic-closed-2026-10-25)
reports a roughly 7.65-CPU container running approximately 48 threads, with
throttling disappearing at four threads. This is upstream evidence; no new GPU
performance measurement was made in this maintenance task.

The blocking batch-one server is also an acknowledged limitation. Amay's newer
S1/S2 servers handle queueing and batching. Those owner-controlled implementations
were not rewritten here.

## Fixes made in Rithvik's portion

- `csnbs/server.py` sets four PyTorch intra-op threads before model initialization.
  `TORCH_THREADS` overrides the default and must be a positive integer in model
  mode. Health metadata records the effective `torch_num_threads`; fake mode
  neither loads nor configures PyTorch. New runs require a fresh baseline.
- `src/resources.ts` no longer queues missed GPU polling ticks behind a slow
  query. Stop prevents new polls, finishes the active observation, preserves its
  success/error record, and closes the file. A delayed-query regression reproduced
  six polls after a stop request before the fix; the corrected case performs one.
  Skipped ticks represent unavailable observations, not reconstructed samples.
- After explicit approval, added `--arrival poisson` with fixed-count seeded
  exponential arrivals, campaign replay, and schedule/duration verification.
  Reports, archive comparisons, plots and accuracy exports retain arrival policy
  and seed. The existing uniform-capacity screen stays inconclusive for this
  different observation policy. The [walkthrough](WALKTHROUGH.md) explains it.

Validation: 21 Python server/Gemma/Qwen checks, 77 TypeScript measurement checks,
100 frontend checks, and 16 plotting/accuracy-export checks passed. One PostgreSQL
integration check was skipped because no `DATABASE_URL` was supplied. TypeScript
checking and measurement/dashboard builds passed. All 58 saved October campaign
runs still pass the updated verified reader. No GPU run, deployment, publication,
or changes to historical evidence were performed.

## Handoff reconciliation

| Request | Current status |
|---|---|
| Seeded Poisson arrivals | Implemented after Rithvik's explicit confirmation. Fixed request count, seeded exponential gaps, independent workload randomness, paired replay and verified schedule/duration. Uniform remains the default. See the [contract](contracts.md#seeded-poisson-arrivals). |
| Seeded shuffled workload | Already implemented in this branch through `--seed` and category-balanced replay. Main's older client lacks it. |
| Off-box comparison / generator CPU | Still outstanding. Existing resource records contain OS load averages, not the generator process's CPU usage. |
| Four-thread server default | Implemented and locally tested; GPU throttling must be checked on the actual deployment. |
| Review or reproduce WP6 | Reviewed below; not signed off as satisfying the current measurement protocol. |
| Longer answers | `MAX_NEW_TOKENS` and server configuration capture already exist here. A longer cap does not establish longer observed output; the comparison remains uncollected. |

## WP6 review: retain as exploratory evidence

The committed `results/timing/wp6_matrix/wp6_matrix_2/run.json` references 74
available run manifests and their request journals. Inspecting those artifacts
and the producing source establishes these limitations:

1. **Warmup overlap:** 56 of 74 trials still had warmup requests in flight when
   the first measured request was sent; the largest overlap was 35.58 seconds.
   Main's older `loadgen.ts` schedules all requests together and later slices
   off the first ten. This branch already completes warmup before measurement.
2. **Tail-policy mismatch:** 66 of 74 rows have fewer than 200 successful measured
   requests. `scripts/wp6_analyze.py` reports their p95 anyway and uses interpolated
   percentiles, while this branch requires 200 successes and nearest-rank p95.
3. **Replication:** of 68 distinct state/token/policy/rate cells, 66 have one
   trial and two have four. These are not the repeated paired blocks required
   for the current pipeline's uncertainty estimates.
4. **Capacity terminology:** the upstream analyzer selects the highest achieved
   throughput over tested rates, using a denominator that includes drain.
   This is observed throughput, not an established sustainable capacity or SLO.
5. **Provenance:** dirty source is recorded, and individual manifests record
   changing commits. The upstream explanation of unrelated edits does not by
   itself establish a single clean immutable execution snapshot.

These findings do not show that batching has no benefit. They limit what the
saved experiment establishes under the current reporting policy. Do not relabel
its observations as a new validated campaign or combine them with October 1's
different execution identities.

Next: prepare a bounded S0/S2 protocol after the health/artifact contracts above
are reconciled, using completed warmup, identical replay, sufficient samples and
repeated paired trials. Freeze its GPU budget and execution source before
collection. The older four-hour driver should not be rerun unchanged.
