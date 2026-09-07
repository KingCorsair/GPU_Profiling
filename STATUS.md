# GPU Profiling Project Status

## Current State

* Current work package: none active — WP1 accepted and closed; WP2 deliberately not started yet (your instruction).
* Status: Between work packages. Waiting for your go-ahead to begin WP2.
* Main objective: establish a trustworthy latency baseline (576 vs 128 visual tokens) that later optimizations can be judged against. Achieved by WP1.
* Current blocker, if any: **yes — WP1 does not meet the roadmap's own WP1 exit criteria.** The roadmap was restructured around WP1–WP7 on 2026-09-07 (commit `a884c18`), and its WP1 §9 gate is defined on `vision_tower` CUDA time, which our harness does not measure. WP2's go/no-go gate depends on the same number. See "Gap: WP1 vs. the restructured roadmap" below. This needs a decision before WP2.
* Last updated: 2026-09-07

## Completed Work

### Prior completed work (pre-WP structure — the profiling/evidence phase that motivated the engineering roadmap)

* **VisPruner concurrency-crash discovery & repro.** Found the shared model instance is unsafe under concurrent access (CUDA device-side assert). Built a 2-thread reproduction harness.
  Output: `scripts/repro_concurrent_crash.py`, `scripts/run_phase0_repro.sh`.
  Completed: 2026-09-05 (commit `cedaeca`).
* **Trace-driven profiling & bug-fixed baseline numbers.** Ran isolated + near-saturation profiler traces at both 576/128 visual-token configs, found and fixed a double-counting bug in the prep-path timing aggregate, and re-prioritized the speed plan off the corrected numbers.
  Output: `AMAY_VISPRUNER_PROFILING_REPORT.md`, `AMAY_TRACE_PLAN.md`, `results/timing/ab_trace_vtn{576,128}_{isolated,near_saturation}*`, `results/timing/prep_trace.json` / `prep_summary.txt`.
  Completed: 2026-09-02 → 2026-09-03 (commits `3daba72`, `c317688`).
* **No-load A/B comparison (Part 1 of the original A/B experiment).** First single-request 576-vs-128 comparison, natural-EOS output, N=10, non-interleaved (block order). Superseded for baseline purposes by WP1 below, but still the reference for natural-EOS behavior.
  Output: `scripts/run_noload_ab_comparison.py`, `results/timing/ab_noload_comparison.json` / `_table.txt`.
  Completed: 2026-09-02.
* **Load-sweep / matplotlib comparisons.** RPS sweeps at multiple load levels (0.5–3 rps) for both configs, with GPU-utilization monitoring and plotted comparisons.
  Output: `scripts/run_ab_load_sweep.py`, `scripts/plot_*`, `results/loadgen/2026-08-29/`, `results/loadgen/2026-09-01/`.
  Completed: 2026-08-29 → 2026-09-01.
* **AMAY_ENGINEERING_ROADMAP.md written.** Full phase-by-phase roadmap (Phase 0 → Phase 6) mapping every proposed optimization to a category, with dependencies, success metrics, and the 576/128 validation formula.
  Completed: 2026-09-05. **Restructured 2026-09-07** (commits `86a4063`, `46de55e`, `a884c18`) so the month-one plan is expressed natively as WP1–WP7; the old Phase 0–6 material is preserved under "Deferred / Month-Two Engineering Work and Reference". Phase references elsewhere in this file have been updated to the WP numbering.

### WP1 — Controlled Benchmark

* Built `scripts/run_wp1_controlled_benchmark.py`, reusing the model-load-once / flip-`visual_token_num` pattern and the model's existing CUDA-event timing instrumentation (no model code changed).
* Fixed the two things `run_noload_ab_comparison.py` couldn't answer: forced output length (`min_new_tokens == max_new_tokens`, not natural-EOS) and genuine trial-level interleaving/randomization between 576 and 128 (not block order).
* Ran 10 warm-up + 30 measured trials per config (60 measured requests total), real GPU run (NVIDIA A40), same 10 fixed requests/images for both configs.
* Result: median latency 576=1868.1ms, 128=1741.2ms (6.8% gain). Noise floor: robust (MAD-based) stdev is 9.2ms (576) / 11.8ms (128) → a ~20-30ms real change would be detectable at both configs (robust MDE 6.8ms and 8.7ms respectively, both ≤20ms). Raw `statistics.stdev` is noisier (18.7ms / 27.8ms) due to a handful of single-trial outliers per config whose component timings (prep/prefill/decode) were normal — attributed to one-off scheduling hiccups, not to `visual_token_num`, and does not affect the 30ms target either way.
  Output: `scripts/run_wp1_controlled_benchmark.py`, `results/timing/wp1_controlled_benchmark_{raw.jsonl,summary.json,table.txt}`.
  **Accepted by Amay on 2026-09-07** for measuring ~30ms effects.
  Committed 2026-09-07 in `5ce804b` (rebased onto the restructured roadmap); the run itself was executed at HEAD `cedaeca`. The raw per-forward-call timing and the refreshed `prep_trace.json` went in separately as `b8d2a26`.

## Current Work

**No WP is being executed.** WP1 is accepted and closed; WP2 has not been started, per your instruction. But one unresolved item now stands between WP1 and WP2:

### Gap: WP1 vs. the restructured roadmap (open, needs your decision)

The roadmap was restructured around WP1–WP7 on 2026-09-07, *after* WP1 was built and accepted. Its WP1 section defines a different exit gate than the one WP1 was accepted against, and WP2 depends on that gate. Nothing here undoes WP1's acceptance for what it does measure — this is about what it does **not** measure.

| Roadmap WP1 asks for | What we have | Status |
|---|---|---|
| §9: run-to-run spread on **`vision_tower` CUDA time** under ~5ms | Whole-request wall-latency spread and MDE | **Not measured** |
| §8: **three** baseline runs, compared against each other | One run of 60 interleaved trials | **Not done** |
| §11: deliverable `scripts/bench_dev.py` | `scripts/run_wp1_controlled_benchmark.py` | Different name, different scope |
| §11: three baseline result JSONs committed before any code change | One summary JSON committed | Partially met |

Why this is not just a naming mismatch:

* **There is no `vision_tower` timing anywhere in the codebase.** The model emits only `multimodal_prep_time`, `lm_forward_time`, and `generate_time`. `vision_tower` is a *sub-component* of `multimodal_prep` (which also covers image preprocessing, the projector, pruning, and the embedding merge). The roadmap's "vision_tower = 43.888ms of 48.270ms multimodal_prep at 576" came from a one-off profiler trace, not from a re-runnable harness.
* **WP2's own stop rule needs that number.** Roadmap §WP2 step 4–5: run the WP1 harness, check `vision_tower` CUDA time, and *"if it did not drop by at least 20%, stop and re-profile."* As things stand, WP2 cannot evaluate its own go/no-go criterion.
* **CLAUDE.md rule 8 says the same thing independently:** *"Know the noise floor. Run the baseline repeatedly first."* One run of 60 interleaved trials establishes within-run spread; it does not establish run-to-run spread, which is what both rule 8 and roadmap §8 ask for. Model load, CUDA context init, and thermal state are constant *within* a run and vary *between* runs — so the variance that matters for a before/after comparison across two separate runs is exactly the one we have not measured.
* **Whole-request latency is a weak substitute here.** The roadmap's own estimate for WP2 is 20–30ms saved. Our n=40 raw MDE is 11.9ms (576) and 17.6ms (128) — so a 20ms gain at the 128 config would clear the noise floor by a hair, and only if it lands at the top of the predicted range. Component-level timing would resolve it cleanly; whole-request timing would leave the result arguable.

Options (your call):

1. **WP1 addendum before WP2** — add `vision_tower` CUDA-event timing to the harness and do the three-run repeatability check. Small and contained; makes WP2's gate measurable and its result unarguable.
2. **Proceed to WP2 on whole-request latency only** — accept that a 20ms gain may be marginally distinguishable at 128, and that WP2's "20% drop in vision_tower" rule cannot be evaluated as written.
3. **Amend the roadmap** — if the wall-latency criterion is what you actually want for WP1, change roadmap §9/§11 to match, so the document and the artifacts agree.

Recommendation: option 1. WP2 is the one change whose entire effect lands inside `vision_tower`, so this is the worst place to be missing that measurement.

### Benchmark protocol (standing, applies to WP2 onward)

### Benchmark protocol (standing, applies to WP2 onward)

* **Measured trials per configuration: 40** (up from WP1's 30). Warm-up stays at 10 per config, discarded.
* Why: at n=30 the ordinary/raw `statistics.stdev` gave a 20.4ms MDE at the 128 config — just over the 20ms line — so the ~20ms detectability claim had to lean on the MAD-based robust estimate. At n=40 the raw-variance MDE clears 20ms on its own at both configs, so the ~20ms claim no longer depends primarily on the robust estimate.
* Projected MDE at n=40 (alpha=0.05, power=0.80, same formula as WP1):

  | Config | raw MDE @ n=30 | raw MDE @ n=40 | robust MDE @ n=40 |
  |---|---|---|---|
  | 576 | 13.78ms | **11.88ms** | 5.84ms |
  | 128 | 20.45ms | **17.63ms** | 7.47ms |

  Both configs clear 20ms on raw variance, with the binding config (128) at ~12% margin. This is consistent with WP1's own sizing output, which already reported `required_n_per_group_for_20ms_raw` = 32 (128) and 15 (576).
* The robust/MAD estimate does **not** go away — it stays reported alongside raw stdev and p95, now as a cross-check rather than as the primary basis for the 20ms claim.
* Assumption behind the projection: WP2+ runs show variance similar to WP1's. If a future run's raw stdev comes in materially higher, n=40 may not be enough and the run should re-report `required_n_per_group_for_20ms_raw` rather than assume the target is met.
* **Implemented in code (2026-09-07).** `scripts/run_wp1_controlled_benchmark.py` now uses `N_MEASURED_TRIALS_PER_CONFIG = 40`, adds a `--trials` override, and its **success criterion is now raw-variance-primary**: pass = raw MDE ≤ 20ms at *both* configs, with the robust estimate still computed and reported as a cross-check (`success_criterion_met_robust_crosscheck`). If raw fails while robust passes, the run says so explicitly and points at the outlier trials — per the watch item above, that pattern means a real system-level noise source to investigate, not one to discount. Not re-run (measurement-only change; requires the GPU pod).
* **Hazard to know before the next run:** the script writes to fixed output paths (`results/timing/wp1_controlled_benchmark_{raw.jsonl,summary.json,table.txt}`). Re-running it overwrites the **accepted WP1 baseline artifacts** in place. Copy them aside, or add an output-prefix flag, before running WP2's before/after.

## Next

* **Decide on the WP1/roadmap gap above** (recommendation: add `vision_tower` timing + the three-run check as a WP1 addendum). This gates WP2's go/no-go rule.
* Preserve the accepted WP1 baseline artifacts before the next run (see hazard above) — they live at fixed paths the script overwrites.
* Begin **WP2 — FP16 Vision Tower** (roadmap §WP2, which supersedes the old Phase 1a) — **only on your explicit go-ahead. Do not start WP2 yet.**
* WP2's before/after comparison must use 40 measured trials per config.

## Month-One Progress

* [x] WP1 — Controlled Benchmark
* [ ] WP2 — FP16 Vision Tower
* [ ] WP3 — Server Diagnostic
* [ ] WP4 — Request Queue
* [ ] WP5 — Static Batching MVP
* [ ] WP6 — Final Load Benchmark
* [ ] WP7 — Final Writeup

## Important Decisions / Findings

* **Prep-path double-counting bug fixed (2026-09-03).** Two aggregates in `prep_summary.txt` were wrong; corrected before re-prioritizing the speed plan. See `AMAY_TRACE_PLAN.md`.
* **Concurrency crash confirmed, not yet fixed.** Shared model instance is unsafe under real concurrent access; this is WP3 territory (the old Phase 0 material), still open.
* **WP1 accepted; measured trials raised to 40 for all future before/after experiments (2026-09-07, your decision).** WP1 is accepted as a valid baseline for measuring ~30ms effects. Going forward, n=40 per config so the ~20ms detectability claim rests on ordinary/raw variance instead of relying primarily on the MAD-based robust estimate. This supersedes the open question below, which is now closed.
* **WP1 noise-floor finding: raw stdev is outlier-sensitive at n=30, robust (MAD) estimate is not.** A handful of individual trials per config (576: 4/30, 128: 8/30) ran 30-100ms+ slower than their peers with otherwise-normal component timings — a system/scheduling artifact, not a `visual_token_num` effect. Using raw `statistics.stdev` as the sole success metric would have called WP1 "too noisy" at 128 (raw MDE 20.4ms vs. the 20ms target) even though the underlying signal is clean. Decision at the time: report both, use the robust estimate as the pass/fail criterion, keep raw stdev/p95 fully visible per the project's own rule 6. **Superseded 2026-09-07:** WP1 stands as accepted on that basis, but from WP2 onward n=40 makes the raw estimate sufficient on its own, so pass/fail no longer depends on the robust estimate. The underlying watch item still holds — **if WP2+ shows outlier rates climbing, that is a real system-level noise source worth chasing** rather than one to discount, and it would also invalidate the n=40 projection above.
* **Fixed-output-length off-by-one.** Requesting `max_new_tokens=64` actually yields 65 generated tokens under this model's `generate()` override (it calls HF's generate with `inputs_embeds` only, which shifts the internal count by one). Harmless for WP1 (constant across every trial, both configs) but worth knowing before WP2+ scripts assume `new_tokens == requested`.
