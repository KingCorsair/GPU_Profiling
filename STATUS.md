# GPU Profiling Project Status

## Current State

* Current work package: none active. WP1 is closed, now including the 2026-09-11 addendum (component timing plus the three-run noise floor). WP2 has not been started (your instruction).
* Status: Between work packages. Waiting for your go-ahead to begin WP2.
* Main objective: establish a trustworthy latency baseline (576 vs 128 visual tokens) that later optimizations can be judged against. **Achieved, and WP1 now meets the roadmap's own §9 gate.**
* Current blocker, if any: none. The "WP1 vs. the restructured roadmap" gap is closed; see the addendum below.
* Last updated: 2026-09-11

## Completed Work

### Prior completed work (pre-WP structure — the profiling/evidence phase that motivated the engineering roadmap)

* **VisPruner concurrency-crash discovery & repro.** Found the shared model instance is unsafe under concurrent access (CUDA device-side assert). Built a 2-thread reproduction harness.
  Output: `scripts/repro_concurrent_crash.py`, `scripts/run_phase0_repro.sh`.
  Completed: 2026-09-05 (commit `cedaeca`).
* **Trace-driven profiling & bug-fixed baseline numbers.** Ran isolated and near-saturation profiler traces at both 576/128 visual-token configs, found and fixed a double-counting bug in the prep-path timing aggregate, and re-prioritized the speed plan off the corrected numbers.
  Output: `AMAY_VISPRUNER_PROFILING_REPORT.md`, `AMAY_TRACE_PLAN.md`, `results/timing/ab_trace_vtn{576,128}_{isolated,near_saturation}*`, `results/timing/prep_trace.json` / `prep_summary.txt`.
  Completed: 2026-09-02 → 2026-09-03 (commits `3daba72`, `c317688`).
* **No-load A/B comparison (Part 1 of the original A/B experiment).** First single-request 576-vs-128 comparison, natural-EOS output, N=10, non-interleaved (block order). Superseded for baseline purposes by WP1 below, but still the reference for natural-EOS behavior.
  Output: `scripts/run_noload_ab_comparison.py`, `results/timing/ab_noload_comparison.json` / `_table.txt`.
  Completed: 2026-09-02.
* **Load-sweep / matplotlib comparisons.** RPS sweeps at multiple load levels (0.5–3 rps) for both configs, with GPU-utilization monitoring and plotted comparisons.
  Output: `scripts/run_ab_load_sweep.py`, `scripts/plot_*`, `results/loadgen/2026-08-29/`, `results/loadgen/2026-09-01/`.
  Completed: 2026-08-29 → 2026-09-01.
* **AMAY_ENGINEERING_ROADMAP.md written.** Full phase-by-phase roadmap (Phase 0 → Phase 6) mapping every proposed optimization to a category, with dependencies, success metrics, and the 576/128 validation formula.
  Completed: 2026-09-05. **Restructured 2026-09-07** (commits `86a4063`, `46de55e`, `a884c18`) so the month-one plan is expressed natively as WP1–WP7; the old Phase 0–6 material is preserved under "Deferred / Month-Two Engineering Work and Reference".

### WP1 — Controlled Benchmark

**Part 1 — whole-request baseline (accepted 2026-09-07).**

* Built `scripts/run_wp1_controlled_benchmark.py`, reusing the model-load-once / flip-`visual_token_num` pattern and the model's existing CUDA-event timing instrumentation (no model code changed).
* Fixed the two things `run_noload_ab_comparison.py` couldn't answer: forced output length (`min_new_tokens == max_new_tokens`, not natural-EOS) and genuine trial-level interleaving/randomization between 576 and 128 (not block order).
* Ran 10 warm-up + 30 measured trials per config, NVIDIA A40, 10 fixed requests, 64 output tokens.
* Result: median latency 576=1868.1ms, 128=1741.2ms (6.8% gain). Robust (MAD) stdev 9.2ms / 11.8ms; raw stdev 18.7ms / 27.8ms.
  Output: `results/timing/wp1_controlled_benchmark_{raw.jsonl,summary.json,table.txt}`. Accepted by Amay on 2026-09-07 for measuring ~30ms effects. Committed in `5ce804b` / `b8d2a26`; the run executed at `cedaeca`.
* **Caveat found 2026-09-11:** this run's `multimodal_prep` is bimodal (576: ~34.7ms floor, 11/30 trials at 38–94ms; 128: ~46ms floor, 9/30 at 100–116ms). That is the CPU-throttling stall described below, not model behavior. Its whole-request outliers may share the cause (a stall landing outside prep); not verified.

**Part 2 — addendum: component timing and three-run noise floor (2026-09-11).** Closes the gap against roadmap §WP1 §8/§9/§11.

* `scripts/bench_dev.py` now times `vision_tower`, `mm_projector` and `multimodal_prep` with CUDA events:
  * forward hooks on the `CLIPVisionTower` wrapper and on `mm_projector`;
  * an instance-level wrapper on `prepare_inputs_labels_for_multimodal`, so there are no edits to model code;
  * event records only, no syncs, read once after `generate()`'s own closing sync.

  Validated: exactly 1 / 1 / 32 event pairs per request, and hooks on vs. off leave prefill and decode unchanged (110.17 vs 110.48ms; 30.77 vs 30.62ms per token).
* Roadmap spec, plus the standing protocol:
  * 20 fixed dev images, 32 forced output tokens;
  * 10 warm-up + 40 measured per config, trial-interleaved with a fixed seed;
  * per-region p50/p95, with git commit, GPU, CPU-thread count and cgroup throttle counts recorded.
* **Three baseline runs**, each a separate process:
  * commit `469a071`, NVIDIA A40, clean tree (only a docs-only `CLAUDE.md` edit outstanding);
  * no other GPU processes, zero CPU-throttled trials, fixed length held on every trial.

  Run-to-run noise floor (p50 of each run):

  | Region | 576: three runs | 576 spread | 128: three runs | 128 spread |
  |---|---|---|---|---|
  | `vision_tower` | 33.96 / 33.90 / 33.91 | **0.07ms** | 33.93 / 33.85 / 33.86 | **0.08ms** |
  | `mm_projector` | 0.33 / 0.33 / 0.33 | 0.00ms | 0.39 / 0.38 / 0.38 | 0.01ms |
  | `multimodal_prep` | 35.34 / 35.26 / 35.31 | 0.08ms | 47.98 / 47.85 / 48.01 | 0.16ms |
  | prefill | 110.17 / 110.29 / 110.30 | 0.13ms | 44.46 / 44.30 / 44.38 | 0.17ms |
  | decode (31 steps) | 949.70 / 949.30 / 949.91 | 0.62ms | 906.47 / 906.57 / 906.44 | 0.13ms |
  | end-to-end | 1112.70 / 1111.89 / 1115.76 | 3.87ms | 1014.37 / 1015.42 / 1015.17 | 1.05ms |

* **§9 gate: PASS at both configs.** The gate is `vision_tower` spread ≤ ~5ms; measured 0.07 / 0.08ms.
* **Noise-floor note (the §11 deliverable).** On this pod (A40, torch pinned to 4 CPU threads), a no-change rerun moves `vision_tower` p50 by at most 0.08ms and whole-request p50 by at most 3.9ms. Within a run:
  * `vision_tower` stdev is 0.17–0.38ms, so a 40-vs-40 before/after comparison resolves a ~0.25ms change (two-sample MDE, α=0.05, power 0.80);
  * whole-request stdev is 4.4–7.0ms, which gives an MDE of ≤ 4.4ms.

  WP2's stop rule (`vision_tower` must drop ≥ 20%, i.e. ~6.8ms) is ~85× the run-to-run spread. **These numbers hold only with the CPU-thread pin.** Without it, a ~55–70ms throttling stall dominates prep (below). Compared with Part 1's 13.8 / 20.4ms raw MDE, most of the improvement is the removed stall; the output length also halved (64 → 32 tokens).
* Other measurements, controlled benchmark only (not serving behavior):
  * **576→128 end-to-end gain is 8.8%** at 32 output tokens, vs 6.8% at 64 in Part 1. Consistent with decode diluting the prefill saving as output grows.
  * **DERIVED:** VisPruner's selection work (`prep_other`) costs 1.0–1.1ms at 576 and 13.6–13.8ms at 128. So at 128 the diversity loop gives back ~12.6ms of the 65.9ms prefill saving.
* Output: `scripts/bench_dev.py` (commit `469a071`) and `results/timing/bench_dev/`:
  * `baseline_{1,2,3}.{json,txt}`;
  * `compare_baseline_1_baseline_2_baseline_3.{json,txt}`;
  * stall diagnosis: `scripts/diagnose_prep_stall.py`, `results/timing/bench_dev/prep_stall_{ab.txt,threads48.json,threads4.json}`.

## Current Work

**No WP is being executed.** WP2 waits for your go-ahead.

### Benchmark protocol (standing, applies to WP2 onward)

* **Use `scripts/bench_dev.py` for before/after comparisons.** It writes per `--run-id` (refuses to overwrite without `--overwrite`), and `--compare` computes the spread and the `vision_tower` gate.
* **40 measured trials per config, 10 warm-up** (unchanged since 2026-09-07).
* **Keep torch pinned to 4 CPU threads** (`bench_dev.py` default, `--cpu-threads`). A run that reports any `throttled_trial_positions` is contaminated. Find out what else was using the container's CPU and rerun it.
* **Compare only against baselines from the same pod.** The noise floor above was measured on this pod's host and CPU quota. On a new pod, rerun `baseline_{1,2,3}` first. (This session's pod needed the weights re-downloaded, and its decode runs ~30.6ms/token vs ~26.5ms in Part 1, so hosts do differ.)
* Commit the harness before running; a dirty tree marks the run unattributable. Docs-only (`*.md`) edits don't count.
* `run_wp1_controlled_benchmark.py` still writes to fixed paths and overwrites Part 1's accepted artifacts if re-run; it has no CPU-thread pin, so it will reproduce the stall. Treat it as historical.

## Next

* **WP3 heads-up, the most important finding here.** Nothing in `csnbs/` or `scripts/start_model_server.sh` sets a torch thread count, so the FastAPI server very likely takes the same ~55–70ms container-throttling stall on every request, with the GPU idle meanwhile. That competes directly with the event-loop hypothesis for the ~34% idle GPU. WP3 should read `/sys/fs/cgroup/cpu/cpu.stat` `nr_throttled` before and after its rps 2.0 run (roadmap WP3 §3 has the note). The fix is one line (`torch.set_num_threads` or `OMP_NUM_THREADS`), but it is in Rithvik's server, so it needs agreeing with him.
* **WP2 re-estimate.** `vision_tower` is 33.9ms, not the trace's 43.9ms, so WP2's expected saving is ~15–23ms, not 20–30ms (roadmap WP1 status block). Its "before" can be `baseline_{1,2,3}` if WP2 runs on this same pod.
* Suggested: add the CPU-throttling gotcha to `CLAUDE.md`'s "Gotchas already hit" list. Not done, because `CLAUDE.md` currently carries an uncommitted edit of yours.
* Commits `469a071` and the addendum commit are local and **not pushed**.
* Begin **WP2 — FP16 Vision Tower** only on your explicit go-ahead.

## Month-One Progress

* [x] WP1 — Controlled Benchmark (whole-request baseline 2026-09-07; component timing + noise floor 2026-09-11)
* [ ] WP2 — FP16 Vision Tower
* [ ] WP3 — Server Diagnostic
* [ ] WP4 — Request Queue
* [ ] WP5 — Static Batching MVP
* [ ] WP6 — Final Load Benchmark
* [ ] WP7 — Final Writeup

## Important Decisions / Findings

* **Container CPU throttling explains the prep-path "noise" (2026-09-11).**
  * The pod's container is CFS-capped at 7.65 CPUs (`cpu.cfs_quota_us=765000` per 100ms), but torch sizes its intra-op pool from the host: 48 threads, with 96 inter-op.
  * One 48-thread burst per request exhausts the quota, and the kernel freezes the whole container for the rest of the 100ms period.
  * **Diagnosis** (`scripts/diagnose_prep_stall.py`): at default threads, 60 of 60 requests stalled, and every one incremented `nr_throttled` during prep. At 4 threads, 0 of 60 did. Prep at 576 fell from ~88ms to 35.3ms and its spread from ±10 to ±0.3ms. Garbage collection, allocator retries and OS preemption were ruled out first.
  * The 4-thread cap costs nothing on the GPU path: prefill and decode are unchanged.
* **`vision_tower` is 33.9ms under CUDA events (2026-09-11)**, identical at 576 and 128 because pruning happens after the tower. The roadmap's 43.888ms came from a profiler trace and includes its overhead; vision-path estimates derived from it are high by ~25%.
* **Prep-path double-counting bug fixed (2026-09-03).** Two aggregates in `prep_summary.txt` were wrong; corrected before re-prioritizing the speed plan. See `AMAY_TRACE_PLAN.md`.
* **Concurrency crash confirmed, not yet fixed.** Shared model instance is unsafe under real concurrent access; this is WP3/WP4 territory, still open.
* **WP1 accepted; measured trials raised to 40 for all future before/after experiments (2026-09-07, your decision).** The robust-vs-raw MDE question from Part 1 is moot for `bench_dev.py`: with the thread pin its raw whole-request MDE is ≤ 4.4ms at n=40.
* **Part 1 noise-floor finding (2026-09-07): raw stdev was outlier-sensitive at n=30, the robust (MAD) estimate was not.** A handful of trials ran 30–100ms+ slow. The watch item, "if outlier rates climb, that is a real system-level noise source worth chasing", turned out to be right: see the throttling finding above.
* **Fixed-output-length off-by-one.** Requesting `max_new_tokens=64` yields `output_ids` of length 65 under this model's `generate()` override (HF's `generate` with `inputs_embeds` only). Harmless (constant across trials). `bench_dev.py` checks forced length via the forward-call count instead (32 calls for 32 requested tokens, on every trial).
