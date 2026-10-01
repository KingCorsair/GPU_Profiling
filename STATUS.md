# GPU Profiling Project Status

## Current State

* **Current work package: WP2 — FP16 Vision Tower. In progress, paused 2026-09-22.**
* Done: the implementation, and the performance A/B (4 same-pod `bench_dev` runs plus an interleaved diagnostic). The ≥20% `vision_tower` gate passes by a wide margin.
* Not done: profiler verification, Layer-1 correctness, Layer-2 TextVQA, a Layer-3 request to Sribhav, and re-baselining the numbers in the docs.
* **Next task:** profiler verification with `scripts/profile_multimodal_prep.py`. The FP32 state is prepared on branch `wp2-profile-fp32`. See `HANDOFF.md`.
* WP3 has not started.
* Blockers: none.
* Last updated: 2026-09-22

## Month-One Progress

* [x] WP1 — Controlled Benchmark (2026-09-07; component timing and noise floor 2026-09-11)
* [~] WP2 — FP16 Vision Tower: implementation and performance done; verification and correctness open
* [ ] WP3 — Server Diagnostic
* [ ] WP4 — Request Queue
* [ ] WP5 — Static Batching MVP
* [ ] WP6 — Final Load Benchmark
* [ ] WP7 — Final Writeup

## WP2 — FP16 Vision Tower

### The change (commit `caf19fe`, parent `e295fd0`)

One line changed: `clip_encoder.py:30`, where `CLIPVisionModel.from_pretrained(...)` gains `torch_dtype=torch.float16`.

Why the tower was FP32:

* The CLIP tower isn't in the LLaVA checkpoint. It loads separately from `openai/clip-vit-large-patch14-336`, whose config says float32.
* The FP16 cast that was meant to fix this, at `builder.py:155-156`, sits behind `if device_map != 'auto'`. Every call site uses `device_map='auto'`, so the cast never ran.

Nothing else changed. The wrapper's casts at `clip_encoder.py:64,70` follow `self.dtype`.

### Performance A/B — same pod, NVIDIA A40

**`bench_dev` runs.**

* Execution order: FP32_1 (`e295fd0`) → FP16_1 (`caf19fe`) → FP16_2 (`caf19fe`) → FP32_2 (`e295fd0`). The reverse order in the second pair was deliberate, to counter run-order bias.
* All four runs:
  * clean tree;
  * 40 measured and 10 warm-up trials per config;
  * 4 CPU threads, no throttled trials, no other GPU process;
  * fixed 32-token output;
  * identical trial order.
* Files: `results/timing/bench_dev/wp2_{fp32,fp16}_samepod_{1,2}.{json,txt}` and `compare_*`.

| `vision_tower` p50 | FP32_1 | FP16_1 | FP16_2 | FP32_2 | Pooled change |
|---|---|---|---|---|---|
| vtn=576 | 33.93 | 14.29 | 14.41 | 34.04 | 33.98 → 14.35 ms, **−57.8%, 2.37×** |
| vtn=128 | 33.94 | 13.79 | 13.57 | 34.07 | 34.00 → 13.68 ms, **−59.8%, 2.49×** |

* Reproducibility: rerun-to-rerun p50 moved by at most 0.13 ms for FP32 and at most 0.21 ms for FP16.
* Every FP32/FP16 pairing gives a 57.5–60.2% reduction (2.35–2.51×).
* **Gate (≥20% reduction): PASS.** Even the slowest FP16 trial (19.86 ms) is 41% below the fastest FP32 trial (33.74 ms).
* Prefill is unchanged (−0.7% / −0.8%). `mm_projector` is unchanged at its floor.

**The FP16 runs had a problem: per-trial wander.**

* FP16 tower time ranged 10.5–19.9 ms per trial (stdev 2.3–2.9 ms), against 0.10–0.17 ms for FP32.
* `prep_other` and decode slowed in the same trials, even though the change never touches them.
* In the bench_dev A/B, end-to-end latency went **−0.2%**, which is within FP32's own 5 ms run-to-run spread. So the end-to-end result from these runs is **not reportable**.

**Interleaved diagnostic, `wp2_diag_1`** (`scripts/diag_wp2_fp16_jitter.py`, run at `02e749e`).

* Design: one process, with both towers loaded and swapped per trial. vtn=128, 60 trials per arm, randomly interleaved.
* Per trial it also records a CPU probe, the SM clock, and the tower's CPU issue time.
* Files: `results/timing/wp2_diag/wp2_diag_1.{json,txt}`.
* Results:
  * **`vision_tower` p50: 34.20 → 11.02 ms, −67.8%, 3.10×.**
  * FP16 tower CUDA-event time equals its CPU issue time (ratio **1.00**; FP32: 3.22). With FP16 the tower is **CPU/kernel-launch-bound**: the GPU waits for Python/HF to issue kernels.
  * FP16 minus neighbouring FP32 trials: `prep_other` **+0.14 ms**, decode **−0.04 ms/token**. FP16 does not change any downstream work.
  * SM clock held at 1740 MHz with no throttle reasons, so GPU clock changes are ruled out.
  * End-to-end at 128: 1019.6 → 995.3 ms (−24.3 ms, −2.4%). That is roughly what Amdahl predicts: about 20 ms saved out of about 1.1 s.

**Conclusion.**

* The wander in the FP16 A/B came from **host CPU slowdowns** (neighbouring tenants and/or CPU frequency), not from FP16.
* The slowdowns come in episodes lasting tens of seconds, with decode shifting in discrete steps (about +0.15 and +0.38 ms/token).
* They hit FP32 runs too. The FP32 tower hid them, because 34 ms of GPU work always had more work queued behind it. The FP16 tower is launch-bound, so it exposes them.
* The FP16 bench_dev runs happened to catch more of these episodes: 42–71% of their trials were in the slow state, against 0–5% for FP32.
* Neither cgroup throttling nor steal time sees these slowdowns. `bench_dev`'s checks cannot detect them.

### How to use these numbers

| Result | Status |
|---|---|
| `vision_tower` −58% / −60% (2.4–2.5×), bench_dev 4-run p50 | **Official WP2 A/B component result** |
| `vision_tower` 34.20 → 11.02 ms (3.10×), `wp2_diag_1` | Interleaved diagnostic: controls for host noise. Quote it *with* that label. |
| ≥20% gate | **PASS** |
| End-to-end ≈ −2% | **Indicative only**, from the diagnostic and from post-hoc fast-state filtering. The bench_dev A/B cannot resolve it. |
| FP16 tower is launch-bound | Measured in the diagnostic (event time / CPU issue time = 1.00) |

None of these are serving numbers. Reportable serving results come from WP6 and Rithvik's harness.

### Remaining before WP2 can close

1. **Profiler verification** (roadmap WP2 §5 step 6, §9 criterion 2). Produce an FP32 trace and an FP16 trace, and confirm:
   * `ampere_sgemm_*` is gone, replaced by FP16 tensor-core GEMMs with FP32 accumulate;
   * softmax and LayerNorm dtypes;
   * softmax is not `<Half,Half,Half>`, i.e. it still accumulates in FP32;
   * the tower's time in the trace is consistent with the benchmark.
2. **Layer-1 correctness.** No script exists yet.
   * exact-text match, greedy, 90 dev images, both configs (bar: ≥85%);
   * token-selection equivalence;
   * cosine similarity of pre-projector features > 0.999;
   * determinism.

   Note: `clip_encoder.forward` returns `image_attentions` without casting, so VisPruner now ranks tokens on FP16 attention. The selected tokens may differ; this check will show whether they do.
3. **Layer-2 TextVQA.** 1,000-question subsample, FP32 vs FP16 at 576, scored with `m4c_evaluator.py`. Revert the change if it regresses beyond noise.
4. **Layer 3.** Request per-category accuracy from Sribhav. Asynchronous; don't block on it.
5. **Re-baseline** the vision-path numbers in `AMAY_ENGINEERING_ROADMAP.md` and `AMAY_TRACE_PLAN.md`. Then close WP2 here.
6. Decide whether to report an end-to-end figure, or leave it to the WP6 load benchmark.

## Completed Work

### Pre-WP profiling and evidence phase (2026-08-29 → 2026-09-05)

* **Concurrency crash:** a shared model instance is unsafe under concurrent access. Repro: `scripts/repro_concurrent_crash.py`. Still unfixed; it's WP3/WP4 territory.
* **Trace-driven profiling**, including the prep-path double-counting fix: `AMAY_VISPRUNER_PROFILING_REPORT.md`, `AMAY_TRACE_PLAN.md`.
* **Original prep trace:** `results/timing/prep_trace.json` / `prep_summary.txt`, taken at `c317688` on an older pod without the CPU-thread pin. This is the source of the "75.4% FP32 sgemm" figure. Keep it unchanged; WP2's traces go to `results/timing/wp2_profile/`.
* **Load sweeps** (0.5–3 rps): `results/loadgen/2026-08-29/`, `2026-09-01/`.
* `AMAY_ENGINEERING_ROADMAP.md`, restructured into WP1–WP7 on 2026-09-07.

### WP1 — Controlled Benchmark (closed)

* **Harness:** `scripts/bench_dev.py`.
  * CUDA-event timing of `vision_tower`, `mm_projector` and `multimodal_prep`, with no syncs inside;
  * 20 fixed images, fixed 32-token output;
  * 10 warm-up + 40 measured trials per config, interleaved;
  * records commit, GPU, CPU threads and throttle counts.
* **Noise floor:** three baselines at `469a071`. Rerun spread: `vision_tower` p50 at most 0.08 ms; whole request at most 3.9 ms. §9 gate PASS.
* **CPU-throttling fix.**
  * The container is capped at 7.65 CPUs, but torch spawned 48 threads.
  * Each request's burst exhausted the quota, causing a ~55–70 ms stall.
  * Fix: pin torch to 4 threads. Diagnosed with `scripts/diagnose_prep_stall.py`.
* **Part 1 whole-request baseline:** `results/timing/wp1_controlled_benchmark_*`, from 2026-09-07. Historical; its prep numbers include the stall.

## Standing benchmark protocol (WP2 onward)

* Use `scripts/bench_dev.py` with a fresh `--run-id`. Never use `--overwrite` or `--allow-shared-gpu`.
* 40 measured and 10 warm-up trials per config. Torch pinned to 4 threads. Any throttled trial contaminates the run.
* Compare only against baselines taken on the same pod. Commit the harness before running.
* **New after WP2:**
  * **Randomise run order across states** (e.g. ABBA), per rule 10.
  * For short, launch-bound regions like the FP16 tower, **interleave both states in one process** when possible. Host CPU slowdowns aren't caught by the throttle check.
* Code states used in WP2: FP32 = `e295fd0`, FP16 = `caf19fe`. They differ by one line.

## Important Decisions / Findings

* **The FP16 vision tower is CPU-launch-bound (2026-09-22).** Any further vision-path win needs fewer kernel launches (CUDA graphs, `torch.compile`, fused attention), not faster GEMMs. From now on, host CPU speed shows up directly in its latency. That matters for WP3–WP5, where the server process competes for CPU.
* **Host CPU slowdowns are invisible to the throttle check (2026-09-22).** They appear as discrete jumps in decode per token and as slowdowns in `prep_other` at 128. See the WP2 section above.
* **Merged `origin/main` instead of rebasing (2026-09-21).** A rebase would have given new SHAs to commits that bench_dev JSONs record as `git_commit`.
* **WP1 findings still hold:**
  * the container CPU-throttling diagnosis and the 4-thread pin;
  * `vision_tower` is 33.9 ms under CUDA events, not the trace's 43.9 ms, because the trace includes profiler overhead;
  * the fixed-output-length off-by-one (`max_new_tokens=64` gives 65 `output_ids`), which is harmless.

## Open carry-overs

* **WP3.** Nothing in `csnbs/` or `scripts/start_model_server.sh` sets a torch thread count, so the server likely hits the ~55–70 ms throttling stall on every request.
  * Now that the FP16 tower is CPU-bound, CPU contention in the server matters even more.
  * Check `nr_throttled` before and after the rps 2.0 run.
  * The fix is in Rithvik's server, so agree it with him.
* **Git host key.** This pod's `~/.ssh/known_hosts` lacked GitHub's host key; it was added by hand on 2026-09-21. Add it to the Dockerfile's boot `CMD` so it survives pod restarts. Not done.
* **`CLAUDE.md`.** Add two entries to "Gotchas already hit": the CPU-throttling stall and the missing `known_hosts` entry. Not done.
