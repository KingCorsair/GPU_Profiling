# GPU Profiling Project Status

## Current State

* **WP3, WP4 and WP5 are closed (2026-10-25).** See their sections below.
* **Current work package: WP6 — Final Load Benchmark.** Pilot `wp6_pilot_1` done (`17e6ab0`); noise-floor repeat blocks added (`ac6a827`). Full matrix `wp6_matrix_1` started 2026-10-25 05:19 UTC: 74 blocks in seeded random order via `scripts/wp6_load_matrix.py`, output in `results/timing/wp6_matrix/wp6_matrix_1/`.
* Open, not blocking: the Layer-3 per-category request (WP2) and the WP5 one-answer EOS difference, both with Sribhav (`kingcorsair/SRIBHAV_ACCURACY_REQUESTS.md`).
* Blockers: none.
* Last updated: 2026-10-25

## Month-One Progress

* [x] WP1 — Controlled Benchmark (2026-09-07; component timing and noise floor 2026-09-11)
* [x] WP2 — FP16 Vision Tower (2026-10-02; Layer-3 confirmation from Sribhav still to come)
* [x] WP3 — Server Diagnostic (2026-10-25, `wp3_diag_2`)
* [x] WP4 — Request Queue (2026-10-25)
* [x] WP5 — Static Batching MVP (2026-10-25, `wp5_val_2`; one EOS rounding difference accepted)
* [ ] WP6 — Final Load Benchmark (matrix `wp6_matrix_1` running)
* [ ] WP7 — Final Writeup

## WP3 — Server Diagnostic (closed 2026-10-25)

**Method.** `csnbs/server.py` served **unmodified**, instrumented from outside by `scripts/wp3_diag_server.py`: ASGI stage timestamps, shims around b64 decode / tokenizer / `process_images` / `generate`, and a 5 ms event-loop heartbeat. `scripts/wp3_run_diag.py` drove it with `csnbs/measure` at 2.0 rps, 60 s per session, {576, 128} × {torch default threads, 4 threads} × 2 repeats in random order. Run `wp3_diag_2` at `cad5444`, clean tree, A40. Files: `results/timing/wp3_diag/wp3_diag_2/summary.{txt,json}`.

| Pooled over repeats | vtn=576, default | vtn=576, 4 thr | vtn=128, default | vtn=128, 4 thr |
|---|---|---|---|---|
| Overlappable CPU (`t_preprocess_done − t_handler_entry`) | 23.7 ms | 25.2 ms | 26.3 ms | 28.3 ms |
| of which `process_images` | 19.3 ms | 20.5 ms | 21.5 ms | 23.1 ms |
| `generate()` | 408.8 ms | 386.2 ms | 378.5 ms | 335.2 ms |
| Service time (handler span) | 433.7 ms | 412.4 ms | 406.3 ms | 364.6 ms |
| **Overlappable CPU / service** | **5.5%** | **6.1%** | **6.5%** | **7.8%** |
| CFS `nr_throttled`, 2 sessions | 195 | 0 | 170 | 0 |
| GPU utilisation (nvidia-smi) | 78% | 74% | 70% | 72% |

**Findings (MEASURED).**

* **Go for WP4, with a small expected gain.** Overlappable CPU work is 5.5–7.8% of service time, just above the 5% gate. The most a queue can recover from overlap is about that much capacity.
* **The thread pin matters more than the queue.** With torch's default threads the server is CFS-throttled about once per request (83–102 events per 120 requests). Pinning to 4 threads removes it entirely and cuts `generate()` by 23 ms at 576 and 43 ms at 128. WP4 onward and every WP6 server state pin 4 threads.
* **The event loop is blocked for the whole request.** Heartbeat blocked time ≈ handler time in every session. 93–95% of requests are not even *received* (ASGI call start) until the previous response is sent.
* **Analysis correction made during WP3.** A queue-based "service interval" cannot be measured from server-side timestamps, because arrivals are invisible while the loop is blocked. Service time is the handler span; `wp3_analyze.py` was changed accordingly before the numbers above were taken.

## WP4 — Request Queue (closed 2026-10-25)

* **Engine:** `vis_pruner_copy/llava/serve/batch_engine.py`. One inference worker thread owns every CUDA call; bounded `queue.Queue` (full → HTTP 503); `submit()` returns a `Future`.
* **Server (S1/S2):** `vis_pruner_copy/llava/serve/batch_server.py`. Same HTTP contract as `csnbs/server.py`, so `csnbs/measure` works against it unchanged. b64 / PIL / `process_images` / tokenization run in a 4-thread pool; the event loop never blocks. `/health` reports the configuration, which `csnbs/measure` records in `run.json`. Rithvik's `csnbs/server.py` was not modified.
* **Shared-state fix:** `llava_llama.py` timing lists are now thread-local instead of on the shared model instance.
* **Output identity (S1 vs S0):** 90/90 identical at both 576 and 128 (`wp5_val_1` and `wp5_val_2`, `engine.generate_batch` at B=1 vs `csnbs/server.py`'s exact code path).
* S1's load numbers are in WP6.

## WP5 — Static Batching MVP (closed 2026-10-25)

**Fixes needed for B>1 (all MEASURED by the checks below):**

1. **B>1 crash.** `prepare_inputs_labels_for_multimodal` masked the whole `(B, N, C)` batch at once, flattening every image's kept tokens into one row: the first sequence got all of them and the second indexed past the end. Now split per image (`llava_arch.py`).
2. **Left padding.** `builder.py` sets `tokenizer_padding_side = "left"` at load; a no-op at B=1.
3. **Batch-dependent token selection.** `wp5_val_1` at vtn=128: every image kept a different token set when batched, and only 79–80/90 answers matched B=1. `scripts/wp5_selection_diag.py` (`wp5_seldiag_1`) showed this is batch-shape numerics, **not** cross-request mixing: 8 copies of one image in a batch agree with each other (24/24), and differ from B=1 by the same 74% kept-token overlap as random partners. Running only the tower per image did not fix it (0/24), which points at `argsort` tie-breaking in the selection code as well as tower rounding. **Fix:** `encode_images` runs per image; the LLM stays batched.

**Validation, `wp5_val_2`** (`scripts/wp5_batch_validation.py`, `6a471cb`, 90 dev images, greedy, max 64 tokens; files `results/wp5_validation/wp5_val_2.{json,txt}`). The run recorded `dirty=True` only because `generate()` appended rows to the tracked `results/timing/llava_llama_timing.json`; serving and validation now write that file to `/tmp`.

| | vtn=576 | vtn=128 |
|---|---|---|
| B=1 via engine vs S0 | 90/90 | 90/90 |
| B=2 / 4 / 8 vs single request | 90 / 90 / 89 | 90 / 90 / 90 |
| Kept-token set changed | 0 | 0 |
| Contamination (anchor with short-prompt and long-output partners, both orders) | 10/10 | 10/10 |

* **1 of 540 batched answers differs:** `textvqa_34610` at 576, B=8, same kept tokens. The first sentence is identical; the single-request answer emits EOS after it, the batched one continues. This is LLM-side FP16 rounding of the EOS decision. Roadmap criterion 1 ("identical, no exceptions") is therefore **missed by one answer**. Accepted because it is a rounding difference and not contamination, and sent to Sribhav for a scoring check (`kingcorsair/SRIBHAV_ACCURACY_REQUESTS.md`).

**Batched `generate()` throughput, no server** (fixed 64-token output, CUDA events, 10 reps after 5 warm-up, interleaved B; throwaway scaling curve, MEASURED but not a serving number):

| B | vtn=576 | vtn=128 | 128 / 576 |
|---|---|---|---|
| 1 | 0.48 req/s | 0.52 req/s | 1.07× |
| 4 | 1.39 (2.9×) | 1.77 (3.4×) | 1.28× |
| 8 | 2.05 (4.3×) | 3.01 (5.8×) | 1.47× |
| 16 | 2.69 (5.6×) | 4.52 (8.7×) | 1.68× |

* **Pruning's advantage grows with batch size** (1.07× → 1.68×). At B=1 decode is memory-bound and pruning-independent; batching amortises the weight reads, so prefill and the attention/KV traffic over visual tokens become a larger share. DERIVED interpretation, consistent with the roadmap's "decode dominates at B=1" story; WP6 tests it under load.
* Per-image selection costs about 5% of batch time at 128, B=16 (3368 → 3538 ms, `wp5_val_1` vs `wp5_val_2`).

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

### Profiler verification (2026-10-02)

`scripts/profile_multimodal_prep.py`, FP32 at `945abe6` (run 2026-10-01) and FP16 at `be4ed77` (run 2026-10-02), same pod. Files: `results/timing/wp2_profile/wp2_profile_{fp32,fp16}_{prep_summary,tower_kernels}.txt`. The raw `*_trace.json` files are git-ignored.

| | FP32 | FP16 |
|---|---|---|
| `ampere_sgemm_*` share of prep-region GPU time (summed from the raw trace) | **75.4%** | **0.0%** |
| `ampere_sgemm_*` share of tower-only GPU time | 81.2% | 0.0% |
| FP16 tensor-core GEMM share, tower only | 0.0% | 66.8% |
| Tower GPU busy per call (kernel rows) | 34.47 ms | 9.05 ms |
| Tower CUDA-event p50, 50 back-to-back calls | 35.38 ms | 10.40 ms |
| Prep region GPU busy per call | 35.7 ms | 11.4 ms |
| `prepare_inputs_labels_for_multimodal` wall, mean of 10 | 49.74 ms | 24.47 ms |
| CPU time / GPU busy in the prep region | 2.01× | 4.63× |

* The FP16 GEMMs are `ampere_fp16_s16816gemm_*`, `ampere_fp16_s1688gemm_*` and `cutlass_75_tensorop_f16_s1688gemm_*`: FP16 tensor cores with **FP32 accumulate**. No FP16-accumulate (`h16816`) kernel appears.
* Softmax is `softmax_warp_forward<c10::Half, c10::Half, float>`: FP16 in and out, **FP32 accumulate**. LayerNorm is `vectorized_layer_norm_kernel<c10::Half, float>`.
* The tower's time in the trace agrees with the benchmark (FP32 ≈ 34 ms, FP16 ≈ 10–14 ms).
* Do not compare the two summaries' `generate(max_new_tokens=1)` lines (207.73 vs 89.02 ms). That loop has no warm-up of its own, so it includes a cold first call.

### Layer-1 correctness (2026-10-02)

`scripts/wp2_layer1_correctness.py`, run `wp2_layer1_1`: all 90 dev images, greedy, both towers in one process. Files: `results/wp2_correctness/wp2_layer1_1.{json,txt}`.

| Check | Result | Bar | |
|---|---|---|---|
| Exact text, FP32 vs FP16, vtn=576 | 90/90 identical | ≥85% | PASS |
| Exact text, FP32 vs FP16, vtn=128 | 80/90 identical (88.9%) | ≥85% | PASS |
| Determinism, FP16 run twice, both configs | 90/90 identical output ids | identical | PASS |
| Softmax still accumulates in FP32 (from the trace) | yes | yes | PASS |
| Pre-projector feature cosine, per image over the 576-token map | min 0.99816, median 0.99978; 6 of 90 images at or below 0.999 | >0.999 | **MISS** |
| Token selection at vtn=128 | 0/90 images keep the identical token set; 74.4% of kept tokens shared | report only | — |

* **The ten changed answers at 128 are not spread evenly:** OCR 5 of 15, counting 2 of 15, object presence 2 of 15, spatial reasoning 1 of 15. Two counting answers flip between yes and no. Whether they got better or worse is Sribhav's call (Layer 3).
* **Why the token set changes on every image.** The attention-ranked half is 99.25% shared, so the top-attention tokens barely move. The other half comes from VisPruner's duplicate-removal loop (`llava_arch.py`, `encode_images`), which pairs tokens by their position in the attention ranking. Read from the code, not measured: a few near-tied tokens swapping rank would re-pair everything after them, so a tiny numeric change can produce a large change in which tokens survive.
* **Why the cosine bar is missed.** A follow-up on 20 images (not saved as a script) found no NaN or infinity, and peak activations of about 210, far below FP16's limit of 65,504. The difference builds up gradually from layer 11 onward. The mean per-token cosine stays above 0.9995 at every layer; a median of 13 of 576 tokens per image fall below 0.999 and 3 below 0.99. This is rounding accumulating in a few tokens, not an overflow.
* The roadmap attaches no stop rule to the cosine bar. The deciding gate is Layer 2.

### Layer-2 TextVQA gate (2026-10-02)

`scripts/wp2_layer2_textvqa.py`, run `wp2_layer2_1`: 1,000 questions from `eval/textvqa/llava_textvqa_val_v051_ocr.jsonl` (seed 20261002), official `m4c_evaluator.py`, every question answered by both towers in one process. Files: `results/wp2_correctness/wp2_layer2_1*`.

| Config | FP32 | FP16 | Difference | Paired 95% interval | Answers changed | Scores changed |
|---|---|---|---|---|---|---|
| vtn=576 (the roadmap's gate) | 57.62% | 57.65% | +0.03 pp | 0.00 to +0.09 pp | 7 | 1 (FP16 higher) |
| vtn=128 (added after Layer 1) | 56.50% | 56.75% | +0.25 pp | −0.74 to +1.25 pp | 76 | 33 (19 higher, 14 lower) |

* **Gate: PASS.** No regression outside noise at either config, so the change is not reverted.
* The 22 questions on images that are in the locked test set were removed before sampling (rule 11).
* `python -m llava.eval.eval_textvqa` on the four answer files reproduces the same four accuracies.
* The TextVQA images are not in git. `scripts/download_textvqa_subset.py` fetches the annotations and the 916 images this subsample uses into `/workspace/datasets/textvqa`.

### How to use these numbers

| Result | Status |
|---|---|
| `vision_tower` −58% / −60% (2.4–2.5×), bench_dev 4-run p50 | **Official WP2 A/B component result** |
| `vision_tower` 34.20 → 11.02 ms (3.10×), `wp2_diag_1` | Interleaved diagnostic: controls for host noise. Quote it *with* that label. |
| ≥20% gate | **PASS** |
| End-to-end ≈ −2% | **Indicative only**, from the diagnostic and from post-hoc fast-state filtering. The bench_dev A/B cannot resolve it. |
| FP16 tower is launch-bound | Measured in the diagnostic (event time / CPU issue time = 1.00) |
| FP32 sgemm share 75.4% → 0.0% | Trace pair, `wp2_profile_{fp32,fp16}` |
| TextVQA 57.62% → 57.65% at 576; 56.50% → 56.75% at 128 | Layer-2 gate, 1,000-question subsample. A development gate, not the project's accuracy result; that is Sribhav's. |

None of these are serving numbers. Reportable serving results come from WP6 and Rithvik's harness.

### Closing WP2 against roadmap §9

| Criterion | Result |
|---|---|
| 1. `vision_tower` CUDA time drops by at least 20% | **PASS**: −58% / −60% |
| 2. `ampere_sgemm_*` replaced by the FP16 GEMM family | **PASS**: 75.4% → 0.0% of prep GPU time |
| 3. Layer-2 TextVQA unchanged within noise | **PASS**: +0.03 pp at 576 |
| 4. Output deterministic across repeated runs | **PASS**: 90/90 at both configs |

Left open:

* **Layer 3.** Per-category accuracy on Sribhav's **dev split**, FP32 (`e295fd0`) vs FP16 (`caf19fe`), with OCR and counting at 128 in particular. *Corrected 2026-10-25:* this said "locked eval set", but a keep/revert decision taken from test results contaminates the test set (rule 11). Request written up in `kingcorsair/SRIBHAV_ACCURACY_REQUESTS.md`. If OCR regresses, revert.
* **End-to-end figure.** Not reported from WP2. About −2% is indicative only; the reportable number comes from the WP6 load benchmark.
* **Feature-cosine bar.** Missed in 6 of 90 images, as described above. Accepted on the strength of Layer 2.

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
* **New after WP2's correctness checks:** for an accuracy A/B of a change inside the model, load both states in one process and answer every question with both, so the comparison is paired. Exclude locked-test images from any gate that decides whether a change is kept.

## Important Decisions / Findings

* **The FP16 vision tower is CPU-launch-bound (2026-09-22).** Any further vision-path win needs fewer kernel launches (CUDA graphs, `torch.compile`, fused attention), not faster GEMMs. From now on, host CPU speed shows up directly in its latency. That matters for WP3–WP5, where the server process competes for CPU.
* **Host CPU slowdowns are invisible to the throttle check (2026-09-22).** They appear as discrete jumps in decode per token and as slowdowns in `prep_other` at 128. See the WP2 section above.
* **Merged `origin/main` instead of rebasing (2026-09-21).** A rebase would have given new SHAs to commits that bench_dev JSONs record as `git_commit`.
* **WP1 findings still hold:**
  * the container CPU-throttling diagnosis and the 4-thread pin;
  * `vision_tower` is 33.9 ms under CUDA events, not the trace's 43.9 ms, because the trace includes profiler overhead;
  * the fixed-output-length off-by-one (`max_new_tokens=64` gives 65 `output_ids`), which is harmless.

## Open carry-overs

* **Thread pin in Rithvik's server.** WP3 confirmed the stall (`nr_throttled` 170–195 at default threads, 0 at 4). `batch_server.py` pins 4 threads, but `csnbs/server.py` and `scripts/start_model_server.sh` still set none. Anything served through S0 is still throttled. The fix is in Rithvik's server, so agree it with him.
* **`/workspace` quota.** A 7 GB download failed with a write error on 2026-10-02 when `du -sh /workspace` reached about 59 GB, so the volume appears to be capped near 60 GB, not the ~100 GB `CLAUDE.md` states. About 53 GB is in use. Check the pod's configured volume size.
* **Stashed timing rows.** `git stash list` holds 10 rows that an interrupted profile run appended to `results/timing/llava_llama_timing.json` on 2026-10-01. They are log lines with no use; `git stash drop` removes them.
* **Git host key.** The Dockerfile now pins GitHub's ed25519 host key into `/root/.ssh/known_hosts` (fingerprint `SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU`, checked against GitHub's published value, 2026-10-25). It takes effect once CI rebuilds the image and the pod is redeployed on it.
* **`CLAUDE.md` gotchas.** The CPU-throttling stall and `known_hosts` entries were added 2026-10-25.
