# Handoff — resume point (paused 2026-09-22)

The full detail is in `STATUS.md`. This file covers only what's needed to pick up again. The previous handoff, about fixing the `vis_pruner_copy` submodule and download, is resolved; it's in git history before this commit.

## Where we are

**WP2 — FP16 Vision Tower, in progress.** The implementation and the performance A/B are done. Verification and correctness are open. WP3 has not started.

## What WP2 changed

* Commit `caf19fe`, parent `e295fd0`. One line: `clip_encoder.py:30`, `CLIPVisionModel.from_pretrained(..., torch_dtype=torch.float16)`.
* Why it was needed: the CLIP tower loads separately from the LLaVA checkpoint, using a float32 config. The intended FP16 cast (`builder.py:155`) is gated on `device_map != 'auto'`, so it never ran.
* Code states: FP32 = `e295fd0`, FP16 = `caf19fe`. They differ only by that line.

## Measured results (NVIDIA A40, same pod)

* **`bench_dev` A/B**, 4 runs in order FP32 → FP16 → FP16 → FP32. `vision_tower` p50:
  * vtn=576: 33.98 → 14.35 ms (**−57.8%, 2.37×**)
  * vtn=128: 34.00 → 13.68 ms (**−59.8%, 2.49×**)
  * Rerun spread: at most 0.21 ms.
  * **The ≥20% gate passes.**
* **Interleaved diagnostic `wp2_diag_1`**, vtn=128: **34.20 → 11.02 ms (−67.8%, 3.10×)**.
* Prefill and `mm_projector` are unchanged.

## What the jitter diagnostic established

* **The FP16 tower is CPU/kernel-launch-bound.** Its CUDA-event time equals its CPU issue time (ratio 1.00; FP32: 3.22).
* **The FP16 runs' per-trial wander came from host CPU slowdowns** that lasted tens of seconds. FP16's launch-bound tower exposes them; FP32's GPU-bound tower hid them.
* **FP16 does not change downstream work.** Against neighbouring FP32 trials in the same process: `prep_other` +0.14 ms, decode −0.04 ms/token.
* **No GPU clock effect:** the SM clock held at 1740 MHz with no throttle reasons.
* The throttle check can't see these slowdowns.

## What's reportable

* **Official WP2 component result:** the bench_dev 4-run p50, −58% / −60% (2.4–2.5×).
* **The 3.10× figure:** a diagnostic that controls for host noise. Always quote it with that label.
* **End-to-end ≈ −2%: indicative only.** The bench_dev A/B shows −0.2%, which is inside run-to-run noise.
* None of these are serving numbers; those come from WP6 and Rithvik's harness.

## Remaining before WP2 can close

1. **Profiler verification — the next task.**
2. **Layer-1 correctness.** A script is still to be written.
   * exact-text match on 90 dev images (≥85%);
   * token-selection equivalence;
   * pre-projector cosine > 0.999;
   * determinism.

   Watch-out: `image_attentions` isn't cast back to FP16's input dtype, so VisPruner now ranks tokens on FP16 attention.
3. **Layer-2 TextVQA:** 1,000-question subsample, FP32 vs FP16 at 576, `m4c_evaluator.py`.
4. **Layer 3:** request per-category accuracy from Sribhav (asynchronous).
5. **Re-baseline** the vision-path numbers in `AMAY_ENGINEERING_ROADMAP.md` and `AMAY_TRACE_PLAN.md`, then close WP2 in `STATUS.md`.

## Next task: profiler verification (prepared, not yet run)

**Goal:** show that the FP32 `ampere_sgemm_*` vision kernels were replaced by FP16 tensor-core GEMMs (names containing `s16816gemm`, `xmma`, `f16f16_f16f32` or `tensorop_f16`), with FP32 accumulate.

Also check:

* softmax and LayerNorm dtypes, including that softmax is **not** `<c10::Half, c10::Half, c10::Half>`;
* any FP16-accumulate GEMMs (`h16816`), which the script flags;
* whether the tower's time in the trace matches the benchmark (FP32 ≈ 34 ms, FP16 ≈ 11 ms).

**Script:** `scripts/profile_multimodal_prep.py`, commit `71010a5`.

* It now takes `--run-id` and writes to `results/timing/wp2_profile/<run-id>_{prep_trace.json, prep_summary.txt, tower_trace.json, tower_kernels.txt}`. It refuses to overwrite.
* It adds a **tower-only** profile. This is needed because VisPruner's selection loop runs FP16 matmuls in both states.
* The old `results/timing/prep_trace.json` / `prep_summary.txt` (`c317688`, older pod, no thread pin) must stay unchanged.

**States:**

| Run | Branch | Commit | Model state |
|---|---|---|---|
| FP32 | `wp2-profile-fp32` (local) | `945abe6` = `e295fd0` + the profiler-script commit, cherry-picked | FP32 tower |
| FP16 | `main` | contains `caf19fe` and the same profiler script | FP16 tower |

The two states differ in model code only by the one `clip_encoder.py:30` line. The profiler script is identical in both.

If the branch is ever lost, recreate it:

```bash
git checkout -b wp2-profile-fp32 e295fd0 && git cherry-pick 71010a5
```

**Before each run:**

* the tracked tree is clean;
* `nvidia-smi` shows no compute apps;
* no python, server or load-generator process is running;
* the run ID is unused.

**Commands.** Run FP32 first, then validate, then run FP16:

```bash
cd /workspace/GPU_Profiling
git checkout wp2-profile-fp32
python scripts/profile_multimodal_prep.py --run-id wp2_profile_fp32
# validate, then:
git checkout main
python scripts/profile_multimodal_prep.py --run-id wp2_profile_fp16
```

**Expect:**

* FP32: `vision_tower dtype=torch.float32`, `ampere_sgemm_*`, `softmax_warp_forward<float, float, float…>`, `vectorized_layer_norm_kernel<float, float>`.
* FP16: `torch.float16`, `ampere_fp16_s16816gemm_*` or `sm80_xmma_gemm_f16f16_f16f32_*`, `softmax_warp_forward<c10::Half, c10::Half, float…>`.

## Open carry-overs (not WP2)

* **WP3:** the server sets no torch thread count, so it likely hits the throttling stall. This matters more now that the FP16 tower is CPU-bound. It needs Rithvik.
* **Git host key:** add GitHub's host key to `~/.ssh/known_hosts` in the Dockerfile's boot `CMD`, so pushes work after a pod restart.
* **`CLAUDE.md`:** add two gotchas, the CPU-throttling stall and the missing `known_hosts` entry.
