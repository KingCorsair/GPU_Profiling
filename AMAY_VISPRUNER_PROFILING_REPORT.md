# The Algorithmic-to-Serving Gap in VisPruner: A GPU Profiling Investigation of LLaVA-1.5 Inference

**Author:** Amay (with Claude Code) · **System:** LLaVA-1.5-7B + VisPruner on NVIDIA A40 · **Status:** Profiling/research phase complete, engineering phase not started

---

## Abstract

VisPruner discards most of the 576 visual tokens LLaVA-1.5 produces for an image, cutting the compute needed to process them by a large, easily-measured margin. This report asks a narrower and more useful question: **how much of that algorithmic reduction actually shows up as a faster response, and how much survives once the system is under realistic load?**

We instrumented the model with `torch.profiler` at increasing scope — a single narrow function, then a full request, then a live HTTP server under an open-loop load sweep — and ran a controlled A/B comparison between `visual_token_num=576` (effectively unpruned) and `visual_token_num=128` (VisPruner) with everything else held fixed. The measured result: cutting visual tokens 4.5x reduces prefill compute by roughly 56-60%, but end-to-end request latency improves by only about 13% (after correcting a timing bug described in full below), and the server's sustainable request-handling capacity under load improves by only about 4% — a figure we treat as provisional, not final. The gap is explained by two separate, both measured, mechanisms: decode (which pruning barely touches) dominates total request time, and the current server processes exactly one request at a time, so its capacity ceiling is set by mean service time rather than peak compute.

This document is the consolidated record of that investigation: what was measured, how, where the evidence lives, what turned out to be wrong or imprecise along the way, and what it implies for the next engineering phase. No optimizations were implemented as part of this work.

---

## 1. Motivation and Research Question

LLaVA-1.5 turns one image into 576 visual tokens before the language model ever sees the question. VisPruner (FasterVLM-style: rank tokens by `[CLS]` attention, keep a subset) discards most of them. The natural assumption — fewer tokens, proportionally less work, proportionally faster answers — is exactly the assumption this investigation set out to test rather than accept.

**Refined research question**, worded to match what the evidence actually supports:

> *Does reducing LLaVA's visual-token count with VisPruner translate proportionally into (a) shorter complete requests, and (b) more requests served per second under load — and where, specifically, does the gap between the algorithmic reduction and either of those outcomes come from?*

The word "proportionally" matters. A 60% cut to prefill's FLOPs does not entitle you to expect a 60% cut to the full request — that would only hold if prefill were the *only* cost in a request. It isn't. This report treats four things as genuinely different quantities, conflating them is the single most common way this kind of result gets overstated:

| Quantity | What it means | Does pruning act on it directly? |
|---|---|---|
| **Algorithmic / FLOPs reduction** | Fewer visual tokens → less matrix-multiply work in the parts of the model that scale with token count | Yes — this is what VisPruner *is* |
| **Prefill-phase execution time** | Wall-clock time for the one forward pass over the whole prompt | Yes, but not linearly (see Experiment 6) |
| **Complete request latency** | Time from receiving a request to returning an answer — includes prefill *and* decode *and* preprocessing | Only partially — decode dominates and pruning barely touches decode |
| **Serving throughput / capacity** | How many requests per second the system can sustain before queueing degrades | Only as much as complete-request latency improves, and only if the serving architecture can actually convert a faster request into more capacity |

A FLOPs reduction never "disappears" — it is real at every step of this chain. What can happen, and what this report finds does happen, is that it gets diluted at each translation from one quantity to the next.

---

## 2. System Under Test

All values below are drawn from experiment metadata files, not restated from memory. Where a setting varies by experiment, the per-experiment section states the actual value used.

| Property | Value | Source |
|---|---|---|
| Model | LLaVA-1.5-7B (fp16, CLIP ViT-L/14@336 vision encoder) | `results/timing/full_request_settings.json` |
| VisPruner implementation | Vendored in `vis_pruner_copy/llava/model/llava_arch.py`, `encode_images()` / `prepare_inputs_labels_for_multimodal()` | Code inspection |
| GPU | NVIDIA A40 (47.7GB) | `full_request_settings.json`, all `*_settings.json` files |
| PyTorch | 2.2.2+cu121 | `full_request_settings.json` |
| CUDA | 12.1 | `full_request_settings.json` |
| cuDNN | 8902 | `full_request_settings.json` |
| Transformers | 4.37.2 | `full_request_settings.json` |
| Attention implementation | `sdpa` | `full_request_settings.json` (`attn_implementation`) |
| dtype | fp16 | `full_request_settings.json` |
| Batch size | 1, throughout every experiment | All settings files — no batched-request path exists in this codebase yet |
| `important_ratio` | 0.5, held fixed across every experiment | All settings files |
| `visual_token_num` configurations tested | 576 (effectively unpruned — CLIP always emits exactly 576 patches) and 128 (VisPruner) | `CLAUDE.md`, all A/B experiment settings |
| Server | `csnbs/server.py` (FastAPI/uvicorn, single process, single asyncio event loop) — **Rithvik's file**, one line made configurable for this work (see §3) | Code inspection |
| Load generator | `csnbs/measure` (TypeScript, open-loop, fixed arrival rate) — **Rithvik's tool**, used unmodified | Code inspection |
| Dataset / workload | `vis_pruner_copy/vispruner_eval_dataset/dev.json`, 90 TextVQA-style image+question records | All load-sweep metadata |
| Output-token policy (load sweep, A/B traces) | `do_sample=False`, `max_new_tokens=64`, natural EOS stopping | `csnbs/server.py`, `ab_trace_*_settings.json` |
| Output-token policy (Experiment 2 only) | `max_new_tokens=24` forced (`min_new_tokens=24`) — **not** the natural-stopping policy used everywhere else; noted where it matters | `full_request_settings.json` |

---

## 3. Experimental Methodology

**Key terms, briefly, before the methodology gets technical:**

| Term | Meaning |
|---|---|
| **Prefill** | The one forward pass over the entire prompt (image tokens + question tokens) at the start of a request. Produces the first output token. |
| **Decode** | Every forward pass after prefill — one per output token, each depending on the last, each reading the model's full weight matrix again. |
| **CUDA kernel launch** | Each individual GPU operation (a matrix multiply, an add, a copy) has to be separately handed off from the CPU to the GPU. That handoff has a small, fixed cost regardless of how much work the kernel actually does — many small kernels means that fixed cost adds up, even if the total GPU compute is tiny. |
| **KV cache** | The model's running memory of every prior token's key/value attention vectors, so decode doesn't have to recompute attention over the whole sequence from scratch at every step. It grows by one token's worth of data per decode step. |
| **Queueing** | What happens when requests arrive faster than the system can finish them — they wait in line, and that wait time gets added on top of the time the request itself actually takes to run. |
| **p50 / p95 / p99** | Percentile latency: p50 is the median request (half were faster, half slower); p95 and p99 describe the slower tail (only 5% or 1% of requests were slower than this). A system can have a fine p50 and a terrible p99 at the same time — that's exactly what happens under queueing. |
| **Throughput** | Requests successfully completed per second, sustained over a time window — distinct from the *arrival rate* (RPS) offered to the system, which throughput may or may not keep up with. |

**`torch.profiler`** (`torch.profiler.profile(activities=[CPU, CUDA])`) records every CPU-side operator dispatch and every CUDA kernel launch during a profiled window, with wall-clock timestamps. It answers "where did the time go" at the operator level. Every experiment here that produces a trace uses this tool with `record_shapes=True`.

**`record_function()`** labels are custom markers inserted around a region of code so the profiler output can be grouped by *meaning* (e.g. "prefill", "vision_tower") rather than only by raw operator name. Every profiling script in this investigation installs these labels via **runtime monkey-patching** — reassigning `model.forward`, `prepare_inputs_labels_for_multimodal`, submodule `.forward` methods, and `DynamicCache.update` to wrapped versions for the duration of the profiled call, then restoring the originals. No vendored LLaVA or `transformers` source file was ever edited to add instrumentation.

**CUDA events** (`torch.cuda.Event(enable_timing=True)`) are the correct way to time GPU work directly, because CUDA execution is asynchronous — a plain `time.perf_counter()` around a GPU call measures how long Python took to *queue* the work, not how long the GPU took to run it (`AMAY_TIMING_NOTES.md`, rule 1). `llava_llama.py`'s own `generate()` method already contains CUDA-event-based timing (`prep_start_evt`/`prep_end_evt`, `fwd_start_evt`/`fwd_end_evt`), writing one JSON line per `generate()` call to a file controlled by the `LLAVA_TIMING_FILE` environment variable. Experiment 4 reuses this existing instrumentation rather than re-implementing it.

**Chrome/Perfetto traces** are the full per-kernel timeline exported via `prof.export_chrome_trace()`, viewable at `ui.perfetto.dev`. These are large (tens to hundreds of MB) and are the primary-evidence artifact behind every summary table in this report — summary tables are a *reduction* of the trace, not a separate measurement.

**The load generator** (`csnbs/measure`) fires HTTP requests at a fixed rate regardless of whether earlier requests have finished (open-loop), which is what allows a queue to form and be observed — a closed-loop client would back off exactly when the system struggles and hide the problem (`AMAY_TIMING_NOTES.md`, rule 5).

**GPU/CPU monitoring** during load sweeps samples `nvidia-smi` and `/proc/stat` once per second in a background thread for the duration of each load level.

**Warm-up**: every experiment discards 6-10 initial requests (CUDA context init, memory pool growth, kernel selection) before beginning measurement, per the project's rule 3.

**A/B comparison methodology** (Experiments 4-6): the model is loaded once; `visual_token_num` and `important_ratio` are read fresh from `self.get_visual_token_num()`/`self.get_important_ratio()` on every call (`llava_arch.py:148-149`), so flipping `model.visual_token_num` between passes is equivalent to a fresh load for anything downstream of that read, without paying two model-load costs. This was verified by code inspection before relying on it.

**MEASURED vs. DERIVED vs. INTERPRETATION**, used consistently through this report:
- **MEASURED** — a number read directly off an instrument (profiler, CUDA event, load-test summary) with no arithmetic beyond simple aggregation.
- **DERIVED** — a number computed from measured numbers via an explicit formula (e.g. a percentage gain, or the Part-1 timing correction), disclosed as such.
- **INTERPRETATION** — a claim about *why* a measured or derived number looks the way it does, which requires code-reading or reasoning beyond direct observation, and may be wrong.

---

## Experiment 1 — Narrow multimodal-preparation profiling

### Research question
Is `prepare_inputs_labels_for_multimodal()` (the function VisPruner's real-removal gather/fixup logic lives in) actually launch-overhead-bound, before committing to writing a Triton kernel for it?

### Experimental setup
- **Script:** `scripts/profile_multimodal_prep.py` (pre-existing, not written as part of this later phase, but foundational to it)
- `visual_token_num=128`, `important_ratio=0.5`, batch=1, `prompt_len=66`
- 10 warm-up iterations, 10 profiled iterations of `prepare_inputs_labels_for_multimodal()` in isolation, plus a separate 10-iteration timing of `generate(max_new_tokens=1)` for context

### Files produced
| File | Contents |
|---|---|
| `scripts/profile_multimodal_prep.py` | SCRIPT |
| `results/timing/prep_trace.json` (45MB) | PROFILER TRACE — full Chrome/Perfetto timeline |
| `results/timing/prep_summary.txt` | SUMMARY — human-readable top-ops table + aggregate stats |

### Results
| Metric | Value | Source |
|---|---|---|
| `prepare_inputs_labels_for_multimodal` as % of a 1-token prefill request | **43.1%** | `prep_summary.txt` |
| Share of that region's CUDA time in non-matmul ops (gather/index/argsort/cat/pad) | **63.2%** | `prep_summary.txt` |
| CPU time / CUDA time ratio in that region | **1.01x** | `prep_summary.txt` |

### Interpretation
The region is a meaningful fraction of a prefill request, and most of its GPU time sits in small, non-matmul ops rather than large matmuls — consistent with a launch-overhead-bound characterization. **However**, the region's own CPU/CUDA ratio (1.01x) does not meet the script's own stated threshold for that conclusion (its inline comment states ">>1 means launch/dispatch overhead dominates actual GPU work"). This is preserved here as an open internal tension rather than smoothed over: the 63.2%-non-matmul finding and the ~1x CPU/CUDA ratio are both real, measured facts from the same file, and they don't fully agree on how to characterize the region. The most defensible reading is that the region contains many small kernels whose *individual* CPU dispatch cost is comparable to their GPU execution cost, which is still consistent with "launch overhead is a real, non-trivial cost here" without being as unambiguous as ">>1" would imply.

**Correction to the planning record:** `AMAY_SPEED_PLAN.md` (line 353) states this figure as "40.7%". The canonical artifact, `prep_summary.txt`, says **43.1%**. This report uses the artifact's number. The discrepancy's source is not established (possibly a different run, possibly a transcription error) and is flagged rather than silently resolved.

### Limitations
Single function in isolation, not a full request. Says nothing about decode, serving, or the 576-vs-128 comparison (this experiment only ever ran at 128).

### Status
**VALID WITH CAVEAT** — the underlying measurement is sound, but the summary percentage differs from what the planning document claims, and the "launch-overhead-bound" framing rests more on the 63.2% figure than on the (weaker) CPU/CUDA ratio evidence in the same file.

---

## Experiment 2 — Full isolated request profiling

### Research question
Experiment 1 only covered one function. Where does latency go across the *entire* request path — image preprocessing, multimodal prep, vision tower, projector, prefill, every decode step, and completion?

### Experimental setup
- **Script:** `scripts/profile_full_request.py`
- `visual_token_num=128`, `important_ratio=0.5`, batch=1, `prompt_len=66`
- **Note:** `max_new_tokens=24` **forced** (`min_new_tokens=24`) to guarantee multiple visible decode steps — this is a different output-token policy from every later experiment in this report, which uses natural EOS stopping. Not directly comparable to Experiments 4-6 on absolute decode cost for that reason.
- 10 warm-up requests, 1 profiled request, `record_shapes=True`

### Files produced
| File | Contents |
|---|---|
| `scripts/profile_full_request.py` | SCRIPT |
| `results/timing/full_request_trace.json` (97MB) | PROFILER TRACE |
| `results/timing/full_request_settings.json` | METADATA — exact run configuration |
| `results/timing/full_request_stage_breakdown.txt` | SUMMARY — per-stage CPU/CUDA time table |
| `results/timing/full_request_summary_by_{cuda_time,count,cpu_time}.txt` | SUMMARY — full operator tables, three sort orders |
| `results/timing/full_request_run_log.txt` | LOG — raw stdout of the run |

### Results
| Stage | Calls | CPU ms | CUDA ms | Source |
|---|---:|---:|---:|---|
| image_preprocessing | 1 | 106.6 | 0.0 | `full_request_stage_breakdown.txt` |
| vision_tower | 1 | 78.2 | 36.2 | same |
| mm_projector | 1 | 0.2 | 0.3 | same |
| multimodal_prep | 24 | 112.5 | 46.2 | same |
| prefill | 1 | 36.7 | 60.8 | same |
| decode (sum of 23 steps) | 23 | 951.9 | 686.2 | same |
| kv_cache_legacy_rebuild | 768 | 3.0 | ~0.0 | same |
| kv_cache_append | 736 | 38.4 | 11.8 | same |
| request_completion | 1 | 0.9 | 0.0 | same |

| Aggregate metric | Value | Source |
|---|---:|---|
| Total `cudaLaunchKernel` calls | **34,722** | `full_request_summary_by_cuda_time.txt` |
| `cudaLaunchKernel` self-CPU time | 156.9ms (12.66% of total self-CPU) | same |
| `aten::mm` CUDA time (share of total CUDA time) | 631.6ms (**85.1%** self, of 722.8ms total self-CUDA) | same |
| Self CPU time total (whole profiled region) | 1.240s | same |
| Self CUDA time total (whole profiled region) | 722.8ms | same |

### Interpretation
Decode dominates (686ms of the total CUDA time, vs. 61ms for prefill) — but this run forced 23 decode steps, which inflates decode's share relative to what a natural, short-answer request would show (see Experiment 4/6 for that). Within decode specifically, CPU time exceeds CUDA time every single step (e.g. step 1: 38.9ms CPU vs 38.0ms CUDA; by step 12, 40.4ms CPU vs 28.9ms CUDA) — a real, measured launch-dispatch overhead signature, distinct from and more clear-cut than Experiment 1's ambiguous 1.01x ratio. In aggregate, matmuls still dominate total CUDA time (85%), meaning the system is not *globally* launch-bound — but the *decode phase specifically* shows a consistent, repeated pattern of CPU time exceeding GPU time, which is the pattern CUDA Graphs is designed to address.

**KV-cache mechanism (new finding at this stage):** `DynamicCache.update()` is called twice per layer per decode step, not once. Splitting by whether the incoming key tensor represents the whole running sequence (`kv_cache_legacy_rebuild`, ~0 GPU cost, pure Python/list-append overhead) or a single new token (`kv_cache_append`, real `torch.cat`-based reallocation, 11.8ms CUDA / 38.4ms CPU over 736 calls) shows the mechanism: `llava_llama.py`'s custom `forward()` does not retain a persisted `Cache` object across `generate()` steps — `past_key_values` crosses the step boundary as a legacy tuple, forcing `DynamicCache.from_legacy_cache()` to reconstruct the cache object from scratch every single decode step, immediately before the real incremental append. The 32-layer, per-step call-count split (416 rebuild-shaped + 384 append-shaped per generate() call under this experiment's settings) is measured directly in `full_request_stage_breakdown.txt`. **The exact interactive stack-trace confirmation of this call chain (showing `modeling_llama.py:1016 → DynamicCache.from_legacy_cache → cache.update`) was produced via an ad-hoc diagnostic during this investigation and is not preserved in a saved file — only the resulting measurement (the count split) and the mechanism explanation (recorded as a code comment in `profile_full_request.py:161-176`) are canonical artifacts.**

### Limitations
Single request, single configuration (128 only — no 576 comparison at this stage). Forced decode length means the decode-vs-prefill *ratio* here should not be read as representative of a natural request (Experiment 4/6 correct for this).

### Status
**VALID WITH CAVEAT** — the stage-level measurements and kernel-launch counts are solid; the decode/prefill *ratio* specifically is not representative due to the forced-length policy, and the KV-cache mechanism's deepest verification (the stack trace) is not independently re-checkable from a saved artifact.

---

## Experiment 3 — Serving / load profiling (single configuration, pre-A/B)

### Research question
How does the system behave under realistic concurrent load, and what limits it? (Run once, at `visual_token_num=576`, before the A/B structure existed.)

### Experimental setup
- **Scripts:** `scripts/run_concurrent_load_test.py` (load sweep, orchestrates Rithvik's `csnbs/measure` + `scripts/start_model_server.sh` unmodified) and `scripts/profile_concurrent_load.py` (in-process trace of repeated requests)
- Server settings (hardcoded at the time): `visual_token_num=576`, `important_ratio=0.5`, `max_new_tokens=64`
- Load sweep: RPS ∈ {0.5, 1.0, 1.5, 2.0, 3.0}, 30s per level, execution order randomized
- Trace: originally attempted as **genuine multi-thread concurrency**; this crashed (see below) and was replaced with a **sequential back-to-back** reproduction of 3 requests

### Files produced
| File | Contents |
|---|---|
| `scripts/run_concurrent_load_test.py` | SCRIPT |
| `scripts/profile_concurrent_load.py` | SCRIPT |
| `results/timing/concurrent_load_summary.txt` | SUMMARY — per-RPS-level throughput/latency/GPU-CPU table |
| `results/timing/concurrent_load_metadata.json` | METADATA — full per-level detail + links to raw loadgen runs |
| `results/loadgen/2026-09-01/{02-35-26Z_rps-1,02-35-58Z_rps-3,02-36-40Z_rps-2,02-37-12Z_rps-1.5,02-37-44Z_rps-0.5}_*/` | RAW DATA — Rithvik's harness output, one directory per RPS level (`run.json` summary + `requests.jsonl` per-request detail) |
| `results/timing/loadgen_monitor/rps-{0.5,1.0,1.5,2.0,3.0}.csv` | RAW DATA — 1Hz GPU/CPU utilization samples per level |
| `results/timing/concurrent_trace.json` (431MB) | PROFILER TRACE — 3 sequential requests |
| `results/timing/concurrent_trace_settings.json` | METADATA |
| `results/timing/concurrent_request_timeline.json` | RAW DATA — per-request start/done timestamps |
| `results/timing/concurrent_summary_by_{cuda_time,count}.txt` | SUMMARY |
| `results/timing/concurrent_profile_run_log.txt` (434KB) | LOG — **contains the crash evidence**, 2,752 lines of `indexSelectSmallIndex` assertion failures |
| `results/timing/concurrent_load_run_log.txt` | LOG |

### Results
| RPS | Throughput | p50 | p95 | p99 | GPU util % | CPU busy % | Source |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 0.500 | 861ms | 1070ms | 1070ms | 33.8 | 5.9 | `concurrent_load_summary.txt` |
| 1.0 | 1.000 | 603ms | 818ms | 1026ms | 48.4 | 6.4 | same |
| 1.5 | 1.500 | 588ms | 2332ms | 2389ms | 55.1 | 7.0 | same |
| 2.0 | 2.000 | 2291ms | 5877ms | 6877ms | 79.0 | 7.5 | same |
| 3.0 | **2.166** (below target) | 7138ms | 10914ms | 19599ms | 87.1 | 7.7 | same |

**Thread-safety finding (MEASURED):** launching truly-simultaneous Python threads, each calling `model.generate()` on the shared model instance, crashed with `../aten/src/ATen/native/cuda/Indexing.cu:1237: indexSelectSmallIndex: Assertion 'srcIndex < srcSelectDimSize' failed` — an out-of-bounds `index_select` inside the vision-tower/token-pruning gather path — followed by `CUDA error: CUBLAS_STATUS_EXECUTION_FAILED` and `device-side assert triggered`. Full output preserved in `concurrent_profile_run_log.txt`.

**Back-to-back (sequential) trace, 3 requests:** inter-request gap = **0.05-0.1ms** (`concurrent_trace_settings.json`, `concurrent_request_timeline.json`) — negligible dead time between one request finishing and the next starting.

### Interpretation
First clear degradation at **rps≈1.5** (p95 jumps from 818ms to 2332ms while p50 barely moves — the classic early-saturation signature: the *median* customer doesn't notice yet, the *tail* customer already does). Hard saturation at **rps=3.0** (achieved throughput plateaus at 2.166, below the 3.0 offered rate — the queue is no longer draining). GPU utilization never exceeds 87% even at the worst level tested, meaning the GPU is not yet the hard ceiling.

The crash is not an obstacle that was worked around silently — it is itself evidence: it demonstrates that the current single shared model instance is **not safe for genuine concurrent execution**, which is very likely *why* the server (`csnbs/server.py`) is built to process requests strictly one at a time rather than this being an accidental limitation. The negligible inter-request gap in the back-to-back trace separately rules out "per-request Python setup/teardown overhead" as a meaningful contributor to queueing — the dominant cost really is each request's own compute, serialized.

### Limitations
Single configuration (576 only, at this stage — the A/B comparison came later in Experiment 5). The "near-saturation" trace is back-to-back, not genuinely concurrent GPU execution — see §9 for the full discussion of what this can and cannot establish.

### Status
**VALID** for the 576-token load-sweep numbers and the thread-safety finding. **DIAGNOSTIC ONLY** for the back-to-back trace's applicability to "real concurrency" claims (it establishes per-request serial behavior, not concurrent-execution behavior, which the crash shows cannot currently be tested this way).

---

## Experiment 4 — 576-vs-128 no-load A/B comparison

### Research question
With load removed as a variable, how much does VisPruner actually improve a single request, using the model's own existing CUDA-event timing rather than a new profiler pass?

### Experimental setup
- **Script:** `scripts/run_noload_ab_comparison.py`
- Model loaded once; `model.visual_token_num` flipped between 576 and 128 (verified equivalent to a fresh load per §3)
- 10 fixed representative requests (same 10 `question_id`s, same order, for both configs) drawn from `dev.json`: `textvqa_34628, 34620, 34614, 34638, 34633, 34610, 34608, 34618, 34616, 34622`
- 10 warm-up requests per config, `important_ratio=0.5` fixed
- `max_new_tokens=64`, natural EOS stopping (matches the live server's policy)
- Timing reused directly from `llava_llama.py generate()`'s own CUDA-event instrumentation, redirected per-config via `LLAVA_TIMING_FILE`

### Files produced
| File | Contents |
|---|---|
| `scripts/run_noload_ab_comparison.py` | SCRIPT |
| `results/timing/ab_noload_comparison.json` | RAW DATA — full per-request detail, both configs |
| `results/timing/ab_noload_comparison_table.txt` | SUMMARY — the headline comparison table |
| `results/timing/ab_noload_timing_{576,128}.jsonl` | RAW DATA — the model's own CUDA-event timing lines, one per request |
| `results/timing/ab_noload_run_log.txt` | LOG |

### Results
| Metric | 576 | 128 | gain % | Source |
|---|---:|---:|---:|---|
| latency_median_ms **(bug-affected — see §9.1)** | 652.1 | 553.0 | 15.2% | `ab_noload_comparison_table.txt` |
| latency_mean_ms | 887.6 | 778.4 | 12.3% | same |
| latency_stdev_ms | 565.1 | 494.8 | 12.4% | same |
| multimodal_prep_mean_ms | 49.7 | 68.4 | **−37.8%** (128 slower) | same |
| prefill_lm_forward_mean_ms | 111.5 | 44.4 | **60.2%** | same |
| decode_lm_forward_mean_ms | 711.8 | 652.1 | **8.4%** | same |
| output_token_count_mean **(includes +1 synthetic BOS — see §9.3)** | 25.2 | 24.3 | 3.6% | same |
| gpu_util_pct_mean_sampled **(weak methodology — see §9.4)** | 95.9 | 98.5 | −2.7% | same |

### Interpretation
The 60.2%/8.4% split (prefill vs. decode) is the central mechanism of this entire investigation: pruning does exactly what it claims to prefill, and almost nothing to decode, and decode is roughly 6-16x larger than prefill in this workload, so the end-to-end number is dominated by the part pruning doesn't touch. The `multimodal_prep` result going the *wrong way* (128 costing more than 576) is explained mechanistically in Experiment 6 below — it isn't noise, it's a real, code-verifiable property of VisPruner's own diversity-selection loop.

### Limitations
See §9 in full — this experiment's headline latency numbers are affected by a confirmed timing bug (§9.1), and its ordering was not randomized within itself (fixed 576-then-128 sequence).

### Status
**VALID WITH CAVEAT** — see the dedicated correction in §9.1 and §10.

---

## Experiment 5 — 576-vs-128 load A/B comparison

### Research question
Does the ~13-15% isolated advantage survive under load, and does VisPruner raise the server's sustainable capacity?

### Experimental setup
- **Script:** `scripts/run_ab_load_sweep.py`, parameterized by `visual_token_num`
- **One-line change to `csnbs/server.py`** (Rithvik's file) was required to make this possible: `visual_token_num=int(os.environ.get("VISUAL_TOKEN_NUM", "576"))`, replacing a hardcoded `576`. Unset, this reproduces the prior behavior exactly. No other line of `csnbs/server.py`, and nothing in `csnbs/measure`, was touched.
- Same RPS levels, duration, and dataset as Experiment 3. **The 576-token leg was reused verbatim from Experiment 3 rather than re-run** (identical settings; only the env-var mechanism changed, which defaults to the same 576). The 128-token leg is a fresh sweep.
- Verified independently (by reading `csnbs/measure/src/loadgen.ts`) that request selection is deterministic (`payloads[sequence % payloads.length]`, dataset read in file order, no shuffle) — so "same request sequence" across the two legs is a **code-confirmed fact**, not an assumption.

### Files produced
| File | Contents |
|---|---|
| `scripts/run_ab_load_sweep.py` | SCRIPT |
| `results/timing/concurrent_load_summary_vtn128.txt` | SUMMARY — 128-token load sweep table |
| `results/timing/concurrent_load_metadata_vtn128.json` | METADATA |
| `results/timing/ab_load_gain_table.json` | DERIVED — per-RPS-level gain percentages, both directions |
| `results/timing/loadgen_monitor/vtn128_rps-{0.5,1.0,1.5,2.0,3.0}.csv` | RAW DATA — GPU/CPU monitoring |
| `results/loadgen/2026-09-01/{03-19-29Z_rps-3,03-20-10Z_rps-0.5,03-20-40Z_rps-2,03-21-12Z_rps-1,03-21-43Z_rps-1.5}_*/` | RAW DATA — Rithvik's harness output for the 128-token leg |
| `results/timing/ab_load_sweep_128_run_log.txt` | LOG |
| `csnbs/server.py` (modified, 1 line) | The env-var parameterization that made this experiment possible |

### Results

| RPS | 576 p50 | 128 p50 | p50 gain | 576 p95 | 128 p95 | p95 gain | 576 p99 | 128 p99 | p99 gain | throughput gain | n (576/128) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 861 | 652 | 24.3% | 1070 | 861 | 19.6% | 1070 | 861 | 19.6% | 0.0% | 5/5 |
| 1.0 | 603 | 492 | 18.4% | 818 | 767 | 6.1% | 1026 | 920 | 10.4% | 0.0% | 20/20 |
| 1.5 | 588 | 475 | 19.3% | 2332 | 2333 | **−0.1%** | 2389 | 2489 | **−4.2%** | 0.0% | 35/35 |
| 2.0 | 2291 | 1374 | 40.0% | 5877 | 5376 | 8.5% | 6877 | 8009 | **−16.5%** | 0.0% | 50/50 |
| 3.0 | 7138 | 7630 | **−6.9%** | 10914 | 10951 | −0.3% | 19599 | 13418 | 31.5% | **3.8%** | 80/80 |

*(All values `results/timing/ab_load_gain_table.json`; derived from the two summary files.)*

| Metric | 576 | 128 |
|---|---:|---:|
| First clear degradation (p95 jump) | rps ≈ 1.5 | rps ≈ 1.5 |
| Hard saturation (achieved throughput < offered) | rps = 3.0, achieves 2.166 | rps = 3.0, achieves 2.249 |
| GPU util at worst observed level | 87.1% | 65.8% |
| GPU memory (roughly flat across all levels) | ~15.2-15.4GB | ~15.1-15.2GB |

### Interpretation
The isolated per-request advantage (Experiment 4) does **not** stay constant under load — at the median it fluctuates (24→18→19→**40**→**−7%**) without a clean trend; at the tail (p95/p99) it degrades toward zero and **reverses sign** starting at rps≈1.5, exactly where both configurations show their first sign of queueing simultaneously. The **rps=2.0 p50 gain of 40.0%** is the single largest positive reading in the table, sitting between two much smaller neighbors (19.3% and −6.9%) — flagged here as more likely evidence of high run-to-run variance at that load level than a stable effect, pending repeat trials.

**Capacity finding (the central result of this experiment):** the throughput ceiling is **2.166 → 2.249 rps, a +3.8% increase** — an order of magnitude below the 56-60% prefill reduction and even below the ~13% (corrected) isolated latency gain. First-degradation onset is **identical** between configs. GPU memory is nearly flat across both configs and all load levels (~200-300MB difference, consistent with a KV cache that is a small fraction of the ~14GB weight footprint at batch=1). The server serializes requests (verified by reading `csnbs/server.py` — `_infer_model` calls `model.generate()` synchronously inside an `async def` with no `await`/executor handoff), so its throughput ceiling is set by `1 / mean_service_time`. Since decode dominates mean service time and pruning barely touches decode, cutting prefill compute in half barely moves the ceiling.

### Limitations
**The two legs of this sweep were not temporally interleaved.** The 576 data was collected roughly 50 minutes before the 128 data, in a separate task, with substantial other GPU activity (Experiment 4, four traces from Experiment 6) in between. This is the weakest form of ordering control (a block design, not the alternation the project's own rules call for) and means thermal drift or other machine-state changes are not ruled out as a partial confound on the ~3.8% capacity-gain figure specifically. See §9.2.

### Status
**VALID WITH CAVEAT.** The within-sweep comparisons (percentiles, saturation onset, queueing pattern) are sound because they come from the same, internally-consistent measurement method applied to both configs. The **capacity-gain magnitude (3.8%) is PROVISIONAL** pending a properly interleaved re-run.

---

## Experiment 6 — 576-vs-128 profiler traces: isolated and near-saturation

### Research question
What does the kernel-level trace show for each configuration, both with no load and under repeated (queued-like) execution — and specifically, where inside the pruning pipeline does the algorithmic reduction get diluted?

### Experimental setup
- **Script:** `scripts/profile_ab_trace.py <visual_token_num> <mode>`, run four times: `576 isolated`, `128 isolated`, `576 near_saturation`, `128 near_saturation`
- `important_ratio=0.5`, `max_new_tokens=64`, natural EOS stopping (matches Experiment 4/5's policy — **not** Experiment 2's forced-length policy)
- `isolated` = 1 request; `near_saturation` = 5 requests run **back-to-back sequentially in one process** (not genuine concurrency — see the crash finding in Experiment 3, which is why this design was chosen deliberately rather than by default)
- 10 warm-up requests before each profiled run

### Files produced
| File pattern | Contents |
|---|---|
| `scripts/profile_ab_trace.py` | SCRIPT |
| `results/timing/ab_trace_vtn{576,128}_{isolated,near_saturation}.json` | PROFILER TRACE (52-449MB each) |
| `results/timing/ab_trace_vtn{576,128}_{isolated,near_saturation}_settings.json` | METADATA |
| `results/timing/ab_trace_vtn{576,128}_{isolated,near_saturation}_stage_breakdown.txt` | SUMMARY — per-stage table |
| `results/timing/ab_trace_vtn{576,128}_{isolated,near_saturation}_summary_by_{cuda_time,cpu_time,count}.txt` | SUMMARY — full operator tables |

### Results — isolated, stage-level (same question both configs)

| Stage | 576 CUDA ms | 128 CUDA ms | Source |
|---|---:|---:|---|
| vision_tower | 43.9 | 36.6 | `ab_trace_vtn{576,128}_isolated_stage_breakdown.txt` |
| mm_projector | 1.1 | 0.3 | same |
| multimodal_prep (total, incl. above) | 48.3 | 43.7 | same |
| prefill | **117.4** | **51.5** | same |
| kv_cache_append (384 calls) | 17.2 | 5.6 | same |
| decode total (12 steps) | 372.6 (31.0/step) | 349.5 (29.1/step) | same |

### Results — the diversity-selection loop, derived from raw trace JSON

The while-loop at `llava_arch.py:160-176` shrinks `residual_indices` by at most 8 per pass, exiting immediately when `important_token_num + diverse_token_num == N` (true exactly when `visual_token_num == 576`). At `visual_token_num=128`, `important_token_num=64`, `diverse_token_num=64`, `residual_indices` starts with 512 entries — predicting **⌈(512−64)/8⌉ = 56 iterations**. Parsing the already-saved `ab_trace_vtn{576,128}_isolated.json` trace files for operator counts inside the `multimodal_prep` span (this is a re-analysis of Experiment 6's own raw artifact, performed during the subsequent engineering-analysis phase — not a new profiling run):

| Operator | 576 (measured) | 128 (measured) |
|---|---:|---:|
| `aten::argsort` | 1 | **57** |
| `aten::max` | 0 | **56** |
| `aten::matmul` | 0 | **56** |
| `aten::bmm` | 48 (vision-tower attention) | 104 |
| `aten::cat` | 7 | **63** |

56-57 measured iterations against a 56-iteration prediction from reading the code. This directly explains Experiment 4's counter-intuitive `multimodal_prep` result (128 costing *more* than 576): the "unpruned baseline" configuration's diversity loop provably does zero merge work, while the pruned configuration's does ~56 iterations of real work.

### Results — near-saturation (5 back-to-back requests, slot 0 shown, same question as isolated)

| Stage | 576 isolated | 576 near-sat | 128 isolated | 128 near-sat |
|---|---:|---:|---:|---:|
| decode total (12 steps) | 372.6ms (31.0/step) | **462.4ms (38.5/step)** | 349.5ms (29.1/step) | **426.0ms (35.5/step)** |

Decode-step CUDA time is **~22-24% higher** in the back-to-back trace than the isolated trace, for both configurations, by a similar relative amount.

### Interpretation
This experiment answers three of the questions this investigation was designed to ask directly, at the code level rather than by inference:
1. **Does expensive work happen on tokens later discarded?** Yes — `mm_projector` runs on all 576 tokens before the mask/gather is applied (`llava_arch.py:185` runs before `:271`), confirmed both by reading the code and by its near-identical, config-independent cost in the trace.
2. **Does token selection itself add meaningful overhead?** Yes, substantially, and specifically at low `visual_token_num` — the diversity loop is the single clearest "pruning tax" found in this investigation.
3. **Does a shorter visual context reduce downstream work efficiently?** Partially. `kv_cache_append` tracks token count reasonably well (17.2→5.6ms, roughly proportional). Prefill does not scale as cleanly: the sequence shrinks by roughly 4.2x (≈587→≈139 tokens including the ~11 text tokens) but prefill time only drops 2.3x (117.4→51.5ms) — a real, measured sub-linear-scaling gap, consistent with a meaningful fixed per-call overhead component that doesn't shrink with sequence length.

The isolated-vs-near-saturation decode gap (~22-24%, symmetric across both configs) is flagged as a genuine, measured effect whose *cause* is not established — its near-identical magnitude across two independently-run configurations argues against a token-count-specific mechanism and toward a shared confound (thermal drift over a long GPU session, or allocator-state effects from repeated `generate()` calls), but this is **INTERPRETATION**, not confirmed.

**One further honest discrepancy, not resolved:** this isolated-trace pair shows `multimodal_prep` at 576 (48.3ms) *higher* than 128 (43.7ms) — the opposite direction from Experiment 4's 10-request average (576=49.7ms lower than 128=68.4ms). The N=10 average is the more trustworthy figure for that specific sub-claim (a single-question trace carries more per-question variance), but the direction genuinely differs between the two measurements and both are reported as measured, not reconciled by selectively citing one.

### Limitations
"Near-saturation" is back-to-back sequential execution in one process, not genuine concurrent GPU execution (which the Experiment 3 crash shows cannot currently be tested safely). It approximates the *per-request* behavior of a queued server, not GPU-level contention between simultaneously-active requests.

### Status
**VALID** for the isolated-trace stage breakdown and the diversity-loop mechanism (both directly reproduced from code + raw trace data). **DIAGNOSTIC ONLY** for the near-saturation decode-time increase (real, measured, cause not established).

---

## Experiment 7 — Post-hoc correctness audit and corrected baseline

### Research question
Do these experiments, taken together, actually contain any bugs — and if so, how much do they change the headline numbers?

### What this was
Not a new profiling run. A line-by-line review of `run_noload_ab_comparison.py` and cross-checks of the other scripts' output against their own claims, prompted by an explicit request to check for mistakes rather than confirm the existing conclusions.

### Findings
See §9 for the full, itemized list. In summary: one real measurement bug was found and quantified (§9.1), one methodology gap was surfaced and its consequence stated plainly (§9.2), and one long-standing unexplained oddity from Experiment 2 was finally root-caused (§9.3).

### The correction
Confirmed directly: `run_one()` in `run_noload_ab_comparison.py` calls `build_request()` (image load + CLIP preprocessing + CUDA transfer) **before** starting its timer, so its "latency" measures `generate()` only, not a complete request. The excluded cost was measured directly, post-hoc, on the same pipeline: **86.7ms average** (10-sample measurement, script re-run for this purpose only — not a re-run of the profiling experiment itself, a standalone timing check).

**DERIVED ESTIMATE** — not a new measurement, simple arithmetic applied to already-measured numbers:

```
raw (measured, generate()-only):
    576 = 652.1 ms
    128 = 553.0 ms

estimated complete request (derived: raw + 86.7ms, applied equally):
    576 ≈ 652.1 + 86.7 = 738.8 ms
    128 ≈ 553.0 + 86.7 = 639.7 ms

corrected gain = (738.8 − 639.7) / 738.8 × 100 = 13.4%   (down from the raw 15.2%)
absolute savings = 99.1 ms, unchanged (adding the same constant to both sides
                                        doesn't change the gap)
```

**Why we did not simply re-run the experiment properly instead:** the correction is small, well-understood, and doesn't change any qualitative conclusion (decode still dominates, the 60%→13% dilution story is unaffected in kind). Re-running would cost GPU time to fix a ~2-percentage-point estimate when a clean, from-scratch end-to-end benchmark is going to be needed anyway as the *first validation step* of the engineering phase (per the project's own validation rule — before/after benchmarks are required for every future optimization). Spending that GPU time now, on a number that will be re-measured properly regardless, was judged not worth it.

### Status
**VALID** as a correctness finding. The corrected 13.4% figure is a **DERIVED ESTIMATE**, explicitly not a new measurement, and should be replaced by a clean instrumented benchmark (timer started before image load) the first time this baseline needs to be re-confirmed precisely.

---

## 9. Experimental Limitations and Corrections

| # | Issue | Affected experiment(s) | Affected result | Severity | Changes relative conclusions? | Correction | Future clean benchmark needed? |
|---|---|---|---|---|---|---|---|
| **9.1** | No-load timer starts *after* image preprocessing | Experiment 4 | `latency_median_ms`, all latency-based gain % in Exp. 4 | **Moderate** | No — direction and rough magnitude of the isolated gain are unchanged; the *percentage* was overstated | DERIVED correction applied in Experiment 7: 15.2% → **13.4%** | Yes — instrumented re-run with the timer moved before `build_request()` |
| **9.2** | 576 and 128 load sweeps (Experiment 5) were not temporally interleaved — collected ~50 minutes apart with substantial intervening GPU activity | Experiment 5 | The 3.8% capacity-gain figure specifically; the whole side-by-side load table to a lesser degree | **Moderate** | No — within-sweep patterns (saturation onset, queueing shape) are internally consistent regardless; but the *precise* capacity-gain number carries an uncontrolled confound | None applied; figure explicitly marked **PROVISIONAL** throughout this report | Yes — a properly alternated/interleaved re-run |
| **9.3** | `output_ids.shape[1]` includes a synthetic leading BOS token (HF's `generate()` seeds `input_ids` with a placeholder when only `inputs_embeds` is given) that was never produced by a forward pass | Experiments 2, 4 (any place "output token count" is reported) | Absolute token-count figures are off by exactly +1 | **Low** | No — it's a constant offset applied identically to every measurement in both configs, so every *relative* (576-vs-128) comparison is unaffected | Root-caused via direct reproduction (`output_ids[0][0] == tokenizer.bos_token_id == 1`); no data correction needed, only the explanation was missing until this audit | No |
| **9.4** | Experiment 4's `gpu_util_pct_mean_sampled` is a single un-synchronized `nvidia-smi` call taken immediately *before* each request, not sampled continuously *during* it | Experiment 4 | `gpu_util_pct_mean_sampled: 95.9 / 98.5` | **Low** | No — these numbers were never load-bearing for any conclusion | None; flagged as lower-confidence than the load-sweep's proper background-thread monitor (Experiments 3/5) | Only if this specific metric is needed precisely |
| **9.5** | Low sample counts at low RPS — e.g. n=5 at rps=0.5 — mean p95 and p99 are mathematically forced to collapse to the same value (the max of 5 samples) | Experiment 5 | p95/p99 columns at rps=0.5, and to a lesser extent 1.0/1.5 | **Low-moderate** | No conclusion rests solely on these cells | None; flagged per-cell via sample counts in Experiment 5's results table | Yes, if precise low-RPS tail behavior is ever needed — requires a much longer run |
| **9.6** | "Near-saturation" profiler traces (Experiments 3, 6) use back-to-back sequential requests in one process, not genuine concurrent GPU execution | Experiments 3, 6 | Any claim about "concurrent" kernel-level behavior | **Moderate** | No conclusion in this report claims genuine GPU-level concurrency was observed — this was a deliberate design choice, stated as such in every relevant section | None needed; this is by design given 9.7 | A genuinely concurrent-safe profiling method would require first fixing 9.7 |
| **9.7** | A prior attempt at genuine multi-thread concurrent `generate()` calls against the shared model instance **crashed** with a CUDA device-side assert | Experiment 3 | Establishes that concurrency cannot currently be tested this way — this is itself a finding, not a gap to route around | **High** (as an engineering finding, not a measurement error) | This crash is load-bearing evidence for why the server serializes requests and why real batching (not naive threading) is the correct next step | None — preserved as-is in `concurrent_profile_run_log.txt` | A fix (thread-safety audit / real tensor-level batching) is required before this can be re-attempted |
| **9.8** | Natural EOS stopping means different requests (and different configs on the same question) generate slightly different numbers of tokens | Experiments 4, 6 | Any decode-time comparison implicitly assumes similar output length; Experiment 4's 128-config generated on average 0.9 fewer tokens than 576's | **Low-moderate** | Partially — a portion of the measured 8.4% decode-time gain in Experiment 4 is attributable to 128 generating slightly fewer tokens (different greedy trajectory), not purely a per-token speed difference; a rough per-token normalization suggests the *per-token* decode gain is closer to ~5% than the raw 8.4% | Not corrected in the canonical tables (both are reported as measured); this note is the correction | A token-count-matched decode comparison (as in Experiment 6's isolated trace, where both configs happened to generate exactly 14 tokens) is the cleaner comparison to cite for decode-specifically claims |
| **9.9** | AMAY_SPEED_PLAN.md's own prose states Experiment 1's headline percentage as 40.7%; the canonical artifact (`prep_summary.txt`) says 43.1% | Experiment 1 | The percentage cited in the planning document | **Low** | No — both readings support the same qualitative conclusion (the region is a meaningful, launch-overhead-flavored chunk of prefill) | This report uses the artifact's number (43.1%) as canonical | No — source of the discrepancy not investigated further, not load-bearing enough to justify it |

---

## 10. Canonical Results

The load-bearing numbers of this investigation, each traceable to a specific file.

| Metric | Value | Experiment | Source file | Type | Confidence | Caveat |
|---|---|---|---|---|---|---|
| Prefill reduction, 576→128 (isolated trace) | 117.4ms → 51.5ms (**56%**) | 6 | `ab_trace_vtn{576,128}_isolated_stage_breakdown.txt` | MEASURED | High | Single-question trace |
| Prefill reduction, 576→128 (10-request average) | 111.5ms → 44.4ms (**60.2%**) | 4 | `ab_noload_comparison_table.txt` | MEASURED | High | Not affected by the §9.1 bug |
| Decode reduction, 576→128 (10-request average) | 711.8ms → 652.1ms (**8.4%**) | 4 | `ab_noload_comparison_table.txt` | MEASURED | High | Partially confounded by output-length differences, §9.8 |
| Decode reduction, 576→128 (isolated, token-count-matched, 14 tokens both) | 372.6ms → 349.5ms (**6.2%**) | 6 | `ab_trace_vtn{576,128}_isolated_stage_breakdown.txt` | MEASURED | High | Cleaner comparison than the above, per §9.8 |
| No-load latency gain, raw | 652.1ms → 553.0ms (**15.2%**) | 4 | `ab_noload_comparison_table.txt` | MEASURED | Medium — excludes preprocessing (§9.1) | Superseded by the corrected estimate below |
| Excluded image-preprocessing cost | **86.7ms average** | 7 | Not a saved artifact — computed post-hoc during the Experiment 7 audit; see Experiment 7 for the exact reproduction method | MEASURED | Medium | Standalone 10-sample check, not part of the original profiling run |
| No-load latency gain, corrected | 738.8ms → 639.7ms (**13.4%**) | 7 | (arithmetic; not a file) | **DERIVED ESTIMATE** | Medium | Needs a clean instrumented re-run to firm up |
| Diversity-loop iterations at T=128 | **56-57 measured** (56 predicted from code) | 6 | Derived from `ab_trace_vtn128_isolated.json` (raw trace) | MEASURED + code-derived prediction | High | — |
| Total CUDA kernel launches, one 24-token request | **34,722** | 2 | `full_request_summary_by_cuda_time.txt` | MEASURED | High | Forced-length policy, not natural stopping |
| `kv_cache_legacy_rebuild` calls per generate() (23 decode steps) | **768**, ~0ms GPU, 3.0ms CPU | 2 | `full_request_stage_breakdown.txt` | MEASURED | High | — |
| `kv_cache_append` calls, cost | **736 calls**, 11.8ms CUDA / 38.4ms CPU | 2 | `full_request_stage_breakdown.txt` | MEASURED | High | — |
| Queueing/tail degradation onset | **rps ≈ 1.5**, both configs | 3, 5 | `concurrent_load_summary.txt`, `concurrent_load_summary_vtn128.txt` | MEASURED | High | — |
| Hard saturation | rps = 3.0; throughput plateaus at 2.166 (576) / 2.249 (128) | 3, 5 | same | MEASURED | High | — |
| Serving-capacity gain | **+3.8%** | 5 | `ab_load_gain_table.json` | MEASURED, but **PROVISIONAL** | **Low-Medium** — see §9.2 | Not temporally interleaved |
| GPU memory footprint, both configs | ~15.1-15.4GB, nearly flat | 3, 5 | `concurrent_load_summary*.txt` | MEASURED | High | KV cache is a small fraction of total footprint at batch=1 |
| Server concurrency model | Strictly serial (verified by code reading) | 3 | `csnbs/server.py` `_infer_model` | MEASURED (code inspection) | High | — |
| Thread-safety crash under genuine concurrency | Confirmed, `indexSelectSmallIndex` assertion | 3 | `concurrent_profile_run_log.txt` | MEASURED | High | — |

---

## 11. Combined Findings

```
576 visual tokens (baseline)
        │
        ▼
128 visual tokens (VisPruner)
        │
        ▼
LARGE reduction in prefill compute            ~56-60%   (MEASURED, Experiments 4 & 6)
        │
        ▼
decode remains ~6-16x larger than prefill,
and pruning barely reduces it                  ~6-8%    (MEASURED, Experiments 4 & 6)
        │
        ▼
therefore much smaller complete-request
latency improvement                            ~13%     (DERIVED ESTIMATE, Experiment 7,
                                                           correcting Experiment 4's 15.2%)
        │
        ▼
server processes exactly one request at a
time (verified from source code)                        (MEASURED — code inspection, Exp. 3)
        │
        ▼
therefore very little of even the 13% shows
up as additional serving capacity              ~4%      (MEASURED but PROVISIONAL, Experiment 5)
```

**1. What does VisPruner clearly improve?** Prefill compute, unconditionally and substantially (56-60%, consistent across two independent measurement methods — Experiment 4's 10-request CUDA-event average and Experiment 6's single-question trace). `kv_cache_append` cost, roughly proportionally (Experiment 6).

**2. What does it barely affect?** Decode (6-8%, the dominant cost in a short-answer request). Vision-tower cost (necessarily — CLIP always processes all 576 patches regardless of downstream pruning). `mm_projector` cost (currently runs on all 576 tokens before the mask is applied — a known, not-yet-fixed inefficiency, per AMAY_SPEED_PLAN.md and confirmed here at the code level).

**3. Why is end-to-end improvement much smaller than prefill improvement?** Decode dominates total request time for the short, factual answers in this dataset, and pruning's mechanism (fewer visual tokens) has no leverage over decode's dominant cost (reading the full weight matrix once per output token).

**4. Why does the serving improvement become even smaller still?** The server is architecturally serial — one request in flight at a time, verified by reading `csnbs/server.py`, not inferred. Its throughput ceiling is `1/mean_service_time`, and mean service time is decode-dominated. A faster prefill barely moves a ceiling set mostly by an unchanged decode cost.

**5. Which effects are VisPruner-specific?** The prefill reduction. The `kv_cache_append` reduction. The diversity-loop's *own* cost (paradoxically higher at low `visual_token_num` — Experiment 6's most surprising and best-evidenced finding).

**6. Which effects are properties of the current inference/serving implementation, not of VisPruner?** Decode's weight-read-dominated cost (a property of autoregressive decoding generically). The server's one-request-at-a-time architecture. The `DynamicCache.from_legacy_cache()` per-step rebuild (present, identically, at both token counts). The prefill's sub-linear scaling with sequence length (~4.2x fewer tokens → only ~2.3x less time — some other, non-pruning-specific overhead is capping how much of the token-count reduction converts into time saved).

---

## 12. Engineering Implications

This section connects measured findings to open engineering questions. It does not choose an implementation order or claim any fix will work — that is deliberately out of scope here (see `AMAY_SPEED_PLAN.md` for the roadmap that consumes these findings).

### Triton / custom kernels

**MEASURED:** The diversity-selection loop runs ~56-57 iterations (≈330+ extra kernel launches: `argsort`×57, `max`×56, `matmul`×56, `cat`×63) at `visual_token_num=128`, and effectively zero at `visual_token_num=576` (Experiment 6).

**IMPLICATION:** VisPruner's own token-selection mechanism gets *more* expensive, not less, the more aggressively it prunes — this is the one place in the pipeline where the pruning-specific cost currently works against the pruning-specific benefit.

**ENGINEERING QUESTION:** Can the iterative, ≤8-per-pass merge loop be replaced with a single fused/vectorized (Triton or otherwise) operation that does the same duplicate-removal in one launch, without changing the selection semantics (raster-order preservation, per the project's own correctness rule)?

**MEASURED:** `mm_projector` runs on all 576 tokens regardless of `visual_token_num` (code-confirmed, `llava_arch.py:185` precedes the mask/gather at `:271`); measured cost is small in absolute terms at this model's projector size (~0.3-1.1ms).

**IMPLICATION:** Real FLOPs are spent on tokens later discarded, though the current absolute cost of this specific inefficiency is small.

**ENGINEERING QUESTION:** Is reordering to project-after-gather worth doing given its small measured absolute impact, and does the `spatial+unpad` anyres branch's existing constraint (projection must precede the `image_newline` concat on that path, per AMAY_SPEED_PLAN.md) make a clean global reorder impractical?

### KV-cache optimization

**MEASURED:** `DynamicCache.update()` is called twice per layer per decode step — once as a near-zero-cost list rebuild (`kv_cache_legacy_rebuild`, 768 calls, ~0ms GPU) and once as the real incremental append (`kv_cache_append`, 736 calls, 11.8ms CUDA / 38.4ms CPU) — identically at both `visual_token_num` settings (Experiment 2).

**IMPLICATION:** `past_key_values` crosses `generate()`'s step boundary as a legacy tuple rather than a persisted `Cache` object, forcing a full cache-object reconstruction every step. This cost is config-independent — it is not a pruning inefficiency, it is a property of how this codebase's custom `generate()`/`forward()` interact with `transformers`' cache API.

**ENGINEERING QUESTION:** Would retaining a persisted `Cache` object across steps (instead of round-tripping through a legacy tuple) remove this overhead, and — separately — is doing so a *necessary precondition* for CUDA Graphs (which need stable tensor addresses across replay, which a per-step-rebuilt cache object cannot provide)?

### CUDA Graphs

**MEASURED:** 34,722 CUDA kernel launches for a single 24-token request (Experiment 2); decode-step CPU time exceeds CUDA time at every step in that trace (e.g. step 1: 38.9ms CPU vs 38.0ms CUDA); this pattern is present at both `visual_token_num` settings in Experiment 6's traces, with similar relative magnitude.

**IMPLICATION:** Decode contains a highly repetitive, per-layer, per-step GPU dispatch pattern whose CPU-side launch cost is comparable to or exceeds its GPU execution cost — a textbook signature for graph-capture-and-replay to address.

**ENGINEERING QUESTION:** Can CUDA Graph capture reduce this launch overhead, and does it require the KV-cache persisted-object work above as a prerequisite (fixed tensor addresses across the replayed steps)?

### Batching / serving

**MEASURED:** Server processes exactly one request at a time (code-confirmed — `_infer_model` calls `model.generate()` synchronously inside an `async def` with no yield point); throughput ceiling improved only 2.166→2.249 rps (+3.8%, provisional) despite a 56-60% per-request compute reduction; genuine multi-thread concurrent execution against the shared model instance crashed with a CUDA device-side assert (Experiment 3).

**IMPLICATION:** The serving architecture, not model compute, is the binding constraint on capacity at every load level tested (GPU utilization never exceeded 87%). Naive concurrency (more threads/coroutines calling `generate()`) is not just insufficient but currently unsafe.

**ENGINEERING QUESTION:** What would it take to make the shared model instance's pruning path safe for concurrent access (or to isolate per-request state), as a prerequisite for any real tensor-level batching — and separately, would real batching (stacking multiple requests into one forward pass) actually convert the existing per-request compute savings into throughput, given decode's dominance?

### Additional category: prefill's sub-linear scaling

**MEASURED:** Sequence length drops ~4.2x (≈587→≈139 tokens) between configs, but prefill time drops only ~2.3x (117.4→51.5ms) (Experiment 6).

**IMPLICATION:** Some cost in the prefill path does not scale down with sequence length — plausibly per-call dispatch/kernel-launch overhead, similar in character to the decode-side finding above but not yet isolated to a specific operation.

**ENGINEERING QUESTION:** What specifically is the non-scaling component of prefill cost, and is it addressable by the same class of fix as the decode-side launch-overhead findings, or is it a separate mechanism?

---

## 13. Conclusion

VisPruner's algorithmic compute reduction is real, large, and does not degrade under any condition tested here — the 56-60% prefill-compute cut measures the same whether the system is idle or under heavy load. What changes is how much of that reduction survives translation into outcomes a user or operator would actually notice. Two measured, code-verified mechanisms explain the gap at each step: decode (unaffected by visual-token pruning) dominates total request time, diluting a 56-60% algorithmic win into a ~13% (corrected) end-to-end latency win; and the server's strictly-serial architecture (verified from source) caps capacity at `1/mean_service_time`, converting that ~13% into a provisional ~4% capacity gain. Neither dilution is evidence that VisPruner "doesn't work" — both are properties of *where in the system* the algorithmic saving lands, not of the algorithm itself. This is the baseline gap the next engineering phase exists to close, and this report's job was to establish that gap precisely enough, and honestly enough about its own uncertainties, to be worth building against.

---

## Appendix A — Experiment Artifact Index

```
GPU_Profiling/
│
├── AMAY_VISPRUNER_PROFILING_REPORT.md      ← this document (canonical research record)
├── AMAY_SPEED_PLAN.md                       ← engineering roadmap (separate document, see §14)
├── AMAY_TIMING_NOTES.md                     ← study notes on timing methodology, referenced throughout §3
│
├── prompts/
│   └── GENERATE_VISPRUNER_PROFILING_REPORT.md   ← reproducibility prompt for this report
│
├── csnbs/
│   └── server.py                            ← Rithvik's file; ONE line made configurable (VISUAL_TOKEN_NUM
│                                               env var) to enable Experiment 5 — see Experiment 5's setup
│
├── scripts/
│   ├── profile_multimodal_prep.py           ← SCRIPT, Experiment 1 (pre-existing)
│   ├── profile_full_request.py              ← SCRIPT, Experiment 2
│   ├── run_concurrent_load_test.py          ← SCRIPT, Experiment 3 (load sweep orchestration)
│   ├── profile_concurrent_load.py           ← SCRIPT, Experiment 3 (trace capture; docstring documents
│   │                                           the thread-safety crash and the sequential-fallback design)
│   ├── run_noload_ab_comparison.py          ← SCRIPT, Experiment 4
│   ├── run_ab_load_sweep.py                 ← SCRIPT, Experiment 5 (generalizes Exp. 3's sweep by config)
│   └── profile_ab_trace.py                  ← SCRIPT, Experiment 6 (isolated + near-saturation traces)
│
└── results/
    ├── timing/
    │   ├── prep_trace.json / prep_summary.txt                    ← Experiment 1: TRACE / SUMMARY
    │   ├── full_request_trace.json                                ← Experiment 2: TRACE
    │   ├── full_request_settings.json                              ← Experiment 2: METADATA
    │   ├── full_request_stage_breakdown.txt                        ← Experiment 2: SUMMARY (open this first)
    │   ├── full_request_summary_by_{cuda_time,count,cpu_time}.txt  ← Experiment 2: SUMMARY (full op tables)
    │   ├── full_request_run_log.txt                                ← Experiment 2: LOG
    │   ├── concurrent_load_summary.txt                             ← Experiment 3: SUMMARY (576 load sweep)
    │   ├── concurrent_load_metadata.json                           ← Experiment 3: METADATA
    │   ├── concurrent_trace.json                                   ← Experiment 3: TRACE (near-sat, 576)
    │   ├── concurrent_trace_settings.json                          ← Experiment 3: METADATA
    │   ├── concurrent_request_timeline.json                        ← Experiment 3: RAW DATA
    │   ├── concurrent_summary_by_{cuda_time,count}.txt              ← Experiment 3: SUMMARY
    │   ├── concurrent_profile_run_log.txt                          ← Experiment 3: LOG — contains the crash
    │   ├── concurrent_load_run_log.txt                              ← Experiment 3: LOG
    │   ├── loadgen_calibration.json                                ← Experiment 3: RAW DATA (pre-sweep calibration)
    │   ├── loadgen_monitor/rps-*.csv                                ← Experiment 3: RAW DATA (GPU/CPU, 576)
    │   ├── ab_noload_comparison.json                                ← Experiment 4: RAW DATA
    │   ├── ab_noload_comparison_table.txt                           ← Experiment 4: SUMMARY (open this first)
    │   ├── ab_noload_timing_{576,128}.jsonl                         ← Experiment 4: RAW DATA
    │   ├── ab_noload_run_log.txt                                    ← Experiment 4: LOG
    │   ├── concurrent_load_summary_vtn128.txt                       ← Experiment 5: SUMMARY (128 load sweep)
    │   ├── concurrent_load_metadata_vtn128.json                     ← Experiment 5: METADATA
    │   ├── ab_load_gain_table.json                                  ← Experiment 5: SUMMARY (open this first)
    │   ├── loadgen_monitor/vtn128_rps-*.csv                         ← Experiment 5: RAW DATA (GPU/CPU, 128)
    │   ├── ab_load_sweep_128_run_log.txt                            ← Experiment 5: LOG
    │   ├── ab_trace_vtn{576,128}_{isolated,near_saturation}.json    ← Experiment 6: TRACE (4 files)
    │   ├── ab_trace_vtn{576,128}_{isolated,near_saturation}_settings.json        ← Experiment 6: METADATA
    │   ├── ab_trace_vtn{576,128}_{isolated,near_saturation}_stage_breakdown.txt  ← Experiment 6: SUMMARY (open first)
    │   ├── ab_trace_vtn{576,128}_{isolated,near_saturation}_summary_by_*.txt     ← Experiment 6: SUMMARY (full op tables)
    │   └── llava_llama_timing.json                                  ← shared, append-only log written by every
    │                                                                    generate() call across all experiments
    │                                                                    (not a per-experiment artifact)
    │
    └── loadgen/
        ├── 2026-08-29/*/                     ← pre-dates this investigation (earlier RPS-sweep smoke tests)
        ├── 2026-09-01/02-3{5,6,7}-*_rps-*/   ← RAW DATA, Experiment 3 (576-token load sweep)
        └── 2026-09-01/03-{19,20,21}-*_rps-*/ ← RAW DATA, Experiment 5 (128-token load sweep)
            each directory contains: run.json (SUMMARY — percentiles, throughput, git commit, GPU model)
                                      requests.jsonl (RAW DATA — one line per individual request)
```

**File-type definitions used throughout this report:**
- **SCRIPT** — the Python/shell code that produced an experiment. Re-runnable (though this report's instructions were not to re-run anything).
- **RAW DATA** — unprocessed per-event or per-request output (e.g. `requests.jsonl`, `.csv` monitoring logs). Large, not meant to be read directly, but the ground truth behind every summary.
- **PROFILER TRACE** — a Chrome/Perfetto-format `.json` file, the complete kernel-level timeline. Open in `ui.perfetto.dev`, not as text. Files range 45MB-449MB.
- **SUMMARY** — a human-readable reduction of raw data or a trace into a table. **Start here** when investigating any experiment.
- **METADATA** — the exact settings (GPU, versions, config values) a given run used. Needed to interpret a summary correctly.
- **LOG** — raw stdout/stderr of a run. Mostly archival; useful for debugging or, in one case (`concurrent_profile_run_log.txt`), as the primary evidence for a specific finding (the crash).

**Worth opening manually:** every `*_stage_breakdown.txt`, `*_comparison_table.txt`, `*_summary.txt` (non-op-table), `ab_load_gain_table.json`, and `concurrent_load_summary*.txt`. **Mainly archival:** the `.json` trace files (open only in Perfetto when investigating a specific kernel-level question), `*_run_log.txt` files, and the `loadgen/*/requests.jsonl` per-request detail (aggregate `run.json` is normally sufficient).

---

## Appendix B — How to Regenerate or Update This Report

1. **Repository root:** `/workspace/GPU_Profiling`. All paths in this report are relative to that root unless given as absolute paths.

2. **Files/directories to inspect before writing or updating this report:** `AMAY_SPEED_PLAN.md`, `AMAY_TIMING_NOTES.md`, `CLAUDE.md` (for ownership rules and project context), every `scripts/*.py` file whose name starts with `profile_`, `run_ab_`, `run_concurrent_`, or `run_noload_`, and everything under `results/timing/` and the relevant subdirectories of `results/loadgen/`.

3. **Files that must never be modified as part of this documentation work:** any `results/timing/*` or `results/loadgen/*` raw artifact (traces, summaries, metadata, logs) — these are the evidence base; editing them breaks traceability. `csnbs/server.py` beyond the one already-made, already-documented line. Anything inside `sribhav/`, `kingcorsair/`, `csnbs/` (other than the one documented server.py line, and only with the same care), or `ScienceQA/` — these belong to other contributors.

4. **How to distinguish "my" files from other contributors':** per `CLAUDE.md`'s ownership table, Amay owns the serving path, real token removal, batching, KV cache, Triton kernels, and Playground UI. Scripts in `scripts/` whose name matches a profiling/timing/A-B pattern (see #2) and everything under `results/timing/` from this investigation are Amay's. `csnbs/` is Rithvik's (load generation, timing harness, server). `sribhav/` is Sribhav's (eval/scoring). When in doubt, check `git log` / file modification dates against the investigation's timeframe, or check `CLAUDE.md`'s "Team and ownership" table directly.

5. **How to preserve raw measurements when updating:** never overwrite an existing `results/timing/*` file with new data from a different run — if you re-run an experiment, write to a new, distinctly-named file (e.g. add a date suffix or increment a version) and update this report's citations, rather than silently replacing evidence a past conclusion depended on.

6. **How to add a new experiment:** follow the per-experiment template used in Experiments 1-7 (Research question / Experimental setup / Files produced / Results / Interpretation / Limitations / Status). Add it as a new numbered experiment section, in whatever position in the chronological/logical sequence makes sense — don't force it into the existing 1-7 numbering if it belongs earlier or in between. Update Appendix A's tree and this report's Table of Contents implicitly (the section headers).

7. **How to update old conclusions when later evidence supersedes them:** do not delete the superseded section. Change its **Status** to `SUPERSEDED`, add a one-line pointer to the experiment that superseded it, and add the new evidence as a new experiment section. This report's own §9/Experiment 7 (the bug audit) is the template for this — the original Experiment 4 numbers were not deleted when the bug was found, they were kept, labeled, and corrected in a new section.

8. **How to keep MEASURED / DERIVED / INTERPRETATION distinct:** every numerical claim in a Results table should be traceable to a file with no arithmetic beyond simple aggregation (MEASURED), or explicitly labeled with the formula used if arithmetic was applied (DERIVED), and every claim about *why* a number looks the way it does belongs in an Interpretation subsection, not a Results table, and should be phrased with appropriate hedging (see Experiment 6's "flagged as INTERPRETATION, not confirmed" pattern) when the mechanism isn't independently verified.

9. **How to update the evidence index (Appendix A):** re-list any new files under the correct experiment, using the same SCRIPT/RAW DATA/PROFILER TRACE/SUMMARY/METADATA/LOG categorization. Group repetitive directories (like `loadgen/*/`) rather than listing every file individually, as done above.

10. **How to avoid rerunning GPU experiments unless explicitly requested:** this report itself was produced entirely by reading existing files and, in one case (Experiment 7's 86.7ms preprocessing measurement), running a small, explicitly-scoped standalone timing check that was clearly distinguished from "re-running the profiling experiment." Default to reading before running. If a claim cannot be supported by an existing artifact and re-measuring it would require substantial GPU time, write **"Not currently preserved in a canonical repository artifact"** (as done in Experiment 2's KV-cache stack-trace note) rather than fabricating support or silently re-running a full experiment.

### Claude Prompt Used to Generate This Report

```
I want you to consolidate the entire VisPruner/GPU profiling phase into one canonical research-style technical report.

Do NOT rerun GPU experiments.
Do NOT implement optimizations.
Do NOT delete or modify raw experiment evidence.

Create:

/workspace/GPU_Profiling/AMAY_VISPRUNER_PROFILING_REPORT.md

This should become the canonical written record of the profiling/research phase.

The goal is that months from now I should be able to open this one document and understand:

- what question we were investigating
- what experiments were conducted
- how each experiment was conducted
- where its scripts/raw outputs/traces are stored
- what each experiment found
- what methodological problems were discovered
- what numbers are trustworthy
- what conclusions survived those problems
- why those findings motivate the next engineering phase

It should read like a clear research/engineering paper rather than a collection of notes.

==================================================
1. INSPECT THE EXISTING EVIDENCE FIRST
==================================================

Before writing the report, inspect all relevant project-owned files and experiment outputs.

At minimum inspect:

- AMAY_SPEED_PLAN.md
- AMAY_TIMING_NOTES.md

Relevant scripts, including where present:

- scripts/profile_multimodal_prep.py
- scripts/profile_full_request.py
- scripts/run_concurrent_load_test.py
- scripts/profile_concurrent_load.py
- scripts/run_noload_ab_comparison.py
- scripts/run_ab_load_sweep.py
- scripts/profile_ab_trace.py
- any later scripts that clearly belong to this same profiling investigation

Relevant outputs:

- results/timing/*
- relevant results/loadgen/*
- profiler summaries
- stage breakdowns
- settings/metadata files
- raw A/B results
- traces
- monitoring outputs
- logs

Do not assume the old planning documents are completely correct.

Where possible, derive claims from the experiment outputs themselves.

Do not include unrelated work belonging to other contributors except where their existing harness/tool was used as part of these experiments.

==================================================
2. REPORT FORMAT
==================================================

Write the report using the following research-style structure.

# Title

Use a descriptive technical title centered on VisPruner, LLaVA, GPU profiling, and the gap between algorithmic/token reduction and real serving performance.

# Abstract

In simple English summarize:

- the system investigated
- the core question
- the main experiments
- the main result
- the main engineering implication

Keep this concise.

# 1. Motivation and Research Question

Explain why this investigation was performed.

The central question is approximately:

"Does reducing LLaVA visual tokens with VisPruner translate into proportional end-to-end latency and serving-performance improvements, and if not, where do the unrealized gains go?"

Refine this wording if the evidence suggests a more accurate version.

Clearly distinguish:

- algorithmic/token reduction
- prefill improvement
- complete request latency
- serving throughput/tail latency

Do not imply that a reduction in visual-token work theoretically guarantees an equal reduction in complete request latency.

# 2. System Under Test

Document:

- model
- VisPruner implementation
- GPU
- PyTorch version
- CUDA version
- Transformers version
- dtype
- attention implementation
- batch size
- important_ratio
- visual_token_num configurations
- server architecture
- load-generation architecture
- dataset/workload
- output-token settings where relevant

Use exact values from experiment metadata wherever available.

# 3. Experimental Methodology

Explain the profiling tools used:

- torch.profiler
- CUDA events where applicable
- record_function regions
- Chrome/Perfetto traces
- load generator
- GPU/CPU monitoring
- warmup
- A/B comparison methodology

Explain what each measurement means.

Clearly distinguish:

MEASURED
DERIVED
INTERPRETATION

==================================================
3. DOCUMENT EVERY MAJOR EXPERIMENT
==================================================

Create one full section per major experiment.

The experiment sequence appears to include approximately:

Experiment 1 — Narrow multimodal preparation profiling

Experiment 2 — Full isolated request profiling

Experiment 3 — Serving/load profiling

Experiment 4 — 576-vs-128 VisPruner no-load A/B comparison

Experiment 5 — 576-vs-128 load A/B comparison

Experiment 6 — 576-vs-128 profiler traces / isolated and near-saturation comparison

Experiment 7 — Any later correction/validation analysis that materially changed interpretation

Change this organization if repository evidence shows a better structure.

For EVERY experiment include:

## Research question

What exactly were we trying to learn?

## Experimental setup

- script
- model settings
- workload
- number of requests
- load/RPS if applicable
- visual_token_num
- output-token behavior
- relevant profiler settings

## Files produced

List exact repository paths for:

- script
- raw data
- profiler trace
- summary
- metadata
- logs

Explain what each file contains.

## Results

Give the important numerical results in tables.

## Interpretation

Explain what those numbers mean in simple English.

## Limitations

Explain what this experiment cannot establish.

## Status

Classify it as one of:

VALID
VALID WITH CAVEAT
DIAGNOSTIC ONLY
SUPERSEDED

Explain why.

==================================================
4. PRESERVE THE IMPORTANT MEASURED RESULTS
==================================================

Include a canonical results section collecting the important numbers.

Examples include, where supported:

- 576 → 128 visual tokens
- prefill latency reduction around 60%
- generate-only median latency improvement around 15.2%
- excluded image preprocessing cost around 86.7 ms
- mathematically corrected estimated full end-to-end improvement around 13.4%
- decode contribution to total latency
- vision tower contribution
- CUDA kernel launch count
- CPU CUDA-launch overhead
- KV-cache legacy rebuild behavior
- KV-cache torch.cat append behavior
- queueing/tail degradation beginning around 1.5 RPS
- observed throughput behavior at higher load
- approximately 3.8% observed capacity improvement, with its methodological caveat
- GPU memory findings
- serial serving behavior

Do NOT blindly use these values from this prompt.

Verify each one against the repository evidence.

For every important result give:

Metric
Value
Experiment
Source file
Measurement type
Confidence
Caveat

==================================================
5. HANDLE THE PART 1 TIMING BUG CORRECTLY
==================================================

Document the confirmed no-load timing bug carefully.

The original no-load timer began AFTER image preprocessing.

Therefore the reported 652.1 ms vs 553.0 ms values measured generate/model-path latency rather than complete request latency.

Image load + preprocessing + CUDA transfer was separately measured at approximately 86.7 ms average and should apply roughly equally to both configurations.

Preserve the raw numbers.

Then show the mathematical correction explicitly:

raw:
576 = 652.1 ms
128 = 553.0 ms

estimated complete request:
576 ≈ 652.1 + 86.7
128 ≈ 553.0 + 86.7

Then calculate the corrected percentage improvement.

Label this:

DERIVED ESTIMATE

not a newly measured result.

Explain why we deliberately chose not to rerun this profiling phase and instead will require a clean end-to-end benchmark during final optimization validation.

==================================================
6. DOCUMENT ALL KNOWN METHODOLOGY ISSUES
==================================================

Create a dedicated section:

# Experimental Limitations and Corrections

Include all confirmed issues, such as:

1. Part 1 preprocessing excluded from timer
2. 576 and 128 load sweeps were not temporally interleaved
3. output_token_count included a synthetic BOS token
4. isolated nvidia-smi utilization measurement was weak
5. low-RPS p95/p99 had tiny sample counts
6. near-saturation profiler experiment used back-to-back requests rather than real concurrent GPU execution
7. naive shared-model threaded generation previously crashed and is not evidence of successful concurrency
8. natural EOS caused small differences in generated output length in some comparisons
9. any other issue supported by the repository

For each issue include:

- affected experiment
- affected result
- severity
- whether relative conclusions change
- correction, if any
- whether a future clean benchmark is required

==================================================
7. SYNTHESIZE THE RESULTS
==================================================

Create a major section:

# Combined Findings

Explain the whole chain in simple English.

It may approximately be:

576 visual tokens
        ↓
128 visual tokens
        ↓
large reduction in prefill cost
        ↓
but decode remains dominant
        ↓
therefore much smaller complete-request latency improvement
        ↓
current serial serving architecture causes queueing
        ↓
therefore little additional serving capacity is realized

But derive the exact wording from evidence.

Answer explicitly:

1. What does VisPruner clearly improve?
2. What does it barely affect?
3. Why is end-to-end improvement much smaller than prefill improvement?
4. Why does the serving improvement become even smaller?
5. Which effects are VisPruner-specific?
6. Which effects are properties of the current inference/serving implementation?

==================================================
8. CONNECT THE EVIDENCE TO THE NEXT ENGINEERING PHASE
==================================================

Create:

# Engineering Implications

Do not design the final implementation roadmap here.

Instead connect each measured finding to an engineering question.

Organize findings under:

## Triton / custom kernels

## KV-cache optimization

## CUDA Graphs

## Batching / serving

## Additional categories

Only add an additional category if the profiling evidence justifies one.

For each:

Measured evidence
→ suspected systems problem
→ engineering question

Example format:

MEASURED:
34k+ CUDA launches during the representative request.

IMPLICATION:
Decode contains highly repetitive GPU dispatch.

ENGINEERING QUESTION:
Can CUDA Graph capture reduce CPU dispatch/launch overhead?

Do NOT say an optimization will work before it is implemented and benchmarked.

==================================================
9. EVIDENCE INDEX / APPENDIX
==================================================

Create an appendix:

# Appendix A — Experiment Artifact Index

This should solve the problem of finding files.

Give a repository tree showing:

GPU_Profiling/
│
├── AMAY_VISPRUNER_PROFILING_REPORT.md
├── AMAY_SPEED_PLAN.md
├── AMAY_TIMING_NOTES.md
│
├── scripts/
│   ├── ...
│
└── results/
    ├── timing/
    │   ├── ...
    └── loadgen/
        └── ...

For every important file give a one-line explanation.

Group repetitive loadgen result directories instead of listing hundreds of files.

Explain the difference between:

SCRIPT
RAW DATA
PROFILER TRACE
SUMMARY
METADATA
LOG

Also identify which files are normally worth opening manually and which are mainly archival/raw evidence.

==================================================
10. MAKE THE REPORT TRACEABLE
==================================================

An important requirement:

Every major numerical claim should point to the exact repository file supporting it.

Do not allow important results to exist only because Claude said them in a conversation.

If a conclusion can be reconstructed from raw files, reconstruct it.

If a claim cannot currently be supported by a repository artifact, explicitly write:

"Not currently preserved in a canonical repository artifact."

Do not fabricate support.

==================================================
11. ADD REGENERATION INSTRUCTIONS
==================================================

This report must be self-documenting.

Create:

# Appendix B — How to Regenerate or Update This Report

Explain step by step how another Claude Code session should regenerate or update this report later.

Include:

1. repository root
2. files/directories that should be inspected
3. which files must never be modified
4. how to distinguish my files from other contributors' files
5. how to preserve raw measurements
6. how to add new experiments
7. how to update old conclusions when later evidence supersedes them
8. how to keep MEASURED / DERIVED / INTERPRETATION distinct
9. how to update the evidence index
10. how to avoid rerunning GPU experiments unless explicitly requested

==================================================
12. EMBED THE EXACT CLAUDE PROMPT
==================================================

At the end of Appendix B include:

## Claude Prompt Used to Generate This Report

Include the complete exact prompt from this message verbatim.

The report should therefore contain its own regeneration prompt.

Do not summarize it.
Do not shorten it.
Preserve the full prompt.

This means someone should be able to:

1. open AMAY_VISPRUNER_PROFILING_REPORT.md
2. copy the prompt from Appendix B
3. give it to Claude Code in the repository
4. regenerate/update the report from the evidence

==================================================
13. ALSO SAVE THE GENERATION PROMPT SEPARATELY
==================================================

In addition to embedding the prompt in the report, create:

/workspace/GPU_Profiling/prompts/GENERATE_VISPRUNER_PROFILING_REPORT.md

Create the prompts/ directory if necessary.

This file should contain:

- purpose of the prompt
- when to use it
- the complete exact Claude prompt
- a note that AMAY_VISPRUNER_PROFILING_REPORT.md is the output

Do not place experimental results in this prompt file.

Its purpose is reproducibility.

==================================================
14. LINK THE DOCUMENTATION TOGETHER
==================================================

Update AMAY_SPEED_PLAN.md minimally near the beginning so it links to:

AMAY_VISPRUNER_PROFILING_REPORT.md

Explain:

AMAY_VISPRUNER_PROFILING_REPORT.md
= completed profiling/research evidence and experimental history

AMAY_SPEED_PLAN.md
= current engineering implementation roadmap

prompts/GENERATE_VISPRUNER_PROFILING_REPORT.md
= instructions for rebuilding/updating the profiling report

Do not substantially rewrite AMAY_SPEED_PLAN.md as part of this task.

==================================================
15. READABILITY
==================================================

The report should be technically rigorous but understandable to someone learning GPU inference.

Use:

- short paragraphs
- tables
- diagrams
- equations where useful
- simple English
- explicit definitions

Avoid writing it like raw profiler notes.

When introducing terms such as:

- prefill
- decode
- CUDA kernel launch
- KV cache
- queueing
- p50/p95
- throughput

briefly explain them.

==================================================
16. FINAL DOCUMENT ORGANIZATION
==================================================

The final report should approximately contain:

Title

Abstract

1. Motivation and Research Question
2. System Under Test
3. Experimental Methodology
4. Experiment 1
5. Experiment 2
6. Experiment 3
7. Experiment 4
8. Experiment 5 / later experiments as needed
9. Experimental Limitations and Corrections
10. Canonical Results
11. Combined Findings
12. Engineering Implications
13. Conclusion

Appendix A — Experiment Artifact Index
Appendix B — How to Regenerate or Update This Report
    - including the full Claude prompt

Change numbering if needed based on actual experiments.

==================================================
17. FINAL RESPONSE TO ME
==================================================

After creating everything, tell me:

1. exact report path
2. exact saved prompt path
3. experiments documented
4. major canonical findings
5. important caveated/superseded results
6. whether any evidence is missing
7. what was changed in AMAY_SPEED_PLAN.md
8. which document I should read first

Do NOT rerun experiments.
Do NOT implement performance optimizations.

This task is strictly consolidation, documentation, reproducibility, and research-report creation.
```
