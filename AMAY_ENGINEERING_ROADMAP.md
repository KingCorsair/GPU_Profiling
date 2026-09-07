# AMAY_ENGINEERING_ROADMAP.md — Closing the VisPruner Algorithmic-to-Serving Gap

**Status:** Planning document. Nothing in this roadmap has been implemented. All evidence cited comes from `AMAY_VISPRUNER_PROFILING_REPORT.md` (the completed profiling phase) and direct, read-only inspection of the current codebase. Restructured 2026-09-07 around WP1-WP7; the older Phase 0-6 material is preserved under Deferred / Month-Two Engineering Work and Reference.

---

## The central project story

> VisPruner cuts LLaVA-1.5's visual-token count 4.5x (576→128) and, measured directly, cuts prefill compute by **56-60%**. That algorithmic reduction is real and does not degrade under any condition tested. But it converts into only a **~13% (corrected estimate) complete-request latency improvement**, because decode — which pruning does not touch — dominates total request time. Under load, that ~13% converts into only a **~4% (provisional) serving-capacity improvement**, because the current server processes exactly one request at a time, so its capacity ceiling is set by decode-dominated service time rather than by peak compute. This roadmap profiles-then-engineers: it identifies which parts of the pipeline are wasting pruning's own benefit (the diversity-selection loop, most notably), fixes those first, then addresses the remaining decode- and serving-side bottlenecks only where measurement — not intuition — justifies the effort.

This is a technically precise version of the original hypothesis, and the profiling phase supports it **with two corrections that must be preserved, not smoothed over**:

| Quantity | Naive expectation | What was actually measured | Source |
|---|---|---|---|
| Prefill compute reduction | "big" | **56-60%**, confirmed by two independent methods | `AMAY_VISPRUNER_PROFILING_REPORT.md` §10 |
| Complete-request latency, **raw measurement** | proportional to prefill | 15.2% (Experiment 4) — **this number has a known bug**: its timer excluded ~86.7ms of image preprocessing | same, §9.1 |
| Complete-request latency, **corrected estimate** | — | **~13.4%, a DERIVED ESTIMATE, not a fresh measurement** | same, Experiment 7 |
| Serving-capacity improvement under load | proportional to latency gain | **~3.8%, PROVISIONAL** — the two load sweeps behind this number were not temporally interleaved | same, §9.2, Experiment 5 |

Every phase below that touches the 576-vs-128 comparison inherits this same discipline: report the raw number, label any correction as derived, and never claim a bigger effect than the evidence supports.

---

---

## Month-One Execution Plan

*Restructured 2026-09-07. The month-one plan is now expressed entirely as WP1–WP7. The old
Phase 0–Phase 6 numbering is no longer an execution order. Phases that are still relevant
were merged into the WP sections below; phases that are not month-one work were moved to
**Deferred / Month-Two Engineering Work**, where their original detail is preserved.*

**There is exactly one execution order in this document:**

**WP1 → WP2 → WP3 → WP4 → WP5 → WP6 → WP7**

Any older "recommended implementation order" appearing later in this file is historical and
labelled as such.

### Month-One Roadmap at a Glance

| Order | Work package | In simple English | Why we are doing it | Expected time |
|---|---|---|---|---|
| **WP1** | Controlled benchmark | Build a stopwatch that actually works. Force every request to produce the same number of output words so timings stop jumping around. | Right now the measurement noise (±179 ms) is six times bigger than the biggest win available (~30 ms). Until this exists, we cannot tell if anything we do helped. | **0.5 day** |
| **WP2** | FP16 vision tower | Flip the image encoder from 32-bit to 16-bit maths. It is one line — the code meant to do this is behind an `if` that never runs. | The image encoder is doing 75% of its work in the slow number format on a GPU built for the fast one. It is the cheapest real speed-up in the whole roadmap. | **2.5 days** |
| **WP3** | Server diagnostic | Watch the server under load and find out where the time actually goes on a single request. | The GPU sits ~34% idle at maximum load. We think we know why, but we should prove it before writing any code to fix it. | **1.5 days** |
| **WP4** | Request queue | Stop the server doing image decoding and inference in a single-file line. Let it prepare the next request while the GPU works on the current one. | This is the suspected reason the GPU idles. It is a small change and it is also required before batching is possible at all. | **2.5 days** |
| **WP5** | Static batching | Let the server collect a few requests and run them through the GPU together in one go, instead of one at a time. | Running requests together is the only way to actually fill the idle GPU. This is the biggest and riskiest piece of the month. | **7–9 days** |
| **WP6** | Final load benchmark | Run the full before/after test properly — all three server versions, both token settings, both short and natural-length answers, all shuffled together. | This is the result. It is also the run that makes the earlier load numbers trustworthy, because the originals were run in separate blocks. | **4 days** |
| **WP7** | Writeup | Write down what we measured, what we calculated, and what we only guessed — kept clearly apart. | An unexplained number is not a result. This is what makes the month defensible. | **1.5 days** |
| | | | **Total** | **19.5–21.5 working days** |

### The month in one sentence

**Build a reliable benchmark → take the clearest measured latency win → diagnose why the
server wastes GPU capacity → fix the serving structure → add batching → prove the result
under load → write it up.**

### Day-12 decision rule

**If WP5 has not begun by Day 12, cut batching from the month-one commitment and finish the
project around the FP16 + server-structure result instead.**

That is already a complete, reportable outcome: *the serving layer was not structured to
use the GPU time the pruned configuration leaves idle, and here is how much of it structure
alone recovers.* Do not start WP5 late and leave it half-finished — a half-written batching
path produces no result at all.

### Governing objective

This project measures two different things. They must never be mixed up.

1. **Isolated inference performance.** How fast one request finishes when nothing else is
   running.
2. **Under-load serving performance.** How many requests per second the server sustains,
   and what the tail latency does as the arrival rate climbs.

With no batching, these are the same measurement — throughput is just `1 / service_time`.
The evidence below shows they have already come apart. That gap is where the month's
headroom is.

### The evidence base

Every WP below cites from this. It is stated once here so the WP sections stay short.

**MEASURED — load sweep** (`results/timing/ab_load_gain_table.json`):

| Offered RPS | achieved 576 | achieved 128 | GPU util 576 | GPU util 128 | p95 gain |
|---|---|---|---|---|---|
| 0.5 | 0.500 | 0.500 | 33.8% | 33.3% | +19.6% |
| 1.0 | 1.000 | 1.000 | 48.4% | 39.5% | +6.1% |
| 1.5 | 1.500 | 1.500 | 55.1% | 48.7% | −0.05% |
| 2.0 | 2.000 | 2.000 | 79.0% | 70.1% | +8.5% |
| 3.0 | 2.166 | 2.249 | 87.1% | **65.8%** | −0.34% |

**PROVISIONAL — this whole table.** Per profiling report §9.2 the two sweeps behind it were
**not temporally interleaved**. They ran as separate blocks, so thermal drift and machine
state are confounded with the configuration variable. Until WP6 reruns them interleaved,
describe the result, do not explain it:

> "the existing measurements show a 21.3 percentage-point utilisation difference between
> the two configurations at saturation"

**not** "pruning freed 21 points of utilisation." The second is a causal claim the data does
not yet support.

**MEASURED — single-request breakdown** (`results/timing/ab_noload_comparison.json`,
n=10, warmup=10, A40):

| | 576 | 128 |
|---|---|---|
| mean latency | 887.6 ms | 778.4 ms |
| median latency | 652.1 ms | 553.0 ms |
| stdev | 565.1 ms | 494.8 ms |
| multimodal_prep | 49.7 ms | **68.4 ms** |
| prefill | 111.5 ms | 44.4 ms |
| decode | 711.8 ms | 652.1 ms |
| output tokens | 25.2 | 24.3 |

**DERIVED from the above:** decode is **80.2%** of mean request latency. The whole prep path
— everything VisPruner touches — is 5.6%. Decode costs 28.2 ms/token. Stdev 565.1 ms at
n=10 gives a standard error of ±179 ms, which is 20% of the mean.

**MEASURED — GPU memory** is flat at ~15.3 GB of 46 GB (33%) at every load level and both
configs. Nothing here is memory-constrained.

**MEASURED — two code defects**, both found by tracing the numbers above back into the
source, both cheap to fix, both load-bearing:

- **The vision tower runs FP32.** `clip_encoder.py:29` calls `CLIPVisionModel.from_pretrained`
  with no `torch_dtype`, so HuggingFace defaults to FP32. The cast that should fix this,
  `vision_tower.to(dtype=torch.float16)` at `builder.py:155-156`, sits behind
  `if device_map != 'auto'` — and `'auto'` is the default at `builder.py:26`. The cast never
  runs. Confirmed in `prep_trace.json`: FP32 `ampere_sgemm_*` is **265.95 ms = 75.4% of
  prep-path kernel time**, against 6.18 ms (1.8%) in the FP16 GEMM family, which is
  `mm_projector` alone. → **WP2**
- **The server blocks its own event loop.** `csnbs/server.py:91` — `async def _infer_model(...)`
  calls `model.generate()` synchronously with no `await`. FastAPI does not offload
  `async def` handlers to a threadpool, so per-request CPU work (base64 decode `:176`, PIL
  decode `:106`, `process_images` `:132`) cannot overlap GPU work. → **WP3, WP4**

**MEASURED — deployment topology.** `scripts/start_model_server.sh:11` runs
`exec python -m uvicorn csnbs.server:app --host … --port …` with **no `--workers` flag**, so
uvicorn defaults to one process and one asyncio event loop. `csnbs/server.py:193` is the
same. No gunicorn or other process manager exists in the repo. The Dockerfile `CMD` only
stages SSH keys and keeps the container alive. `scripts/run_ab_load_sweep.py:177` starts the
server through that same script, so the sweeps above ran on this topology, and it is
recorded per-run in the `server_concurrency_model` metadata field. **One event loop, so one
request's GPU work at a time, whatever the arrival concurrency.** If the deployment ever
moves to `--workers N`, redo this analysis — N processes means N event loops and N copies of
the model.

### Two corrections to earlier assumptions in this document

Both are code-verified and both change what is worth doing.

- **Phase 3a (persisted KV cache object) is worth 1.85 ms.**
  `ab_trace_vtn576_isolated_stage_breakdown.txt` shows `kv_cache_legacy_rebuild` at 416
  calls, **1.848 ms CPU and 0.000 ms CUDA** — 0.2% of request latency. It is enabling work
  for a fixed-size buffer and CUDA Graphs, not a latency optimization. It is deferred.
- **The R4 padding-waste premise does not hold for this pruner.** `llava_arch.py:148-151`:
  `visual_token_num` is a global scalar, and `index_masks.scatter_` sets exactly `T` True
  per row. **Every image in a batch keeps an identical visual-token count.** Padding waste
  therefore comes only from text-prompt and output-length variance, not from pruning. The
  version of R4 described in `CLAUDE.md` applies to per-image adaptive methods, not to
  VisPruner. Re-derive it before scheduling any packed-layout work.

### Why serving work comes before kernel work

The largest measured headroom is in serving structure, not in per-request compute:

- Decode is 80.2% of single-request latency and nothing affordable this month touches it.
  That caps every single-request optimization in this document at a few percent.
- At saturation the 128 config shows ~34% idle GPU with p50 latency at 7.6 s, and 33% of
  GPU memory in use. There is room, and nothing is blocking it except how the server is
  built.

So queue and batching work moves ahead of Triton, KV-cache, and CUDA-Graph work. Those are
now in **Deferred / Month-Two Engineering Work**.

**Why not just run `--workers 3` instead of building batching?** Name it rather than ignore
it — an interviewer will ask. Two reasons it is not the answer. Each worker is a separate
process loading its own ~14 GB of weights, so three workers is ~42 GB of 46 GB with almost
nothing left for KV cache and activations. And separate processes time-slice the same SMs
rather than aggregating work — that improves overlap but not compute efficiency, whereas
batching combines requests into single larger kernel launches. Worth running as a cheap
control if time allows. Not a substitute for WP5.

### Ownership boundary with Rithvik

**Agree this before WP4 starts. It is not a default.**

`CLAUDE.md` assigns "serving path" to Amay. `csnbs/` and the FastAPI server are Rithvik's.
Both readings are defensible, so it needs a decision.

Proposed split:

- **Amay writes** `vis_pruner_copy/llava/serve/batch_engine.py`, exposing
  `submit(image_tensor, input_ids) -> Future`. It owns the queue, the worker thread,
  micro-batch collection, batched `generate()`, and output splitting. **Zero lines inside
  `csnbs/`.**
- **Rithvik owns** `server.py`: the HTTP layer, the threadpool prep call, importing
  `batch_engine`, and `BATCH_SIZE` / `MAX_WAIT_MS` / `VISUAL_TOKEN_NUM` env config.
- **Rithvik owns all reportable numbers.** The WP6 sweep runs on his harness, per
  `CLAUDE.md`.
- Integration is one function signature, agreed on day 1.

If Rithvik would rather own the queue as well, WP4+WP5 shrinks to roughly 4–5 days and the
month gains real slack.

### How to read the rest of this document

- **WP1–WP7** below are the month-one plan. Read them in order. Each is self-contained.
- **Deferred / Month-Two Engineering Work** holds the phases that are not month-one, with
  their original detail intact.
- **Reference — Background and Historical Context** holds the category taxonomy, the
  dependency analysis, the 576/128 validation formula, and a note marking the old phase
  ordering as superseded.

---

## WP1 — Controlled Benchmark

**0.5 day. No dependencies. Nothing else in the month is measurable until this exists.**

### 1. Problem

We cannot currently measure whether any optimization worked.

The existing A/B harness lets each request stop when the model decides to stop. Different
requests produce different numbers of output tokens. Decode costs 28.2 ms per token, so a
five-token difference in output length moves the measurement by 141 ms.

### 2. Why this matters

The biggest single win available anywhere in this roadmap is about 30 ms. The current
measurement noise is ±179 ms. We would not be able to see a real improvement, and worse, we
could easily "see" one that is not there.

`CLAUDE.md` rule 8 says this directly: if run-to-run variance is 8% and an optimisation
shows 5%, nothing has been measured.

### 3. Evidence

- **MEASURED** — `ab_noload_comparison.json`: mean latency 887.6 ms, stdev 565.1 ms, n=10.
  That is a 64% coefficient of variation.
- **DERIVED** — standard error of the mean = 565.1/√10 = ±179 ms, or 20% of the mean.
- **MEASURED** — `output_token_policy` in that file's metadata is
  `"natural EOS stopping, capped at max_new_tokens"`, and `output_token_count_mean` is 25.2.
- **DERIVED** — decode is 711.8 ms for 25.2 tokens = 28.2 ms/token, so output-length
  variance is the dominant variance term.

### 4. What we are trying to achieve

A benchmark that can reliably detect a 20 ms change in a single component.

### 5. What Amay will actually do

1. Write `scripts/bench_dev.py`.
2. Load a fixed set of 20 images from `vispruner_eval_dataset/dev.json`. Same images every
   run, in the same order.
3. Set **`min_new_tokens == max_new_tokens == 32`**. This is the key line. It forces every
   request to generate exactly 32 tokens, which removes the output-length variance.
4. Run 10 warm-up iterations and throw them away. This covers CUDA context init, memory
   pool warm-up, and cuDNN autotuning.
5. Run 30 measured iterations.
6. Wrap these regions in CUDA events: `vision_tower`, `mm_projector`, `multimodal_prep`,
   `prefill`, `decode`. Use `time.perf_counter()` for end-to-end wall time.
7. Call `torch.cuda.synchronize()` **once, outside the measurement loop**. Never inside it —
   an inner sync drains the pipeline and inflates fast kernels.
8. Run both `visual_token_num` settings, 576 and 128, with trials **interleaved** rather
   than run as two blocks.
9. Report p50 and p95. **Do not report p99** — 30 samples cannot support it.
10. Record the git commit and GPU model in the output JSON.
11. Run the whole thing three times against unmodified code and record the spread. This is
    the noise floor, and everything later is compared against it.

### 6. Code/files involved

| File | Why it matters |
|---|---|
| `scripts/bench_dev.py` *(new)* | The harness itself |
| `vis_pruner_copy/vispruner_eval_dataset/dev.json` | The fixed 20-image input set |
| `vis_pruner_copy/llava/model/builder.py` | `load_pretrained_model()` — how the model is loaded, including the `visual_token_num` argument |
| `scripts/time_sqa_sweep.py` | Existing harness to copy conventions from, not to modify |

### 7. Concepts Amay needs to understand

- **Why CUDA events, not `time.time()`.** CUDA is asynchronous. A wall clock measures how
  long Python took to *queue* work, not how long the GPU took to do it.
- **Why synchronize outside the loop.** Syncing inside the loop forces the pipeline to drain
  on every iteration, which makes fast kernels look slower than they are.
- **Why warm-up matters.** The first several iterations pay one-off costs that will never
  recur in steady state.
- **Why p99 needs ~1000 samples.** With 30 samples, p99 and the maximum are the same number,
  and that number is noise.

### 8. Tests and benchmarks

| Run | Question it answers |
|---|---|
| Three baseline runs, unmodified code, both configs | What is this machine's noise floor? |
| Compare the three runs' `vision_tower` CUDA times | Is the component-level measurement stable enough to detect ~20 ms? |

### 9. Success criteria

- Run-to-run spread on `vision_tower` CUDA time is **under ~5 ms** across the three baseline
  runs.
- The harness writes a JSON with per-region p50/p95, git commit, and GPU model.
- Rerunning it produces the same numbers within that spread.

### 10. Failure / stop / re-ranking criteria

- **If the spread on `vision_tower` exceeds ~5 ms**, do not proceed to WP2. Fix the harness
  first. Likely causes: another process on the GPU, thermal throttling, or a missed
  synchronization.
- **If any other process is on the GPU**, stop and coordinate. `CLAUDE.md` rule 9 — one
  person on the GPU during benchmark runs.

### 11. Deliverables

- `scripts/bench_dev.py`
- Three baseline result JSONs, both configs, committed before any code changes
- A one-paragraph note stating the measured noise floor

### 12. What Amay should be able to explain afterward

- Why does fixing the output length matter more than any other change to this benchmark?
- Why do CUDA events give a different answer than a wall clock?
- What is this machine's noise floor, and how do you know?
- Why is p99 not reported here?

### 13. Time breakdown

| | |
|---|---|
| Implementation | 2 h |
| Debugging | 0.5 h |
| Validation (three baseline runs) | 1 h |
| Documentation | 0.5 h |
| **Total** | **0.5 day** |

---

## WP2 — FP16 Vision Tower

**2.5 days. Depends on WP1. Supersedes the old Phase 1a.**

### 1. Problem

The CLIP vision tower runs in FP32. It should be running in FP16.

This is not a design decision anyone made. It is a bug. The code that was supposed to cast
the tower to FP16 sits behind an `if` that is never true.

### 2. Why this matters

The vision tower is the single largest piece of GPU work in the image-processing path. The
A40's FP16 tensor cores are several times faster than its FP32 SGEMM path. We are leaving
that on the table for no reason.

It also has to land before anything else is benchmarked, because it changes every
vision-path number in this document. Every later measurement would otherwise be against a
stale baseline.

### 3. Evidence

- **MEASURED (code)** — `clip_encoder.py:29` calls `CLIPVisionModel.from_pretrained(...)`
  with no `torch_dtype`. HuggingFace defaults to FP32.
- **MEASURED (code)** — `builder.py:155-156` has `if device_map != 'auto': vision_tower.to(...,
  dtype=torch.float16)`. `builder.py:26` defaults `device_map="auto"`. The cast never runs.
- **MEASURED (trace)** — `prep_trace.json`, 9,760 kernel events, 352.66 ms total GPU time.
  FP32 `ampere_sgemm_*` accounts for **265.95 ms = 75.4%**. The FP16 GEMM family accounts
  for 6.18 ms = 1.8%, and that is `mm_projector`, which is already Half.
- **MEASURED (trace)** — `vectorized_layer_norm_kernel<float, float>` (500 calls) and
  `softmax_warp_forward<float, float, float>` (19.42 ms) confirm the tower's *activations*
  are FP32 too, not only its weights.
- **MEASURED** — `vision_tower` is 43.888 ms of 48.270 ms `multimodal_prep` CUDA time at 576.
- **DERIVED** — the vision tower is ~45.2 ms, or ~5.1% of the 887.6 ms mean request latency.
- **HYPOTHETICAL** — A40 FP32 SGEMM peak is 37.4 TFLOPS against 149.7 TFLOPS for FP16 tensor
  cores, a 4× theoretical ratio. On a saturated GEMM at 96.3% occupancy, a realistic
  achieved speedup is **1.8–3×**.
- **DERIVED** — that gives 20–30 ms saved, which is **2.3–3.4% of mean latency** and
  3.1–4.6% of median. Small, and that smallness is itself a finding: decode is 80.2% of the
  request and nothing here touches it.

### 4. What we are trying to achieve

Move the vision tower's matrix multiplies onto the FP16 tensor-core path, and prove it did
not damage answer quality — especially on text-in-image questions.

### 5. What Amay will actually do

1. Print `vision_tower.dtype` after loading. Confirm it says `torch.float32`. Do not skip
   this — it is the one-line proof the bug is real.
2. Add `torch_dtype=torch.float16` to the `CLIPVisionModel.from_pretrained` call at
   `clip_encoder.py:29`.
3. Assert `vision_tower.dtype == torch.float16` after load.
4. Run `scripts/bench_dev.py` from WP1, both configs. Check `vision_tower` CUDA time.
5. If it did not drop by at least 20%, stop and re-profile. Do not continue.
6. Re-run `scripts/profile_multimodal_prep.py`. Confirm the `ampere_sgemm_*` rows are gone
   and the `f16`/`xmma` GEMM family has replaced them.
7. Run the Layer 1 correctness checks (see **Tests and benchmarks** below).
8. Run the Layer 2 OCR gate (see **Tests and benchmarks** below).
9. Re-baseline the vision-path numbers quoted in this document and in
   `AMAY_TRACE_PLAN.md`.

**Why `torch_dtype` at construction rather than `.half()` afterwards:** it never
materialises FP32 weights in memory. **Why not just relax the `if` at `builder.py:155`:**
that line also moves devices and has other callers, so changing it has a wider blast radius
than the one-line fix.

**Do not touch** `mm_projector` (already Half), the language model, the image processor, the
pruning code in `llava_arch.py`, the anyres / `spatial+unpad` branch, or `builder.py:155-156`
itself.

### 6. Code/files involved

| File | Why it matters |
|---|---|
| `vis_pruner_copy/llava/model/multimodal_encoder/clip_encoder.py:29` | The one line to change. `load_model()` constructs the tower here |
| `vis_pruner_copy/llava/model/multimodal_encoder/clip_encoder.py:64,70` | `images.to(dtype=self.dtype)` upcasts the incoming FP16 tensor to FP32 on entry, then `:70` casts back down for the projector. That round-trip disappears with the fix |
| `vis_pruner_copy/llava/model/builder.py:26,155-156` | Where the `device_map='auto'` default and the skipped cast live. Read, do not change |
| `scripts/profile_multimodal_prep.py` | Produces the before/after trace pair |
| `vis_pruner_copy/llava/eval/m4c_evaluator.py` | The official TextVQA scorer, already vendored |
| `eval/textvqa/llava_textvqa_val_v051_ocr.jsonl` | 5,000 TextVQA questions, annotations already present |

### 7. Concepts Amay needs to understand

- **FP32 SGEMM vs FP16 tensor cores.** Why an A40 runs half-precision matrix multiplies
  several times faster, and how to recognise each in a trace by kernel name
  (`ampere_sgemm_*` vs the `f16`/`xmma` family).
- **Where FP16 is numerically risky.** LayerNorm accumulation and attention softmax are the
  two places. HuggingFace's CLIP usually upcasts softmax internally; the trace will show
  whether it did.
- **Amdahl's law, concretely.** A 2× speedup on 5.1% of the request is a 2.5% end-to-end
  win. This is the number that must be reported, not the component-level one.

### 8. Tests and benchmarks

**Layer 1 — local development gate. Minutes. Run on every iteration. No dependency on
anyone.**

| Check | Question it answers | Bar |
|---|---|---|
| Exact-text comparison, greedy, 90 dev images, both configs | Did the model's answers change? | Report % identical. **Not expected to be 100%** |
| Token-selection equivalence — dump `selected_indices` from `encode_images()`, FP32 vs FP16, 20 fixed images | Did the pruner pick different tokens? | Report exact-match rate |
| Cosine similarity of pre-projector `image_features` | Did the image representation drift? | > 0.999 |
| Softmax dtype in the new trace | Is the numerically risky op still accumulating in FP32? | FP32 accumulate preserved. If it becomes `<half,half,half>`, stop |
| Run twice, same seed | Is it still deterministic? | Byte-identical |

**Layer 2 — local OCR acceptance gate. ~15 min GPU. No dependency on anyone.**

Official TextVQA VQA-accuracy on a fixed 1,000-question subsample of
`eval/textvqa/llava_textvqa_val_v051_ocr.jsonl`, scored with the already-vendored
`m4c_evaluator.py`, FP32 vs FP16 at 576. Run the full 5,000 if the subsample moves.

This answers: *did FP16 damage the model's ability to read text in images?* That is the
specific failure mode FP16 vision encoding is known for.

Running an official vendored scorer is not designing scoring logic, so this is **not** in
Sribhav's protected area under `CLAUDE.md`.

**Layer 3 — final locked accuracy gate. Sribhav. Asynchronous.**

Per-category accuracy on the locked eval set, OCR broken out, before vs after. **This
remains the formal acceptance criterion inherited from the old Phase 1a.** Request it when
the change lands; do not block on it, because Layer 2 already gives a high-power OCR gate
that Amay controls. When reporting it, note that per-category n=15 gives a standard error
around 12 pp, so Layer 2 is the stronger evidence and Layer 3 is confirmation.

**Performance runs:** `scripts/bench_dev.py` both configs, before and after; and a
before/after `profile_multimodal_prep.py` trace pair.

### 9. Success criteria

All four must hold:

1. `vision_tower` CUDA time drops **by at least 20%**.
2. The trace's `ampere_sgemm_*` rows are replaced by the FP16 GEMM family, and the FP32
   sgemm share falls well below the 75.4% baseline.
3. Layer 2 TextVQA accuracy is unchanged within noise.
4. Output is still deterministic across repeated runs.

### 10. Failure / stop / re-ranking criteria

- **`vision_tower` CUDA time drops <20%** → the tower was not compute-bound the way the
  trace suggests. Stop, re-profile, and treat every other trace-derived conclusion in this
  document as suspect until explained.
- **Exact-text match <85% on greedy dev** → numerics are unstable. Investigate LayerNorm and
  softmax before going further.
- **TextVQA accuracy regresses outside noise (Layer 2), or per-category OCR regresses
  (Layer 3)** → **revert the dtype change.** Speed is not worth silent quality loss on the
  exact categories this project uses to argue pruning is safe.
- **Softmax becomes `<half,half,half>`** → pin softmax to FP32 accumulate before accepting
  any timing result.
- **`device_map='auto'` plus accelerate refuses the dtype at construction** → fall back to an
  explicit `.half()` after `load_model()`. Same outcome, slightly more memory during load.

### 11. Deliverables

- The one-line change at `clip_encoder.py:29`
- Before/after trace pair, with the FP32-sgemm share stated against the 75.4% baseline
- `scripts/bench_dev.py` results, before and after, both configs
- Exact-text comparison report, 90 dev images
- TextVQA VQA-accuracy, FP32 vs FP16
- Token-selection equivalence report
- Re-baselined vision-path numbers in this file and `AMAY_TRACE_PLAN.md`

### 12. What Amay should be able to explain afterward

- Why was the vision tower FP32, in one sentence, naming the exact branch?
- How do you tell FP32 from FP16 GEMMs by looking at a profiler trace?
- Why is a 2× speedup on the vision tower only ~3% end-to-end?
- Which operations inside a vision transformer are risky to run in FP16, and how did you
  check?
- Why is exact-text match not expected to be 100%, and what would make you revert?

### 13. Time breakdown

| | |
|---|---|
| Implementation | 0.25 day |
| Debugging | 0.25 day |
| Correctness validation (Layers 1 and 2) | 1.25 days |
| Performance validation, traces, documentation | 0.75 day |
| **Total** | **2.5 days** |

---

## WP3 — Server Diagnostic

**1.5 days. No code changes. This WP decides whether WP4 and WP5 happen at all.**

### 1. Problem

At maximum load the GPU is about a third idle, and we do not know why with certainty.

At an offered rate of 3 rps the server achieves 2.249 rps with the pruned config, and GPU
utilisation sits at 65.8% while p50 latency is 7.6 seconds. Requests are queueing. The GPU
is not busy. Something between the two is wasting time.

We have a strong suspicion, but it is a hypothesis derived from reading code.

### 2. Why this matters

WP4 and WP5 are 9.5–11.5 days of the month. Both are built on the assumption that the idle
GPU is caused by the server's structure. If that assumption is wrong, both are wasted work.

One and a half days of measurement to de-risk ten days of implementation is the right trade.

### 3. Evidence

- **PROVISIONAL** — at offered 3 rps: 87.1% GPU utilisation at 576, 65.8% at 128. Not
  temporally interleaved; see the evidence base above.
- **MEASURED** — median latency improved 15.2% from 576 to 128, but the throughput ceiling
  improved only 3.8%. On a strictly serial server those should match. They do not, so the
  ceiling is set by something other than model service time.
- **DERIVED** — 80 requests completed in 41.37 s at the 3 rps level, so effective service
  time is ~517 ms, against a 652 ms median model latency. Requests are partly overlapping,
  and whatever is *not* overlapping is what sets the ceiling.
- **MEASURED (code)** — `csnbs/server.py:91`, `async def _infer_model(...)`, contains a
  blocking `model.generate()` and no `await`. FastAPI does not offload `async def` handlers
  to a threadpool.
- **MEASURED (code)** — single uvicorn process, single event loop; see the topology note in
  the evidence base.
- **HYPOTHETICAL** — per-request CPU work (base64 decode, PIL decode, `process_images`,
  tokenization) is serialising behind GPU work instead of overlapping with it, and that is
  what leaves the GPU idle.

**Note on what is and is not in question.** That a blocking call inside an `async def` blocks
the event loop is a certainty of asyncio semantics, not a hypothesis. What is genuinely
unknown is **how much CPU work exists outside `generate()`**. If it is 5 ms, the queue is
pointless. If it is 90 ms, it is the ceiling. That quantity is what this WP measures.

### 4. What we are trying to achieve

A number: how many milliseconds per request are spent on CPU work that could be overlapping
with GPU work but currently is not.

### 5. What Amay will actually do

1. Add a background heartbeat coroutine to the server: `await asyncio.sleep(0.005)` in a
   loop, recording actual versus expected wake times to a file. When the loop is blocked,
   the gaps show up directly and their size equals the blocking duration.
2. Add per-request server-side timestamps, written to JSONL:
   `t_handler_entry`, `t_b64_done`, `t_pil_done`, `t_preprocess_done`, `t_generate_start`,
   `t_generate_end`, `t_response`.
3. Run the existing load sweep at rps 2.0, both configs.
4. Compute `mean(t_preprocess_done − t_handler_entry)`. **This is the overlappable CPU
   work.** Compare it against measured service time.
5. Check whether `t_handler_entry[N+1] ≈ t_generate_end[N]`. If requests are not even being
   *received* until the previous one finishes, that is the confirmation.
6. Write up the service-time decomposition: HTTP receive, base64, PIL, preprocessing,
   `generate()`, response.

These are diagnostics. Remove them, or put them behind a flag, before WP4's timing runs.

### 6. Code/files involved

| File | Why it matters |
|---|---|
| `csnbs/server.py:91` | `_infer_model` — the `async def` with the blocking call |
| `csnbs/server.py:106,132,176` | PIL decode, `process_images`, base64 decode — the CPU work in question |
| `scripts/run_ab_load_sweep.py` | Existing sweep driver; reuse it, do not rewrite it |
| `scripts/start_model_server.sh:11` | Confirms the single-worker topology |
| `results/timing/loadgen_monitor/*.csv` | Existing GPU-utilisation samples to compare against |

**Ownership note:** this WP adds temporary instrumentation to Rithvik's file. Agree it with
him first, and remove it afterwards.

### 7. Concepts Amay needs to understand

- **Why `async def` plus a blocking call is worse than a plain `def`.** FastAPI runs `def`
  handlers in a threadpool. It runs `async def` handlers directly on the event loop. So the
  `async` keyword here does the opposite of what it looks like it does.
- **What "blocking the event loop" actually stops.** Not just other inference — also
  accepting connections, reading request bodies, and writing responses.
- **Open-loop vs closed-loop load.** The existing generator is open-loop, which is correct.
  A closed-loop generator would back off exactly when the server struggles and hide the
  problem entirely.

### 8. Tests and benchmarks

| Run | Question it answers |
|---|---|
| Heartbeat trace at rps 2.0 | Is the event loop actually blocked, and for how long at a time? |
| Per-request timestamp JSONL at rps 2.0 | How much CPU work per request could overlap with GPU work? |
| `t_handler_entry[N+1]` vs `t_generate_end[N]` | Are requests being received only after the previous one finishes? |

### 9. Success criteria

This WP succeeds by producing a clear answer, in either direction.

- The heartbeat trace shows gaps, and their size matches `generate()` duration.
- Overlappable CPU work is quantified as a number and a percentage of service time.
- The service-time decomposition accounts for most of the measured ~517 ms.

### 10. Failure / stop / re-ranking criteria

**This is the month's most important decision gate.**

- **If overlappable CPU work is under 5% of service time, do not proceed to WP4.** The
  event-loop hypothesis is dead. The idle GPU has another cause and it must be found before
  any serving code is written. Next suspects, in order: per-request H2D transfer, image
  preprocessing happening on the load-generator side, HTTP and base64 overhead, and the
  load generator itself being the bottleneck.
- **If the heartbeat shows no gaps**, something is wrong with the instrumentation, since
  asyncio semantics say there must be gaps. Fix the instrument before drawing conclusions.
- **If overlappable CPU work is large but service time still does not add up**, keep
  decomposing before building anything.

### 11. Deliverables

- Heartbeat trace file plus a plot or table of loop-blocking gaps
- Per-request timestamp JSONL at rps 2.0, both configs
- A service-time decomposition table: how many ms each stage costs
- **A written go/no-go for WP4**, with the overlappable-CPU-work number stated explicitly
- Instrumentation removed or flag-gated

### 12. What Amay should be able to explain afterward

- What problem am I investigating, in one sentence?
- Why do I think it exists — what in the code causes it?
- What exactly did I measure, and what number came out?
- What result would have told me I was wrong?
- Why does `async def` make this worse rather than better?

### 13. Time breakdown

| | |
|---|---|
| Implementation (instrumentation) | 0.5 day |
| Debugging | 0.25 day |
| Running the sweep and analysing | 0.5 day |
| Documentation and go/no-go writeup | 0.25 day |
| **Total** | **1.5 days** |

---

## WP4 — Request Queue

**2.5 days. Depends on WP3's go decision. This is server state S1.**

### 1. Problem

The server does everything for one request before it starts the next. Decoding the image,
preprocessing it, running the GPU, and sending the response all happen in a single file.

While the GPU is working, nothing else can happen. While the CPU decodes the next image, the
GPU sits idle.

### 2. Why this matters

Two reasons, and the second is the more important one.

First, if WP3 confirms the CPU work is significant, overlapping it with GPU work raises the
throughput ceiling for very little code.

Second, **you cannot batch requests you have not queued.** WP5 is impossible without this.
Even if the direct win were zero, this would still be on the critical path.

It also isolates a result. If we went straight to batching, we could never tell how much
capacity was lost to bad server structure and how much was gained by batching. Measuring S1
separately answers that.

### 3. Evidence

- Everything from WP3. **Do not start this WP before WP3 has produced its number.**
- **MEASURED (code)** — the blocking `async def` at `csnbs/server.py:91`.
- **MEASURED** — 33% of GPU memory in use, so nothing here is memory-constrained.

### 4. What we are trying to achieve

Let the server prepare request N+1 on the CPU while the GPU is still working on request N,
without ever letting two threads touch the model at the same time.

### 5. What Amay will actually do

1. Create `vis_pruner_copy/llava/serve/batch_engine.py`.
2. Expose one function: `submit(image_tensor, input_ids) -> Future`.
3. Inside it, run a **single** worker thread that pulls from a `queue.Queue` and calls
   `model.generate()`. One thread, always. B=1 for now.
4. In `server.py` (Rithvik's, coordinate first), change the handler so that base64 decode,
   PIL decode, `process_images`, and tokenization run in a threadpool, then call
   `submit(...)` and await the future.
5. Before any timing run, fix the shared-state bug at `llava_llama.py:176-177` — see below.
6. Measure GPU utilisation at saturation, before and after.

**The boundary, precisely.** This is the part to get right.

| Stage | Where it runs | Why |
|---|---|---|
| HTTP receive, base64 decode | threadpool, before the queue | Pure CPU, no model state |
| PIL decode | threadpool, before the queue | Pure CPU |
| `process_images` (CLIP resize/normalise) | threadpool, before the queue | Pure CPU, produces a CPU tensor. `image_processor` is stateless per call |
| `tokenizer_image_token` | threadpool, before the queue | Pure CPU. Sentencepiece encoding is thread-safe |
| **H2D copy** | **inference worker** | First CUDA call. Keep all CUDA in one thread |
| `prepare_inputs_labels_for_multimodal`, vision tower, pruning, prefill, decode | inference worker | All CUDA |
| `batch_decode` | inference worker | Simplest to keep with the rest |

H2D stays in the worker deliberately. Overlapping copies from another thread would need
pinned memory and a separate stream, and would put a second thread into CUDA — reintroducing
exactly the hazard described under WP5's B>1 note. Not worth it for this.

**Execution timeline** (DERIVED, using a placeholder 87 ms prep and 360 ms GPU; real numbers
come from WP3):

```
before:  N    [prep 87][====== generate 360 ======][resp]
         N+1                                       [prep 87][====== generate 360 ======]
                                    service = 452 ms  ->  2.21 rps

after:   N    [prep 87][====== generate 360 ======][resp]
         N+1           [prep 87]                   [====== generate 360 ======]
                                    service = 360 ms  ->  2.78 rps   (+26%)
```

**One existing bug must be fixed first.** `llava_llama.py:176-177` assigns
`self._timing_forward_events = []` and `self._timing_prepare_inputs_calls = []` on the
**shared model instance** on every `generate()` call. With one worker this is safe. With any
concurrency it silently corrupts the timing output. Remove it or make it thread-local before
WP4's measurements.

### 6. Code/files involved

| File | Why it matters |
|---|---|
| `vis_pruner_copy/llava/serve/batch_engine.py` *(new)* | Amay's file. Queue, worker thread, `submit()` |
| `csnbs/server.py` | Rithvik's. Handler becomes async accept → threadpool prep → `submit()` |
| `vis_pruner_copy/llava/model/language_model/llava_llama.py:176-177` | Shared-instance timing lists. Fix before measuring |
| `vis_pruner_copy/llava/mm_utils.py` | `process_images`, `tokenizer_image_token` — the CPU work moving to the threadpool |
| `scripts/run_ab_load_sweep.py` | Reuse for the S1 measurement |

### 7. Concepts Amay needs to understand

- **Why one worker thread and not several.** Several threads in `model.generate()` is
  exactly the configuration that produced the CUDA device-side assert. One worker sidesteps
  it completely.
- **What is safe to run concurrently and what is not.** CPU preprocessing touches only
  stateless helpers. Anything touching CUDA or the model stays in the worker.
- **Producer/consumer with a bounded queue.** What happens when the queue fills, and why an
  unbounded queue turns a throughput problem into an out-of-memory problem.
- **Why this raises throughput without improving single-request latency.** For one request
  with no contention, nothing overlaps, so nothing improves. It may even get slightly worse
  from queue handoff.

### 8. Tests and benchmarks

| Run | Question it answers |
|---|---|
| Single request, S0 vs S1 | Did the queue add latency when there is nothing to overlap with? |
| Load sweep at all five rps levels, S1, both configs | Did the throughput ceiling move? |
| GPU utilisation sampling during that sweep | Did the idle GPU actually fill up? |
| Output comparison, S0 vs S1, 90 dev images | Are the answers identical? They must be — nothing about the model changed |
| Sustained run with the queue at capacity | Does anything deadlock or leak when the queue is full? |

### 9. Success criteria

- Output is **byte-identical** to S0 on the same inputs. This is not negotiable — no model
  code changed.
- GPU utilisation at saturation rises measurably above the S0 figure.
- The throughput ceiling moves above ~2.2 rps.
- No deadlocks, no unbounded memory growth under sustained load.
- Single-request latency is not meaningfully worse.

### 10. Failure / stop / re-ranking criteria

- **Any output difference from S0** → a correctness bug. Stop and fix. There is no
  performance justification for a wrong answer.
- **If GPU utilisation does not move**, the overlappable CPU work was not the constraint
  after all, even though WP3 said it was. Re-open WP3's analysis before starting WP5.
- **If S1 alone reaches ≥90% GPU utilisation at saturation**, batching's remaining headroom
  is small. Consider spending WP5's budget on a batch-size sweep and tail-latency
  characterisation instead of pushing for larger batches.
- **If single-request latency regresses noticeably**, the handoff is too expensive. Look at
  the queue implementation before blaming the design.

### 11. Deliverables

- `vis_pruner_copy/llava/serve/batch_engine.py` with a single-worker queue
- The `server.py` change (Rithvik's, co-designed)
- The `llava_llama.py:176-177` shared-state fix
- S1 load-sweep results, both configs, with GPU-utilisation curves
- S0-vs-S1 output-identity report
- A short note stating how much capacity was recovered by structure alone

### 12. What Amay should be able to explain afterward

- Which work moved off the event loop, and which deliberately did not?
- Why does all CUDA work stay in one thread?
- Why is this required before batching, even if its own gain were zero?
- Why does this improve throughput but not single-request latency?
- How much of the capacity gap was structural, and how do you know?

### 13. Time breakdown

| | |
|---|---|
| Implementation | 1 day |
| Debugging | 0.5 day |
| Validation (output identity, stability) | 0.5 day |
| Benchmarking and documentation | 0.5 day |
| **Total** | **2.5 days** |

---

## WP5 — Static Batching MVP

**7–9 days. Depends on WP4. This is server state S2. Biggest and riskiest item in the
month.**

**This is a deliberately reduced version of the old Phase 5, not the same work.** See
"Scope, and what was cut" below.

### 1. Problem

The server runs one request through the GPU at a time. Even with WP4's queue, each
`generate()` call processes a single request.

A 7B model decoding one token at a time barely uses the GPU. Most of the hardware sits idle
waiting on memory. Running several requests through the same forward pass costs almost the
same as running one.

### 2. Why this matters

This is the only mechanism that aggregates GPU work. WP4 lets the CPU and GPU overlap.
Batching makes the GPU do more per unit of time.

It is also what makes the project's central claim testable. The pruned config leaves GPU
time unused. Batching is the mechanism that turns unused GPU time into capacity. Without
it, the project ends with a diagnosis and no treatment.

### 3. Evidence

- **PROVISIONAL** — 21.3-point utilisation difference between configs at saturation, and
  ~34% idle GPU at 128.
- **MEASURED** — GPU memory flat at 15.3 GB of 46 GB. Room for many concurrent requests.
- **MEASURED (code)** — `llava_arch.py:350-379` already builds a padded batch, an
  `attention_mask`, and `position_ids` for **arbitrary B**, with a configurable padding
  side. `encode_images()` is written with `B` throughout. The hard part is already done.
- **MEASURED (code)** — every eval script in the repo uses `batch_size=1`. **B>1 has never
  been run through this pruning path.**
- **MEASURED (code)** — `llava_arch.py:360` reads
  `getattr(self.config, 'tokenizer_padding_side', 'right')`, and that config value is only
  ever set in `train.py:923`, never at inference. It therefore defaults to **right padding**.
- **HYPOTHETICAL** — batching raises throughput substantially. **State no target.** See the
  no-target rule below.

### 4. What we are trying to achieve

Let the inference worker take up to N queued requests, run them through one `generate()`
call, and return each answer to the right requester — with every answer identical to what
that request would have produced alone.

### 5. What Amay will actually do

**Step 1 — B>1 shape-safety smoke. Do this first, before writing anything else.**

Run the pruning path directly at B = 2, 4, 8 with a plain script, no server. Confirm
`encode_images()` and `prepare_inputs_labels_for_multimodal()` survive and produce sane
shapes. If it crashes, see the stop criteria — this changes the month.

**Step 2 — Fix the padding side.**

Set `model.config.tokenizer_padding_side = "left"` at load, and left-pad `input_ids`.
Right-padding a decoder-only model for batched generation produces garbage for every
sequence shorter than the longest, and it fails **silently**.

**Step 3 — Micro-batch collector.**

In `batch_engine.py`, have the worker drain up to `BATCH_SIZE` requests from the queue, or
wait up to `MAX_WAIT_MS`, whichever comes first. Both fixed and configurable by env var.

**Step 4 — Batched call.**

Stack image tensors (already the same shape — CLIP input is fixed size). Left-pad
`input_ids` to the batch's longest. Build the `attention_mask`. Call `generate()` once.

**Step 5 — Output splitting.**

Strip left padding and the prompt from each sequence. Handle per-sequence EOS — sequences
finish at different points and HF pads the finished ones. Route each answer to the right
future.

**Step 6 — Equivalence validation.**

Every request's output must match what it produces at B=1.

**Step 7 — Integration and load testing.**

Wire into the server, run the sweep, debug under real traffic.

#### Scope, and what was cut

**Included:** single image per request, fixed `BATCH_SIZE`, fixed max-wait window, greedy
decoding, fixed `max_new_tokens=64`, left padding.

**Excluded:** bucketing, length sorting, packed or varlen layout, continuous batching,
multi-image requests, per-request generation parameters, preemption, priority scheduling,
paged KV.

**Why this is 7–9 days when the old Phase 5 said 9–14.** One code-verified reason: the old
estimate assumed the batched multimodal padding path had to be built. It already exists at
`llava_arch.py:350-379`, written and shipped by LLaVA upstream. That was the hard part. What
remains is the collector, output splitting, and validation.

| Task | Days |
|---|---|
| B>1 shape-safety smoke | 1–2 |
| Left-padding fix | 1 |
| Micro-batch collector and future routing | 1.5 |
| Per-request output extraction | 1 |
| B=1 vs B>1 equivalence validation | 1.5 |
| Integration and load-test debugging | 1.5–2 |

### 6. Code/files involved

| File | Why it matters |
|---|---|
| `vis_pruner_copy/llava/serve/batch_engine.py` | Amay's. Gains the collector and the batched call |
| `vis_pruner_copy/llava/model/llava_arch.py:350-379` | Already builds the padded batch, mask, and position IDs for arbitrary B. Read it before writing anything |
| `vis_pruner_copy/llava/model/llava_arch.py:360` | The `tokenizer_padding_side` default that must become `"left"` |
| `vis_pruner_copy/llava/model/llava_arch.py:141-187` | `encode_images()` — the pruning path, written with `B` but never run at B>1 |
| `vis_pruner_copy/llava/model/builder.py` | Where to set `tokenizer_padding_side` at load |
| `csnbs/server.py` | Rithvik's. Gains `BATCH_SIZE` and `MAX_WAIT_MS` env config |

### 7. Concepts Amay needs to understand

- **Why decode is memory-bound, and why that makes batching nearly free.** Decoding reads
  the whole weight matrix to produce one token. Reading it once for eight requests costs
  about the same as reading it once for one.
- **Left vs right padding in decoder-only generation.** Why right-padding breaks generation,
  and why the failure is silent rather than an error.
- **Static batching's weakness.** A static batch runs until its longest sequence finishes.
  Short requests wait for the straggler. This is the whole argument for continuous batching,
  and WP6 is designed to expose it.
- **Cross-request contamination.** Why one request seeing another's tokens is a correctness
  and security bug, not a performance tradeoff.

### 8. Tests and benchmarks

| Run | Question it answers |
|---|---|
| B>1 smoke at B = 2, 4, 8 | Does the pruning path survive batched shapes at all? |
| B=1 vs B>1 output comparison, 90 dev images | Does batching change any answer? |
| Deliberate mixed-length batch | Does a short request get another request's tokens? |
| Batched forward latency vs batch size, no server | How does GPU time scale with B? |
| Load sweep, S2, both configs | Did the throughput ceiling move? |
| GPU utilisation during that sweep | Did the idle GPU fill? |

### 9. Success criteria

1. **Every request's output is identical to its B=1 output.** No exceptions.
2. **No cross-request contamination**, verified explicitly on mixed-length batches.
3. Throughput ceiling measurably exceeds S1's.
4. GPU utilisation at saturation rises above S1's.
5. Behaviour is stable across a sustained run.

### 10. Failure / stop / re-ranking criteria

- **If the B>1 smoke crashes** — most likely with
  `indexSelectSmallIndex: Assertion 'srcIndex < srcSelectDimSize' failed`, the same assert
  seen under multi-threaded access — **stop.** Promote the concurrency and correctness
  investigation (old Phase 0) to month-one work, reproduce under `CUDA_LAUNCH_BLOCKING=1` to
  get a real stack trace, and **descope batching.** Ship the S0-vs-S1 comparison instead.
  That is still a complete result.
- **Any cross-request contamination** → blocking correctness bug. Not a tradeoff.
- **Any output difference from B=1** → stop and fix before measuring anything.
- **If WP5 has not begun by Day 12** → cut it. See the Day-12 rule near the top.
- **If throughput does not improve at any batch size** → report that. A measured negative
  result about static batching on this workload is a legitimate finding.

**No throughput target appears anywhere in this WP, deliberately.** The utilisation headroom
shows that room exists. It does **not** imply throughput scales with utilisation. Batch
efficiency, right-padding waste, memory-bandwidth saturation at larger B, collation cost,
and scheduling delay can all break that relationship. Any number obtained by scaling
throughput by a utilisation ratio is **HYPOTHETICAL** and must never be stated as an
expectation.

### 11. Deliverables

- `batch_engine.py` with micro-batch collection and batched `generate()`
- The `tokenizer_padding_side = "left"` fix
- B>1 shape-safety smoke script and its result
- B=1 vs B>1 equivalence report across 90 dev images
- Cross-request contamination test and result
- Batched forward latency vs batch size, measured without the server
- S2 load-sweep results with GPU-utilisation curves

### 12. What Amay should be able to explain afterward

- Why does batching help decode so much more than it helps prefill?
- What breaks if you right-pad, and why does nothing raise an error?
- What already existed in LLaVA that made this smaller than it looks?
- What is the weakness of static batching, and how did you measure it?
- How do you know request 3 did not see request 5's tokens?

### 13. Time breakdown

| | |
|---|---|
| Implementation | 3.5–4 days |
| Debugging | 1.5–2.5 days |
| Validation (equivalence, contamination) | 1.5 days |
| Benchmarking and documentation | 0.5 day |
| **Total** | **7–9 days** |

---

## WP6 — Final Load Benchmark

**4 days. Depends on WP2, WP4, WP5. This is the project's result.**

### 1. Problem

Two things are wrong with the numbers we currently have.

First, the existing load sweeps ran as two separate blocks rather than interleaved, so run
order is confounded with the configuration being tested. Everything derived from them is
PROVISIONAL.

Second, we now have three server versions and no comparison across them.

### 2. Why this matters

This run produces the deliverable, and it is the run that either confirms or destroys the
premise the whole month was built on.

Measuring S0, S1, and S2 separately is what lets us say how much capacity was lost to server
structure versus gained by batching. Without S1 in the middle, a structural fix gets
credited to batching.

### 3. Evidence

- **PROVISIONAL** — the entire existing load table. Per profiling report §9.2, the sweeps
  were not temporally interleaved.
- **MEASURED** — the existing p99 figures come from as few as five samples per level, which
  is why p99 and p95 collapse together (report §9.5).
- **MEASURED** — `output_token_policy` is natural EOS, so output length varies (report §9.8).
- **MEASURED** — an earlier timing measurement excluded ~86.7 ms of image preprocessing
  (report §9.1). The timer must start before image load this time.

### 4. What we are trying to achieve

One trustworthy matrix of results, produced under conditions where run order, output length,
and sample count cannot distort the comparison.

### 5. What Amay will actually do

Run the full matrix: **{S0, S1, S2} × {576, 128} × {fixed-length, natural-EOS}**.

Design requirements, each fixing a specific known weakness:

| Requirement | Fixes |
|---|---|
| Timer starts before image load, ends after response text is produced | profiling report §9.1 timing bug |
| Trials **interleaved or randomised at the individual-trial level**, never run as blocks | profiling report §9.2 non-interleaved sweeps |
| ≥10 discarded warm-up iterations per configuration | rule 3 |
| Enough repeated trials to establish this system's noise floor first | rule 8 |
| Load levels sized so p95 has ≥100 samples; **p99 only where ≥1000 samples exist, otherwise omitted entirely** | profiling report §9.5 sample-count collapse |
| RPS levels spanning clearly-under-capacity to clearly-saturated | profiling report §9.2 |
| Achieved vs offered rate reported at every level | preserved methodology |
| Continuous GPU utilisation and memory sampling throughout | preserved methodology |
| Full CPU+CUDA profiler trace at 576 and 128, isolated and near-saturation, using the same `record_function` labels as the profiling phase so it diffs cleanly | before/after comparison |
| Total `cudaLaunchKernel` count before vs after, both configs | validates the WP2 claim |
| Exact-text comparison against the pre-optimization baseline for every non-generic change | rule 15 |

**Both output policies, explicitly labelled.** This is not optional, and here is why.

**A static batch runs until its longest sequence finishes.** Under a fixed 64-token
workload, every sequence in a batch ends together, which hides that behaviour completely —
a fixed-length-only sweep would systematically flatter WP5. Under natural EOS, a batch of
eight where one request generates 64 tokens and seven generate 10 makes those seven wait for
the straggler. That interaction is the most important tail-latency property of static
batching. It has to be measured, not assumed.

Cost is small. At ~100 s per level for ≥100 p95 samples, five levels across six
server/config cells is roughly one hour of GPU per output policy, plus model reloads. The
second policy is about **one extra hour**.

**Note the relationship to WP1.** WP1 is the *controlled* benchmark —
`min_new_tokens == max_new_tokens` removes decode-length variance so a 20–30 ms component
change is visible. It is the right instrument for the WP2 A/B and for iterating during
development. **It is not a representative serving workload and must never be described as
one.** WP6's natural-EOS policy is the representative one.

**Ownership.** Reportable numbers come from Rithvik's harness, per `CLAUDE.md`. Amay supplies
the server states and the configuration; Rithvik drives the sweep.

### 6. Code/files involved

| File | Why it matters |
|---|---|
| `scripts/run_ab_load_sweep.py` | Existing sweep driver. Extend for three server states and two output policies |
| `csnbs/measure/src/loadgen.ts` | Rithvik's open-loop load generator |
| `scripts/profile_ab_trace.py` | Produces the profiler traces to diff against the profiling phase |
| `results/timing/ab_load_gain_table.json` | The PROVISIONAL table this run replaces |
| `results/timing/loadgen_monitor/*.csv` | Existing utilisation samples for comparison |
| `results/loadgen/` | Where run artifacts land |

### 7. Concepts Amay needs to understand

- **Why interleaving matters.** Run two configs as separate blocks and thermal drift becomes
  indistinguishable from the effect you are measuring.
- **Coordinated omission.** Why the load generator must be open-loop. A closed-loop generator
  backs off exactly when the server struggles, so the queue never forms and the numbers look
  great and mean nothing.
- **Why p99 needs ~1000 samples.** With 60 samples, p99 is the maximum, and the maximum is
  noise.
- **Achieved vs offered rate.** Once achieved falls below offered, the server is saturated
  and latency numbers describe the queue, not the model.

### 8. Tests and benchmarks

| Run | Question it answers |
|---|---|
| Noise-floor run before anything else | Can this setup detect the effect at all? |
| S0 vs S1, both configs, both policies | How much capacity did bad server structure cost? |
| S1 vs S2, both configs, both policies | How much did batching add on top? |
| 576 vs 128 within each state | Does pruning's benefit change once the server can use it? |
| Fixed-length vs natural-EOS at S2 | How much does output-length variance hurt static batching? |
| Profiler trace diff vs the profiling phase | What changed at kernel level, and what did not? |
| Exact-text vs pre-optimization baseline | Did any of this change the answers? |

### 9. Success criteria

- The full matrix is populated, interleaved, with ≥100 p95 samples per cell.
- Every cell records git commit and GPU model.
- p99 appears only where the sample count supports it, and is omitted elsewhere without
  comment or apology.
- The utilisation difference between configs either reproduces or does not — **both are
  acceptable outcomes**, and the number is reported either way.

### 10. Failure / stop / re-ranking criteria

- **If the interleaved rerun does not reproduce the utilisation difference, say so plainly
  and immediately.** The original table was confounded by run order, the premise behind
  putting serving work first was wrong, and the priority order in this document must be
  rebuilt from the corrected numbers. This is the single most important honesty checkpoint
  in the month. Do not bury it.
- **If exact-text output differs from the pre-optimization baseline** in a way not already
  explained by WP2's dtype change, stop and find out why before reporting any timing.
- **If noise-floor variance exceeds the effect being measured**, the result is "not
  measurable on this setup," not a number.

### 11. Deliverables

- The full matrix: throughput, p50, p95 at {S0,S1,S2} × {576,128} × {fixed, natural-EOS}
- GPU-utilisation-vs-offered-rate curves for all three states
- Achieved-vs-offered-rate table at every level
- Profiler trace pair, isolated and near-saturation, diffable against the profiling phase
- Kernel-launch counts, before vs after
- Exact-text comparison against the pre-optimization baseline
- All run artifacts under `results/`, with commit and GPU recorded

### 12. What Amay should be able to explain afterward

- Why were the original load numbers not trustworthy?
- Why measure S1 separately instead of going straight from S0 to S2?
- What is coordinated omission, and why is the generator open-loop?
- Why is p99 missing from some cells?
- What did the natural-EOS run show that the fixed-length run could not?

### 13. Time breakdown

| | |
|---|---|
| Harness extension (three states, two policies) | 1 day |
| Debugging | 0.5 day |
| Benchmark execution | 1.5 days |
| Analysis and tables | 1 day |
| **Total** | **4 days** |

---

## WP7 — Final Writeup

**1.5 days. Depends on WP6.**

### 1. Problem

A number without an explanation is not a result. A number whose provenance is unclear is
worse than no number, because someone will build on it.

### 2. Why this matters

The project's value is that its claims are defensible. That only survives if every figure
carries its status and its source.

It is also the last chance to catch a claim that quietly outgrew its evidence — most likely
one attributing a serving gain to VisPruner.

### 3. Evidence

Everything produced by WP1–WP6.

### 4. What we are trying to achieve

A written conclusion stating plainly how much of the 60% → 13% → 4% gap was closed, by what,
and what remains open.

### 5. What Amay will actually do

1. Label every figure **MEASURED**, **DERIVED**, **PROVISIONAL**, or **HYPOTHETICAL**.
   - MEASURED: came directly out of a profiler or benchmark run.
   - DERIVED: calculated from measured values. Show the arithmetic.
   - PROVISIONAL: measured, but under conditions with a known methodological weakness.
   - HYPOTHETICAL: plausible, not verified. Never stated as an expectation.
2. Keep the four attribution claims separate (below).
3. Confirm every script used for final validation is in `scripts/` and runs from a clean
   checkout. Not one-off shell commands.
4. Write the conclusion: what closed the gap, by how much, and what did not.
5. Update this document's re-baselined numbers so it does not contradict the results.
6. Record what was deferred and why, so month two starts from evidence rather than memory.

#### The four claims, kept separate

1. **Pruning reduces compute.** MEASURED: prefill 111.5 → 44.4 ms, −60.2%. This is
   VisPruner's effect.
2. **Pruning also costs something.** MEASURED: `multimodal_prep` 49.7 → 68.4 ms, +37.8%. The
   diversity-selection loop runs ~56 iterations at 128 and zero at 576. Pruning makes the
   prep path slower. **Report this beside the gain, not beneath it.**
3. **FP16 is a generic inference optimization.** It helps 576 and 128 by a similar absolute
   amount. It must never appear inside a "VisPruner advantage" delta.
4. **Queueing and batching are serving optimizations.** A throughput gain from batching is
   **not** evidence that VisPruner improved. The honest framing: *the pruned configuration
   leaves GPU time unused; batching is the mechanism that converts unused GPU time into
   capacity.* Neither is sufficient alone — which the S0/S1/S2 × 576/128 matrix shows
   directly, because it contains the cells where each is present without the other.

For any change touching per-request latency, apply the validation formula in the Reference
section before claiming an optimization recovered VisPruner's advantage.

### 6. Code/files involved

| File | Why it matters |
|---|---|
| `AMAY_ENGINEERING_ROADMAP.md` | This file. Re-baseline its numbers against the results |
| `AMAY_VISPRUNER_PROFILING_REPORT.md` | The discipline to match, and the baseline to diff against |
| `AMAY_TRACE_PLAN.md` | Phase-A numbers that WP2 invalidates |
| `scripts/` | Every final-validation script must live here and rerun cleanly |
| `results/` | Raw artifacts backing every figure |

### 7. Concepts Amay needs to understand

- **The difference between measured and derived**, and why conflating them is how honest
  projects become dishonest ones.
- **Attribution.** Which optimizations are pruning-specific, which are generic, and which
  are serving-level.

### 8. Tests and benchmarks

No new runs. One verification: check out the repo clean and confirm every script named in
the writeup runs and reproduces its figure.

### 9. Success criteria

- Every figure carries a status label and a source file.
- The four claims are separated, and no serving gain is attributed to VisPruner.
- All scripts rerun from a clean checkout.
- The conclusion states how much of the gap closed, including where the answer is "less than
  hoped" or "not at all."

### 10. Failure / stop / re-ranking criteria

- **If a figure cannot be traced to a run**, remove it. Do not soften it.
- **If a script does not reproduce**, fix it or drop the claim it supports.
- **If WP6 invalidated the serving premise**, the writeup says that clearly and the roadmap's
  priority order is rebuilt. Reporting a failed premise is the correct outcome, not a
  setback.

### 11. Deliverables

- Final written conclusion with MEASURED / DERIVED / PROVISIONAL / HYPOTHETICAL throughout
- Re-baselined numbers in this file and `AMAY_TRACE_PLAN.md`
- All final-validation scripts in `scripts/`, verified to rerun
- A deferred-work note for month two, with the evidence that justifies each item

### 12. What Amay should be able to explain afterward

- How much of the 60% → 13% → 4% gap closed, and what closed it?
- Which of your changes helped VisPruner specifically, and which helped everything equally?
- Why is a throughput gain from batching not evidence that pruning got better?
- Which of your numbers are still provisional, and what would it take to firm them up?
- What did you not do, and why?

### 13. Time breakdown

| | |
|---|---|
| Writing | 1 day |
| Re-baselining this document | 0.25 day |
| Script reproducibility check | 0.25 day |
| **Total** | **1.5 days** |

---

## Month-One Completion Criteria

This replaces the older "Definition of Project Complete", which was written around the
Phase 0–6 ordering.

**Complete** means all of:

- WP1 landed, with a stated noise floor
- WP2 landed and validated, or explicitly reverted with the evidence that caused it
- WP3 produced a go/no-go with a quantified overlappable-CPU-work figure
- WP4 landed with byte-identical output to S0, or WP3 said don't build it and that is
  documented
- WP5 landed with B=1 equivalence proven, **or** cut by the Day-12 rule or the B>1 crash
  criterion, with the reason recorded
- WP6 run interleaved, with the matrix populated for whichever server states exist
- WP7 written, every figure labelled, every script reproducible

**Note what is not required.** No throughput target. No minimum speedup. A month that ends
with "we measured it properly and the gain was smaller than expected" is complete. A month
that ends with a large unexplained number is not.

---

## Deferred / Month-Two Engineering Work

**None of this is month-one work. It is kept here with its original detail intact so month
two starts from evidence rather than memory.**

The phase numbering below is the original numbering from earlier drafts of this roadmap. It
is retained for these deferred items only. **It is not an execution order** — the month-one
execution order is WP1 → WP7, above.

### Why each item is deferred

| Item | Original phase | Why deferred | What would bring it back |
|---|---|---|---|
| Triton diversity-loop fusion | 2a | 5–9 days, and DERIVED at ~2.4% end-to-end, only on the 128 path. Real, pruning-specific, but unaffordable alongside the serving work | Month two. It is the strongest remaining pruning-specific item and the only one that involves kernel authoring |
| Projector reorder + prefill investigation | 1b | `mm_projector` measured at 0.3–1.1 ms — DERIVED at under 1% end-to-end. The investigation half may honestly conclude "not worth pursuing" | Cheap enough to fold into month two alongside 2a, since both touch the same file |
| Persisted `Cache` object | 3a | MEASURED at 1.848 ms CPU, 0.000 ms CUDA — 0.2% of request latency. Enabling work, not a latency win | Only as a prerequisite for 3b and CUDA Graphs |
| Fixed-size KV buffer | 3b | 4–6 days. Its value is stable tensor addresses for CUDA Graphs, not the ~4% it saves directly | Together with Phase 4, as one block |
| CUDA Graphs | 4 | 6–9 days, hard-depends on 3a+3b, so 12–19 days as a chain. **Largest single-request prize in the roadmap — DERIVED at 8–16%** — but unfundable in one month | Month two or three, as a dedicated block. Run the decode roofline diagnostic first |
| Triton gather/fixup kernel | 2b | Withdrawn to Optional in an earlier revision. The region is 0.079 ms/call, 0.2% of prep GPU busy. The 63.2% figure that once justified it was a double-counting artifact | New evidence only |
| Paged KV allocator | — | MEASURED: GPU memory flat at 15.3 GB of 46 GB (33%) at every load level, both configs. No memory pressure exists to solve | Only if WP5's batching at larger B creates real KV-cache memory pressure. Measure during WP6 |
| Packed / varlen batching | — | **The premise does not hold for this pruner.** `visual_token_num` is a global scalar, so every image in a batch keeps an identical token count. Padding waste comes only from text and output-length variance | Re-derive the premise first. If WP6's natural-EOS run shows large padding waste from *output* length, that is a different and real argument — for continuous batching, not packed layout |
| vLLM port | — | Out of scope until the toy versions exist to compare against, per the original plan's own reasoning | After a batching implementation exists to compare |

### One diagnostic worth doing before committing to the KV-cache chain

Before spending 12–19 days on 3a → 3b → 4, spend **half a day** measuring decode against the
memory-bandwidth roofline.

**MEASURED:** decode is 711.8 ms for 25.2 tokens = 28.2 ms/token.
**DERIVED:** a 7B model in FP16 is ~13.5 GB of weights; the A40 has 696 GB/s of memory
bandwidth; so a purely memory-bound decode step has a floor around 19.4 ms. That puts the
current implementation at roughly 69% of the bandwidth roofline.

If decode is already near the roofline, CUDA Graphs recovers launch overhead but cannot beat
the memory wall, and the 8–16% estimate is optimistic. If it is far from the roofline, the
chain is worth the three weeks. **Deciding that with half a day of measurement instead of
three weeks of implementation is the highest-leverage thing available after month one.**

### Note on cross-references inside preserved blocks

The preserved text below and in the Reference section still cites the earlier draft's
section numbers. Those sections no longer exist under those numbers. The mapping:

| Old reference | Now |
|---|---|
| §5 (Core / Strong Extension / Optional) | "Why each item is deferred", above |
| §7 (phase-by-phase roadmap) | WP1-WP7, plus this section |
| §8 (576/128 validation formula) | Reference -> The 576/128 validation formula |
| §9 (Definition of Project Complete) | Month-One Completion Criteria |
| §10 (final benchmark suite) | WP6 |
| §11 (master table, recommended order) | Month-One Roadmap at a Glance, plus the deferral table above |

References to `§9.x` mean sections of `AMAY_VISPRUNER_PROFILING_REPORT.md`, not this file,
and remain valid.

### Original phase specifications, preserved

The four sections below are unchanged from the earlier draft of this roadmap. They contain
the detailed engineering design, evidence, dependencies, and estimates for each deferred
item.

### Phase 1b — Pruning Dataflow / Algorithm-System Co-Design

| Field | Detail |
|---|---|
| **Category** | Pruning Dataflow / Algorithm-System Co-Design (new) |
| **Problem being solved** | `mm_projector` runs on all 576 tokens before pruning discards most of them; prefill's cost doesn't scale proportionally with token count and the reason isn't yet identified |
| **Why it matters** | Directly wastes compute on discarded tokens (small in current absolute measured terms, but zero-cost and correctness-neutral to fix); the prefill-scaling gap, if understood, could unlock further gains cheaply |
| **Measured evidence** | `mm_projector` cost ~0.3-1.1ms, nearly config-independent (code-confirmed it processes all 576 tokens regardless of `visual_token_num`); prefill 117.4ms (576) vs. 51.5ms (128) — a 2.3x reduction against a ~4.2x token-count reduction |
| **Source of evidence** | `llava_arch.py:185` vs. `:271` (code); `ab_trace_vtn{576,128}_isolated_stage_breakdown.txt` |
| **Exact files/functions involved** | `llava_arch.py`, `encode_images()` (line 185) and `prepare_inputs_labels_for_multimodal()`'s single-image branch (lines 269-272); the `spatial+unpad` anyres branch (constraint: projection must stay before the `image_newline` concat on that specific path, per `AMAY_SPEED_PLAN.md`'s prior analysis, preserved here) |
| **Proposed engineering design** | (a) Reorder: apply the boolean-mask gather to the pre-projection, `mm_hidden_size`-wide features, then project only the retained tokens — for the `flat` merge type and single-image branch only, leaving the anyres `spatial+unpad` branch's existing ordering untouched per its documented constraint. (b) Separately, instrument prefill in isolation across 2-3 `visual_token_num` values to identify the specific non-scaling cost component (fixed per-call dispatch overhead vs. something else) |
| **Prerequisite knowledge** | The exact tensor shapes at each stage (`mm_hidden_size` vs. `hidden_size` — CLIP's dim vs. the LLM's dim) so the reordered gather operates on the correct shape |
| **Dependencies** | None |
| **Implementation tasks** | 1. Reorder the gather-then-project sequence for the `flat`/single-image path only. 2. Verify the anyres `spatial+unpad` branch is untouched and still passes its existing behavior. 3. Build a standalone prefill-only microbenchmark sweeping `visual_token_num` ∈ {576, 384, 256, 128} to see whether the 2.3x-vs-4.2x gap is a smooth curve or a step function (a step function would point to a specific fixed cost; a smooth sub-linear curve would point to something more structural). 4. Document findings — this task may conclude "not worth pursuing further" and that is an acceptable, honest outcome. |
| **Correctness tests** | Output text identical to pre-reorder baseline on a fixed set of test images (the reorder must not change *which* tokens get selected, only *when* they get projected); explicit check that the anyres branch's output is byte-identical to before |
| **Microbenchmark** | `mm_projector` cost alone, before/after, both configs |
| **Full-system benchmark** | Four-way (576/128 × before/after) on the full no-load request suite (reusing the corrected version of Experiment 4's methodology, §10) |
| **Before/after metrics** | `mm_projector` cost; `multimodal_prep` total; prefill time; end-to-end latency; VisPruner advantage before/after (see §8 formula) |
| **Success criterion** | `mm_projector` cost drops in proportion to `128/576` at the 128 setting (from ~constant to genuinely `visual_token_num`-scaled); no regression in 576's numbers or in output correctness |
| **Rejection/rollback criterion** | If the reorder produces any output difference on the anyres path, revert immediately — this path's ordering constraint is load-bearing, not incidental |
| **Difficulty** | Low (reorder) / Medium (prefill investigation) |
| **Estimated implementation time** | 0.5 day (reorder) + 1 day (investigation) |
| **Estimated debugging time** | 0.5 day |
| **Expected performance impact** | Small for the reorder alone; the prefill investigation's impact is unknown until scoped |
| **Deliverables** | Reordered `encode_images()`/`prepare_inputs_labels_for_multimodal()` for the flat/single-image path; a written finding on prefill's scaling behavior, whether or not it leads to further work |

---

### Phase 2 — Triton Kernel Fusion

**Milestone 2a: Diversity-loop fusion (core)**

| Field | Detail |
|---|---|
| **Category** | Triton |
| **Problem being solved** | The diversity-selection while-loop costs *more* at aggressive pruning ratios, working against pruning's own benefit |
| **Measured evidence** | ~56-57 real iterations at T=128, 0 at T=576; `argsort`×57, `max`×56, `matmul`×56, `cat`×63 vs. `argsort`×1 and 0 of the rest |
| **Source of evidence** | Derived from `ab_trace_vtn{576,128}_isolated.json` (raw trace op counts inside the `multimodal_prep` span) |
| **Exact files/functions involved** | `llava_arch.py:160-176`, inside `encode_images()` |
| **Proposed engineering design** | A Triton kernel performing the equivalent iterative deduplication (rank residual tokens by pairwise similarity, discard the most-similar-to-something-already-kept, repeat until `diverse_token_num` remain) in one launch, operating on the full `residual_indices` set at once rather than shrinking by ≤8 per Python-level pass |
| **Prerequisite knowledge** | Triton's programming model (grid/block indexing, shared memory); the exact pairwise-similarity-and-argsort algorithm currently implemented in Python, well enough to reproduce its selection *exactly* (same tokens selected, same tie-breaking) — not just something similar |
| **Dependencies** | None (independent of Phase 1a/1b) |
| **Implementation tasks (PR-sized)** | 1. Extract the current Python while-loop into a standalone, testable function taking `(image_normalized, residual_indices, diverse_token_num)` and returning the same signature, for use as the correctness reference. 2. Study `megablocks/backend/kernels.py`'s `_padded_copy`/`gather()` as a kernel-authoring template (per `AMAY_SPEED_PLAN.md`'s own reading list — preserved as sound advice). 3. Write a Triton kernel for the pairwise-similarity + argsort + selection step. 4. Handle the loop's iterative shrinking (`R` decreases by `r` each pass) — either as a single kernel that internally loops, or as a fixed small number of kernel launches replacing the ~56 Python-level ones. 5. Validate exact index-set equality against the Python reference across a range of images and `visual_token_num` settings. 6. Integrate behind a feature flag so the old path remains available for comparison. |
| **Correctness tests** | Exact selected-index-set equality (not just "similar accuracy") against the Python reference, across ≥20 test images and at least 3 `visual_token_num` settings (e.g. 64, 128, 256); explicit raster-order preservation check per the project's rule 16 |
| **Microbenchmark** | `multimodal_prep` region alone, isolated (reuse `scripts/profile_multimodal_prep.py`'s pattern), before/after, at multiple `visual_token_num` settings |
| **Full-system benchmark** | Four-way (576/128 × before/after) per §10; specifically track `multimodal_prep_mean_ms` |
| **Before/after metrics** | `multimodal_prep_mean_ms` (576, 128); kernel-launch count in that region; end-to-end latency; VisPruner advantage before/after |
| **Success criterion** | `multimodal_prep_mean_ms` at T=128 drops to at or below T=576's (reversing Experiment 4's counter-intuitive finding); `additional_realized_pruning_gain` (§8 formula) is positive and outside the project's established noise floor |
| **Rejection/rollback criterion** | Any selected-index-set mismatch against the reference on any test image; if the Triton kernel is *slower* than the Python loop at any tested `visual_token_num`, keep the Python path for that range and gate the kernel to only the settings where it wins |
| **Difficulty** | High |
| **Estimated implementation time** | 3-5 days |
| **Estimated debugging time** | 2-4 days |
| **Expected performance impact** | Moderate-significant relative to `multimodal_prep`'s own cost; small relative to total end-to-end latency (decode still dominates) |
| **Deliverables** | `triton_diversity_select.py` (new file, Amay's to write per `CLAUDE.md`'s ownership list); correctness test suite; before/after benchmark report |

**Milestone 2b: Gather/fixup kernel (OPTIONAL — demoted 2026-09-03)**

> **Demoted from Strong Extension to Optional.** The measured evidence this milestone rested on was an artifact of the double-counting bug in `profile_multimodal_prep.py` (report §9.10, Experiment 8). Corrected, the region it targets is 0.2% of the prep path's GPU time. It is kept here, fully scoped, because the CPU-side sync/launch cost is real and because the design work is already done — but it should not be built before Phase 1a (FP16 vision tower), which touches 93.1% of the same path's GPU time. Estimated payoff no longer justifies 5-8 days ahead of anything else in this document.

| Field | Detail |
|---|---|
| **Category** | Triton |
| **Problem being solved** | The original AMAY_SPEED_PLAN.md target — separate gather/argsort/cat/pad ops in the post-selection fixup path |
| **Measured evidence** | ~~63.2% of the region's CUDA time in non-matmul ops~~ **— WITHDRAWN.** Corrected: phase D (this milestone's exact target) is **0.079 ms/call, 0.2% of GPU busy**. Remaining evidence is CPU-side only: ~2.2 ms profiled wall, 9 stream syncs, 17 `nonzero`, 5 `.item()` per call |
| **Source of evidence** | `prep_summary.txt` (regenerated 2026-09-03), report Experiment 8 / §9.10, `AMAY_TRACE_PLAN.md` phase table |
| **Exact files/functions involved** | `llava_arch.py:272` (single-image gather), `:350-390` (padding/position-id/mask rebuild loop) |
| **Proposed engineering design** | As already scoped in `AMAY_SPEED_PLAN.md`: augment `encode_images()` to also return `selected_indices` directly (already computed, currently discarded into a boolean mask); write `gather_kernel()`/`triton_gather()` in a new `triton_fixup.py` doing gather + position-id remap + mask construction in one pass, operating on the shape produced by Phase 1b's reorder (post-projection width) if Phase 1b has landed, or the pre-reorder shape otherwise |
| **Dependencies** | Soft: Phase 1b's projector-reorder decision (changes the kernel's expected input width) |
| **Implementation tasks** | 1-4 as already itemized in `AMAY_SPEED_PLAN.md`'s own plan (augment `encode_images()`'s return, swap the single-image `else` branch's gather call, add the 3-way-unpack `, _` at the multi-image call site, write `gather_kernel`/`triton_gather`) — this roadmap does not re-derive that design, it cites it as still valid. 5. Extend to the anyres/multi-image branch (AMAY_SPEED_PLAN.md's own noted gap: "My augmentation only patches the single-image else branch; this whole branch is unaddressed"). 6. Address the per-batch main-loop syncs (`.sum()`, `.tolist()`) if batching (Phase 5) makes batch size > 1 relevant by this point. |
| **Correctness tests** | Output embeddings, position IDs, and attention mask bit-identical to the pre-kernel path, across single-image and multi-image/anyres inputs |
| **Microbenchmark / Full-system benchmark / Before-after metrics** | Same protocol as Milestone 2a, applied to the fixup region and to `prefill_lm_forward_mean_ms` |
| **Success criterion** | Sync count in the fixup region drops from 9 to ~0 and profiled CPU wall for the region drops measurably. **Not** a CUDA-time criterion — there is only 0.079 ms/call of GPU time available to win, which is below any plausible noise floor, so a GPU-time success criterion here would be unfalsifiable |
| **Difficulty** | High |
| **Estimated implementation/debugging time** | 3-5 days / 2-3 days |
| **Deliverables** | `triton_fixup.py`; correctness suite covering both single-image and anyres paths |

---

### Phase 3 — KV-Cache Optimization

**Milestone 3a: Persisted Cache object (core)**

| Field | Detail |
|---|---|
| **Category** | KV-Cache |
| **Problem being solved** | `DynamicCache.from_legacy_cache()` reconstructs the cache object from scratch every decode step |
| **Measured evidence** | `kv_cache_legacy_rebuild`: 768 calls in a 23-decode-step request, identical mechanism at both token counts |
| **Source of evidence** | `full_request_stage_breakdown.txt`; mechanism explained in `scripts/profile_full_request.py`'s code comments (the exact interactive stack-trace confirmation is not preserved in a file, per the profiling report's own traceability note) |
| **Exact files/functions involved** | `llava_llama.py`'s `generate()`/`forward()`; interaction with `transformers.cache_utils.DynamicCache` |
| **Proposed engineering design** | Retain the constructed `Cache` object as instance/generation-loop state across steps instead of converting to and from a legacy tuple between calls |
| **Dependencies** | None |
| **Implementation tasks** | 1. Reproduce the current stack trace with logging to confirm the exact call path before changing anything. 2. Modify `generate()`/`forward()` to pass and retain a `Cache` object directly. 3. Confirm `_validate_model_kwargs` and other HF-side machinery that inspects `past_key_values`'s type still behave correctly. 4. Remove the now-dead `from_legacy_cache` call path for this model. |
| **Correctness tests** | Generated text identical to before, across the same 10-request A/B set used in the profiling phase |
| **Microbenchmark** | Decode-step CPU time, before/after, isolated from the rest of the request |
| **Full-system benchmark** | Four-way (576/128 × before/after), tracking decode CPU time and `kv_cache_legacy_rebuild` call count specifically (expect it to drop to 0) |
| **Success criterion** | `kv_cache_legacy_rebuild` call count → 0; decode-step CPU/CUDA ratio improves at both configs by a similar absolute amount (confirming this is correctly classified PRUNING-ENABLING, not PRUNING-SPECIFIC — if the 576-vs-128 relative gain moves a lot, that's worth explaining, not expected) |
| **Rejection/rollback criterion** | Any change in generated text vs. the reference; any HF-generation-loop compatibility break |
| **Difficulty** | Medium |
| **Estimated implementation time** | 1-2 days |
| **Estimated debugging time** | 1-2 days |
| **Deliverables** | Modified `generate()`/`forward()`; before/after decode-CPU benchmark |

**Milestone 3b: KV-cache append speedup (strong extension)**

| Field | Detail |
|---|---|
| **Problem being solved** | `torch.cat`-based per-step reallocation, once the rebuild overhead above is removed |
| **Proposed design** | A preallocated, fixed-size buffer (sized to a maximum sequence length for this project's use case) that the cache writes into in place, avoiding the reallocate-and-copy pattern — this is also the direct prerequisite CUDA Graphs (Phase 4) needs |
| **Dependencies** | Milestone 3a |
| **Success criterion** | `kv_cache_append` CUDA time drops; tensor addresses become stable across steps (verified by address inspection, not just timing) — this stability check is itself the acceptance test for "ready for Phase 4" |
| **Difficulty / time** | Medium-high / 2-3 days implementation, 2-3 days debugging |

---

### Phase 4 — CUDA Graphs (strong extension)

| Field | Detail |
|---|---|
| **Category** | CUDA Graphs |
| **Problem being solved** | 34,722 kernel launches per request; decode-step CPU time exceeds CUDA time at every step |
| **Measured evidence** | Same as above |
| **Dependencies** | **Hard: Phase 3 (both milestones)** |
| **Proposed engineering design** | Capture a single decode-step forward pass (fixed-shape single-token input, fixed-shape KV buffer from Phase 3b) as a CUDA graph; replay it for every subsequent decode step within a `generate()` call |
| **Implementation tasks** | 1. Confirm KV-buffer address stability (Phase 3b's acceptance test) as a precondition. 2. Capture the graph on a representative decode step. 3. Handle the varying-length final step / EOS detection outside the captured region (graphs can't contain Python-level control flow). 4. Validate correctness across multiple decode lengths. |
| **Correctness tests** | Generated text identical across a range of `max_new_tokens` values, with and without graph capture |
| **Microbenchmark / benchmark** | Decode ms/step, before/after, both configs |
| **Success criterion** | Decode ms/step drops meaningfully at both configs, by a similar absolute amount (GENERIC classification — expect the 576-vs-128 relative gain to stay roughly flat) |
| **Rejection/rollback criterion** | Any silent correctness break (graphs can fail silently if shapes drift) — validate exhaustively before trusting speed numbers |
| **Difficulty** | High |
| **Estimated implementation/debugging time** | 3-4 days / 3-5 days (graph-capture correctness bugs are notoriously silent) |
| **Deliverables** | Graph-capture implementation; before/after decode-ms/step benchmark, both configs |

---

## Reference — Background and Historical Context

**Nothing in this section is an execution plan.** It holds the analysis the month-one plan
was derived from, plus material kept for context.

### Corrections that override the preserved text below

Two statements in the preserved material contradict the current plan. The current plan wins:

- **"The concurrency-safety audit is a precondition for any form of multi-request
  execution."** Not as stated. WP4's queue uses one worker thread at B=1, so no two threads
  ever enter the model and the multi-threading fault cannot apply. WP5 needs only a narrow
  B>1 shape-safety smoke, budgeted as its first task. The full audit is deferred. What
  remains true, and is preserved in WP5, is that the crash's root cause is **unknown** -
  device-side asserts are asynchronous, so the reported location is unreliable without
  `CUDA_LAUNCH_BLOCKING=1`. If the B>1 smoke crashes, the audit is promoted immediately.
- **"Packed/varlen batching addresses padding waste from per-image token counts."** It does
  not apply to this pruner: `visual_token_num` is a global scalar, so every image in a batch
  keeps an identical token count. See the corrections in the Month-One Execution Plan.

### Historical note — the superseded phase ordering

Earlier drafts of this roadmap organised the work as **Phase 0 → 1a → 1b → 2a → 3a → 3b →
4 → 5 → 6**, with a "Core / Strong Extension / Optional" split and a master table of
per-phase estimates.

**That ordering is superseded.** It ranked work by category rather than by measured
end-to-end value, and three of its assumptions did not survive checking:

- Phase 3a was listed as Core on latency grounds. It is MEASURED at 1.848 ms CPU and
  0.000 ms CUDA — 0.2% of request latency.
- Phase 5 (batching) was listed as a Strong Extension, gated behind Phase 0. The load data
  shows serving structure holds the largest measured headroom, and a single-worker queue
  does not need Phase 0 at all.
- Phase 5's 9–14 day estimate assumed the batched multimodal padding path had to be built.
  It already exists at `llava_arch.py:350-379`.

The current execution order is **WP1 → WP7**, defined above. Where any older statement in
this file conflicts with it, the WP order wins.

The deferred phases keep their original numbering in the section above purely as labels, so
their detailed specifications remain findable.

### The 576/128 validation formula

Still in force. Apply it in WP7 for any change touching per-request latency.

For every phase above that touches per-request latency, before claiming an optimization "recovered VisPruner's advantage," compute:

```
VisPruner advantage before = (L576_before - L128_before) / L576_before
VisPruner advantage after  = (L576_after  - L128_after ) / L576_after
additional realized pruning gain = advantage_after - advantage_before
```

Report this **alongside**, not instead of, the absolute ms improvement for each config. A PRUNING-ENABLING or GENERIC fix (Phases 0, 3, 4) that improves both configs by a similar absolute amount will correctly show `additional_realized_pruning_gain ≈ 0` — that is the expected, correct outcome for those phases, and should be reported as "reduced total latency, did not change the realized pruning advantage," not left ambiguous or oversold. Only Phases 1 and 2 (Pruning Dataflow, Triton) are expected to show a clearly positive `additional_realized_pruning_gain`; if they don't, that is itself an important, reportable finding, not a result to suppress.

Every phase's benchmark should track, where relevant: complete request latency, generate/model-path latency, multimodal-prep time, prefill time, decode time, decode ms/token, GPU time, CPU dispatch time, CUDA kernel count, GPU utilization, throughput, p50/p95 (p99 only where sample count supports it, per the project's own rule 6), and saturation behavior — the same metric set the profiling phase already established, applied consistently so before/after comparisons are apples-to-apples.

---

---

### How This Roadmap Differs From Earlier Plans

`AMAY_SPEED_PLAN.md` was written *before* the systematic profiling phase existed (per its own "Status (2026-08-29)" section, it reflects one profiling pass — `scripts/profile_multimodal_prep.py`, Experiment 1 in the profiling report). It is the right document to compare against, not to edit.

| What the old plan assumed | What profiling later established | What changed |
|---|---|---|
| `prepare_inputs_labels_for_multimodal` is "40.7%" of a 1-token prefill request, and is launch-overhead-bound (>>1 CPU/CUDA ratio implied) | **Both the old plan's framing and this roadmap's earlier correction of it were wrong** — the script producing all of these figures double-counted GPU time (report §9.10, Experiment 8). Re-measured: the region is **76.6% GEMM** on the GPU, i.e. compute-bound, not launch-overhead-bound; "everything else" is **23.4%**, and most of that is CLIP ViT softmax, not the fixup path. The CPU/GPU-busy ratio is **1.90x** (real dispatch overhead does exist — in phase B's 706 tiny kernels and 112 stream syncs, not in the fixup). The percentage-of-prefill figure is **withdrawn** in both its 40.7% and 43.1% forms (§9.11 — un-warmed denominator). | This roadmap no longer cites any percentage-of-prefill figure, and treats the region as **compute-bound with a real but separately-located CPU dispatch cost**. The change is load-bearing: it is why the gather/fixup kernel is demoted from priority 3 to 9 below. |
| The gather/fixup kernel (targeting `llava_arch.py:272`) was the primary, best-scoped Triton target | Profiling of the *whole* pipeline (Experiment 6) found a **larger, more surprising** target: the diversity-selection while-loop (`llava_arch.py:160-176`) runs ~56-57 real iterations at `visual_token_num=128` and **zero** at 576 — it is the single clearest place where VisPruner's own selection mechanism costs *more*, not less, as pruning gets more aggressive. | The gather/fixup kernel remains a valid target (Experiment 1's evidence for it still stands), but the diversity-loop fusion is now the **higher-priority** Triton target — new evidence, not present in the old plan. |
| `mm_projector` running on all 576 tokens before pruning was flagged as "a bigger, kernel-free win... at aggressive ratios this is easily >4x more projector matmul than necessary" | Measured directly: `mm_projector`'s cost is **small in absolute terms** at this model's projector size (~0.3-1.1ms, isolated trace) — real, and directionally correct to fix, but not the large win the old plan's FLOP-counting argument implied. | The reorder remains worth doing (near-zero cost, correctness-neutral, frees the fixup kernel to operate on the smaller post-projection shape), but this roadmap does not claim it moves the needle much on its own — the old plan's estimate of its size was not confirmed by measurement. |
| CUDA graph capture around the decode loop, blocked on "a preallocated, fixed-size KV cache buffer" (correctly anticipated as the hard part) | Profiling found the *specific mechanism* forcing cache instability: `past_key_values` round-trips through a legacy tuple between `generate()` steps, forcing `DynamicCache.from_legacy_cache()` to rebuild the cache object from scratch every step (768 rebuild-shaped calls measured in one 23-decode-step request), immediately before the real `torch.cat`-based append (736 calls). | The old plan's instinct (KV cache needs fixing before CUDA Graphs) is confirmed and now has a named, specific root cause rather than a general "KV cache grows every step" framing. |
| A toy paged-KV allocator and vLLM port were sequenced as steps 3-5, following CUDA graphs | GPU memory was measured nearly flat (~15.1-15.4GB) across every load level and both token configurations — the KV cache is a small fraction of the ~14GB weight footprint at batch=1, and **nothing in the load-sweep evidence shows memory pressure or a concurrency ceiling driven by memory**. | Paged KV is **not currently justified** by any measurement in this repository. It is demoted to Optional (§5), pending a future experiment that actually exercises multiple concurrent requests' KV caches at once — which nothing built so far does. |
| Hand-rolled CUDA graph capture and a toy paged-KV allocator were the named "speed" architecture; real serving concurrency was implicitly assumed to be safe to build toward | A genuine multi-thread concurrent `generate()` attempt against the shared model instance **crashed** with a CUDA device-side assert (`indexSelectSmallIndex` out-of-bounds in the pruning gather path) | This is an entirely new finding with no analog in the old plan. It motivates a **new engineering category** (§7) that did not exist before: serving correctness / concurrency safety, which now gates any batching work. |
| Batching (R4 on the project roadmap) was framed primarily as a padding-vs-packing efficiency question | The measured capacity ceiling (2.166→2.249 rps, +3.8%) shows the *current* bottleneck is that the server never runs more than one request at a time at all — padding/packing efficiency is a question that only becomes answerable *after* some form of batching exists to measure it against | Packed/varlen batching remains Optional (§5) — not because it's a bad idea, but because nothing has been built yet to expose whether padding waste is even a real problem here. |

**Old ideas that remain unchanged:** the sequencing instinct (dataflow/algorithmic work → Triton → KV cache → CUDA Graphs → batching) was directionally right and survives, refined with hard evidence for *why* each dependency exists rather than intuition. The self-test bar ("explain this from memory, with your own numbers") and the benchmarking rules in `AMAY_TIMING_NOTES.md` are unchanged and this roadmap inherits them without modification.

**New engineering categories that appeared:** two categories with no clean home in the original "Triton / KV-cache / CUDA Graphs / Batching" framing are introduced in §7 — **Pruning Dataflow / Algorithm-System Co-Design** and **Serving Correctness / Concurrency Safety**.

---

---

### Every Proposed Change, Mapped to a Category

**Classification legend** (used throughout this document):
- **PRUNING-SPECIFIC** — expected to disproportionately improve the 128-token path relative to 576.
- **PRUNING-ENABLING** — removes overhead that currently masks pruning's benefit, but the fix itself is not pruning-aware and may help 576 by a similar absolute amount.
- **GENERIC INFERENCE OPTIMIZATION** — makes the model faster without changing VisPruner's relative advantage.
- **SERVING OPTIMIZATION** — improves multi-request throughput, queueing, or tail latency; orthogonal to per-request pruning gains.

| Engineering change | Measured bottleneck | Category | Why it belongs there | Classification | Expected impact | Dependency | Priority |
|---|---|---|---|---|---|---|---|
| **Run the CLIP vision tower in FP16 instead of FP32** | Phase A (CLIP ViT forward) is **33.02 ms of 35.49 ms** per-call GPU busy — **93.1%** — at 96.3% GPU occupancy, and every ViT GEMM is an FP32 `ampere_sgemm_*` while the projector two ops later is already `c10::Half` (`AMAY_TRACE_PLAN.md` P1, corroborated by report Experiment 8) | **Pruning Dataflow / Algorithm-System Co-Design** | No kernel authoring — it is a dtype/config change on an already-saturated GEMM, the only lever available on compute-bound work | GENERIC INFERENCE OPTIMIZATION | **Large** — the single biggest measured target in the prep path | None | **1** |
| Fuse/vectorize the diversity-selection while-loop | ~56-57 iterations, ~330+ kernel launches at T=128, zero at T=576 (`ab_trace_vtn128_isolated.json` op counts) | **Triton** | Requires writing a new fused kernel to replace an iterative Python loop — this is kernel authoring, not a reorder | PRUNING-SPECIFIC | Moderate-significant | None (can start immediately) | **2** |
| Gather + position-id + mask fixup kernel | ~~63.2% non-matmul CUDA time~~ — **withdrawn (§9.10).** Corrected: the region this kernel targets (phase D — gather, splice, pad/mask rebuild) is **0.079 ms/call, 0.2% of GPU work** (`AMAY_TRACE_PLAN.md`); what remains is a CPU-side cost of ~2.2 ms profiled wall, 9 syncs, 17 `nonzero`, 5 `item` | **Triton** | Also kernel authoring; fuses several small ops (gather/argsort/cat/pad) into one launch | PRUNING-SPECIFIC | **Very small on GPU time; the case is now launch/sync reduction only** | Soft: benefits from Pruning Dataflow's mm_projector-reorder decision being made first (changes the kernel's input shape) | **9** (was 3) |
| Move `mm_projector` to run after the token-selection gather, not before | Projector runs on all 576 tokens unconditionally (`llava_arch.py:185`, before the mask at `:271`); measured cost small (~0.3-1.1ms) | **Pruning Dataflow / Algorithm-System Co-Design** | Zero kernel writing — this is purely a reorder of two existing operations | PRUNING-SPECIFIC (in direction; small in current measured magnitude) | Small | None | 3 |
| Investigate prefill's sub-linear scaling (~4.2x fewer tokens → only ~2.3x less prefill time) | 117.4ms→51.5ms vs. an ideal ~4.2x-implied ~28ms (Experiment 6) | **Pruning Dataflow / Algorithm-System Co-Design** | Diagnostic first — not yet a scoped fix; belongs with the other "how is data organized/moved" questions | Unknown until investigated | Unknown | None | 4 |
| Retain a persisted `Cache` object across `generate()` steps instead of round-tripping through a legacy tuple | `kv_cache_legacy_rebuild`: 768 calls/request, ~0ms GPU but real CPU overhead, identical at both token counts (Experiment 2) | **KV-Cache Optimization** | Directly changes cache-object lifecycle management | PRUNING-ENABLING (removes overhead equally from both configs; enables the next phase) | Small standalone; large as an enabler | None | 5 |
| Speed up the KV-cache incremental append itself (`torch.cat` per layer per step) | `kv_cache_append`: 736 calls, 11.8ms CUDA / 38.4ms CPU (576) vs. 5.6ms CUDA (128) — the one KV-cache cost that already tracks token count | **KV-Cache Optimization** | Same subsystem as the item above | PRUNING-SPECIFIC in effect (already responds to token count) but the fix itself (fusing launch overhead) is generic | Small-moderate | Soft: cleaner after the persisted-Cache-object work | 6 |
| CUDA Graph capture of the decode step | 34,722 total kernel launches in one 24-token request; decode-step CPU time exceeds CUDA time at every step, both configs, similar magnitude | **CUDA Graphs** | Textbook capture-and-replay target | GENERIC INFERENCE OPTIMIZATION | Significant (total-latency) | **Hard: requires the persisted-Cache-object work above** (stable tensor addresses) | 7 |
| Root-cause and fix the shared-model thread-safety crash (`indexSelectSmallIndex` assertion) | Genuine multi-thread `generate()` calls against the shared model instance crash (`concurrent_profile_run_log.txt`) | **Serving Correctness / Concurrency Safety** (new category) | This is a correctness/safety audit, not a performance optimization in itself | N/A — prerequisite work | N/A (unblocks capacity work) | None | **0** |
| Real tensor-level batched forward passes (multiple requests stacked into one `generate()` call) | Throughput ceiling 2.166→2.249 rps (+3.8%) despite 56-60% per-request compute cut; server verified strictly serial by reading `csnbs/server.py` | **Batching / Serving** | The actual mechanism that would let per-request savings become capacity savings | SERVING OPTIMIZATION | Significant, but for total capacity, not for the 576-vs-128 delta specifically | **Hard: requires the concurrency-safety fix above** | 8 |

---

---

### How the Categories Connect

**A. Dependency table**

| Category | Hard depends on | Soft (practical) depends on | Can run in parallel with |
|---|---|---|---|
| Pruning Dataflow / Co-Design | — | — | Serving Correctness, KV-Cache work |
| Triton (diversity loop) | — | — | Pruning Dataflow, Serving Correctness, KV-Cache work |
| Triton (gather/fixup kernel) | — | Pruning Dataflow's projector-reorder decision (changes the kernel's input shape) | KV-Cache work |
| KV-Cache (persisted object) | — | — | Pruning Dataflow, Triton, Serving Correctness |
| KV-Cache (append speedup) | — | KV-Cache (persisted object), for a cleaner target | — |
| CUDA Graphs | **KV-Cache (persisted object)** | — | Batching's early design work (not its implementation) |
| Serving Correctness audit | — | — | Everything else — do early, informs later work |
| Batching (real tensor-level) | **Serving Correctness audit** | Benefits from Triton/KV-Cache/CUDA-Graphs being done first (batches a cheaper request) but does not require them | — |

**B. Diagram, derived from the table above — two independent tracks that only meet at the very end:**

```
                     ┌─────────────────────────────┐
                     │   Serving Correctness /      │
                     │   Concurrency Safety audit   │   (Phase 0 — cheap, high-leverage,
                     │   (root-cause the crash)     │    informs both tracks below)
                     └───────────────┬───────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     │                                 │
                     ▼                                 ▼
     ┌───────────────────────────┐        ┌─────────────────────────────┐
     │  TRACK 1: per-request      │        │  TRACK 2: serving capacity   │
     │  compute (independent of   │        │  (depends on Track 0 above)  │
     │  serving architecture)     │        │                               │
     │                             │        │                               │
     │  Pruning Dataflow /         │        │                               │
     │  Co-Design (projector       │        │                               │
     │  reorder, prefill           │        │                               │
     │  investigation)             │        │                               │
     │         │                   │        │                               │
     │         ▼                   │        │                               │
     │  Triton (diversity loop,    │        │                               │
     │  then gather/fixup kernel)  │        │                               │
     │                             │        │                               │
     │  KV-Cache (persisted        │        │                               │
     │  object, then append        │        │                               │
     │  speedup) ── independent    │        │                               │
     │  of the above, same track   │        │                               │
     │         │                   │        │                               │
     │         ▼                   │        │                               │
     │  CUDA Graphs (needs the     │        │                               │
     │  persisted-Cache-object     │        │                               │
     │  work directly above)       │        │                               │
     └───────────────┬─────────────┘        └───────────────┬───────────────┘
                      │                                       │
                      └───────────────────┬───────────────────┘
                                           ▼
                          ┌─────────────────────────────────┐
                          │   Batching (real tensor-level)   │
                          │   — combines a now-cheaper        │
                          │   per-request cost with real       │
                          │   concurrency, safely              │
                          └─────────────────────────────────┘
                                           │
                                           ▼
                          ┌─────────────────────────────────┐
                          │   Final consolidated benchmark    │
                          │   suite (§10)                      │
                          └─────────────────────────────────┘
```

**Answers to the specific questions asked:**
- *Does early pruning/dataflow work need to happen before Triton fusion?* Only for the gather/fixup kernel (soft dependency — its input shape changes if the projector reorder happens). The diversity-loop fusion is independent and can start first or in parallel.
- *Is Triton independent of KV-cache work?* Yes — completely different subsystems (vision/pruning path vs. LM decode path). No shared code, no shared state.
- *Does KV-cache cleanup need to happen before CUDA Graphs?* Yes, hard dependency — graph capture requires stable tensor addresses across replay, and the current per-step cache rebuild cannot provide that.
- *Does CUDA Graph capture require static/stable buffers first?* Yes — this is exactly what the KV-Cache (persisted object) phase produces.
- *Does batching depend on solving a current correctness problem?* Yes, hard dependency — the crash demonstrates the shared model instance is not safe for concurrent access as currently written.
- *Does batching actually depend on CUDA Graphs, or can they be developed independently?* **Independent.** Batching's hard dependency is the concurrency-safety fix, not CUDA Graphs. They can be built in either order or in parallel; doing Graphs first is a *soft* preference (batch a cheaper request), not a requirement.
- *Which pieces can be implemented independently?* Pruning Dataflow, Triton (diversity loop), KV-Cache (persisted object), and the Serving Correctness audit are all mutually independent and can be started in any order or in parallel.
- *Which pieces should be sequential?* Triton (fixup kernel) after Pruning Dataflow's reorder decision; CUDA Graphs after KV-Cache (persisted object); Batching after Serving Correctness.

---

---

### New Engineering Categories

#### Pruning Dataflow / Algorithm-System Co-Design

**Why this needs its own category rather than being folded into Triton:** every item here is a question of *when* and *in what order* existing, already-correct operations run — not a question of *how* an operation is implemented at the kernel level. Forcing "move the projector call" into "Triton" would misrepresent it as kernel-authoring work; it's a two-line reorder with a correctness-preserving constraint (the `spatial+unpad` anyres branch needs projection before its `image_newline` concat, per `AMAY_SPEED_PLAN.md`'s own prior finding — this constraint is preserved, not re-litigated, in this roadmap). Justified by: `llava_arch.py:185` vs. `:271` (code), and the measured near-identical projector cost regardless of config (Experiment 6).

#### Serving Correctness / Concurrency Safety

**Why this needs its own category rather than being folded into Batching:** this work produces no latency or throughput number of its own — its deliverable is "the shared model instance no longer crashes under concurrent access." That is a prerequisite for batching, not a form of batching. Conflating them would make it impossible to benchmark-gate each piece independently (§11's one-optimization-at-a-time rule). Justified by: `concurrent_profile_run_log.txt` (2,752 lines of `indexSelectSmallIndex` assertion failures, reproduced and preserved in the profiling report's Experiment 3).

---

---

### The Four Original Areas, In Detail

#### Triton / Custom Kernel Fusion

**Exact operations targeted:** (1) the diversity-selection while-loop, `llava_arch.py:160-176`, inside `encode_images()`; (2) the post-selection gather + position-id remap + attention-mask rebuild, `llava_arch.py:272` and `:350-390`, inside `prepare_inputs_labels_for_multimodal()`.

**Profiler evidence:** For (1): `aten::argsort`×57, `aten::max`×56, `aten::matmul`×56, `aten::cat`×63 measured inside the `multimodal_prep` span at `visual_token_num=128`, vs. `aten::argsort`×1 and zero of the others at `visual_token_num=576` (`ab_trace_vtn{576,128}_isolated.json`, parsed directly). For (2): **the original evidence — "63.2% of the region's CUDA time in non-matmul ops" — is withdrawn** (report §9.10 / Experiment 8: the script double-counted GPU time). Re-measured, non-GEMM work is 23.4% of the region's GPU busy time, and the fixup region specifically (phase D) is **0.079 ms/call — 0.2%**. What survives as evidence for (2) is CPU-side, not GPU-side: ~2.2 ms profiled CPU wall, 9 `cudaStreamSynchronize`, 17 `nonzero`, 5 `.item()` in that region (`AMAY_TRACE_PLAN.md` phase D). That is a launch/sync-reduction argument of small absolute size, not the compute-share argument originally claimed.

**Is it pruning-specific?** (1) is unambiguously pruning-specific — its cost is a direct, monotonic function of how many tokens get discarded, and it is provably zero-cost at the unpruned baseline. (2) is pruning-specific in the sense that its magnitude scales with how much gets pruned, though the underlying inefficiency (many small ops instead of one fused op) is a generic pattern.

**What is being fused/restructured:** (1) replaces an iterative, `≤8`-tokens-per-pass Python loop (index_select → batched matmul → max → argsort → cat, repeated ~56 times) with a single kernel doing the equivalent iterative deduplication in one launch. (2) replaces a sequence of separate gather/argsort/cat/pad calls with one kernel producing the gathered embeddings, remapped position IDs, and rebuilt attention mask together.

**Expected benefit:** for (1), removing ~330+ kernel launches per pruned request; for (2), removing the launch-overhead majority of a region already shown to be ~43% of a prefill request.

**Benchmark required:** four-way (576/128 × before/after) on `multimodal_prep_mean_ms` specifically, plus the standard latency/prefill/decode set — see §8, Phase 2.

#### KV-Cache Optimization

**Current behavior:** `llava_llama.py`'s custom `forward()`/`generate()` do not retain a `transformers.cache_utils.DynamicCache` object across decode steps. Instead, `past_key_values` is passed back to `generate()` as a legacy tuple, and on every subsequent step, `LlamaModel.forward()` calls `DynamicCache.from_legacy_cache(past_key_values)`, which loops over all 32 layers reconstructing the cache object from that tuple — each layer's reconstruction is a cheap Python list-append (since the freshly-constructed cache object starts empty), not itself a GPU-bound operation, but 32 extra Python-level calls per step nonetheless. Immediately after, the real incremental update happens via `DynamicCache.update()`'s `else` branch: `torch.cat([self.key_cache[layer_idx], key_states], dim=-2)` — a full reallocate-and-copy, per layer, per step.

**Is `torch.cat`-based growth still present?** Yes, confirmed directly (`kv_cache_append`, 736 calls in a 23-decode-step request, 11.8ms CUDA / 38.4ms CPU at `visual_token_num=576`; 5.6ms CUDA at 128 in the isolated trace comparison). This is the one KV-cache-related cost that already responds to token count — a shorter retained context means less to copy each step.

**Proposed cache architecture:** retain the constructed `DynamicCache` object across `generate()`'s Python-level step boundary (avoiding the `from_legacy_cache()` round-trip entirely). This alone does not eliminate the per-step `torch.cat` reallocation — that requires a further step (a preallocated, fixed-size buffer that the cache writes into in place, sized to the maximum sequence length this project's use case needs), which is also the specific prerequisite CUDA Graphs needs.

**Why this matters:** it is the one KV-cache finding in this investigation that is *config-independent* (present, in nearly identical call counts, at both 576 and 128) — fixing it is PRUNING-ENABLING, not PRUNING-SPECIFIC. It should not be sold as a pruning win; it should be sold as the unblock for CUDA Graphs.

**Is static/persistent KV justified?** Yes — directly evidenced by the `from_legacy_cache` call-count measurement, and independently motivated as the CUDA-Graphs prerequisite.

**Is paged KV justified — or not yet?** **Not yet.** GPU memory was measured nearly flat (~15.1-15.4GB) across every load level and both token configurations in the load-sweep experiments — nothing in this repository's evidence shows a memory-driven concurrency ceiling. Paged KV solves a problem (memory fragmentation under many concurrent, variable-length KV caches) that this project has not yet built the conditions to observe. It belongs in §5's Optional bucket, revisited only once real batching (Phase 5) creates actual concurrent memory pressure.

#### CUDA Graphs

**Repeated decode structure:** every decode step repeats an identical sequence of per-layer operations (32 layers × {QKV projection, RoPE apply, KV-cache update, attention, output projection, MLP}) — confirmed by the near-identical steady-state per-step CUDA time across many consecutive steps in Experiment 2's trace (steps 7-23 settle to ~27.5-28.9ms each).

**Kernel launch count:** 34,722 `cudaLaunchKernel` calls for one 24-token request (Experiment 2); 156,439 for 3 back-to-back 64-token-capable requests (Experiment 3's near-saturation trace) — both measured directly.

**CPU launch overhead:** decode-step CPU time exceeds CUDA time at every single step measured, in both the 576 and 128 traces, by a similar relative margin — this is the specific signature (CPU-bound dispatch, not GPU-bound compute) that graph replay is designed to collapse.

**Prerequisites:** the KV-Cache persisted-object work above — CUDA Graphs require the same tensor addresses to be valid across every replay of the captured graph, which the current per-step cache rebuild cannot guarantee.

**Interaction with KV-cache stabilization:** graph capture should target the *steady-state* decode step (post-cache-stabilization), using the fixed-size preallocated buffer produced by the KV-Cache phase, not the current dynamically-reallocating cache.

**What should actually be captured:** the single decode-step forward pass (one call to the LM's forward, given a fixed-shape KV buffer and a fixed-shape single-token input) — not the whole `generate()` loop, and not prefill (which has a genuinely variable input shape per request and is a poor graph-capture candidate for that reason).

**Benchmark required:** decode ms/step, before/after, at both 576 and 128 — expected to improve both configs by a similar *absolute* amount (this is a GENERIC optimization; if the 576-vs-128 relative gain shifts substantially, that result would itself need explaining, not assumed).

#### Batching / Serving

**Current server execution model:** `csnbs/server.py`'s `/infer` handler (`_infer_model`) is an `async def` that calls `model.generate()` synchronously with no `await`/executor handoff — verified by reading the source, not inferred. A single uvicorn worker, single asyncio event loop, therefore services exactly one request's GPU work at a time regardless of arrival concurrency.

**Serialization/queueing evidence:** derived average in-flight concurrency (Little's Law: throughput × mean latency) grows from 0.45 at rps=0.5 to 16.17 at rps=3.0 (`ab_load_gain_table.json`); GPU utilization never exceeds 87% even at the worst load level tested — the GPU is not the hard ceiling, the serialization is.

**Why naive Python threading is not the solution:** directly demonstrated, not assumed — launching genuinely simultaneous Python threads against the shared model instance crashed with a CUDA device-side assert inside the pruning/gather path (`concurrent_profile_run_log.txt`). This rules out "just add more concurrent callers" as a fix.

**Proper tensor batching:** stacking multiple requests' input tensors (images, token sequences) into one batched forward pass, so the GPU genuinely processes several requests' work in the same kernel launches, rather than several Python-level callers racing to use the same un-batched, stateful code path.

**Prerequisites:** the Serving Correctness / Concurrency Safety audit (§4) — understanding and fixing whatever shared/mutable state in the pruning path causes the crash is a precondition for any form of multi-request execution, batched or otherwise.

**Initial basic batching design (padded, not packed):** pad every request in a batch to the longest sequence present (the simplest correct design), process the batch in one forward pass, and separate outputs by their original per-request boundaries afterward. This is deliberately not the padding-efficient (packed/varlen) version — that refinement is Optional (§5) until this basic version exists and padding waste can actually be measured against it.

**When would packed/varlen batching become justified?** Only after the basic padded-batching version above is built and benchmarked, and *that* benchmark shows meaningful throughput loss attributable specifically to padding waste (e.g. comparing batches of similar-length vs. very-different-length requests at a fixed batch size). Nothing in this repository currently measures that.

---
