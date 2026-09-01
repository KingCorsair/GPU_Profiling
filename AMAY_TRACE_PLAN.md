# Trace-driven speed plan (Amay)

Supersedes the ordering in `AMAY_SPEED_PLAN.md`. That doc was written before profiling; this
one is written from `results/timing/prep_trace.json` and re-orders accordingly. The old doc's
*reasoning* mostly survives; its *priorities* do not.

Source of truth: `results/timing/prep_trace.json` (46MB chrome trace, 143,661 events),
produced by `scripts/profile_multimodal_prep.py`. NVIDIA A40 (sm86, 84 SMs, 47.7GB),
`visual_token_num=128`, `important_ratio=0.5`, batch=1, prompt_len=66, 10 warm-up + 10
profiled iterations, single image, single-image code path only.

---

## Measured baseline

One call to `prepare_inputs_labels_for_multimodal`: **48.0 ms real wall**, **35.49 ms GPU busy**,
1319 GPU events. GPU busy was stable to ±0.13 ms across all 10 iterations.

| Phase | Code | GPU busy | % GPU | GPU events | CPU-side character |
|---|---|---|---|---|---|
| **A** CLIP ViT forward | `encode_images` :142 | **33.02 ms** | **93.1%** | 561 | GPU 96.3% busy — saturated, compute-bound. All FP32. |
| **B** pruning selection | `encode_images` :154-183 | 2.06 ms | 5.8% | 706 | GPU **94.5% idle** over a 37.5 ms span. Avg kernel **2.94 µs**. **112 `cudaStreamSynchronize`**, 9.49 ms host-blocked. 5630 aten ops, 857 runtime calls. |
| **C** `mm_projector` (576 tok) | `encode_images` :185 | 0.32 ms | 0.9% | 3 | negligible |
| **D** gather + splice + pad/mask | `prepare_inputs...` :272, :296-390 | **0.079 ms** | **0.2%** | 49 | 2.18 ms profiled CPU wall. 9 syncs, 17 `nonzero`, 5 `item`. |

**The overhead budget is 12.5 ms.** Real wall (48.0) minus GPU busy (35.5). That is the entire
prize available to every launch-overhead / sync / fusion fix in this document combined, and
essentially all of it sits in phase B.

**Everything else is a GPU-work problem, and 93% of the GPU work is phase A.**

### Two numbers in `prep_summary.txt` are wrong — don't quote them

`profile_multimodal_prep.py` sums `self_cuda_time_total` over `prof.key_averages()`, which
contains both aten-level rows (`aten::addmm`) and kernel-level rows (`ampere_sgemm_128x64_tn`).
It double-counts; its own table footer says 354.9 ms where the script prints 739.6 ms.

- `63.2% everything else (gather/index/cat/argsort/pad)` — wrong. The classifier only matched
  `aten::` names, so every `ampere_sgemm_*` / `sm80_xmma_*` row landed in "everything else."
- `CPU time / CUDA time ratio: 1.01x` — wrong. Actual 744/355 = **2.10x**.

Fixed in P0. Every other line in that summary holds.

### Profiler-inflation caveat, applies to every CPU number below

Profiled wall 74.9 ms vs real 48.0 ms → ~3.1 µs of overhead per aten op across 8597 ops.
**GPU-side numbers are trustworthy. Every CPU-side number in the trace is an upper bound.**
Where a CPU figure is quoted below it is labelled "(profiled)".

---

## Priority order and why it changed

| | `AMAY_SPEED_PLAN.md` order | Trace-driven order | Reason for the move |
|---|---|---|---|
| 1 | Triton gather kernel at `:272` | **FP32→FP16 vision tower** | Not in the old plan at all. Touches 93.1% of GPU time. |
| 2 | (projector reorder, "bigger win") | **`torch.arange` device fix** | Not in the old plan. 112 stream syncs/request, one line. |
| 3 | CUDA graphs | **Pruning loop restructure** | Was listed as "noted for later, not part of this task." It is the second-largest cost. |
| 4 | Paged KV | **Projector reorder** | Correct reasoning, but it is 0.9% of GPU time, not "a bigger FLOP-level win." |
| 5 | Packed batching | **Triton kernel — retargeted at the loop** | The gather at `:272` is ~10 µs of GPU time. See P5. |

---

## P0 — Noise floor and a repeatable prep microbenchmark

**Problem being fixed.** Nothing in this document is currently measurable. The profile is N=10,
one run, one config, unreplicated. Rule 8 says if run-to-run variance exceeds the effect size,
nothing has been measured — and the variance is unknown. The existing sweeps in `results/`
carry 2–5 questions per setting, far below rule 6's ~1000 samples for a stable tail.
Additionally `profile_multimodal_prep.py` prints two wrong aggregates (above).

**Why it matters.** P4 and P5 have expected effects of ~0.25 ms and <1 ms against a 48 ms
baseline. If the noise floor is 5%, those are unmeasurable and should not be built on a
latency justification. You need the number before you can decide.

**Exact code location.** `scripts/profile_multimodal_prep.py` (aggregate block, ~lines 130-150);
`vis_pruner_copy/llava/model/language_model/llava_llama.py` `forward()` ~:231 (`prep_start_evt`/
`prep_end_evt` already record `multimodal_prep_time` per forward call) and `generate()` ~:179-209
(`multimodal_prep_time_generate`, currently `perf_counter()` + `synchronize()`).

**Implementation change.**
1. Fix the double-count: classify by whether the event is an aten-level op or a kernel-level
   row, and sum exactly one level. Report GPU busy from kernel/memcpy/memset rows only.
2. Add a phase segmentation to the script's output (A/B/C/D as defined above), so every later
   change reports which phase moved. Segment markers: `aten::mean` (start of B), `aten::scatter_`
   (end of B), last `aten::linear` (end of C).
3. Switch `generate()`'s top-level number to read `forward_calls[0]["multimodal_prep_time"]`
   (the prefill entry, already recorded off CUDA events) instead of `perf_counter()`.
4. Run the unchanged baseline 10× at fixed config, randomised order (rule 10), report p50/p95
   and run-to-run spread. Record git commit + GPU model (rule 7).

**Custom kernel work?** No.

**Correctness risks.** None — measurement only.

**Benchmark afterward.** This *is* the benchmark. Output: the noise floor as a percentage.

**Definition of done.** A stated noise floor number, in this file, e.g. "prep p50 = 48.0 ms,
run-to-run spread 2.1%." Any later change smaller than that number is declared unmeasurable
rather than reported as a win.

**Scope.** Small. Half a day.

**Dependencies.** None. Blocks the "done" criterion of every other item.

> **Ownership note.** Core timing/profiling code is yours per CLAUDE.md, not something I write
> unprompted. This item is specified, not implemented. Ask explicitly if you want it written.

---

## P1 — FP32 → FP16 vision tower

**Problem being fixed.** The CLIP ViT runs in FP32. Trace evidence: all 144 ViT `aten::addmm`
rows carry `Input type: ['float','float','float']`; the patch embed is
`implicit_convolve_sgemm<float,...>`; the GEMM kernels are `ampere_sgemm_128x64_tn` (480 calls)
and `ampere_sgemm_32x128_tn` (960 calls) — the FP32 SGEMM family. Two ops later the projector is
`c10::Half` / `sm80_xmma_gemm_f16f16`, so this is the vision tower specifically, not the model.

**Why it matters.** Phase A is 33.02 ms = 93.1% of all GPU work in prep, and it is genuinely
compute-bound (96.3% GPU busy — no bubbles to fuse away). The only lever on a saturated GEMM
phase is doing the arithmetic in a cheaper precision. ViT-L/14@336 is ~381 GFLOP per forward;
33.0 ms is 11.5 TFLOP/s against an A40 FP32 peak of 37.4. FP16 tensor-core peak is 149.7.

**Exact code location.** `vis_pruner_copy/llava/model/builder.py:151-156`:
```python
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model(device_map=device_map)   # CLIPVisionModel.from_pretrained, no torch_dtype -> fp32
if device_map != 'auto':
    vision_tower.to(device=device_map, dtype=torch.float16)   # never runs
```
`device_map` defaults to `"auto"` (`builder.py:26`) and **every** caller uses the default —
`model_vqa_science.py:47`, `model_vqa_heterogeneous.py:82`, `model_vqa_loader.py:85`,
`model_vqa_mmbench.py:60`, `model_vqa.py:35`, `run_llava.py:55`,
`scripts/profile_multimodal_prep.py:34`. The cast is dead code in every run ever made.
Downstream, `multimodal_encoder/clip_encoder.py:64` does `images.to(dtype=self.dtype)`, so the
`.half()` input tensor is *upcast* to FP32 on entry, and `:70` casts the output back down.

**Implementation change.** Load the vision tower in the model's dtype rather than casting after
the fact — pass `torch_dtype` through `load_model()` into `CLIPVisionModel.from_pretrained`, so
the weights are never materialised in FP32. Then the `device_map != 'auto'` guard becomes
irrelevant rather than needing a patch. Keep `clip_encoder.py`'s `self.dtype` plumbing as-is;
it will now report `torch.float16` and the input upcast disappears on its own.

**Custom kernel work?** No.

**Correctness risks.** The highest-consequence item in this document for accuracy.
- Token selection is driven by `[CLS]` attention (`encode_images:154-156`:
  `image_attentions.mean(dim=1).argsort(descending=True)`). FP16 attention probabilities plus an
  FP16 mean over 16 heads will perturb the ranking. Near-tied tokens will swap. **The selected
  token set will not be bit-identical.**
- FP16 overflow in the ViT is a known-safe path (this is how LLaVA ships), but the
  `output_attentions=True` route is not the shipped path — attention probs are being read out
  rather than consumed by a fused kernel. Check for NaNs explicitly.
- **Must validate:** (a) Jaccard overlap of `selected_indices` FP32 vs FP16 across ≥200 dev
  images, at `visual_token_num` ∈ {64, 128, 576}; (b) end-to-end accuracy delta on the **dev**
  split only — rule 11, the test split stays locked; (c) this is Sribhav's axis, so the accuracy
  call is his, not yours.
- At `visual_token_num=576` the masks are all-True and selection is a no-op, so the 576 baseline
  should show a pure-numerics delta only. That is a useful control: run it first.

**Benchmark afterward.**
- Must improve: phase A GPU busy, 33.02 ms → target < 15 ms. `multimodal_prep_time` 48 ms →
  target < 30 ms.
- Must stay flat: `lm_forward_time` (the LLM was already FP16 — if this moves, something else
  changed). Token-selection Jaccard ≥ some threshold you and Sribhav agree on up front.
- Watch: peak GPU memory should *drop* (~600 MB of ViT weights), which is an R5 input.

**Definition of done.** Phase A GPU busy re-measured from a fresh trace showing FP16 kernels
(`sm80_xmma_*` / `*_f16f16_*`, no `ampere_sgemm_*` in the ViT), prep p50 down by more than the
P0 noise floor, `lm_forward_time` unchanged within noise, and a signed-off accuracy delta on dev.

**Scope.** Small to write (a handful of lines), medium to validate. The validation is the work.

**Dependencies.** P0 for the "done" criterion. Do this before P2/P3 — it changes the baseline
every later measurement is taken against, and it changes the phase A/B ratio that justifies them.

**Consequence for the project, not just for speed.** Every number in `results/` was produced
with an FP32 vision tower. `scienceqa_timing_sweep.jsonl`, `heterogeneous_timing_sweep.jsonl`,
`scienceqa_acc_vs_tokens.png` — all of them. They are not wrong, but they are not the
configuration you will report. Tell Rithvik before he runs anything for the chart.

---

## P2 — `torch.arange(B)` on the CPU inside the pruning loop

**Problem being fixed.** 112 `cudaStreamSynchronize` calls per prep request, 9.49 ms of host
blocking (profiled), all located inside `aten::copy_`, all in phase B — matched one-for-one by
112 `Memcpy HtoD (Pageable → Device)` in the same phase.

**Why it matters.** `torch.arange(B)` with no `device=` builds a CPU tensor. Using it as an
index operand forces a pageable host-to-device copy, and CUDA performs a **full stream
synchronize before initiating a pageable H2D transfer**. So the pipeline is drained 112 times
per request. This is *the* mechanism behind phase B's 94.5% GPU idle: the CPU cannot run ahead
and queue work, because every loop iteration blocks it twice until the GPU catches up. It is
also why the 12.5 ms wall-minus-GPU-busy gap exists at all.

**Exact code location.** `vis_pruner_copy/llava/model/llava_arch.py`, `encode_images()`:
- `:167` — `image_normalized[torch.arange(B).unsqueeze(-1).expand(-1, R), residual_indices]`
- `:174` — `residual_indices[..., ::2][torch.arange(B).unsqueeze(-1).expand(-1, R // 2 - r), distinct_indices]`

Two per loop iteration × 56 iterations at `visual_token_num=128, important_ratio=0.5` = 112. Exact match.

**Implementation change.** Hoist a single batch-index tensor out of the loop, constructed once
on the correct device (`device=image_features.device`), and reuse it — reshaping/expanding
rather than reallocating. It is `arange(B)` with `B=1` in every path currently profiled; the
tensor is trivial, the cost is entirely the transfer and the sync it forces.

**Custom kernel work?** No. This is a one-line-per-site change.

**Correctness risks.** Essentially none — the index values are identical, only their residency
changes. Assert bitwise-identical `selected_indices` before and after on a fixed image. If they
differ, something else is wrong.

**Benchmark afterward.**
- Must improve: `cudaStreamSynchronize` count per prep call, 121 → 9 (the 9 remaining are in
  phase D, item P6). Pageable H2D memcpys 112 → 0. Wall-minus-GPU-busy gap, 12.5 ms → target < 7 ms.
- Must stay flat: GPU busy total (this removes *stalls*, not *work* — if GPU busy moves, you
  changed something you didn't mean to). Accuracy: unchanged, bitwise.

**Definition of done.** Fresh trace showing zero pageable H2D in phase B and ≤9 stream syncs
per call, with `selected_indices` asserted identical to the pre-change output.

**Scope.** Small. An hour including the trace re-run.

**Dependencies.** P0 for measurement. Independent of P1, but measure it *after* P1 lands so the
gap is quoted against the FP16 baseline.

---

## P3 — Restructure the pruning while-loop

**Problem being fixed.** Phase B: 706 GPU events and 857 CUDA runtime calls to perform **2.06 ms**
of actual GPU work, spread over a 37.5 ms span at 94.5% idle, average kernel duration 2.94 µs.
Confirmed op counts per call: 56 `aten::matmul`, 56 `aten::max`, 57 `aten::argsort`,
112 `aten::index`, 56 `CatArrayBatchedCopy`, 174 DtoD memcpys, 5630 aten ops total.

**Why it matters.** Two separate mechanisms, and they need separating because they have
different fixes:
1. **Launch/dispatch count.** ~8 chained ops × 56 iterations. Each op is 2.94 µs of GPU work
   behind ~3-10 µs of CPU dispatch. This is what fusion fixes.
2. **Redundant memory traffic.** Every pass re-gathers the full `(B, R, C)` residual block out of
   `image_normalized` from scratch (`:167`), where R walks 512 → 64. That is ~33 MB of DtoD
   traffic per request to maintain a list that only ever shrinks by 8 elements per pass. This is
   what a compacted live buffer fixes, no kernel needed.

**Exact code location.** `llava_arch.py:161-176`, the `while diverse_token_num > 0:` block.

**Loop shape, for the record.** `r = min(8, R - diverse_token_num)` caps removals at 8 per pass.
At `visual_token_num=128, important_ratio=0.5`: `T_imp=64`, `T_div=64`, residual starts at 512,
runs (512-64)/8 = **56 passes**. Note the pass count *grows* as pruning gets more aggressive —
at `visual_token_num=576` the loop breaks immediately (0 passes), at 64 it runs 64 passes. The
profile was taken at the 56-pass point.

**Implementation change (design level, semantics-preserving).**
1. **Gather once, compact in place.** Materialise `residual_tokens` once before the loop and
   maintain it as a live compacted buffer, rather than re-indexing `image_normalized` each pass.
   Kills 56 gathers and the bulk of the 174 DtoD memcpys.
2. **`topk` instead of `argsort`.** `scores.argsort(descending=True)[:, r:]` sorts all R/2
   elements to discard r=8 of them. Only the top-r are needed. Replaces a full radix sort with a
   bounded selection.
3. **Fuse `a @ b.T` → `.max(dim=-1)`.** The `(B, R/2, R/2)` score matrix is materialised only to
   take a row-wise max. At R=512 that is a 256×256 intermediate written and re-read for a
   256-element result. Candidate for P5-v1.
4. **Do not change `r`.** Raising the per-pass removal count from 8 would collapse 56 passes into
   ~7 and is by far the largest launch-count win available — but it **changes the algorithm's
   output**, not just its speed. If you want it, it is a method change and belongs to Sribhav's
   axis with a full dev-set evaluation, reported separately. Do not fold it into a speed PR.

**Custom kernel work?** Items 1-2 no. Item 3 is the natural Triton target — see P5.

**Correctness risks.**
- Items 1-2 must be bitwise-identity-preserving. `topk` vs `argsort` tie-breaking can differ when
  scores are exactly equal; with FP16 cosine similarities over 1024 dims, exact ties are
  plausible. Assert identical `selected_indices` on a fixed set of ≥100 dev images and
  investigate any mismatch rather than waving it through.
- **`residual_indices` is in score order, not raster order, and this matters later.**
  `argsort(descending=True)[:, r:]` returns survivors ordered by score, and `:174` reorders the
  index list accordingly. Raster order is restored today only because `:183` scatters into a
  boolean `index_masks` and `:272` does `image_features[index_masks]`, which returns elements in
  ascending index order. Any refactor that keeps the *index* path and drops the *mask* path
  silently violates rule 16. See P4/P5 — this is the single highest-risk trap in the refactor.

**Benchmark afterward.**
- Must improve: GPU events per prep call 1319 → target < 700; phase B aten op count 5630 →
  target < 1500; wall-minus-GPU-busy gap → target < 2 ms.
- Must stay flat: phase B GPU busy should *drop* modestly (less DtoD traffic) but not vanish —
  2.06 ms → ~1.5 ms. `lm_forward_time` unchanged. Accuracy bitwise-identical.

**Definition of done.** Fresh trace with the above counts, `selected_indices` asserted identical
across a 100-image dev sample, and prep p50 improvement exceeding the P0 noise floor.

**Scope.** Medium. Two to three days including validation.

**Dependencies.** P2 (which is a strict subset of this work — do it standalone first so its
effect is separately attributable). P1 first, so the ratios are quoted against the real baseline.

> **Ownership note.** This is the token-removal implementation, which CLAUDE.md lists as yours.
> Specified here, not written. Confirm explicitly if you want it implemented for you.

---

## P4 — Prune before the projector

**Problem being fixed.** `mm_projector` runs on all 576 tokens (`encode_images:185`, trace shows
`aten::addmm` on `[576, 1024] × [1024, 4096]` then `[576, 4096] × [4096, 4096]`) before the
mask/gather down to T kept tokens happens later at `:272`.

**Why it matters.** The FLOP argument in `AMAY_SPEED_PLAN.md` is correct — projecting 576 tokens
to produce 128 is 4.5× more matmul than necessary, and the token-pruning literature places
pruning between the ViT and the projector for exactly this reason. **But the trace sizes it at
0.32 ms, 0.9% of GPU time.** Two linears against 24 transformer layers. The old plan's claim that
this is "a bigger FLOP-level win than the fixup kernel itself" is true and both are noise.

**Exact code location.** `llava_arch.py:185` (`mm_projector` call) and `:272`
(`image_features[index_masks].unsqueeze(0)`), plus the 3-way unpack at the multi-image call site
~`:202`.

**Implementation change.** Move the mask/gather to before `:185` so the projector sees only kept
tokens, and drop the gather at `:272`. The gathered tensor becomes `mm_hidden_size`-wide (1024)
instead of `hidden_size`-wide (4096) — a 4× narrower gather, which also shrinks P5's target
further. **Per-branch, not global:** the `spatial`+`unpad` anyres branch concatenates
`image_newline` (LLM-dimensional) before the mask/reshape logic completes, so on that path the
projection has to stay where it is. Clean for `flat` merge type and the single-image branch.

**Custom kernel work?** No.

**Correctness risks.**
- **Rule 16, again, and this is where it bites.** If you gather with `selected_indices` you get
  score order. Today's `image_features[index_masks]` gives raster order. The replacement must
  sort the indices ascending first, and the test is an exact-equality assert against the current
  boolean-mask output — not an eyeball check of the accuracy number, which would hide it.
- The projector is position-agnostic (a per-token MLP), so projecting before vs after selection
  is mathematically identical up to FP16 rounding. Assert `allclose` with a tight tolerance.
- Do not land this at the same time as P1. Both perturb FP16 numerics; landed together you
  cannot attribute an accuracy delta to either.

**Benchmark afterward.**
- Must improve: phase C GPU busy 0.32 ms → ~0.07 ms at `visual_token_num=128`. Expect ~0.25 ms
  total — **likely at or below the P0 noise floor**, so report it as a FLOP-count result with a
  latency measurement that may be indistinguishable from zero. That is an honest finding, not a
  failure.
- Must stay flat: accuracy exactly (modulo FP16 rounding), `lm_forward_time`, phases A/B/D.

**Definition of done.** Projector `addmm` input shapes in a fresh trace read `[128, 1024]`
rather than `[576, 1024]`, and the gathered output is asserted exactly equal to the previous
boolean-mask result on ≥100 dev images.

**Scope.** Small to medium. The anyres branch split is what makes it not-small.

**Dependencies.** P1 (numerics attribution). Should land before P5, since it changes the shape
and width of what any gather kernel operates on.

---

## P5 — Triton kernel: retarget from the final gather to the pruning loop

### The retargeting decision

`AMAY_SPEED_PLAN.md` targets `image_features[index_masks]` at `:272`. The trace does not support
that as a kernel target:

| Candidate | GPU events | GPU time | Kernel-fusion upside |
|---|---|---|---|
| `:272` gather (old target) | 3 `aten::index` ops in phase D; the `[1,576,4096]` one is **~10 µs** | phase D total **79 µs** | Removing phase D entirely = 0.2% of GPU time, ~1.6 ms of CPU dispatch (profiled) |
| Pruning loop `:161-176` (new target) | **706** | **2.06 ms**, 94.5% idle, 2.94 µs/kernel | Collapses ~450 launches and 56 materialised `(R/2, R/2)` intermediates |

**The pruning loop is the stronger target by roughly two orders of magnitude on every axis.** It
is also the better interview artifact: a fused similarity-reduction kernel with a data-dependent
loop is a substantially more interesting thing to have written than a row gather, and unlike the
gather it produces a number that survives a noise-floor check.

Additionally, after P4 the `:272` gather stops existing in its current form. Building a kernel
for it and then deleting the call site is wasted work.

**Recommendation: build the Triton kernel against the pruning loop. Keep `:272` as a plain
PyTorch gather.** If you still want a gather kernel for its own sake, do it last and justify it
as a learning exercise, not as a latency fix — and do not use `multimodal_prep_time_generate` as
its success criterion, because it will not move measurably.

**Prerequisite status:** `triton==2.2.0` is already pinned at `requirements.txt:91` (commit
b1e4b2e). `AMAY_SPEED_PLAN.md`'s "Triton isn't installed anywhere" note is stale. Verify in the
current image with `python -c "import triton; print(triton.__version__)"` before starting.

### Staged kernel plan

#### v0 — reference PyTorch behaviour (no Triton)

**Purpose.** Freeze the semantics the kernel must reproduce, before any of it moves.

**Deliverable.** A standalone reference function, outside the model, that takes
`(image_normalized, residual_indices, diverse_token_num)` and returns the final
`residual_indices` — a faithful extraction of `:161-176` with nothing changed. Plus a fixture:
saved input tensors and expected output for a handful of real images at
`visual_token_num` ∈ {64, 128, 288}, covering 64, 56, and 36 loop passes.

**Why this stage exists.** Rule 19 — prototype on something fast. A bug in the kernel should
cost seconds against a saved fixture, not a 40-minute model reload and rerun.

**Done when.** The reference reproduces the in-model `selected_indices` bitwise on the fixture.

**Scope.** Small.

#### v1 — smallest correct Triton kernel: fused similarity row-max

**Target.** Steps 3-4 of each pass only:
```
scores = a @ b.transpose(-1, -2)       # (B, R/2, R/2)  materialised
scores = scores.max(dim=-1).values     # (B, R/2)       all that's actually wanted
```

**What the kernel does.** One program per row-block of `a`. Loads a tile of `a`, streams tiles of
`b` through, accumulates the dot products, keeps a running max across the `j` dimension, writes
one value per row. The `(R/2, R/2)` intermediate is never written to global memory. Structurally
this is the fused-softmax tutorial with `max` as the reduction and no second pass — the right
second Triton kernel to write after vector-add.

**Why start here.** It is one kernel with one output, a trivially checkable reference
(`(a @ b.mT).max(-1).values`), and it removes the largest intermediate in the loop. It does not
touch the index bookkeeping, so it cannot break rule 16.

**Correctness tests.**
- `torch.allclose` against the v0 reference on the fixture, FP16 tolerance stated explicitly.
- Shape sweep: R/2 ∈ {32, 33, 128, 256, 257} — non-power-of-2 and non-tile-multiple sizes are
  where masking bugs live. R walks 512→64 in steps of 8, so odd tile remainders occur on nearly
  every pass.
- B > 1, even though nothing currently exercises it, so the kernel does not become a batch-1
  assumption baked into the project.
- A degenerate case: R/2 < one tile.

**Latency benchmark.** Isolated, against the fixture, CUDA events, 10 warm-up discarded, sync
outside the loop (rules 1-4). Compare kernel vs `(a @ b.mT).max(-1).values` across the R values
actually seen. Then in-model: phase B GPU event count and GPU busy.

**Done when.** Bitwise-or-allclose match on all fixture shapes, and phase B GPU events drop by
~112 (the 56 matmuls and 56 max reductions) with no accuracy change.

**Scope.** Medium. This is the real learning.

#### v2 — fused pass: row-max + top-r + index compaction

**Target.** An entire loop pass in one launch: similarity row-max, select the r most-redundant
rows, and emit the compacted `residual_indices` — replacing ~8 launches per pass with 1.

**What it adds over v1.** The top-r selection and the index compaction, which means the kernel
now owns index bookkeeping and *can* violate rule 16. The output ordering must be specified
explicitly and asserted, not assumed.

**Open design question to settle before writing it.** v2 still launches 56 times because the pass
count is data-dependent (`R` shrinks by `r` each pass, and `r` depends on `R`). A v3 that pushes
the loop *inside* one persistent kernel would take it to a single launch — that is the more
impressive version and the harder one. Decide whether v2 is the stopping point based on what v1
and v2 actually measure; do not commit to v3 up front.

**Correctness tests.** Everything from v1, plus: exact `selected_indices` equality against the v0
reference across the full fixture, and an explicit assertion about output ordering (score order
vs raster order) with a comment saying which one downstream code depends on.

**Latency benchmark.** Phase B GPU events 706 → target < 150. Wall-minus-GPU-busy gap → < 1 ms.

**Done when.** Exact index equality on the fixture, event counts hit, prep p50 improvement above
the noise floor, `lm_forward_time` flat.

**Scope.** Large.

**Dependencies.** v1, P3 (restructure first — do not build a kernel for a loop you are about to
reshape), P0 for the noise floor.

> **Ownership note.** Triton kernels are yours per CLAUDE.md. This is a specification. The
> confirm-once gate applies before I write any of v1/v2.

---

## P6 — Batch-dependent splice, padding, and mask rebuild

**Problem being fixed.** The forced CPU-GPU sync pattern in the per-batch-item loops. Trace
confirms the pattern exists in phase D: 17 `aten::nonzero`, 5 `aten::item`, 4
`_local_scalar_dense`, 9 `cudaStreamSynchronize`. It also confirms the multi-op structure of the
pad/mask rebuild: 49 GPU events for 79 µs of work.

**Why it matters — and why it is ranked here rather than higher or lower.** At batch=1 the whole
of phase D is 0.079 ms of GPU time and 9 syncs costing 0.02 ms. On the profile in hand it is
irrelevant. **But the profile cannot see this item's actual cost**: `BATCH_SIZE = 1` is hardcoded
in `profile_multimodal_prep.py:16`, and the script's own comment explains why — `:272` does
`image_features[index_masks].unsqueeze(0)`, which collapses B into the token dimension, so there
is no batched path to profile yet. The sync count in these loops scales with batch size (per-item
`.sum()` read into a Python `if`, and `.tolist()`), so a cost that is 9 syncs at batch=1 is ~144
at batch=16. This is the one item in the document that is plausibly *under*-measured rather than
over-measured, and the trace is silent on it by construction.

**Exact code location.** `llava_arch.py`:
- `:296-297` — `input_ids`/`labels` masked-gather per batch item, before the main loop
- `:302-341` — per-batch-item splice loop; syncs via `.sum()` into a Python conditional and `.tolist()`
- `:350-390` — `max_len` computation (`max(x.shape[0] for x in ...)`, a host-side reduction over
  GPU tensors), then `torch.zeros` + slice-assignment rebuild of `attention_mask` and
  `position_ids` per item

**Implementation change.** Deferred pending measurement — but the shape is: replace per-item
Python iteration with batched ops (a single padded scatter, an `arange` broadcast for position
IDs, a comparison-based attention mask), so the number of syncs becomes O(1) rather than O(batch).
This is also the "position-id remap + mask construction" half of the original problem statement.

**Custom kernel work?** Possibly, as a fused pad+position+mask kernel — but only if the batched
PyTorch version still shows a problem. Do not start with the kernel.

**Correctness risks.** High-surface, low-subtlety: sequence assembly, padding side, position ID
continuity across the spliced image span, and label alignment. Every one is exactly-checkable
against the current implementation's output. Also note this loop is where variable-length
batching (R4) will land, so changes here are load-bearing for a headline result.

**Benchmark afterward.** Cannot be specified until the batch>1 profile exists. Provisionally:
`cudaStreamSynchronize` count per prep call must become independent of batch size.

**Definition of done, stage 1 (do this now).** A profile at batch ∈ {1, 4, 8, 16} showing how
sync count, phase D GPU time, and prep wall scale with batch. **Then** decide whether to build
anything. If sync count is flat in batch, close this item.

**Scope.** Stage 1 small. The fix itself medium.

**Dependencies.** Requires a batched code path to exist at all — `:272`'s `.unsqueeze(0)` means
one does not. That makes this partly blocked on the batching work in `AMAY_SPEED_PLAN.md` #4,
which is a larger reordering than this document covers.

---

## P7 — Anyres / multi-image path

**Problem being fixed.** Unknown. This is the honest answer.

**Why it matters.** `llava_arch.py:198-269` repeats the masked-gather pattern per image tile in a
Python loop, with `permute` / `unpad_image` / `cat` per tile. Structurally it looks like phase B's
problem multiplied by tile count. But **the trace exercises only the single-image `else` branch** —
`profile_multimodal_prep.py` sends one 336×336 image, so this branch never executes and there is
zero evidence about its cost.

**Exact code location.** `llava_arch.py:198-269`, plus `:211` (`index_masks = [x.flatten(0, 1) for x in index_masks]`).

**Implementation change.** Not specifiable until profiled.

**Custom kernel work?** Unknown.

**Correctness risks.** The `spatial`+`unpad` sub-branch concatenates `image_newline` before the
mask/reshape completes (this is the constraint noted in P4), so it has ordering semantics the
single-image path does not. Any change here needs its own equality fixture.

**Benchmark afterward.** Stage 1 is the benchmark: extend
`scripts/profile_multimodal_prep.py` to take a multi-tile image and produce the same A/B/C/D
phase split for the anyres path.

**Definition of done, stage 1.** A phase table for the anyres path comparable to the one at the
top of this document. Then re-prioritise.

**Scope.** Stage 1 small. Unknown thereafter.

**Dependencies.** P0's phase-segmentation output, so the two paths are directly comparable.

---

## Milestone table

| # | Milestone | Item(s) | Expected effect on prep p50 | Cumulative prep p50 | Kernel work | Scope | Depends on |
|---|---|---|---|---|---|---|---|
| M0 | Measurement is trustworthy | P0 | none (baseline 48.0 ms) | 48.0 ms | no | S | — |
| M1 | ViT in FP16 | P1 | **−18 to −23 ms** | ~25-30 ms | no | S code / M validation | M0 |
| M2 | Syncs eliminated | P2 | −3 to −6 ms | ~21-26 ms | no | S | M1 |
| M3 | Loop restructured | P3 | −4 to −7 ms | ~15-20 ms | no | M | M2 |
| M4 | Projector reordered | P4 | −0.25 ms (may be sub-noise) | ~15-20 ms | no | S/M | M1 |
| M5 | Triton v1 (fused row-max) | P5-v0, P5-v1 | −0.5 to −1 ms | ~14-19 ms | **yes** | M | M3 |
| M6 | Triton v2 (fused pass) | P5-v2 | −1 to −2 ms | ~13-17 ms | **yes** | L | M5 |
| M7 | Batch scaling characterised | P6 stage 1 | none (diagnostic) | — | no | S | M0 |
| M8 | Anyres characterised | P7 stage 1 | none (diagnostic) | — | no | S | M0 |

Cumulative figures assume the effects compose, which they roughly should: M1 removes GPU work,
M2/M3/M5/M6 remove stalls and launches, and those are largely independent. **Treat every number
in this table as a hypothesis to be falsified, not a forecast.** The only measured numbers in
this document are in the baseline table at the top.

Floor check: after M1 the GPU-busy floor is ~13-15 ms (FP16 phase A) and the overhead budget is
12.5 ms. Nothing in P2-P6 can take prep below the phase A floor, because **the ViT always
processes all 576 patches regardless of `visual_token_num`** — `[CLS]`-attention scoring requires
the full forward pass. That is an Amdahl ceiling on the entire method, not just on this plan.

---

## Deliverables checklist

**Measurement**
- [ ] `profile_multimodal_prep.py` double-count bug fixed; aggregates recomputed
- [ ] A/B/C/D phase segmentation added to the script's output
- [ ] `generate()` reads `forward_calls[0]["multimodal_prep_time"]` off CUDA events
- [ ] Noise floor measured and written down (10 runs, randomised order, git commit + GPU recorded)
- [ ] Baseline re-run and stored after M1, since every later comparison is against FP16

**Code**
- [ ] Vision tower loads in FP16 (`builder.py:151-156` path)
- [ ] `torch.arange` hoisted to device, both sites (`llava_arch.py:167`, `:174`)
- [ ] Pruning loop: gather-once compacted buffer
- [ ] Pruning loop: `topk` replaces `argsort`
- [ ] Projector moved before selection (single-image + `flat` branches; anyres exempted)
- [ ] `triton_fixup.py` (or better-named) with v1 fused row-max kernel
- [ ] v2 fused-pass kernel

**Validation**
- [ ] v0 reference function + saved fixtures at `visual_token_num` ∈ {64, 128, 288}
- [ ] `selected_indices` bitwise-equality harness (used by P2, P3, P4, P5)
- [ ] Explicit rule-16 assertion: any index-based gather output equals the boolean-mask output
- [ ] FP16-vs-FP32 token-selection Jaccard, ≥200 dev images, three token counts
- [ ] Dev-set accuracy delta for M1, signed off by Sribhav (test split stays locked — rule 11)
- [ ] Triton shape sweep including non-power-of-2 and B>1

**Diagnostics**
- [ ] Batch ∈ {1,4,8,16} profile → sync-count scaling
- [ ] Anyres-path phase table

**Communication**
- [ ] Tell Rithvik that every number in `results/` predates the FP16 fix, before he runs anything
      for the chart
- [ ] Note in the eventual writeup that phase A is invariant to `visual_token_num` and therefore
      caps the achievable speedup

---

## Recommended order for this week

**Day 1 — make measurement trustworthy (P0).** Fix the aggregate bug, add phase segmentation,
run the baseline 10× in randomised order, write down the noise floor. Everything downstream is
unfalsifiable without this.

**Day 1 (afternoon) — land P1.** The code change is small. Get a fresh trace and confirm the ViT
kernels are FP16. This is the single largest win available and it is a configuration bug, not an
optimisation.

**Day 2 — validate P1 with Sribhav.** Jaccard overlap on token selection, then dev-set accuracy.
Run the `visual_token_num=576` control first (selection is a no-op there, so any delta is pure
numerics). Do not proceed to P3/P4 until this is settled — they both perturb numerics further and
you will lose attribution.

**Day 2 (afternoon) — land P2.** One line per site, huge structural effect. Confirm 112 syncs → 0
and `selected_indices` bitwise-identical. Measure against the FP16 baseline.

**Day 3 — P6 stage 1 and P7 stage 1.** Both are cheap profiling runs, both remove a blind spot,
and P6's answer determines whether a whole workstream is needed. Run them while the P1 accuracy
validation is still in flight.

**Days 3-4 — P3.** Gather-once and `topk`. Assert bitwise-identical selection. This is where the
remaining overhead budget gets claimed.

**Day 4 — P4.** Small, and it reshapes P5's target. The rule-16 equality assert is the whole test.

**Day 5 — P5-v0 and start P5-v1.** Reference function and fixtures first (rule 19 — a bug should
cost seconds). Then the fused row-max kernel. v2 is next week.

Rule 9 applies throughout: one person on the GPU during any run whose numbers you intend to
quote. Coordinate with Rithvik and Sribhav before each measurement block.

---

## What I should be able to explain in an interview after each milestone

**After M0.** Why a benchmark without a noise floor is not a measurement. What
`torch.profiler` gives you that `perf_counter()` cannot — the separation of CPU dispatch time
from GPU kernel time — and how to read a chrome trace: gaps on the GPU track while the CPU track
is busy means launch-overhead-bound; one long saturated kernel means bandwidth- or compute-bound.
Why my own summary script's first aggregate was wrong (aten-level and kernel-level rows both
carry CUDA time; summing both double-counts) and how I caught it (the profiler's own table footer
disagreed with my number by 2.08×).

**After M1.** How I found that 93% of multimodal-prep GPU time was a vision tower silently
running in FP32 — not by guessing, but from `Input type` fields and `ampere_sgemm_*` kernel names
in the trace, with the FP16 projector two ops later as the contrast. The root cause: a
`device_map != 'auto'` guard around the `.half()` cast, with `'auto'` as the default in every
caller, so the cast was dead code in every run the project ever made. Why FP32→FP16 is the only
lever available on a phase that is already 96.3% GPU-busy: you cannot fuse away bubbles that
aren't there, so you change the arithmetic. And the accuracy risk that makes it not-free: token
selection ranks `[CLS]` attention scores, so FP16 rounding reorders near-ties and changes which
tokens survive.

**After M2.** What coordinated CPU-GPU stalling looks like in a trace: 706 GPU events for 2.06 ms
of work spread over 37.5 ms at 94.5% idle. Why `torch.arange(B)` without `device=` costs 112 full
stream synchronisations per request — a pageable host-to-device copy forces a stream sync before
it can start, so the CPU cannot run ahead and queue work. Why that one line was worth more than
the Triton kernel I originally planned to write. The general lesson: async execution only helps
if nothing drags the host back into lockstep, and an accidental CPU tensor is the most common way
to do that.

**After M3.** The difference between a launch-count problem and a memory-traffic problem, and why
the same loop had both: ~8 chained ops × 56 passes (dispatch-bound), plus re-gathering the full
residual token block from scratch every pass to shrink it by 8 elements (~33 MB of DtoD traffic
for bookkeeping). Why `argsort` to discard 8 of 256 is the wrong primitive. And the ordering trap
I did not fall into: `residual_indices` is in score order, and raster order is restored only
because the code scatters to a boolean mask and indexes with it — so any refactor to an
index-based gather silently reorders tokens, which is exactly the failure rule 16 exists to
prevent, and it degrades quality without ever throwing.

**After M4.** Why pruning belongs between the ViT and the projector, and why the FLOP argument
being correct did not make it a meaningful latency win here — 4.5× less projector matmul, on two
linear layers, is 0.25 ms against a 48 ms baseline. Being able to say "the theory was right and
the effect was under my noise floor" is the point; it is the same category of result as the
project's own R1.

**After M5.** Why I retargeted the Triton kernel from the final gather to the pruning loop: the
gather was ~10 µs of GPU time and the loop was 2.06 ms across 706 launches, so the original
target had a ceiling of about 0.2% of GPU time. What the kernel actually does — one program per
row-block, streaming the second operand through and keeping a running max, so the (R/2 × R/2)
similarity matrix is never written to global memory. Why the tile-remainder cases matter: R walks
512→64 in steps of 8, so almost every pass has a non-tile-multiple size and masking bugs would
show up as silent wrong answers rather than crashes.

**After M6.** Why the pass count is data-dependent and what that costs: `r = min(8, R - T_div)`
means 56 launches at `visual_token_num=128` and 64 at 64, so the loop gets *longer* as pruning
gets more aggressive — the opposite of the intuition. What it would take to push the loop inside
a single persistent kernel, and why I did or didn't. Where the remaining time went once launches
stopped being the bottleneck.

**Across all of it.** The one number that frames the whole project: the vision tower always
processes all 576 patches, because `[CLS]`-attention scoring needs the full forward pass to rank
tokens. At `visual_token_num=128` that is 33 ms of a 111 ms request that pruning cannot touch —
an Amdahl ceiling on the method itself, before any of the serving-side effects the project set
out to measure. That belongs in the writeup as prominently as any speedup number.
