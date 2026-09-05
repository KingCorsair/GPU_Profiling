# Speed side — what to build next, and why (Amay)

Personal planning doc, not a spec anyone else needs to follow. Written after a session
figuring out which GPU-serving optimizations are worth hand-building myself vs. just
flipping on, given what I actually own (batching, KV cache, Triton kernels, token removal,
vLLM port — see CLAUDE.md's "Team and ownership" and "Amay — don't write these for him
unprompted"). The bar here isn't "what would make timing better" in the abstract, it's
"what can I explain from memory, with my own numbers, in an interview" (the self-test).

Config flags (FlashAttention-2, `torch.compile(mode="reduce-overhead")`) already got
evaluated and ruled out for this doc's purpose — they help, but they're one line, not a
project. This plan is about the things worth actually hand-building.

**Related documents:**
- [`AMAY_VISPRUNER_PROFILING_REPORT.md`](AMAY_VISPRUNER_PROFILING_REPORT.md) — the completed
  profiling/research phase: every experiment, what it measured, where its evidence lives, and
  the findings this plan is built on. Read that first if the *why* behind an item here isn't
  obvious from this doc alone.
- [`prompts/GENERATE_VISPRUNER_PROFILING_REPORT.md`](prompts/GENERATE_VISPRUNER_PROFILING_REPORT.md)
  — the prompt that generates/updates the report above, for when new experiments need folding in.

---

## The four candidates, ranked

### 1. Triton kernel for the post-pruning fixup — do this first

> **Problem statement:** replace `image_features[index_masks]` on `llava_arch.py:272` with
> a Triton kernel that does the same gather, faster.

**The whole list:**

*Study first — two functions from MegaBlocks (`megablocks/backend/kernels.py`):*
1. `_padded_copy` (line 45) — the Triton kernel template: worker looks up its index,
   computes the source pointer, copies the row in blocked chunks.
2. `gather()` (line 141) — the wrapper template: allocate output, launch one program per
   index.

*The changes to my code:*
3. `encode_images()` — return `selected_indices` as a third value.
4. Line 272 — replace `image_features[index_masks].unsqueeze(0)` with
   `triton_gather(image_features, selected_indices)`, add `, _` at the anyres call site,
   add the import, and write the two functions from #1-2 as my own `gather_kernel` +
   `triton_gather` in a new `triton_fixup.py`.

*The 5 findings from the Claude Code report (noted for later, not part of this task):*
5. Projector runs on all 576 tokens before pruning — reorder so it only processes kept
   tokens.
6. Diversity while-loop (`encode_images()` :161-176) — dozens of chained small kernel
   launches.
7. Padding/position/mask rebuild loop (:350-390) — fusable into the kernel as a v2.
8. Per-batch splice loop (:302-341) — forced CPU-GPU syncs via `.sum()` and `.tolist()`.
9. Anyres/multi-image branch (:198-269) — the same masked gather repeated per tile in a
   Python loop.

Rules 15/16 in CLAUDE.md already require, after boolean-mask pruning: re-gather kept
tokens in raster order, fix position IDs, rebuild the attention mask. Right now that's
presumably several separate PyTorch ops (gather, argsort, mask rebuild) — each its own
kernel launch and memory round-trip on small tensors, i.e. launch-overhead- and
bandwidth-bound, not compute-bound.

**Why this one first:** it's *my own* stated bottleneck from the project's own rules, not
a borrowed one. Smallest scope, most directly measurable with the load/prefill/decode
harness I already have.

**Found while digging into this — a bigger, kernel-free win, do it before finalizing the
kernel design:** `encode_images()` runs `mm_projector` on the *full* `(B, N, C)` tensor
(`llava_arch.py:185`) — the mask/gather down to `T` kept tokens doesn't happen until later,
in `prepare_inputs_labels_for_multimodal()`. So the projector always does 576-token work
regardless of `visual_token_num`. Token-pruning literature is consistent that pruning
belongs *between* the ViT and the projector for exactly this reason — projector cost scales
with token count same as the LLM's does. At aggressive ratios this is easily >4x more
projector matmul than necessary, and it's a bigger FLOP-level win than the fixup kernel
itself, with no kernel required — just gather before projecting instead of after.

Two things this changes about the rest of the plan:
- **Sequencing:** do this reorder first, separately from the kernel. It changes what the
  kernel in step 2 below even gathers — pre-projection features are `mm_hidden_size`-wide
  (CLIP's dim, e.g. 1024), not `hidden_size`-wide (LLM's dim, e.g. 4096), and the gather
  moves from `prepare_inputs_labels_for_multimodal:272` into `encode_images` itself. Design
  the kernel against the post-reorder shape, not the current one.
- **Catch on the anyres path:** clean for the `flat` merge type and the single-image branch,
  but the `spatial`+`unpad` branch concatenates `image_newline` (LLM-dimensional) *before*
  the mask/reshape logic finishes, so projection has to stay before that concat on that path
  specifically. Not a blocker — just means this isn't one global move, it's per-branch.

**Plan:**
1. Profile the current fixup op sequence (`torch.profiler`) to confirm it's actually
   launch/bandwidth-bound before committing to a kernel — don't skip this, "prototype on
   something fast before touching something slow" (rule 19).
2. If confirmed: write a fused Triton kernel doing gather + position-id remap + mask
   construction in one pass.
3. Before/after comparison, same harness, same rules (CUDA events, warm-up, noise floor).

**Implementation shape for step 2 — augment vs. new:**

Mapped this out in a session confirming it against the actual code. Two functions get
augmented in place, two get added net-new in a new `triton_fixup.py`. No other files touch.

*Augment (existing functions, modified in place):*
1. `encode_images()` (`llava_arch.py:141`) — augment its return. It already computes
   `selected_indices` (~line 179) before collapsing them into the `index_masks` bool mask
   it currently returns; let it also return `selected_indices` directly. No logic inside
   changes — it's already computed, just wasn't exposed.
2. `prepare_inputs_labels_for_multimodal()` (`llava_arch.py:189`), the single-image `else`
   branch at ~line 272 — swap `image_features[index_masks].unsqueeze(0)` for
   `triton_gather(image_features, selected_indices)`. Also needs a `, _` added to the
   3-way unpack at the multi-image call site (~line 202), since `encode_images` now returns
   three values on that path too — cosmetic, no behavior change there.

*Add (net-new, in `triton_fixup.py`):*
3. `gather_kernel()` — the actual Triton kernel. B·T parallel workers, each looks up its
   assigned row index, computes where that row lives in memory, copies its values to the
   output. Doesn't exist anywhere in the project yet.
4. `triton_gather()` — Python wrapper: sorts indices (preserve raster order — rule 16),
   allocates the output tensor, launches B·T copies of the kernel, returns a tensor shaped
   like the old `image_features[index_masks]` result.

The augmentations exist purely to route data to and from the additions; all the new
capability lives in 3/4. Confirm-once gate before writing 3/4 for real: CLAUDE.md lists
Triton kernels and the token-removal implementation as mine to write, not Claude's, by
default — so this is spec, not something to hand over wholesale.

**Fixup-path costs this scope doesn't cover — found by tracing the whole path, not just the
one line at `:272`:**

The current design (steps 3/4 above) is a plain gather. But the problem statement two
sections up already says the kernel should do "gather + position-id remap + mask
construction" — there's a real gap between that stated goal and what's actually scoped.
Worth deciding explicitly which v1 is: gather-only (matches `:272`, simpler) or actually
fused (matches what the doc already claims). Things a fused version, or a v2, would need to
account for:

1. **The duplicate-token-pruning `while` loop**, `encode_images()` `llava_arch.py:161-176`
   — the biggest one I was missing. Each iteration chains ~7-8 ops (gather, batched matmul,
   max, argsort, two more gathers, a cat), looping multiple times as `residual_indices`
   shrinks by ≤8 per pass. Upstream of the gather kernel entirely, and a bigger pile of
   small launches than the line I'm targeting.
2. **The multi-image/anyres branch**, `llava_arch.py:198-269` — same masked-gather pattern,
   repeated per image tile in a Python loop, plus `permute`/`unpad_image`/`cat` per tile.
   My augmentation only patches the single-image `else` branch; this whole branch is
   unaddressed.
3. **`input_ids`/`labels` masking**, `llava_arch.py:296-297` — same masked-gather-with-
   implicit-sync pattern, done per batch item before the main loop even starts.
4. **The per-batch main loop**, `llava_arch.py:302-341` — at least two forced CPU-GPU syncs
   *per batch item* (`.sum()` read into a Python `if`, and `.tolist()`). A sync-*count*
   problem, separate from the launch-*count* problem the gather kernel solves.
5. **The padding/rebuild loop**, `llava_arch.py:350-390` — ~5-6 launches per batch item to
   do what's structurally one batched pad + arange-broadcast + mask-assign. This is the
   actual "position-id remap + mask construction" the problem statement already names.

None of this needs deciding right now — profiling (step 1) will show which of these
actually cost time. But going in aware of the full list beats discovering #1 or #5 mid-build
and re-scoping the kernel then.

**Prerequisite, not yet done:** Triton isn't installed anywhere in this environment — not in
`requirements.txt`, not in the Dockerfile, no `.triton` cache on disk. Add it to
`requirements.txt` and rebuild the image before step 2 can run at all (see CLAUDE.md:
environment changes go through the Dockerfile, not a pip install on a live pod).

**The metric — what number actually has to go down:**

`multimodal_prep_time_generate`, already logged by `generate()` in
`llava_llama.py` (~lines 179–209, `vis_pruner_copy/llava/model/language_model/llava_llama.py`).
It's measured with `time.perf_counter()` bracketed by a `torch.cuda.synchronize()` call, not
pure CUDA events.

**Correction (checked against current code):** I'd written above that `prep_start_evt`/
`prep_end_evt` were unused — they're not. They're already recorded in `forward()` and read
out into `forward_calls[i]["multimodal_prep_time"]` (`llava_llama.py:~231`). The actual gap
is narrower than I thought: `generate()`'s top-level number still uses `perf_counter()` +
`synchronize()` instead of just reading the first `forward_calls` entry (the prefill one)
off the events that are already being recorded. That's the step-3 fix, not "wire up unused
events from scratch."

There's also a per-decode-step version, `multimodal_prep_time`, inside the `forward_calls`
list — same underlying cost, logged once per call instead of once per generation. Either
works as the target; `multimodal_prep_time_generate` is simpler since it's one number per
question.

**Metrics in the same file that are NOT the target — don't accidentally optimize these:**
- `generate_time` — the whole generate() call, prep + every decode step. Too broad: decode
  time is in here and a kernel touching only the fixup step won't move it. If I benchmark
  against this number I could convince myself the kernel did nothing when it actually worked.
- `model_load_s` — one-time disk/GPU load cost, unrelated to per-request work entirely.
  (Correction: this one isn't actually in `llava_llama.py` — it's computed in the eval
  scripts, `model_vqa_science.py`/`model_vqa_heterogeneous.py`. Doesn't change the point.)
- `lm_forward_time` — the LM's actual forward-pass math. This is what the fixup step feeds
  *into*; the kernel doesn't touch it and it shouldn't move.

**What "done" looks like:** `multimodal_prep_time_generate` (or the per-call
`multimodal_prep_time`) drops after swapping in the fused kernel, at fixed `visual_token_num`,
same harness, same warm-up/noise-floor rules as everything else — and `generate_time` /
`lm_forward_time` stay flat, which is how I'd know the change was isolated to the right place.

**Where the current (pre-kernel) implementation actually lives:** the gather/position-id/
mask-rebuild logic rules 15/16 describe is in `llava/model/llava_arch.py`,
`prepare_inputs_labels_for_multimodal()` (~lines 189–394) and `encode_images()` (~141–186).
Confirmed multi-op, not fused: index selection via `argsort`/`topk`-style slicing into
`index_masks`, then a **per-batch-item Python loop** filtering by `attention_mask`, then
manual padding + `attention_mask`/`position_ids` rebuild via `torch.zeros` + slice assignment.
That per-sample Python loop is itself worth noting in the profiling step — it's not just
"three kernel launches," it's three kernel launches *repeated once per item in the batch*,
which is additional overhead a fused kernel should also collapse.

---

### 2. Hand-rolled CUDA graph capture around the decode loop

Not `torch.compile` — actually calling `torch.cuda.graph()` myself around the forward
pass. The real difficulty: CUDA graphs need fixed tensor addresses/shapes, which fights
with a KV cache that grows every decode step. Solving that tension (preallocated,
fixed-size KV cache buffer) is the actual interview story, and it sets up #3.

**Plan:**
1. Preallocate a fixed-size KV cache buffer (max sequence length known up front for this
   project's use case).
2. Capture the decode step as a CUDA graph against that buffer.
3. Measure per-token decode latency before/after — this is a direct extension of the R2
   prefill/decode split, using CUDA events at the `forward()` boundary as already planned
   in AMAY_TIMING_NOTES.md.

---

### 3. Minimal paged KV cache allocator — before reaching for vLLM

A toy version, no external library: block table mapping logical sequence positions to
physical cache blocks, allocation/free list. Size it against R5 directly — peak KV cache
memory per `visual_token_num` setting, show fragmentation under naive per-sequence
contiguous allocation, then show the paged version raising the concurrency ceiling.

**Why before the vLLM port, not instead of it:** "I used vLLM's PagedAttention" is a much
weaker interview answer than "I built a simplified version, understood exactly what it
solves, then adopted the production implementation and can say what I got right or missed
against it." Do this, then port to vLLM, then compare.

**Depends on:** #2's fixed-size buffer work is a natural stepping stone here.

---

### 4. Length-bucketed / packed-sequence batching for R4

Already on the project roadmap (R4: padded vs. bucketed vs. packed). Padded and bucketed
are the easy versions. Packed is the real one: variable-length sequences concatenated with
a block-diagonal attention mask (or FlashAttention's `varlen` API) instead of padding to
the longest — directly demonstrates why padding destroys pruning's saving, which is the
project's core "gap nobody has closed" claim.

**Plan:** implement padded as the baseline (should already exist), then bucketed, then
packed. Same chart, three bars, tied straight to a result the writeup needs anyway.

---

## Study material — what to read, and when

Don't read everything cover-to-cover before starting. Most of this sticks far better when
pulled up mid-build to answer a specific wall you've hit than read in the abstract first —
open the relevant doc, start building, go back to it when you need it. There's one
exception, called out below.

**Foundational, read enough of before step 1's profiling pass (not full courses):**
- GPU memory hierarchy / roofline model (global vs. shared memory, bandwidth vs. compute)
  — enough to reason about "bandwidth-bound" vs. "launch-overhead-bound." NVIDIA's CUDA C
  Programming Guide (memory hierarchy chapter), or ch. 1–3 of *Programming Massively
  Parallel Processors* (Kirk & Hwu).
- Reading a `torch.profiler` / Nsight Systems trace — gaps between kernel launches means
  overhead-bound, one long kernel with low throughput means bandwidth-bound. Needed before
  the profiling pass that opens step 1, so you can actually confirm the bottleneck instead
  of guessing.

  **Expanded after a walkthrough session (2026-08-29) — what I need to understand before this
  runs, even if I'm not the one writing the harness:**
  - `torch.profiler` separates CPU-side dispatch time from actual GPU kernel time —
    that's the thing a plain `time.perf_counter()` can't give you, and it's the whole reason
    to use it here instead of wall-clock timing.
  - Shape: a `torch.profiler.profile(activities=[CPU, CUDA], record_shapes=True)` context
    manager around the region of interest, with nested `record_function("label")` blocks
    inside it to break the trace down by sub-region instead of getting one undifferentiated
    number. `record_shapes=True` tags each op with the tensor shapes involved — needed to
    later connect "this op is slow" to "because B/T/N was this size."
  - Warm-up still applies here same as rule 3 — run ~10 calls outside the profiler context
    first and throw them away (CUDA context init, kernel-cache warm-up), *then* open the
    profiler for the calls that actually count.
  - **Where the boundary goes matters a lot.** Wrap `prepare_inputs_labels_for_multimodal()`
    specifically (near where `prep_start_evt`/`prep_end_evt` already bracket it in
    `llava_llama.py`) — not the whole `generate()` call. Wrapping `generate()` buries the
    multimodal-prep signal under every decode step's LM forward pass.
  - Two ways to read the output, cheap to detailed:
    1. `prof.key_averages().table(sort_by="cuda_time_total")` — a summary table with a
       **call-count column**. This alone should show the diversity while-loop's
       multiplication effect (`llava_arch.py:161-176`) — if `argsort`/`matmul` show up with
       dozens of calls for one forward pass, that's the loop, visible without reading a
       visual trace at all.
    2. `prof.export_chrome_trace("trace.json")`, opened in Perfetto (`ui.perfetto.dev`) or
       `chrome://tracing` — an actual timeline with CPU and GPU tracks. This is the literal
       picture behind the heuristic above it in this doc: gaps on the GPU track while the CPU
       track stays busy = launch-overhead-bound; one continuous saturated block on the GPU
       track = bandwidth/compute-bound, and the fix there is doing less work, not fusing
       launches.
  - Point of this section: even if someone else (or Claude, after the one-confirmation gate
    CLAUDE.md requires for this category of code) ends up writing the actual harness, I
    should be able to look at its output and say which of the two possibilities it shows,
    and why — that's the self-test bar, not just "the number went down."

**Per build step — skim once, keep open as a reference tab while building:**
- **#1 (Triton kernel):** official Triton tutorials in order — vector-add, fused-softmax,
  matmul. Also re-read `prepare_inputs_labels_for_multimodal` in `llava_llama.py` for the
  actual position-ID/mask shapes before designing the kernel's output.
- **#2 (CUDA graphs):** PyTorch's "CUDA Graphs" docs page for the capture/replay API and
  the fixed-address/fixed-shape constraint. Then read HF's `StaticCache` implementation as
  the reference solution to "KV cache grows every step, graphs need fixed shapes" — the
  exact tension #2 asks me to solve.
- **#4 (packed batching):** FlashAttention's `varlen` docs (`flash_attn_varlen_func`,
  `cu_seqlens`) — the actual mechanism for a block-diagonal mask over concatenated
  variable-length sequences.

**The one required full read, before writing any code for that step:**
- **#3 (toy paged KV allocator):** the PagedAttention paper (Kwon et al., SOSP 2023) —
  short, and close enough to a spec for this exact allocator (block table, logical→physical
  mapping, free list) that skipping it risks reinventing a worse design. Read it completely
  first. Only after the toy version works: skim vLLM's `block_manager` source, as the
  "here's what production does differently" comparison — not as the starting point.

**vLLM port:** read "Inside vLLM: Anatomy of a High-Throughput LLM Inference System" only
after #2 and #3 are built, so it lands as a comparison against my own toy versions instead
of being the abstraction I start from.

---

## Ruled out for now

**Speculative decoding.** Legitimate decode-side optimization (attacks decode's
memory-boundedness directly, unlike pruning), but a correct implementation is close to
*being* the reference implementation — draft model, verify-and-accept logic — so it's hard
to make distinctively mine. Also a detour from the KV-cache/batching thread rather than a
deepening of it. Revisit only if #1–#4 are done and there's time left.

---

## Sequencing

1. Profile the pruning fixup path to confirm the bottleneck (cheap, fast, don't skip).
2. Triton fixup kernel (#1).
3. CUDA graph decode loop with fixed-size KV buffer (#2).
4. Toy paged KV allocator (#3), using #2's buffer work.
5. vLLM port, with the toy allocator as a point of comparison ("here's what production
   does differently").
6. Packed batching for R4 (#4) — can happen in parallel with 3–5, it's independent.

Each step should produce a number, on my own throwaway harness, before moving to the next
— per CLAUDE.md, only Rithvik's harness produces reportable numbers, but I need to know a
change helped before it's worth asking him to measure it properly.

---

## Status (2026-08-29)

> **CORRECTION (2026-09-03) — the numbers in this Status block are wrong, and the conclusion
> they support is backwards.** The script that produced them double-counted GPU time; see
> §9.10 and Experiment 8 in `AMAY_VISPRUNER_PROFILING_REPORT.md`. The original text is kept
> below because `AMAY_ENGINEERING_ROADMAP.md` §1 uses this document as the "what the old plan
> assumed" baseline and explicitly treats it as a document to compare against, not to rewrite.
> **Corrected values, same config, re-measured on commit `c317688`:**
>
> - Of the region's GPU busy time, **76.6% is big matmul/attn kernels** and only **23.4%** is
>   everything else → **compute-bound, not launch-overhead-bound.** The premise this step was
>   run to confirm was *not* confirmed. It was refuted.
> - Most of that 23.4% is the CLIP ViT's own softmax and elementwise work (phase A), not the
>   fixup path. The gather/splice/pad/mask region this plan targets is **0.079 ms/call —
>   0.2% of GPU work** (`AMAY_TRACE_PLAN.md` phase D).
> - CPU/GPU-busy ratio is **1.90x**, not 1.01x. Real dispatch overhead does exist, but it sits
>   in phase B's 706 tiny kernels and 112 stream syncs, not in the fixup.
> - The **40.7% / 43.1%** "share of a prefill request" figure is **withdrawn entirely** — its
>   denominator is an un-warmed `generate()` call that moved 111 → 191 ms between runs (§9.11).
>
> **Consequence for this plan's sequencing:** step 2 (the Triton fixup kernel) has lost the
> evidence that put it first. `AMAY_TRACE_PLAN.md` supersedes the ordering below; the current
> top priority is the FP32→FP16 vision tower, which touches the 93.1% of GPU time this step
> mistakenly attributed to small ops.

**Step 1 (profile the fixup path) — done.** Ran `scripts/profile_multimodal_prep.py`
(`visual_token_num=128`) on the A40. Results:

- ~~`prepare_inputs_labels_for_multimodal` is **40.7%** of a 1-token prefill request.~~ *(withdrawn)*
- ~~Of that region's CUDA time, **63.2%** sits in non-matmul ops (gather/index/argsort/cat/pad),
  only 36.8% in big matmul/attn kernels → **launch-overhead-bound, not compute-bound**.
  Confirms the Triton kernel premise before building it, per the rule at the top of this
  section ("don't skip this").~~ *(wrong — see correction above; actual split is 76.6% / 23.4%)*
- Full table + numbers: `results/timing/prep_summary.txt` (short, readable — **regenerated
  2026-09-03 with the fix; it no longer contains the numbers quoted above**). Full Chrome
  trace: `results/timing/prep_trace.json` (open in `ui.perfetto.dev`, not as text).

**Self-test to pass before moving on:** ~~explain out loud why 63.2% non-matmul time means
launch-bound, and why that's the reason to build the kernel next~~ — replaced by a better one:
explain out loud why a region can be 76.6% GEMM on the GPU *and* carry a 1.90x CPU/GPU-busy
ratio at the same time, and what that combination implies about which of the two available
levers (make the GEMMs cheaper vs. remove launch overhead) is worth pulling first.

**Not yet decided:** whether to add fine-grained `record_function` labels inside
`llava_arch.py` (around the 5 named hotspots — diversity while-loop, anyres branch,
per-batch masking, main-loop syncs, padding/rebuild loop) to break the profiler output down
per-region instead of one blob per iteration. Current run only shows the aggregate; this
would show which specific region dominates before committing to the kernel's exact scope.

**Next up:** step 2, the Triton fixup kernel itself — mine to write, not Claude's, per
CLAUDE.md's protected list, past the one-time confirm already used for the profiling script.
