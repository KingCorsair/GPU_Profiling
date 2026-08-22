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

---

## The four candidates, ranked

### 1. Triton kernel for the post-pruning fixup — do this first

Rules 15/16 in CLAUDE.md already require, after boolean-mask pruning: re-gather kept
tokens in raster order, fix position IDs, rebuild the attention mask. Right now that's
presumably several separate PyTorch ops (gather, argsort, mask rebuild) — each its own
kernel launch and memory round-trip on small tensors, i.e. launch-overhead- and
bandwidth-bound, not compute-bound.

**Why this one first:** it's *my own* stated bottleneck from the project's own rules, not
a borrowed one. Smallest scope, most directly measurable with the load/prefill/decode
harness I already have.

**Plan:**
1. Profile the current fixup op sequence (`torch.profiler`) to confirm it's actually
   launch/bandwidth-bound before committing to a kernel — don't skip this, "prototype on
   something fast before touching something slow" (rule 19).
2. If confirmed: write a fused Triton kernel doing gather + position-id remap + mask
   construction in one pass.
3. Before/after comparison, same harness, same rules (CUDA events, warm-up, noise floor).

**The metric — what number actually has to go down:**

`multimodal_prep_time_generate`, already logged by `generate()` in
`llava_llama.py` (~lines 179–209, `vis_pruner_copy/llava/model/language_model/llava_llama.py`).
It's measured with `time.perf_counter()` bracketed by a `torch.cuda.synchronize()` call, not
pure CUDA events yet — there's an unused event pair (`prep_start_evt`/`prep_end_evt`) in the
same file, in `forward()`, not currently wired into the readout. Worth switching to those
events for the before/after comparison in step 3, since that's the rule-1-compliant way to
time GPU work and avoids relying on a sync I have to place by hand.

There's also a per-decode-step version, `multimodal_prep_time`, inside the `forward_calls`
list — same underlying cost, logged once per call instead of once per generation. Either
works as the target; `multimodal_prep_time_generate` is simpler since it's one number per
question.

**Metrics in the same file that are NOT the target — don't accidentally optimize these:**
- `generate_time` — the whole generate() call, prep + every decode step. Too broad: decode
  time is in here and a kernel touching only the fixup step won't move it. If I benchmark
  against this number I could convince myself the kernel did nothing when it actually worked.
- `model_load_s` — one-time disk/GPU load cost, unrelated to per-request work entirely.
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
