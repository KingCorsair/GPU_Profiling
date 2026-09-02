# AMAY_ENGINEERING_ROADMAP.md — Closing the VisPruner Algorithmic-to-Serving Gap

**Status:** Planning document. Nothing in this roadmap has been implemented. All evidence cited comes from `AMAY_VISPRUNER_PROFILING_REPORT.md` (the completed profiling phase) and direct, read-only inspection of the current codebase. This document does not modify, replace, or append to any existing file — it is new.

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

## 1. How This Roadmap Differs From Earlier Plans

`AMAY_SPEED_PLAN.md` was written *before* the systematic profiling phase existed (per its own "Status (2026-08-29)" section, it reflects one profiling pass — `scripts/profile_multimodal_prep.py`, Experiment 1 in the profiling report). It is the right document to compare against, not to edit.

| What the old plan assumed | What profiling later established | What changed |
|---|---|---|
| `prepare_inputs_labels_for_multimodal` is "40.7%" of a 1-token prefill request, and is launch-overhead-bound (>>1 CPU/CUDA ratio implied) | The canonical artifact (`prep_summary.txt`) actually says **43.1%**, and the region's *own* CPU/CUDA ratio is 1.01x — not >>1. The 63.2%-non-matmul finding still supports "launch-overhead-flavored," but the >>1 framing overstates what that specific file shows. | This roadmap cites 43.1%, not 40.7%, and treats the "launch-overhead-bound" characterization as evidenced-but-imprecise, not settled. |
| The gather/fixup kernel (targeting `llava_arch.py:272`) was the primary, best-scoped Triton target | Profiling of the *whole* pipeline (Experiment 6) found a **larger, more surprising** target: the diversity-selection while-loop (`llava_arch.py:160-176`) runs ~56-57 real iterations at `visual_token_num=128` and **zero** at 576 — it is the single clearest place where VisPruner's own selection mechanism costs *more*, not less, as pruning gets more aggressive. | The gather/fixup kernel remains a valid target (Experiment 1's evidence for it still stands), but the diversity-loop fusion is now the **higher-priority** Triton target — new evidence, not present in the old plan. |
| `mm_projector` running on all 576 tokens before pruning was flagged as "a bigger, kernel-free win... at aggressive ratios this is easily >4x more projector matmul than necessary" | Measured directly: `mm_projector`'s cost is **small in absolute terms** at this model's projector size (~0.3-1.1ms, isolated trace) — real, and directionally correct to fix, but not the large win the old plan's FLOP-counting argument implied. | The reorder remains worth doing (near-zero cost, correctness-neutral, frees the fixup kernel to operate on the smaller post-projection shape), but this roadmap does not claim it moves the needle much on its own — the old plan's estimate of its size was not confirmed by measurement. |
| CUDA graph capture around the decode loop, blocked on "a preallocated, fixed-size KV cache buffer" (correctly anticipated as the hard part) | Profiling found the *specific mechanism* forcing cache instability: `past_key_values` round-trips through a legacy tuple between `generate()` steps, forcing `DynamicCache.from_legacy_cache()` to rebuild the cache object from scratch every step (768 rebuild-shaped calls measured in one 23-decode-step request), immediately before the real `torch.cat`-based append (736 calls). | The old plan's instinct (KV cache needs fixing before CUDA Graphs) is confirmed and now has a named, specific root cause rather than a general "KV cache grows every step" framing. |
| A toy paged-KV allocator and vLLM port were sequenced as steps 3-5, following CUDA graphs | GPU memory was measured nearly flat (~15.1-15.4GB) across every load level and both token configurations — the KV cache is a small fraction of the ~14GB weight footprint at batch=1, and **nothing in the load-sweep evidence shows memory pressure or a concurrency ceiling driven by memory**. | Paged KV is **not currently justified** by any measurement in this repository. It is demoted to Optional (§5), pending a future experiment that actually exercises multiple concurrent requests' KV caches at once — which nothing built so far does. |
| Hand-rolled CUDA graph capture and a toy paged-KV allocator were the named "speed" architecture; real serving concurrency was implicitly assumed to be safe to build toward | A genuine multi-thread concurrent `generate()` attempt against the shared model instance **crashed** with a CUDA device-side assert (`indexSelectSmallIndex` out-of-bounds in the pruning gather path) | This is an entirely new finding with no analog in the old plan. It motivates a **new engineering category** (§7) that did not exist before: serving correctness / concurrency safety, which now gates any batching work. |
| Batching (R4 on the project roadmap) was framed primarily as a padding-vs-packing efficiency question | The measured capacity ceiling (2.166→2.249 rps, +3.8%) shows the *current* bottleneck is that the server never runs more than one request at a time at all — padding/packing efficiency is a question that only becomes answerable *after* some form of batching exists to measure it against | Packed/varlen batching remains Optional (§5) — not because it's a bad idea, but because nothing has been built yet to expose whether padding waste is even a real problem here. |

**Old ideas that remain unchanged:** the sequencing instinct (dataflow/algorithmic work → Triton → KV cache → CUDA Graphs → batching) was directionally right and survives, refined with hard evidence for *why* each dependency exists rather than intuition. The self-test bar ("explain this from memory, with your own numbers") and the benchmarking rules in `AMAY_TIMING_NOTES.md` are unchanged and this roadmap inherits them without modification.

**New engineering categories that appeared:** two categories with no clean home in the original "Triton / KV-cache / CUDA Graphs / Batching" framing are introduced in §7 — **Pruning Dataflow / Algorithm-System Co-Design** and **Serving Correctness / Concurrency Safety**.

---

## 2. Every Proposed Change, Mapped to a Category

**Classification legend** (used throughout this document):
- **PRUNING-SPECIFIC** — expected to disproportionately improve the 128-token path relative to 576.
- **PRUNING-ENABLING** — removes overhead that currently masks pruning's benefit, but the fix itself is not pruning-aware and may help 576 by a similar absolute amount.
- **GENERIC INFERENCE OPTIMIZATION** — makes the model faster without changing VisPruner's relative advantage.
- **SERVING OPTIMIZATION** — improves multi-request throughput, queueing, or tail latency; orthogonal to per-request pruning gains.

| Engineering change | Measured bottleneck | Category | Why it belongs there | Classification | Expected impact | Dependency | Priority |
|---|---|---|---|---|---|---|---|
| Fuse/vectorize the diversity-selection while-loop | ~56-57 iterations, ~330+ kernel launches at T=128, zero at T=576 (`ab_trace_vtn128_isolated.json` op counts) | **Triton** | Requires writing a new fused kernel to replace an iterative Python loop — this is kernel authoring, not a reorder | PRUNING-SPECIFIC | Moderate-significant | None (can start immediately) | **1** |
| Gather + position-id + mask fixup kernel | 63.2% non-matmul CUDA time in `prepare_inputs_labels_for_multimodal` (Experiment 1) | **Triton** | Also kernel authoring; fuses several small ops (gather/argsort/cat/pad) into one launch | PRUNING-SPECIFIC | Moderate | Soft: benefits from Pruning Dataflow's mm_projector-reorder decision being made first (changes the kernel's input shape) | 3 |
| Move `mm_projector` to run after the token-selection gather, not before | Projector runs on all 576 tokens unconditionally (`llava_arch.py:185`, before the mask at `:271`); measured cost small (~0.3-1.1ms) | **Pruning Dataflow / Algorithm-System Co-Design** | Zero kernel writing — this is purely a reorder of two existing operations | PRUNING-SPECIFIC (in direction; small in current measured magnitude) | Small | None | 2 |
| Investigate prefill's sub-linear scaling (~4.2x fewer tokens → only ~2.3x less prefill time) | 117.4ms→51.5ms vs. an ideal ~4.2x-implied ~28ms (Experiment 6) | **Pruning Dataflow / Algorithm-System Co-Design** | Diagnostic first — not yet a scoped fix; belongs with the other "how is data organized/moved" questions | Unknown until investigated | Unknown | None | 4 |
| Retain a persisted `Cache` object across `generate()` steps instead of round-tripping through a legacy tuple | `kv_cache_legacy_rebuild`: 768 calls/request, ~0ms GPU but real CPU overhead, identical at both token counts (Experiment 2) | **KV-Cache Optimization** | Directly changes cache-object lifecycle management | PRUNING-ENABLING (removes overhead equally from both configs; enables the next phase) | Small standalone; large as an enabler | None | 5 |
| Speed up the KV-cache incremental append itself (`torch.cat` per layer per step) | `kv_cache_append`: 736 calls, 11.8ms CUDA / 38.4ms CPU (576) vs. 5.6ms CUDA (128) — the one KV-cache cost that already tracks token count | **KV-Cache Optimization** | Same subsystem as the item above | PRUNING-SPECIFIC in effect (already responds to token count) but the fix itself (fusing launch overhead) is generic | Small-moderate | Soft: cleaner after the persisted-Cache-object work | 6 |
| CUDA Graph capture of the decode step | 34,722 total kernel launches in one 24-token request; decode-step CPU time exceeds CUDA time at every step, both configs, similar magnitude | **CUDA Graphs** | Textbook capture-and-replay target | GENERIC INFERENCE OPTIMIZATION | Significant (total-latency) | **Hard: requires the persisted-Cache-object work above** (stable tensor addresses) | 7 |
| Root-cause and fix the shared-model thread-safety crash (`indexSelectSmallIndex` assertion) | Genuine multi-thread `generate()` calls against the shared model instance crash (`concurrent_profile_run_log.txt`) | **Serving Correctness / Concurrency Safety** (new category) | This is a correctness/safety audit, not a performance optimization in itself | N/A — prerequisite work | N/A (unblocks capacity work) | None | **0** |
| Real tensor-level batched forward passes (multiple requests stacked into one `generate()` call) | Throughput ceiling 2.166→2.249 rps (+3.8%) despite 56-60% per-request compute cut; server verified strictly serial by reading `csnbs/server.py` | **Batching / Serving** | The actual mechanism that would let per-request savings become capacity savings | SERVING OPTIMIZATION | Significant, but for total capacity, not for the 576-vs-128 delta specifically | **Hard: requires the concurrency-safety fix above** | 8 |

---

## 3. How the Categories Connect

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

## 4. New Engineering Categories

### Pruning Dataflow / Algorithm-System Co-Design

**Why this needs its own category rather than being folded into Triton:** every item here is a question of *when* and *in what order* existing, already-correct operations run — not a question of *how* an operation is implemented at the kernel level. Forcing "move the projector call" into "Triton" would misrepresent it as kernel-authoring work; it's a two-line reorder with a correctness-preserving constraint (the `spatial+unpad` anyres branch needs projection before its `image_newline` concat, per `AMAY_SPEED_PLAN.md`'s own prior finding — this constraint is preserved, not re-litigated, in this roadmap). Justified by: `llava_arch.py:185` vs. `:271` (code), and the measured near-identical projector cost regardless of config (Experiment 6).

### Serving Correctness / Concurrency Safety

**Why this needs its own category rather than being folded into Batching:** this work produces no latency or throughput number of its own — its deliverable is "the shared model instance no longer crashes under concurrent access." That is a prerequisite for batching, not a form of batching. Conflating them would make it impossible to benchmark-gate each piece independently (§11's one-optimization-at-a-time rule). Justified by: `concurrent_profile_run_log.txt` (2,752 lines of `indexSelectSmallIndex` assertion failures, reproduced and preserved in the profiling report's Experiment 3).

---

## 5. Core vs. Extension vs. Optional

### CORE PROJECT
Work necessary to establish the central project result (that pruning-specific engineering measurably recovers more of VisPruner's algorithmic advantage than generic work does, and that serving-level fixes are necessary — not optional — to convert per-request gains into capacity gains):
- Serving Correctness / Concurrency Safety audit (Phase 0)
- Pruning Dataflow reorder + prefill investigation (Phase 1)
- Triton: diversity-loop fusion (Phase 2a)
- KV-Cache: persisted `Cache` object (Phase 3)
- The final four-way (576/128 × before/after) benchmark suite (§10), run after every phase

### STRONG EXTENSIONS
Significantly strengthens the project but is not required for the central claim:
- Triton: gather/fixup kernel (Phase 2b) — a second, well-evidenced Triton target; strengthens the Triton story but the diversity-loop fusion alone already demonstrates the pruning-specific-Triton thesis
- KV-Cache: append-speedup (Phase 3b)
- CUDA Graphs (Phase 4) — a strong, generic systems-depth demonstration, gated on Phase 3
- Batching, basic version (Phase 5) — demonstrates the capacity story, gated on Phase 0

### OPTIONAL — ONLY IF MEASUREMENTS JUSTIFY IT
Not currently supported by any measurement in this repository; would require new evidence before starting:
- **Paged KV-cache allocator** — no measurement here shows memory pressure or a memory-driven concurrency ceiling (GPU memory measured flat, ~15.1-15.4GB, across every load level and both configs). Revisit only after real batching (Phase 5) exists and creates actual concurrent-KV-cache memory pressure to measure.
- **Packed/varlen batching** — no batching exists yet to expose a padding-waste problem. Revisit only after Phase 5's basic (padded) batching is built and measured against.
- **Additional Triton kernels beyond the two identified** — no third pruning-path bottleneck has been measured with comparable evidence quality to the diversity loop or the fixup region.
- **A dedicated serving scheduler / request prioritization** — no evidence of a scheduling problem (the current bottleneck is "no concurrency at all," not "poor ordering of concurrent requests").
- **Memory-layout optimization beyond what Phases 2-3 already touch** — not evidenced as a distinct bottleneck; would need its own profiling pass to justify.
- **vLLM port** — remains a long-term comparison point per the original plan's own reasoning ("build a simplified version, understand exactly what it solves, then adopt the production implementation"), but is out of scope until the toy versions above (paged KV, batching) exist to compare against.

---

## 6. The Four Original Areas, In Detail

### Triton / Custom Kernel Fusion

**Exact operations targeted:** (1) the diversity-selection while-loop, `llava_arch.py:160-176`, inside `encode_images()`; (2) the post-selection gather + position-id remap + attention-mask rebuild, `llava_arch.py:272` and `:350-390`, inside `prepare_inputs_labels_for_multimodal()`.

**Profiler evidence:** For (1): `aten::argsort`×57, `aten::max`×56, `aten::matmul`×56, `aten::cat`×63 measured inside the `multimodal_prep` span at `visual_token_num=128`, vs. `aten::argsort`×1 and zero of the others at `visual_token_num=576` (`ab_trace_vtn{576,128}_isolated.json`, parsed directly). For (2): 63.2% of the region's CUDA time in non-matmul ops (`prep_summary.txt`, Experiment 1).

**Is it pruning-specific?** (1) is unambiguously pruning-specific — its cost is a direct, monotonic function of how many tokens get discarded, and it is provably zero-cost at the unpruned baseline. (2) is pruning-specific in the sense that its magnitude scales with how much gets pruned, though the underlying inefficiency (many small ops instead of one fused op) is a generic pattern.

**What is being fused/restructured:** (1) replaces an iterative, `≤8`-tokens-per-pass Python loop (index_select → batched matmul → max → argsort → cat, repeated ~56 times) with a single kernel doing the equivalent iterative deduplication in one launch. (2) replaces a sequence of separate gather/argsort/cat/pad calls with one kernel producing the gathered embeddings, remapped position IDs, and rebuilt attention mask together.

**Expected benefit:** for (1), removing ~330+ kernel launches per pruned request; for (2), removing the launch-overhead majority of a region already shown to be ~43% of a prefill request.

**Benchmark required:** four-way (576/128 × before/after) on `multimodal_prep_mean_ms` specifically, plus the standard latency/prefill/decode set — see §8, Phase 2.

### KV-Cache Optimization

**Current behavior:** `llava_llama.py`'s custom `forward()`/`generate()` do not retain a `transformers.cache_utils.DynamicCache` object across decode steps. Instead, `past_key_values` is passed back to `generate()` as a legacy tuple, and on every subsequent step, `LlamaModel.forward()` calls `DynamicCache.from_legacy_cache(past_key_values)`, which loops over all 32 layers reconstructing the cache object from that tuple — each layer's reconstruction is a cheap Python list-append (since the freshly-constructed cache object starts empty), not itself a GPU-bound operation, but 32 extra Python-level calls per step nonetheless. Immediately after, the real incremental update happens via `DynamicCache.update()`'s `else` branch: `torch.cat([self.key_cache[layer_idx], key_states], dim=-2)` — a full reallocate-and-copy, per layer, per step.

**Is `torch.cat`-based growth still present?** Yes, confirmed directly (`kv_cache_append`, 736 calls in a 23-decode-step request, 11.8ms CUDA / 38.4ms CPU at `visual_token_num=576`; 5.6ms CUDA at 128 in the isolated trace comparison). This is the one KV-cache-related cost that already responds to token count — a shorter retained context means less to copy each step.

**Proposed cache architecture:** retain the constructed `DynamicCache` object across `generate()`'s Python-level step boundary (avoiding the `from_legacy_cache()` round-trip entirely). This alone does not eliminate the per-step `torch.cat` reallocation — that requires a further step (a preallocated, fixed-size buffer that the cache writes into in place, sized to the maximum sequence length this project's use case needs), which is also the specific prerequisite CUDA Graphs needs.

**Why this matters:** it is the one KV-cache finding in this investigation that is *config-independent* (present, in nearly identical call counts, at both 576 and 128) — fixing it is PRUNING-ENABLING, not PRUNING-SPECIFIC. It should not be sold as a pruning win; it should be sold as the unblock for CUDA Graphs.

**Is static/persistent KV justified?** Yes — directly evidenced by the `from_legacy_cache` call-count measurement, and independently motivated as the CUDA-Graphs prerequisite.

**Is paged KV justified — or not yet?** **Not yet.** GPU memory was measured nearly flat (~15.1-15.4GB) across every load level and both token configurations in the load-sweep experiments — nothing in this repository's evidence shows a memory-driven concurrency ceiling. Paged KV solves a problem (memory fragmentation under many concurrent, variable-length KV caches) that this project has not yet built the conditions to observe. It belongs in §5's Optional bucket, revisited only once real batching (Phase 5) creates actual concurrent memory pressure.

### CUDA Graphs

**Repeated decode structure:** every decode step repeats an identical sequence of per-layer operations (32 layers × {QKV projection, RoPE apply, KV-cache update, attention, output projection, MLP}) — confirmed by the near-identical steady-state per-step CUDA time across many consecutive steps in Experiment 2's trace (steps 7-23 settle to ~27.5-28.9ms each).

**Kernel launch count:** 34,722 `cudaLaunchKernel` calls for one 24-token request (Experiment 2); 156,439 for 3 back-to-back 64-token-capable requests (Experiment 3's near-saturation trace) — both measured directly.

**CPU launch overhead:** decode-step CPU time exceeds CUDA time at every single step measured, in both the 576 and 128 traces, by a similar relative margin — this is the specific signature (CPU-bound dispatch, not GPU-bound compute) that graph replay is designed to collapse.

**Prerequisites:** the KV-Cache persisted-object work above — CUDA Graphs require the same tensor addresses to be valid across every replay of the captured graph, which the current per-step cache rebuild cannot guarantee.

**Interaction with KV-cache stabilization:** graph capture should target the *steady-state* decode step (post-cache-stabilization), using the fixed-size preallocated buffer produced by the KV-Cache phase, not the current dynamically-reallocating cache.

**What should actually be captured:** the single decode-step forward pass (one call to the LM's forward, given a fixed-shape KV buffer and a fixed-shape single-token input) — not the whole `generate()` loop, and not prefill (which has a genuinely variable input shape per request and is a poor graph-capture candidate for that reason).

**Benchmark required:** decode ms/step, before/after, at both 576 and 128 — expected to improve both configs by a similar *absolute* amount (this is a GENERIC optimization; if the 576-vs-128 relative gain shifts substantially, that result would itself need explaining, not assumed).

### Batching / Serving

**Current server execution model:** `csnbs/server.py`'s `/infer` handler (`_infer_model`) is an `async def` that calls `model.generate()` synchronously with no `await`/executor handoff — verified by reading the source, not inferred. A single uvicorn worker, single asyncio event loop, therefore services exactly one request's GPU work at a time regardless of arrival concurrency.

**Serialization/queueing evidence:** derived average in-flight concurrency (Little's Law: throughput × mean latency) grows from 0.45 at rps=0.5 to 16.17 at rps=3.0 (`ab_load_gain_table.json`); GPU utilization never exceeds 87% even at the worst load level tested — the GPU is not the hard ceiling, the serialization is.

**Why naive Python threading is not the solution:** directly demonstrated, not assumed — launching genuinely simultaneous Python threads against the shared model instance crashed with a CUDA device-side assert inside the pruning/gather path (`concurrent_profile_run_log.txt`). This rules out "just add more concurrent callers" as a fix.

**Proper tensor batching:** stacking multiple requests' input tensors (images, token sequences) into one batched forward pass, so the GPU genuinely processes several requests' work in the same kernel launches, rather than several Python-level callers racing to use the same un-batched, stateful code path.

**Prerequisites:** the Serving Correctness / Concurrency Safety audit (§4) — understanding and fixing whatever shared/mutable state in the pruning path causes the crash is a precondition for any form of multi-request execution, batched or otherwise.

**Initial basic batching design (padded, not packed):** pad every request in a batch to the longest sequence present (the simplest correct design), process the batch in one forward pass, and separate outputs by their original per-request boundaries afterward. This is deliberately not the padding-efficient (packed/varlen) version — that refinement is Optional (§5) until this basic version exists and padding waste can actually be measured against it.

**When would packed/varlen batching become justified?** Only after the basic padded-batching version above is built and benchmarked, and *that* benchmark shows meaningful throughput loss attributable specifically to padding waste (e.g. comparing batches of similar-length vs. very-different-length requests at a fixed batch size). Nothing in this repository currently measures that.

---

## 7. Detailed Phase-by-Phase Implementation Roadmap

Each phase is designed to be benchmarked in isolation before the next begins, per the project's own "prototype on something fast before touching something slow" rule and this task's explicit one-optimization-at-a-time requirement.

### Phase 0 — Serving Correctness / Concurrency Safety Audit

| Field | Detail |
|---|---|
| **Category** | Serving Correctness / Concurrency Safety (new) |
| **Problem being solved** | The shared LLaVA+VisPruner model instance is not safe for concurrent access — a real crash, not a theoretical concern |
| **Why it matters** | Blocks all real batching work (Phase 5); may also reveal state-sharing issues relevant to how the Triton kernels (Phase 2) should be written |
| **Measured evidence** | `indexSelectSmallIndex: Assertion 'srcIndex < srcSelectDimSize' failed`, out-of-bounds index_select inside the vision-tower/token-pruning gather path, followed by `CUBLAS_STATUS_EXECUTION_FAILED` and `device-side assert triggered` |
| **Source of evidence** | `results/timing/concurrent_profile_run_log.txt` (2,752 matching lines) |
| **Exact files/functions involved** | `llava_arch.py`'s `encode_images()` and `prepare_inputs_labels_for_multimodal()` (the crash is inside the pruning gather path); the calling pattern that triggered it, `scripts/profile_concurrent_load.py`'s original (now-superseded) multi-thread attempt |
| **Proposed engineering design** | (a) Reproduce the crash under controlled conditions with `CUDA_LAUNCH_BLOCKING=1` and device-side assertions enabled for a precise stack trace; (b) audit `encode_images()`/`prepare_inputs_labels_for_multimodal()` for any state not local to a single call (module-level buffers, cached tensors, mutated config attributes); (c) determine whether the fix is "make this code path reentrant" or "serialize access to it with a lock until real batching replaces per-call state entirely" |
| **Prerequisite knowledge** | CUDA's asynchronous execution model and why `device-side assert triggered` poisons the whole CUDA context; Python's GIL behavior around C++/CUDA calls (why threads can still race on GPU-side state despite the GIL) |
| **Dependencies** | None |
| **Implementation tasks** | 1. Reproduce with `CUDA_LAUNCH_BLOCKING=1`, capture exact stack trace. 2. Read every line of `encode_images()`/`prepare_inputs_labels_for_multimodal()` for non-local mutable state. 3. Write a minimal two-thread repro isolating the exact racing operation. 4. Decide and document: reentrant fix vs. explicit lock vs. "wait for real batching to make this moot." 5. If a code fix is chosen, implement and re-run the two-thread repro to confirm no crash. |
| **Correctness tests** | The two-thread repro must run N times without crashing; output correctness (same answer as sequential execution) must be verified per thread |
| **Microbenchmark** | Two-thread repro, timed, to see if the fix has any per-call overhead cost |
| **Full-system benchmark** | Not applicable at this phase — this phase produces a safety fix, not a latency number |
| **Before/after metrics** | Crash: yes → no. Per-request latency: should be unchanged (this phase should not measurably regress single-request performance) |
| **Success criterion** | The two-thread repro completes N/N runs without a CUDA assertion, with correct, order-independent output |
| **Rejection/rollback criterion** | If the only viable fix requires a global lock that serializes all pruning work anyway, that's a valid, honest finding — document it as "true concurrency requires batching, not just thread-safety," don't force a fix that doesn't actually enable Phase 5 |
| **Difficulty** | Medium — debugging a CUDA race condition is unfamiliar-territory work, but the reproduction is already in hand |
| **Estimated implementation time** | 0.5-1 day |
| **Estimated debugging time** | 1-2 days (CUDA race conditions can be non-deterministic) |
| **Expected performance impact** | None directly; this phase's value is unblocking Phase 5 |
| **Deliverables** | A written root-cause explanation (this is exactly the kind of finding worth being able to explain from memory, per the project's self-test); a minimal reproducible two-thread test; either a fix or a documented decision that batching (not a thread-safety patch) is the real solution |

### Phase 1 — Pruning Dataflow / Algorithm-System Co-Design

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
| **Dependencies** | None (independent of Phase 1) |
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

**Milestone 2b: Gather/fixup kernel (strong extension)**

| Field | Detail |
|---|---|
| **Category** | Triton |
| **Problem being solved** | The original AMAY_SPEED_PLAN.md target — separate gather/argsort/cat/pad ops in the post-selection fixup path |
| **Measured evidence** | 63.2% of the region's CUDA time in non-matmul ops |
| **Source of evidence** | `prep_summary.txt`, Experiment 1 |
| **Exact files/functions involved** | `llava_arch.py:272` (single-image gather), `:350-390` (padding/position-id/mask rebuild loop) |
| **Proposed engineering design** | As already scoped in `AMAY_SPEED_PLAN.md`: augment `encode_images()` to also return `selected_indices` directly (already computed, currently discarded into a boolean mask); write `gather_kernel()`/`triton_gather()` in a new `triton_fixup.py` doing gather + position-id remap + mask construction in one pass, operating on the shape produced by Phase 1's reorder (post-projection width) if Phase 1 has landed, or the pre-reorder shape otherwise |
| **Dependencies** | Soft: Phase 1's projector-reorder decision (changes the kernel's expected input width) |
| **Implementation tasks** | 1-4 as already itemized in `AMAY_SPEED_PLAN.md`'s own plan (augment `encode_images()`'s return, swap the single-image `else` branch's gather call, add the 3-way-unpack `, _` at the multi-image call site, write `gather_kernel`/`triton_gather`) — this roadmap does not re-derive that design, it cites it as still valid. 5. Extend to the anyres/multi-image branch (AMAY_SPEED_PLAN.md's own noted gap: "My augmentation only patches the single-image else branch; this whole branch is unaddressed"). 6. Address the per-batch main-loop syncs (`.sum()`, `.tolist()`) if batching (Phase 5) makes batch size > 1 relevant by this point. |
| **Correctness tests** | Output embeddings, position IDs, and attention mask bit-identical to the pre-kernel path, across single-image and multi-image/anyres inputs |
| **Microbenchmark / Full-system benchmark / Before-after metrics** | Same protocol as Milestone 2a, applied to the fixup region and to `prefill_lm_forward_mean_ms` |
| **Success criterion** | Fixup-region CUDA time drops with launch count; end-to-end 128-vs-576 VisPruner advantage improves further |
| **Difficulty** | High |
| **Estimated implementation/debugging time** | 3-5 days / 2-3 days |
| **Deliverables** | `triton_fixup.py`; correctness suite covering both single-image and anyres paths |

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

### Phase 5 — Batching (basic, padded)

| Field | Detail |
|---|---|
| **Category** | Batching / Serving |
| **Problem being solved** | Server capacity ceiling barely moves (2.166→2.249 rps) despite large per-request compute savings |
| **Dependencies** | **Hard: Phase 0.** Soft: benefits from Phases 1-4 (batches a cheaper request), not required |
| **Proposed engineering design** | Modify the request-handling path to accumulate a small window of incoming requests, pad their token sequences and stack their image tensors, run one batched `generate()` call, and split the outputs back out per-request. This is server/scheduling work and — per `CLAUDE.md`'s ownership rules — belongs in `csnbs/server.py`, which is **Rithvik's file**; this phase requires coordinating the design with him, not implementing it unilaterally in his directory. |
| **Implementation tasks (PR-sized)** | 1. Identify the request-batching boundary (a time window? a fixed batch size with a max wait?) — a scheduling decision to make explicitly, not implicitly. 2. Build batched tensor construction: stack images (already matching shape per request, since CLIP's input is fixed-size), pad token sequences to the batch's longest. 3. Support matching/mismatched image counts per request in the same batch (single-image only, initially — multi-image batching is a further extension). 4. Introduce prompt-length bucketing if naive padding proves wasteful (measure first, per §5's Optional-bucket reasoning — don't build this preemptively). 5. Validate output separation — each request's output must map back to the correct requester, including under concurrent completion at different lengths within the batch. 6. Benchmark batch sizes 1/2/4/8 explicitly, not just "batching on vs. off." |
| **Correctness tests** | Per-request output identical to what that same request would produce unbatched; no cross-request contamination (a critical, security-adjacent correctness property, not just an accuracy one) |
| **Microbenchmark** | Batched forward pass latency vs. batch size, in isolation from the server |
| **Full-system benchmark** | Full load sweep (§10), both configs, **properly interleaved this time** (correcting Experiment 5's §9.2 limitation) |
| **Before/after metrics** | Throughput ceiling; p50/p95/p99 at each RPS level; GPU utilization (expect it to climb closer to 100% under load, since the server can now do useful work while previously-idle-between-requests time existed) |
| **Success criterion** | Throughput ceiling measurably exceeds the current ~2.2rps; both 576 and 128's ceilings move, and the *capacity* VisPruner-advantage (per §8's formula, applied to throughput) is recomputed and reported honestly, whatever it shows |
| **Rejection/rollback criterion** | Any cross-request output contamination — treat as a blocking correctness bug, not a performance tradeoff |
| **Difficulty** | Very high |
| **Estimated implementation time** | 5-8 days |
| **Estimated debugging time** | 4-6 days |
| **Expected performance impact** | Significant for total capacity; classified SERVING OPTIMIZATION, not expected to change the per-request 576-vs-128 relative advantage |
| **Deliverables** | Batched request-handling path (co-designed with Rithvik); correctness suite; a properly-interleaved before/after load sweep |

### Phase 6 — Final Consolidated Benchmark

See §10 for the full suite definition. This phase runs every benchmark type defined there, across every optimization landed in Phases 1-5, as one coherent final report — not executed as part of this planning document.

---

## 8. The 576/128 Validation Formula, Applied Throughout

For every phase above that touches per-request latency, before claiming an optimization "recovered VisPruner's advantage," compute:

```
VisPruner advantage before = (L576_before - L128_before) / L576_before
VisPruner advantage after  = (L576_after  - L128_after ) / L576_after
additional realized pruning gain = advantage_after - advantage_before
```

Report this **alongside**, not instead of, the absolute ms improvement for each config. A PRUNING-ENABLING or GENERIC fix (Phases 0, 3, 4) that improves both configs by a similar absolute amount will correctly show `additional_realized_pruning_gain ≈ 0` — that is the expected, correct outcome for those phases, and should be reported as "reduced total latency, did not change the realized pruning advantage," not left ambiguous or oversold. Only Phases 1 and 2 (Pruning Dataflow, Triton) are expected to show a clearly positive `additional_realized_pruning_gain`; if they don't, that is itself an important, reportable finding, not a result to suppress.

Every phase's benchmark should track, where relevant: complete request latency, generate/model-path latency, multimodal-prep time, prefill time, decode time, decode ms/token, GPU time, CPU dispatch time, CUDA kernel count, GPU utilization, throughput, p50/p95 (p99 only where sample count supports it, per the project's own rule 6), and saturation behavior — the same metric set the profiling phase already established, applied consistently so before/after comparisons are apples-to-apples.

---

## 9. Definition of "Project Complete"

### MINIMUM COMPLETE PROJECT
- Phase 0 (correctness audit) completed and documented, whatever its outcome
- Phase 1 (dataflow reorder) and Phase 2a (diversity-loop Triton kernel) implemented, correctness-verified, and benchmarked with the four-way 576/128-before/after protocol
- Phase 3a (persisted Cache object) implemented and benchmarked
- A final load sweep (properly interleaved) confirming whatever the state of batching is at project end
- A final profiler-trace comparison (isolated + near-saturation, both configs) against the original profiling-phase traces, showing what changed and what didn't
- All scripts used for final validation are reproducible (checked into `scripts/`, following this repository's existing conventions) — not one-off, undocumented commands
- A final written conclusion, in the same MEASURED/DERIVED/INTERPRETATION discipline as the profiling report, stating plainly how much of the 60%→13%→4% gap was actually closed, and by how much

### IDEAL EXTENDED PROJECT
Everything above, plus:
- Phase 2b (fixup kernel), Phase 3b (append speedup), Phase 4 (CUDA Graphs), and Phase 5 (basic batching) all implemented and benchmarked
- The Optional-bucket items (§5) revisited with fresh justification if Phase 5's results create the conditions (memory pressure, measured padding waste) those items need to be evaluated honestly
- A comparison against vLLM's actual implementation of the toy versions built here (paged-KV-equivalent, batching-equivalent), per the original plan's own reasoning for why that comparison is valuable

---

## 10. Final Benchmark Suite (defined, not run)

To be executed once, after all landed optimizations, deliberately designed to fix the exploratory phase's known weaknesses (per `AMAY_VISPRUNER_PROFILING_REPORT.md` §9):

| Component | Design requirement | Fixes which prior weakness |
|---|---|---|
| Complete-request timing | Timer starts before image load, ends after response text is produced — no excluded preprocessing | §9.1 (the Part-1 timing bug) |
| Output-length policy | Run twice: once with a fixed output length (for clean, token-count-matched decode comparisons) and once with natural EOS stopping (for realistic behavior) — both explicitly labeled | §9.8 (natural-EOS length variance) |
| 576/128 ordering | Genuinely interleaved or randomized at the individual-trial level, not run as two separate blocks | §9.2 (non-interleaved load sweeps) |
| Warm-up | ≥10 discarded iterations per configuration, per the project's existing rule 3 | (already correctly done throughout; preserved) |
| Repeated trials | Enough trials to establish this specific system's noise floor before trusting any delta, per rule 8 — not assumed from the exploratory phase's single-pass numbers | General rigor |
| Sample count | Load-sweep durations/RPS chosen so p95 has ≥100 samples and p99 has ≥1000 where p99 is reported at all; explicitly omit p99 where it isn't | §9.5 (n=5 p95≈p99 collapse) |
| Load sweep | RPS levels spanning clearly-under-capacity through clearly-saturated, for both configs, run in alternating order | §9.2 |
| Percentiles | p50, p95 always; p99 only where the sample-count requirement above is met | §9.5 |
| Throughput | Achieved vs. offered rate at every level, both configs | Preserved from the existing methodology |
| Profiler trace | Full CPU+CUDA trace at 576 and 128, both isolated and near-saturation, directly diffable against the profiling-phase traces (same `record_function` label scheme) | Enables a clean before/after comparison |
| Kernel-launch comparison | Total `cudaLaunchKernel` count, before vs. after, both configs | Directly validates Phases 2 and 4's claims |
| GPU utilization | Continuous background-thread sampling during every load level, as already established | Preserved from the existing methodology |
| GPU memory | Continuous sampling; specifically watched under Phase 5's batching to see whether paged KV becomes justified | Directly informs the Optional-bucket revisit criterion in §5 |
| Correctness/output checks | Exact-text comparison against the pre-optimization baseline for every non-generic-speedup change (Triton kernels, KV-cache change, batching); accuracy-preserving (not just "looks similar") checks per the project's own rule 15 |

This suite is **defined here, not executed** — per this task's explicit instruction not to run any benchmark now.

---

## 11. Master Table

| Phase | Category | Main technique | Problem solved | Deliverable | Success metric | Dependency | Core/Extension | Est. effort |
|---|---|---|---|---|---|---|---|---|
| 0 | Serving Correctness | Root-cause + fix the concurrency crash | Shared model instance unsafe under concurrent access | Fix or documented decision + 2-thread repro | N/N repro runs pass | None | Core | 1.5-3 days |
| 1 | Pruning Dataflow | Projector reorder + prefill investigation | Wasted projector FLOPs; unexplained prefill sub-linear scaling | Reordered code + written finding | `mm_projector` cost scales with T; prefill mechanism named | None | Core | 1.5-2 days |
| 2a | Triton | Diversity-loop fusion | ~56-iteration Python loop, pruning-specific tax | `triton_diversity_select.py` | `multimodal_prep` at T=128 ≤ T=576's | None | Core | 5-9 days |
| 2b | Triton | Gather/fixup kernel | 63.2% non-matmul time in fixup region | `triton_fixup.py` | Fixup-region launch count drops | Soft: Phase 1 | Strong Extension | 5-8 days |
| 3a | KV-Cache | Persisted Cache object | Per-step `from_legacy_cache` rebuild | Modified `generate()`/`forward()` | Rebuild call count → 0 | None | Core | 2-4 days |
| 3b | KV-Cache | Fixed-size buffer | `torch.cat` reallocation per step | Preallocated buffer | Stable tensor addresses (verified) | 3a | Strong Extension | 4-6 days |
| 4 | CUDA Graphs | Decode-step capture/replay | 34k+ launches/request, CPU>CUDA per step | Graph-capture implementation | Decode ms/step drops, both configs | **3a + 3b** | Strong Extension | 6-9 days |
| 5 | Batching | Basic padded batching | Server serializes; capacity ceiling flat | Batched request path (co-designed w/ Rithvik) | Throughput ceiling measurably exceeds ~2.2rps | **Phase 0** | Strong Extension | 9-14 days |
| 6 | — | Final consolidated benchmark | Validate everything above together | Final report | See §9's completion criteria | All landed phases | Core (as a step; content depends on what landed) | (defined in §10, not estimated here) |

### Direct answers

1. **Exact recommended implementation order:** Phase 0 → Phase 1 → Phase 2a → (2b, 3a in either order or parallel) → 3b → 4 → 5 → 6.
2. **Which tasks can happen independently:** Phase 0, Phase 1, Phase 2a, and Phase 3a are mutually independent — any order or parallel work among them is fine.
3. **Which tasks depend on earlier phases:** Phase 2b softly depends on Phase 1; Phase 3b hard-depends on 3a; Phase 4 hard-depends on 3a+3b; Phase 5 hard-depends on Phase 0.
4. **Where Triton fits:** Phase 2 — independent of KV-cache/CUDA-Graphs/Batching entirely; only its second milestone (2b) has a soft dependency on Phase 1.
5. **Where KV-cache fits:** Phase 3 — independent of Triton and Pruning Dataflow; is the hard prerequisite for CUDA Graphs.
6. **Where CUDA Graphs fit:** Phase 4 — strictly after KV-cache (3a+3b); independent of Batching and of Triton.
7. **Where batching fits:** Phase 5 — hard-depends only on Phase 0 (correctness), not on Triton/KV-cache/CUDA-Graphs, though doing it last means it batches an already-cheaper request.
8. **Which new categories were necessary:** Pruning Dataflow / Algorithm-System Co-Design, and Serving Correctness / Concurrency Safety — both justified by specific code-level and crash-log evidence, not created for taxonomic tidiness.
9. **What counts as project complete:** see §9 — Minimum requires Phases 0, 1, 2a, 3a plus a final interleaved load sweep, trace comparison, reproducible scripts, and an honest written conclusion; Ideal adds 2b, 3b, 4, 5, and a revisit of the Optional bucket.
10. **What the final benchmark suite contains:** see §10 in full — defined, not run.
11. **Which optimization should be implemented first:** **Phase 0** (the correctness audit) — it is the cheapest phase, it de-risks the highest-effort later phase (Batching) early rather than late, and understanding the crash may also inform how Phase 2's Triton kernels should be written to avoid the same class of bug. If a narrower "first *performance* optimization" answer is wanted specifically: **Phase 2a, the diversity-loop Triton fusion** — it has the strongest, most directly quantified evidence of any candidate in this roadmap (56 measured iterations against a 56-iteration code-level prediction) and is the clearest PRUNING-SPECIFIC win available.
