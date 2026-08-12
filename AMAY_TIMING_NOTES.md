# Timing notes (Amay)

Personal study notes on what to time and why, for the speed side of this project. Not a
spec anyone else needs to follow — written so I can explain this from memory later (the
self-test in CLAUDE.md). Read alongside `CLAUDE.md`, especially "Key technical concepts"
and the non-negotiable benchmarking rules.

---

## The core idea

A single inference request breaks into three phases, and the whole reason to time them
separately is that each one responds differently to the thing I'm actually changing
(`visual_token_num`, i.e. `n_tokens`):

| Phase | What it is | Bound by | Reacts to `n_tokens`? |
|---|---|---|---|
| **Load** | Read weights off disk, move to GPU, CUDA context init | Disk/IO, one-time | No — same checkpoint every run |
| **Prefill** | One forward pass over the whole prompt (image + question tokens) | Compute | Yes — this is the phase pruning shrinks |
| **Decode** | Autoregressive, one output token at a time, each step re-reading the weight matrix | Memory bandwidth | No, not meaningfully |

If I only measure total wall time, I'm summing a constant (load) with a variable I care
about (prefill) and a variable I don't (decode). A real speedup in prefill can get buried
by either of the other two, and I can't tell which happened without the split. That's not
a nice-to-have — it's the difference between a measurement and a guess.

Everything else in this file is downstream of that one table.

---

## Why this specific split and not some other one

Compute-bound vs. memory-bound is the actual mechanism that determines whether removing
visual tokens can help at all. It's not an arbitrary place to draw a line:

- Prefill is a big matmul over every token at once → fewer tokens → less compute → faster.
- Decode reads the entire weight matrix per token regardless of how many visual tokens
  survived prefill → pruning doesn't touch this cost.
- Consequence: **the longer the generated output, the more decode dominates, the less
  pruning helps.** ScienceQA answers are one letter → near-pure prefill → an *upper bound*
  on the benefit, not the general case. The heterogeneous set allows up to 128 generated
  tokens → decode is not automatically negligible there, and I should check actual answer
  lengths before assuming it is.

---

## What this feeds (the results table)

Different rows in CLAUDE.md's results table need different timing setups — not all of
this is "the same measurement, more precise":

- **R1 (masked vs. real removal)** — same load/prefill/decode harness, run against FastV
  (masked) and VisPruner (real removal) at matched settings. Not a new timing technique,
  just running the existing one twice to expose the FLOP-vs-wall-clock gap.
- **R2 (prefill/decode split at varying output lengths)** — needs prefill and decode
  measured *separately*, not just generation-as-a-whole. This is the one that actually
  requires the deeper instrumentation below.
- **R3 (single vs. concurrent throughput)** — a different measurement paradigm entirely:
  not one subprocess run sequentially, but a live server under open-loop arrivals with
  percentile latencies. Rithvik's load generator, my serving path.
- **R4 (padded vs. bucketed vs. packed batching)** — same request-latency measurement as
  R3, varying batching layout instead of `n_tokens`.
- **R5 (KV cache / concurrency ceiling)** — sweep concurrency at each `n_tokens` setting,
  watch p99 latency (or memory) as a function of load, not a single-request number at all.

So "what am I timing" always comes back to which row I'm trying to produce evidence for.

---

## Where each granularity of timing actually lives

Three different files, not one:

1. **Load vs. generation split** — inside the eval script itself (`model_vqa_science.py`,
   and now `model_vqa_heterogeneous.py` with Sribhav's go-ahead). Bracket
   `load_pretrained_model()`, bracket the question loop. `perf_counter()` is fine here —
   by the time each bracket closes, the relevant work has actually happened.

2. **Whole-subprocess wall time + the sweep** (repeats, randomized order, CSV output) —
   a harness script in `scripts/`, e.g. `time_sqa_sweep.py`, that launches the eval script
   as a subprocess and times the whole thing. This is exempt from the "CUDA events, not
   `time.time()`" rule — by process exit, every kernel has finished. Entirely my own
   directory, no coordination needed.

3. **Prefill vs. decode split** — not in either eval script. It's inside
   `llava/model/language_model/llava_llama.py`'s `generate()`, at the boundary between
   `prepare_inputs_labels_for_multimodal` (prunes visual tokens, builds `inputs_embeds`)
   and `super().generate(...)` (HuggingFace's generation loop). The first `forward()` call
   inside that loop is prefill; every call after is one decode step. This needs **CUDA
   events, not perf_counter**, since `forward()` is async GPU work — a wall clock here
   would measure how long Python took to queue the call, not how long the GPU took to run
   it. This file is shared by both eval scripts and anything else built on the model, so
   don't touch it until the coarser split actually shows decode is big enough here to be
   worth isolating.

---

## Rules that bite specifically here (from CLAUDE.md)

- **`perf_counter()`, not `time.time()`** — monotonic, can't be skewed by a clock
  adjustment mid-run. Applies to every boundary above except direct GPU-op timing.
- **Sync outside the timing region, never inside a loop** — an inner sync drains the
  pipeline and inflates fast kernels. Only sync right at the boundary I'm measuring.
- **Model load is constant overhead — report it separately, always.** Never let it hide
  inside a "total time" number.
- **Know the noise floor before trusting a delta.** Repeat the baseline (`n=576`) several
  times before comparing against it — matters more on small sets (90 questions) than large
  ones (4241), since there's less to average the noise over.
- **Randomize run order** across settings so thermal drift doesn't get confounded with the
  setting under test.
- **Only Rithvik's harness produces reportable numbers.** Everything in this file is for
  finding bottlenecks and checking whether a change helped — mine is throwaway by design.

---

## Sequencing (what I'm actually doing, in order)

1. Load/generation split on the heterogeneous set, same pattern as ScienceQA.
2. Whole-subprocess sweep harness for the heterogeneous set, reusing the `time_sqa_sweep.py`
   pattern.
3. Check actual generated-answer lengths on this dataset before assuming decode is
   negligible.
4. Only if decode turns out to matter: prefill/decode split inside `llava_llama.py`, with
   CUDA events.
