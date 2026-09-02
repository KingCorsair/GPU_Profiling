# Demo plan — frontend + backend for the project demo

Shared plan, not any one person's doc — the demo needs input from all three of us:
Amay owns "Playground UI" per CLAUDE.md's team table, but the data behind it is Rithvik's
loadgen harness (Speed axis) and Sribhav's eval results (Accuracy axis). Answers: what does
the demo need to show, what already exists to build it on, and what's missing before it can
show real numbers instead of mock data.

---

## What the demo has to prove

Not "pruning makes things faster" — every pruning paper already claims that. The thing
nobody else has shown (CLAUDE.md's problem statement) is that published speedups are
measured single-request and evaporate under concurrent serving load — and that fixing the
serving path (real removal, smaller KV cache, no padding waste) recovers it. The demo's job
is to make that gap visible, live, on screen — not to re-explain it in prose.

**Worth naming explicitly:** CLAUDE.md already states the project's actual deliverable —
*"one chart: accuracy against measured throughput."* That's not a Speed-only chart. Whatever
we build here should either **be** that chart made interactive, or sit right next to it —
not a second, disconnected thing that only tells the speed half of the story. That's a
call worth making as a group, not something to decide unilaterally in this doc (see
"Open questions").

**The other half, if we include it:** accuracy vs. compression, at the same
`visual_token_num` settings — from Sribhav's eval harness. Together: does the model still
get faster (and stay accurate) as compression increases, under real load — the full
problem statement, not just the speed slice of it.

---

## The two plots — decided 2026-08-29

Same axes on both: X = request rate (rps, open-loop), Y = latency (p50/p95/p99).

Dropped the third (real vLLM vs. our optimized+pruned stack) — real infra cost (standing up
and benchmarking an actual vLLM instance) for a stretch payoff. These two fully prove the
thesis on their own; revisit plot 3 later only if there's time to spare.

**Plot 1 — pruning alone doesn't help under load. (required)**
Baseline (576 tokens) vs. pruned, both run on the **current, unoptimized** fixup path (no
Triton kernel, no paged/bucketed batching — naive gather + padded batching). Expected: the
two lines nearly overlap, especially as rps climbs. This is the problem statement, proven —
published speedup theory doesn't survive real serving as-is.

**Plot 2 — our engineering makes the promise real. (required)**
Same two settings, now on the **optimized** pipeline (Triton fixup kernel + paged/bucketed
batching, per AMAY_SPEED_PLAN.md). Expected: pruned pulls away from baseline as rps climbs —
the crossover. This is the fix, working.

Both only need the loadgen sweep already planned above (no new infra).

---

## Interaction design

Two sliders, one chart (speed half):
- **Compression** — `visual_token_num` (576 → e.g. 64), the model's actual pruning setting.
- **Load** — requests/sec, open-loop (rule 5 — no closed-loop coordinated omission).

Chart: latency (p50/p95/p99) vs. rps at the current compression setting, with the 576-token
baseline always shown faint underneath for comparison.

If the accuracy axis is in scope: a second small panel or toggle showing accuracy at that
same compression setting, so the compression slider drives both charts at once.

Two data modes:
- **Replay (default)** — scrub across pre-recorded, validated runs from Rithvik's loadgen
  harness and Sribhav's eval harness. This is what gets shown live in an interview — no GPU
  pod has to be up, and per CLAUDE.md only Rithvik's harness produces speed numbers that
  count, and only Sribhav's validated scorer produces accuracy numbers that count.
- **Live (stretch)** — if a pod is up, hit the real `/infer` endpoint and show it update in
  real time. Cool if it works, not the fallback plan.

---

## What already exists to build on

- `csnbs/server.py` — FastAPI `/infer` endpoint, `fake`/`model` mode. Currently loads the
  model **once at startup** with `visual_token_num=576` hardcoded (`server.py:46`) — not
  configurable per request yet.
- `csnbs/measure/src/loadgen.ts` — Rithvik's open-loop load generator. Takes `--rps`, writes
  `run.json` + `requests.jsonl` per run.
- `results/loadgen/2026-08-29/*/run.json` — 10 real runs already recorded, rps swept 1 → 2,
  `p50/p95/p99` already computed per run. Real, validated data — usable in replay mode
  today, as far as it goes (see gaps below).
- `sribhav/evaluation_results_normalized.json` — real, per-question accuracy scores
  (`bert_f1`, `nli_score`, `composite_score`) across 90 questions, with an aggregate
  `composite_score_accuracy: 0.41`. This is one run's worth of validated accuracy data —
  but it isn't tagged with which `visual_token_num` produced it, so it can't yet be plotted
  against compression.

---

## What's missing before the demo shows real numbers

This is the actual gap, and it's mostly not frontend work:

1. **The compression axis doesn't exist in the loadgen data yet.** Every recorded run has
   `benchmark.pruningRatio` / `tokenRemovalMode` = `null` — all 10 runs are the same
   (baseline) config, only rps varies. The schema already has fields for this
   (`loadgen.ts:81-82`, hardcoded `null` at `:508-509`) — wired for it, just not populated.
2. **The server can't switch `visual_token_num` without a restart.** It's a
   `load_pretrained_model()` argument (`server.py:46`), set once at process start. To sweep
   compression, either:
   - (a) make it a per-request parameter — needs threading through the model's forward
     path, which is core token-removal/serving-path code (Amay's protected list in
     CLAUDE.md, not something to hand off), or
   - (b) run N server processes, one per `visual_token_num` setting, and point the loadgen
     at each in turn — no change to protected model code, just an ops/scripting problem.

   (b) is the pragmatic path: cheaper, doesn't touch protected code, and the loadgen
   already supports pointing at an arbitrary `--endpoint`.
3. **No aggregation step exists yet** to turn N `run.json` files (one per compression × rps
   cell) into the single matrix the frontend chart needs.
4. **Sribhav's eval results aren't swept across compression either** — same shape of gap as
   #1, on the accuracy side. `evaluation_results_normalized.json` is one config's worth of
   scores; the demo needs one accuracy number per `visual_token_num` setting, ideally the
   **same settings** used in the loadgen sweep so both axes line up on one x-axis.
5. **Per rule 12 (include tasks compression should break):** if the accuracy axis goes in,
   it should probably show per-category accuracy (OCR, counting, spatial reasoning vs.
   plain description), not just one overall number — otherwise the chart quietly hides
   exactly the failure mode the project's eval design exists to catch.

---

## Backend for the demo — what it actually is

Not a new inference service. A thin **read layer** in front of both Rithvik's and Sribhav's
stored results:

- A script that walks `results/loadgen/**/run.json`, pulls
  `(visual_token_num, rps) → {p50, p95, p99, achievedRps}`.
- A script that walks Sribhav's eval output, pulls `visual_token_num → {overall accuracy,
  per-category accuracy}`.
- Both merge into one `demo_matrix.json`, keyed by `visual_token_num` so the frontend can
  drive both charts off one slider.
- A small FastAPI app (or even a static file) serving that JSON to the frontend. No model,
  no GPU dependency for replay mode — this is why replay mode can run anywhere, laptop
  included.
- Live mode (stretch): a real endpoint that kicks off a short loadgen run against `/infer`
  on demand and streams results back. Bigger scope — only after replay mode is solid.

This aggregation script and the thin serving API are plumbing, not "serving path" or
"scoring logic" in the CLAUDE.md-protected sense for either Amay or Sribhav — fair game to
write directly, no confirm-once gate needed. Building the sweeps themselves (running
Rithvik's loadgen N times, running Sribhav's eval N times) is each owner's own harness,
same as always.

---

## Frontend

- **Stack:** React + Vite, Recharts — matches Rithvik's stack per CLAUDE.md, so it's
  consistent with the rest of the project's UIs.
- **Components:**
  - `CompressionSlider` — `visual_token_num`, discrete steps matching whatever settings
    actually got benchmarked (not continuous — can't honestly interpolate an un-run config).
    Drives both charts below.
  - `LoadSlider` — rps, discrete steps matching recorded runs. Drives the latency chart only.
  - `LatencyChart` — p50/p95/p99 vs rps, pruned (solid) vs baseline (faint/dashed) overlay.
  - `AccuracyPanel` (if in scope) — accuracy at the current compression setting, overall and
    per-category, pruned vs baseline.
  - `ModeToggle` — replay vs live.
  - `StatStrip` — small readout under the charts: current token count, % tokens dropped,
    measured speedup, measured accuracy delta at this exact setting. The skim-friendly
    summary for a viewer who won't read the charts closely.
- **Data:** fetch `demo_matrix.json` once; everything else is client-side
  slider → lookup → re-render. No backend round-trip per slider move in replay mode — keeps
  it snappy live.

---

## Sequencing

1. Agree as a group whether the accuracy axis is in scope for v1 (see open questions) —
   changes what data needs to exist before frontend work is worth starting.
2. Run the missing sweeps at agreed `visual_token_num` settings:
   - Rithvik: loadgen against N server processes (one per setting).
   - Sribhav: eval harness at the same N settings, same eval set.
3. Aggregation script → `demo_matrix.json`.
4. Frontend: sliders + chart(s) against the real matrix, replay mode only.
5. Stretch: live mode.

---

## Open questions — need a group answer before step 1

- **Is the accuracy axis in v1, or speed-only for now, with accuracy added later?**
  Speed-only is faster to ship; accuracy-included is the actual project deliverable chart,
  interactive. Worth deciding together, not defaulting silently either way.
- Which `visual_token_num` values are worth benchmarking? Needs to be the same list for
  both the loadgen sweep and the eval sweep, agreed by Rithvik and Sribhav together —
  576 baseline + probably 3-4 more points, enough to make the crossover visible without a
  dense sweep.
- Same rps range as the existing 10 runs (1-2), or push higher to guarantee the saturation
  crossover actually shows up?
- Does the aggregation/serving layer live in `kingcorsair/`, or get its own top-level
  `playground/` dir now that it reads from more than one person's results directory? Given
  "don't write code in someone else's directory," and this only *reads* others' results
  without writing to them, a shared top-level dir (not owned by any one person's folder)
  might be the more honest home once it's genuinely a team artifact.
