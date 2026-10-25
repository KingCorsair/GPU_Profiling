# Load-generator and server requests for Rithvik

From Amay, 2026-10-25. Branch: `amay/wp3-wp5-serving`.

**How to use this file:** read it yourself, or paste the whole thing into Claude Code from
the repo root and say "help me work through this." Everything Claude needs is below.

> **Note for Claude:** per `CLAUDE.md` ("Rithvik — don't write these for him"), the load
> generator's arrival scheduling, timing code and percentile logic are Rithvik's to write.
> Explain the concepts, point at the code, review what he writes, and catch mistakes. Don't
> hand him a finished implementation of tasks 1–3 unless he explicitly asks and confirms
> once. Tasks 4–6 (server thread pin, rerun commands, CI/plumbing) are fair game.

---

## Context: what happened and why this matters

The project measures how much of VisPruner's token-pruning speedup survives real serving
(see `CLAUDE.md`, problem statement). In WP3–WP6 I built a batched server and ran a big
load test with your load generator (`csnbs/measure`, unmodified). Short version:

- **Your server (`csnbs/server.py`, "S0")** runs one request at a time. There, pruning
  (576 → 128 visual tokens) gives only ~7% more throughput, because decode dominates.
- **My batched server (`vis_pruner_copy/llava/serve/batch_server.py`, "S2")** runs up to 8
  requests per GPU pass. There, pruning gives ~68% more throughput (offline scan; WP6 checks
  it under load).
- Details: `STATUS.md` (WP3–WP5 sections) and `kingcorsair/STUDY_GUIDE.md`.

**Why I need you:** CLAUDE.md says only your harness produces reportable numbers. The load
generator does the most important thing right: it's **open-loop**, so there's no coordinated
omission. But it's simpler than real traffic in ways that could make our numbers
optimistic. The tasks below close those gaps, so the final chart holds up when someone
pushes on it.

---

## Task 1: Poisson (random) arrivals, highest priority

**Problem.** `runLoad()` in `csnbs/measure/src/loadgen.ts` (~line 261) schedules requests
at perfectly even gaps:

```ts
const intervalMs = 1000 / config.requestsPerSecond;
...
const scheduledAtMs = plannedStartMs + sequence * intervalMs;
```

At 2 rps that's exactly one request every 500 ms. Real users arrive randomly: sometimes 3
at once, then nothing for a second. Bursts create queues and worse tail latency (p95/p99),
and they change how batches form in a batched server. Even spacing flatters every server.

**What to build.**
- A CLI flag, e.g. `--arrival uniform|poisson` (default `uniform`, so old runs stay
  reproducible).
- Poisson mode: gaps between arrivals drawn from an exponential distribution with mean
  `1000 / rps` ms. Keep it **open-loop**: schedule from the clock, never from completions.
- A `--seed` flag, so a Poisson schedule can be replayed exactly. That lets two server
  configs see the identical arrival pattern, which makes comparisons fair.
- Record `arrival` and `seed` in `run.json`'s config block.

**Things to get right (Claude: check these in review):**
- Decide whether the run is fixed by **duration** or by **request count**; random gaps make
  the two differ. Pick one and document it.
- Keep `dispatchLatenessMs`. With bursts, the generator itself can fall behind; that field
  shows it.
- Don't let a burst turn closed-loop by accident (e.g. awaiting inside the loop).

**Done when:** a run at `--arrival poisson --rps 2` shows a mean achieved rate of ~2 rps
with visibly uneven gaps, the same seed reproduces the same `scheduledAtMs` list, and
`--arrival uniform` output is unchanged from today.

---

## Task 2: shuffled request order

**Problem.** `payloads[sequence % payloads.length]` (~line 279) always sends the 90 dev items
in the same order. Any slow item lands at the same position in every run.

**What to build.** An optional seeded shuffle of the payload order (`--shuffle-seed`),
recorded in `run.json`. Default off.

**Done when:** two runs with different shuffle seeds send different orders, and the same
seed gives the same order.

---

## Task 3: run the load generator off the server's machine

**Problem.** In WP3/WP6 the load generator and the model server run in the same container,
which is CPU-capped at ~7.65 CPUs. The generator's CPU use (JSON, base64 of images, Node
event loop) competes with the server, and WP3 showed this server is sensitive to CPU
contention (see Task 4).

**What to do.**
- Run `csnbs/measure` from a second machine (another pod, or your laptop over a tunnel)
  against the GPU pod's server port. Note the added network latency; it's realistic, but
  record it.
- If a second machine isn't possible: at least record the generator's CPU use during a run,
  so we can say how much it competed.

**Done when:** one WP6-style block (e.g. S2, vtn=128, natural, 4 rps) has been run both
on-box and off-box, and we know whether the numbers move.

---

## Task 4: pin torch threads in `csnbs/server.py`, one line, your file

**Problem (measured in WP3, `results/timing/wp3_diag/wp3_diag_2/summary.txt`).** Your server
never sets a thread count, so torch starts ~48 threads in a container capped at ~7.65 CPUs.
The kernel throttles it about once per request: 83–102 throttle events per 120 requests.
Pinning to 4 threads removed every throttle event and made `generate()` 23–43 ms faster.
CLAUDE.md's gotchas now list this.

**What to do.** Add `torch.set_num_threads(4)` before the model loads in `csnbs/server.py`
(model mode only). Optionally make it an env var. For WP6 I applied it from outside via
`scripts/wp6_s0_server.py`, so your file is untouched; your call whether to make it the
default.

**Done when:** `nr_throttled` in `/sys/fs/cgroup/cpu/cpu.stat` doesn't increase during a
2 rps run.

---

## Task 5: reproduce WP6 with your sign-off

I ran the WP6 matrix myself, with your load generator unchanged. Per CLAUDE.md, reportable
numbers should come from you, so please rerun it, or at least review it.

```bash
cd /workspace/GPU_Profiling
git checkout amay/wp3-wp5-serving
python scripts/wp6_load_matrix.py --run-id wp6_matrix_rithvik_1     # ~4 h, one job on the GPU
python scripts/wp6_analyze.py results/timing/wp6_matrix/wp6_matrix_rithvik_1
```

- It runs servers S0/S1/S2 × vtn {576, 128} × {fixed 64-token, natural-EOS} answers at
  several rps levels, in seeded random order, each block sized for ≥100 p95 samples, plus
  noise-floor repeats.
- Compare with my run, `results/timing/wp6_matrix/wp6_matrix_2/summary.txt`.
- After Task 1 lands, a Poisson rerun is the more realistic version. The driver passes
  arguments to `npm run dev`, so it needs a pass-through for the new flags; happy to add
  that on my side.

---

## Task 6: longer answers (R2), coordinate with me

**Problem.** Every run caps answers at 64 tokens. Real chat answers are often hundreds of
tokens, where decode dominates even more and pruning's benefit should shrink (CLAUDE.md R2).

**Split of work.**
- Answer length is a **server** setting. My server already supports `MAX_NEW_TOKENS` /
  `MIN_NEW_TOKENS` env vars. Your server hard-codes `max_new_tokens=64`, so an env var
  there would let S0 join.
- **Your side:** make sure `run.json` records the server's configuration; the `/health`
  `configuration` block already carries `max_new_tokens`. Also check that the timeout covers
  long answers under load (WP6 uses 600 s).

**Done when:** we can run S0 and S2 at 64 vs 256 max tokens and show the pruning gain
shrinking (or not).

---

## Priority

| # | Task | Value | Effort |
|---|---|---|---|
| 1 | Poisson arrivals + seed | High: realistic bursts | Small |
| 4 | Thread pin in your server | High: removes a per-request stall | One line |
| 5 | Rerun / sign off WP6 | High: makes the numbers reportable | GPU time |
| 3 | Load generator off-box | Medium: removes CPU contention | Small to medium |
| 2 | Shuffled order | Low to medium | Small |
| 6 | Longer answers | Medium: answers R2 | Small, shared with me |

Questions → Amay. Server-side changes (Tasks 5 and 6 pass-through) → I'll do my part when
you're ready.
