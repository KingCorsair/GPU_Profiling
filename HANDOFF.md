# Handoff — resume point (2026-10-02)

The full detail is in `STATUS.md`. This file covers only what's needed to pick up again.

## Where we are

**WP2 — FP16 Vision Tower is closed.** All four roadmap §9 criteria hold, and the FP16 change (`caf19fe`) stays. **WP3 — Server Diagnostic is next.**

## Where the WP2 close-out lives

* `scripts/wp2_layer1_correctness.py`, `scripts/wp2_layer2_textvqa.py`, `scripts/download_textvqa_subset.py`
* `results/wp2_correctness/` (Layer 1 and Layer 2 outputs)
* `results/timing/wp2_profile/` (the FP32 and FP16 summaries; the raw `*_trace.json` files are git-ignored and exist only on the pod)

The Layer 1 and Layer 2 outputs record commit `be4ed77` with a dirty tree, because the scripts that produced them were committed afterwards, in the close-out commit. The model code they ran is `be4ed77`'s.

## WP2 results in one place

| | Result |
|---|---|
| `vision_tower` p50, bench_dev A/B | 33.98 → 14.35 ms at 576, 34.00 → 13.68 ms at 128 (−58% / −60%) |
| FP32 `ampere_sgemm_*` share of prep GPU time | 75.4% → 0.0% |
| Softmax and GEMM accumulate | still FP32 |
| Exact text, FP32 vs FP16, 90 dev images | 100% identical at 576, 88.9% at 128 |
| Determinism, FP16 twice | identical at both configs |
| TextVQA, 1,000 questions | 57.62% → 57.65% at 576; 56.50% → 56.75% at 128 |
| Feature cosine > 0.999 | missed in 6 of 90 images (min 0.99816) |
| Token selection at 128 | different on every image; 74.4% of kept tokens shared |

## Left open from WP2

1. **Send the Layer-3 request to Sribhav.** Per-category accuracy on the locked eval set, FP32 (`e295fd0`) vs FP16 (`caf19fe`), with OCR and counting at 128 called out: those are the categories where answers changed in Layer 1. Non-blocking. If OCR regresses there, revert the one-line change.
2. **End-to-end speed-up** is left to WP6. Do not quote one from WP2.

## Next task: WP3 — Server Diagnostic

Roadmap WP3: measure how many milliseconds per request are CPU work that could overlap GPU work but doesn't. It gates WP4.

A first attempt was started and stopped on 2026-10-02. It is parked outside the repo at `/workspace/wp3_parked/` (see its `README.txt`):

* `scripts/wp3_diag_server.py` serves `csnbs/server.py` unmodified and instruments it from outside: an event-loop heartbeat and per-request stage timestamps. Nothing in `csnbs/` changes.
* `scripts/wp3_run_diag.py` drives it with the existing `csnbs/measure` load generator, with the two configs and a 4-thread pin interleaved.
* `scripts/wp3_analyze.py` produces the service-time decomposition.
* `results/timing/wp3_diag/wp3_diag_1` was **stopped early**: 4 of 8 sessions are complete. Do not treat it as a finished run.

Nothing from that attempt has been analysed or reported. To resume, move the scripts back into `scripts/` and start a fresh run ID.

Before running WP3: agree the instrumentation approach with Rithvik, since it concerns his server.

## Open carry-overs

* **`/workspace` quota:** writes failed at about 59 GB used. The volume looks capped near 60 GB; about 53 GB is in use.
* **Stash:** `git stash list` holds 10 stray rows from `results/timing/llava_llama_timing.json`. Safe to drop.
* **WP3:** the server sets no torch thread count, so it likely hits the throttling stall.
* **Git host key:** add GitHub's host key to `~/.ssh/known_hosts` in the Dockerfile's boot `CMD`.
* **`CLAUDE.md`:** add the CPU-throttling stall and the missing `known_hosts` entry to the gotchas.
