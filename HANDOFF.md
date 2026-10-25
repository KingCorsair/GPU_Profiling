# Handoff — resume point (2026-10-25)

The full detail is in `STATUS.md`; the conclusions are in `AMAY_MONTH_ONE_RESULTS.md`. This
file covers only what's needed to pick up again.

## Where we are

**Month one is complete.** WP1–WP7 are closed. All work is on `main` and
`amay/wp3-wp5-serving` (identical) on GitHub.

## The result in one table (MEASURED, `wp6_matrix_2`)

| Max throughput (req/s) | 576 | 128 | Pruning's gain |
|---|---|---|---|
| Old server, one request at a time (S0), fixed / natural | 0.47 / 2.56 | 0.50 / 2.91 | +6% / +14% |
| Batched server, B ≤ 8 (S2), fixed / natural | 1.95 / 4.36 | 2.85 / 6.82 | **+46% / +57%** |

Pruning barely helps one request at a time (decode dominates); batched, it pulls away.
The queue alone (S1) gave no measurable gain.

## Where things live

| What | Where |
|---|---|
| Batched server and engine | `vis_pruner_copy/llava/serve/batch_server.py`, `batch_engine.py` |
| Model fixes for B>1 | `llava_arch.py` (per-image split and selection), `builder.py` (left padding), `llava_llama.py` (thread-local timing) |
| WP3 diagnosis | `scripts/wp3_*.py`, `results/timing/wp3_diag/wp3_diag_2/` |
| WP5 correctness | `scripts/wp5_*.py`, `results/wp5_validation/` |
| WP6 load matrix and plots | `scripts/wp6_*.py`, `results/timing/wp6_matrix/wp6_matrix_2/` |
| Study guide (for Amay) | `kingcorsair/STUDY_GUIDE.md` |
| Requests to teammates | `kingcorsair/SRIBHAV_ACCURACY_REQUESTS.md`, `kingcorsair/RITHVIK_LOADGEN_REQUESTS.md` |

## Open, not blocking

1. **Rithvik:** rerun or sign off WP6 so the numbers are reportable; Poisson arrivals and
   the other load-generator improvements; the one-line thread pin in `csnbs/server.py`.
2. **Sribhav:** WP2 Layer-3 per-category check on his **dev split** (not the locked test
   set; rule 11), and a scoring check of the one batched answer that differs.
3. **Not done:** a clean-checkout rerun of every script (WP7 §5.3).

## Next, if continuing (month two, all optional)

1. **vLLM port.** Plan in `AMAY_ENGINEERING_ROADMAP.md`, "OPTIONAL — vLLM Port". 8–13 days.
2. **Batch size 16 in the server.** The offline scan shows the pruning gap widening
   (1.47× at B=8 → 1.68× at B=16).
3. **Longer answers (R2).** `MAX_NEW_TOKENS=256` on the batched server.
4. **Why S1 recovered nothing,** despite WP3's 5–8% overlap bound.

## Gotchas from this session

* Long GPU runs launched from a Claude Code session die when the session ends. Launch them
  detached: `setsid nohup python scripts/... > results/....log 2>&1 < /dev/null &`.
* `generate()` appends a timing row per call to the tracked
  `results/timing/llava_llama_timing.json` unless `LLAVA_TIMING_FILE` is set. The serving
  and validation scripts now point it at `/tmp`.
* Pushing from the pod works over HTTPS through VS Code's GitHub login. The SSH remote has no
  key on this pod.
