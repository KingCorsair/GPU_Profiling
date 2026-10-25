# Study guide: WP3–WP6 (what was built, and how to learn it)

Everything is on branch `amay/wp3-wp5-serving`. Go top to bottom. Don't move to the next
WP until you can explain the current one out loud without looking.

---

## The big picture, in four lines

- The old server (Rithvik's `csnbs/server.py`) handles **one request at a time** and
  **freezes** while the GPU works.
- WP3 measured *why* the GPU sits idle. WP4 built a server that doesn't freeze.
  WP5 made it run **several requests together** (batching). WP6 measures all three servers
  under load.
- Batching is where the speed comes from: the GPU reads the model weights once per step,
  no matter how many requests share that step.
- Pruning (128 tokens) helps much more once you batch: about 1.07× faster alone, about
  1.7× faster in a batch of 16.

---

## WP3: find where the old server wastes time

**What I did.** I watched the old server under load, without changing its code. I added
timestamps around each step, plus a "heartbeat" that checks every 5 ms whether the server
is frozen.

**What we found.**
- Only ~25 ms per request is CPU work that could run alongside the GPU, which is 5–8% of
  each request. So a queue alone would only help a little.
- The bigger problem: torch used 48 threads in a container capped at ~7.6 CPUs, so the
  server froze once per request. Pinning to 4 threads fixed it and saved 23–43 ms.
- The server doesn't even *receive* the next request until the current one is done.

**Read:**
- `scripts/wp3_diag_server.py`: how to measure someone's server without editing it.
- `results/timing/wp3_diag/wp3_diag_2/summary.txt`: the numbers.
- `csnbs/server.py`: the old server. Find the `async def` that blocks.

**Key idea:** `async def` with a slow blocking call inside freezes *everything*. No new
requests, no responses, nothing.

---

## WP4: a server that doesn't freeze (request queue)

**What I did.** I built a new server. Image prep (decode, resize, tokenize) runs in
background threads. GPU work goes into a queue, and **one** worker thread takes jobs out
and runs them.

**Why one worker thread?** Two threads using the model at once crashed it before
(`scripts/repro_concurrent_crash.py`). One GPU thread = no crash.

**Result.** Answers match the old server exactly: 90/90 at both 576 and 128.

**Read:**
- `vis_pruner_copy/llava/serve/batch_server.py`: the new server. Compare it with
  `csnbs/server.py`.
- `vis_pruner_copy/llava/serve/batch_engine.py`: `submit()`, `_run()`, and the bounded queue.
- `vis_pruner_copy/llava/model/language_model/llava_llama.py`: the `_timing_tls` part.
  Timing notes are now per-thread, so two threads can't erase each other's notes.

**Key idea:** a queue lets the CPU prepare request N+1 while the GPU works on request N.

---

## WP5: run several requests together (batching)

**What I did.** The worker now takes up to 8 requests from the queue (waiting at most
10 ms for more) and runs them through the GPU in one go.

**Three bugs batching exposed, all fixed:**
1. **Crash.** All images' tokens were glued into one row, so request 1 got everything and
   request 2 crashed. Fixed in `llava_arch.py`: each image now gets its own tokens.
2. **Padding.** Short prompts must be padded on the *left*. On the right, new words get
   added after the padding, and the answers become garbage with no error. Fixed in
   `builder.py`.
3. **Token picking changed in a batch.** At 128 tokens, an image kept *different* tokens
   in a batch than alone, which changed about 11 of 90 answers. Cause: tiny rounding
   differences plus near-tied scores flipping order. Fixed in `llava_arch.py` by picking
   tokens one image at a time.

**How we knew bug 3 wasn't requests leaking into each other:** 8 copies of the same image
in one batch all picked the *same* tokens. Leaking would have made them differ.

**Result.**
- Batched answers match single requests 539 of 540 times. The 1 difference is the model
  stopping one sentence earlier, which is a rounding difference.
- Throughput with a batch of 16: 5.6× at 576 tokens, 8.7× at 128 tokens.

**Read:**
- `vis_pruner_copy/llava/serve/batch_engine.py`: `_collect()` (how batches form) and
  `generate_batch()` (padding, one call, splitting answers back out).
- `vis_pruner_copy/llava/model/llava_arch.py`: the top of `encode_images`, and the
  "One (T, C) tensor per image" comment.
- `vis_pruner_copy/llava/model/builder.py`: the left-padding lines.
- `scripts/wp5_batch_validation.py`: `s0_reference`, `MaskRecorder`, and the
  contamination test.
- `results/wp5_validation/wp5_val_2.txt`: the final numbers.

**Key idea:** decode is memory-bound. Reading the weights once for 8 requests costs about
the same as reading them for 1.

---

## WP6: the big load test (running)

**What it does.** It tests 3 servers (old, queue, batched) × 2 token counts (576, 128)
× 2 answer types (natural length, fixed 64 tokens), at many request rates. The order is
random, so time-of-day effects can't fake a result. Every rate gets at least 100 samples
for p95. Some settings are repeated to measure the noise floor.

**Read:** just the top comment of `scripts/wp6_load_matrix.py`. The results will be in
`results/timing/wp6_matrix/wp6_matrix_2/summary.txt` when it finishes.

---

## Files you can skip

These are helper scripts with no new ideas: `wp3_run_diag.py`, `wp3_analyze.py`,
`wp5_selection_diag.py` (just know what it found), `wp6_s0_server.py`, `wp6_analyze.py`.

---

## Self-test: answer out loud, without looking

1. Why did WP3 say a queue would only help 5–8%?
2. Why did the 4-thread pin matter more than the queue?
3. Why does `async def` + a slow call freeze the whole server?
4. Why does all GPU work stay in one thread?
5. What breaks with right padding, and why is there no error?
6. Why did batching change which tokens were kept, and how did we prove it wasn't leaking?
7. Why does batching help 128 tokens (8.7×) more than 576 (5.6×)?
8. What was the one differing answer out of 540, and why is it OK?

## Try it yourself (after WP6 finishes; one job on the GPU at a time)

Run the server and send it requests:
```bash
cd /workspace/GPU_Profiling
PYTHONPATH=vis_pruner_copy MAX_BATCH_SIZE=8 VISUAL_TOKEN_NUM=128 TORCH_THREADS=4 \
  python -m uvicorn llava.serve.batch_server:app --port 8001
curl localhost:8001/health      # batches_run vs requests sent shows batching happening
```

Break it on purpose, and **guess the result before you run it**:
- Set the padding side to `"right"` in `builder.py`, then run
  `python scripts/wp5_batch_validation.py --run-id test_right --skip-scan`.
- Remove the per-image block at the top of `encode_images`, then run the same check and
  watch 128 drop to about 79/90.
- Undo both with `git checkout -- vis_pruner_copy/llava/model/` when done.
