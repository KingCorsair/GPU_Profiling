# CLAUDE.md

Context for Claude Code working in this repository.

---

## What this project is

We take LLaVA-1.5-7B and make it cheaper to run by discarding visual tokens the model
doesn't need. Then we measure two things honestly: how much faster it actually gets under
real serving conditions, and how much accuracy we lose.

**Problem statement:**

> When visual token pruning is implemented as *real token removal* inside a serving stack,
> how much of its theoretical FLOP reduction survives as measured throughput and latency
> improvement under concurrent load — and where does the difference go?

**Why this is worth doing.** An image becomes 500–2,000 tokens; the text question beside it
is about 8. Almost all inference cost is the image. Published methods (FastV, VisPruner /
FasterVLM) show most of those tokens are redundant and report large FLOP reductions.

But FLOPs are not seconds. Two gaps nobody has closed:

1. **FastV's released code masks tokens rather than removing them.** The authors state this
   in their repo — real in-place dropping is listed as future work. Masked tokens still sit
   in the tensor and still consume compute. The accuracy results are valid; the speedup is
   not demonstrated.
2. **Nobody has measured this under real serving conditions.** Papers benchmark single
   requests. Production means concurrency, batching, KV cache pressure, and variable output
   lengths — and compression interacts with all of them.

**The deliverable:** one chart, accuracy against measured throughput, plus an honest writeup
of where the published claims held and where they didn't.

**If the speedup turns out to be small, that is the result.** A negative finding here is
more valuable than a positive one, because it corrects something the field assumes.

---

## Team and ownership

Three people, three vertical slices. Each owns their slice top to bottom.

| Person | Slice | Owns |
|---|---|---|
| **Amay** | Serving | Playground UI, streaming API, real token removal, batching, KV cache, Triton kernels, vLLM port |
| **Rithvik** | Measurement | Benchmark UI, load generation, CUDA-event timing, job queue, storage, deployment, CI |
| **Sribhav** | Evaluation | Eval UI, compression methods, scoring, datasets, regression detection |

**Do not write code in someone else's directory.** If you need something from another slice,
the change belongs to its owner.

---

## Stack

- **Engine + Amay's service:** Python 3.12, PyTorch, Transformers, FastAPI; later vLLM and Triton
- **Sribhav's service:** Java 21, Spring Boot, Maven, JPA
- **Rithvik's service:** Node 20, TypeScript, React, Vite, Recharts
- **Shared:** Postgres (SQLite while developing), Docker, GitHub Actions

The polyglot setup is deliberate — it gives Sribhav and Rithvik resume-relevant stack
experience. Don't "simplify" it to one language.

---

## Model and method

**Base model:** `liuhaotian/llava-v1.5-7b` (~14GB fp16, CLIP ViT-L vision encoder)

We use LLaVA-1.5 rather than SmolVLM because the reference implementations target it, its
CLIP encoder has a real `[CLS]` token (SigLIP does not, which breaks CLS-attention scoring),
and it makes our numbers directly comparable to published ones.

**Primary method:** VisPruner / FasterVLM — uses `[CLS]` attention in the vision encoder to
rank visual tokens, prunes before the LLM. Chosen because it is FlashAttention-compatible
and prunes outside the language model, making real removal tractable.

**Control:** FastV, run as released (masked), to demonstrate the FLOP-vs-time gap.

**Reference repos** (read-only, kept outside the tracked tree):
- https://github.com/Theia-4869/VisPruner
- https://github.com/chenllliang/FastV

---

## Repo layout

```
/engine      shared inference service (Python) — rotating maintainer
/serving     Amay's service
/measure     Rithvik's service + shared infra (auth, CI, deploy)
/eval        Sribhav's service
/frontend    shared React app, three feature areas
/results     benchmark and eval outputs (committed — they're small and they matter)
```

---

## Infrastructure

**GPU:** rented RunPod pod, RTX 4090 (24GB) or A40 (48GB), ~80–100GB disk.
No network volume — it locks you to one datacenter and causes GPU availability waits.
The pod is **disposable**.

**Everything of value lives in git.** Model weights and datasets re-download; code and
results are committed. A lost pod costs a `git clone` and twenty minutes.

**Session start:**
```bash
cd /workspace
git clone https://github.com/KingCorsair/GPU_Profiling.git   # first time only
cd GPU_Profiling && git pull
pip install -r requirements.txt
```

**Session end:** `git add . && git commit && git push`, then **Stop the pod in the console**
(square icon, wait for "Exited"). Closing the browser does not stop billing.

**Never store work in `/root`** — it dies with the pod. Use `/workspace`.

---

## Non-negotiable rules

These are the ones that silently invalidate everything downstream if broken.

### Benchmarking

1. **CUDA events, never `time.time()`**, for GPU work. CUDA is asynchronous — a wall clock
   measures how long Python took to *queue* the work, not to do it.
2. **Warm up.** Discard the first ~10 iterations (CUDA context init, memory pools, Triton
   autotuning).
3. **Synchronize outside the timing loop, never inside it.** An inner sync drains the
   pipeline and inflates fast kernels.
4. **Flush L2 between runs** for small tensors, or you measure cache hits.
5. **Open-loop load generation** at fixed arrival rates. Closed-loop causes *coordinated
   omission*: a slow response delays the next request, so load backs off exactly when the
   system is struggling and the queue never forms. Numbers look great and mean nothing.
6. **Report p50/p95/p99, never means.** Latency is right-skewed; users experience the tail.
   A stable p99 needs ~1000+ samples.
7. **Record the git commit and GPU model with every run.** An unattributable benchmark is
   unrepeatable.
8. **Know the noise floor.** Run the baseline repeatedly first. If run-to-run variance is 8%
   and an optimisation shows 5%, nothing has been measured.
9. **One person on the GPU during benchmark runs.** Concurrent work skews timings.
10. **Only Rithvik's harness produces reportable numbers.** Exploratory timing stays local.

### Evaluation

11. **Split dev/test once, at the start. Lock the test set.** Tune only against dev. Looking
    at test results and then changing the method contaminates it, and every subsequent number
    is dishonest — usually without anyone noticing.
12. **Include tasks compression should break:** OCR, counting, spatial reasoning. An eval set
    of only coarse description will conclude compression is free. It isn't.
13. **Validate the scorer.** Exact match marks correct answers wrong ("the car is red" vs
    "red"). Run multiple scorers, measure agreement (Cohen's kappa), hand-check a sample of
    disagreements. An accuracy figure from an unvalidated scorer is not a measurement.
14. **Report per-category accuracy, not just overall.** A method can hold total accuracy
    steady while OCR collapses, and the average hides it.

### Implementation

15. **Real removal, not masking.** Masked tokens still cost compute. `torch.gather` on kept
    indices produces a genuinely smaller tensor. Then fix everything downstream: position
    IDs, attention masks, KV cache sizing.
16. **Preserve spatial order after top-k.** `topk` returns score-sorted indices. Feeding
    tokens out of raster order silently degrades quality in ways that are miserable to debug.
17. **Always run a `random` scoring baseline.** If random performs as well as the method,
    the method isn't working.
18. **Visualise the importance map before building on it.** If it looks like noise, the
    scoring is broken and every downstream result is meaningless.

---

## Key technical concepts

**Prefill vs decode** — the most load-bearing distinction in this project.
- *Prefill* processes the whole prompt at once. Large matrices, high arithmetic intensity.
  **Compute-bound.** This is what pruning shrinks.
- *Decode* generates one token at a time, each depending on the last. Reads the entire
  weight matrix per token. **Memory-bound.** Pruning does not help it.

Consequence: the longer the output, the more decode dominates and the less compression helps.
A paper reporting 45% FLOP reduction is describing a phase that may be 20% of a real request.

**Variable-length batching** — the biggest gap between paper claims and serving reality.
Different images keep different numbers of tokens. Standard batching pads to the longest, so
a request pruned to 300 tokens in a batch padded to 900 costs full price. Fix with length
bucketing or a packed layout. No paper hits this because papers run one request at a time.

**KV cache** — grows with batch × context. Memory, not compute, is what caps batch size in
practice. Fewer visual tokens means a smaller cache means more concurrent requests fit —
a throughput win papers don't report.

---

## Results we're producing

| ID | Result | Expectation |
|---|---|---|
| R1 | Masked vs real removal | Identical accuracy; near-zero speedup for masked |
| R2 | Prefill vs decode split at varying output lengths | Benefit shrinks as output grows |
| R3 | Single request vs concurrent throughput | Divergence — batching changes the bottleneck |
| R4 | Padded vs bucketed vs packed batching | Padding destroys much of the saving |
| R5 | Peak KV cache and max concurrent requests per ratio | Pruning raises the concurrency ceiling |
| R6 | Accuracy by task category | OCR and counting break first |

---

## How to work with me (Amay)

I'm learning GPU and inference engineering. I'll be interviewed on this verbally, in depth.

**Write freely, without asking:**
- Scaffolding, config, Dockerfiles, CI YAML, requirements files
- Boilerplate: FastAPI routes, DB schemas, argument parsing
- Plotting, reporting, and analysis code
- Environment and dependency debugging
- Test harness *structure* (not the timing logic itself)
- Explanations, code review, "why is this slow" reasoning

**Don't hand me finished implementations of:**
- Batching and scheduling logic
- KV cache management
- Triton kernels
- The core timing measurement code
- The compression method itself

For those: help me think. Ask what I've tried. Point at the concept. Review what I write and
tell me what's wrong with it. Don't hand me a solution even if I ask in a frustrated moment —
especially then.

**When I ask "why is this slow," don't just fix it.** Walk me through diagnosing it: what to
profile, what the numbers mean, what the candidate explanations are.

**Push back.** If a design is wrong, say so. If I'm about to make one of the benchmarking
errors above, stop me.

**Before anything goes on a resume:** close the laptop and explain that component's
performance characteristics out loud, from memory, including the numbers and why they came
out that way. If I can't, it isn't mine yet.

---

## Gotchas already hit — don't repeat these

- `AutoModelForVision2Seq` was renamed `AutoModelForImageTextToText` in recent transformers.
- `pip install pytorch` is wrong; the package is `torch`. On Linux, plain `pip install torch`
  gives a CPU-only build — use the CUDA index URL or a PyTorch pod template.
- Cloning another repo inside this one creates a git submodule and its contents aren't
  tracked. Delete the nested `.git`, or keep reference repos outside the tree.
- `~/.ssh/authorized_keys` edits do not survive a pod restart.
- SSH silently falls back to password auth if key permissions are loose. `chmod 700 ~/.ssh`
  and `chmod 600 ~/.ssh/authorized_keys`.
- `df -h /workspace` reports RunPod's shared cluster (hundreds of TB), not your quota. Use
  `du -sh /workspace` and compare against the disk size you configured.
- Terminate unused pods. Stopped pods still bill for storage.
