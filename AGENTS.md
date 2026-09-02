# AGENTS.md

Context for GPT working in this repository.

---

## What this project is

We take LLaVA-1.5-7B and make it cheaper to run by discarding visual tokens the model
doesn't need. Then we measure two things honestly: how much faster it actually gets under
real serving conditions, and how much accuracy we lose.

**Problem statement:**

> Published visual token pruning methods report speedups measured on single requests or
> uniform offline batches. Under realistic serving — concurrent arrivals, continuous
> batching, mixed-length traffic — how much of that speedup survives, and where does the
> difference go?

**Why this is worth doing.** An image becomes 576 tokens in LLaVA-1.5; the text question
beside it is about 8. Almost all inference cost is the image. Published methods (FastV,
VisPruner / FasterVLM) show most of those tokens are redundant.

Two gaps nobody has closed:

1. **FastV's released code masks tokens rather than removing them.** The authors state this
   in their repo — real in-place dropping is listed as future work. Masked tokens still sit
   in the tensor and still consume compute. The accuracy results are valid; the speedup is
   not demonstrated. (VisPruner, which we use, does real removal via boolean-mask indexing.)
2. **Nobody has measured this under concurrent load.** Papers benchmark single requests or
   fixed offline batches. Production means concurrency, queueing, KV cache pressure, and
   mixed-length traffic — and pruning interacts with all of them. In particular, pruned
   sequences have different lengths, and standard batching pads to the longest, destroying
   much of the saving.

**The deliverable:** one chart — accuracy against measured throughput — plus an honest
writeup of where the published claims held and where they didn't.

**If the speedup turns out small, that is the result.** A negative finding here is more
valuable than a positive one, because it corrects something the field assumes.

---

## Team and ownership

| Person | Responsible for | Owns |
|---|---|---|
| **Amay** | Speed | Serving path, real token removal, batching, KV cache, Triton kernels, vLLM port, Playground UI |
| **Rithvik** | Measurement | Load generation, timing harness, percentile analysis, storage, deployment, CI, benchmark UI |
| **Sribhav** | Accuracy | Eval set design, scoring logic, regression detection, compression methods, eval UI |

**Do not write code in someone else's directory.** If you need something from another
slice, the change belongs to its owner.

**Amay and Rithvik both measure — the split is purpose, not activity.** Amay measures to
find bottlenecks: profiling, rough timing, checking whether a change helped. Throwaway.
Rithvik measures to produce results: controlled conditions, percentiles, stored runs,
statistical comparison. **Only Rithvik's harness produces reportable numbers.** If a figure
appears in a chart or a blog post, it came from his system.

---

## Stack

- **Engine + Amay's service:** Python 3.12, PyTorch 2.2.2, Transformers 4.37.2, FastAPI; later vLLM and Triton
- **Sribhav's service:** Java 21, Spring Boot, Maven, JPA
- **Rithvik's service:** Node 20, TypeScript, React, Vite, Recharts
- **Shared:** Postgres (SQLite while developing), Docker, GitHub Actions

The polyglot setup is deliberate — it gives Sribhav and Rithvik resume-relevant stack
experience. Don't "simplify" it to one language.

---

## Model and method

**Base model:** LLaVA-1.5-7B (~14GB fp16, CLIP ViT-L/14@336 vision encoder)

CLIP always emits exactly **576** patch tokens. When `visual_token_num=576`, the pruning
code's masks stay all-True and nothing is dropped — **576 is the unpruned baseline**, not a
light-pruning setting. Everything is normalised against it.

**Primary method:** VisPruner / FasterVLM — uses `[CLS]` attention in the vision encoder to
rank visual tokens, prunes before the LLM. FlashAttention-compatible, and does real removal.

**Control:** FastV, run as released (masked), to demonstrate the FLOP-vs-time gap.

**Two tunable parameters:**
- `visual_token_num` — total tokens kept. Drives speed. Amay's axis.
- `important_ratio` — split between dominant tokens kept exactly and merged context tokens.
  Affects accuracy only, not speed. Sribhav's axis.

Sweep one at a time. Changing both makes results uninterpretable.

**Reference repos** (read-only, kept outside the tracked tree):
- https://github.com/Theia-4869/VisPruner
- https://github.com/chenllliang/FastV

---

## Infrastructure

**Environment is a Docker image**, built by GitHub Actions on push, published as
`kingcorsair/gpu_profiling_project:latest`. Deploy RunPod pods with that as the custom
container image — no pip installs on the pod.

**To change the environment, edit the Dockerfile and rebuild.** Not `pip install` on a
running pod — that disappears when the container stops.

**GPU:** RunPod, A40 (48GB) or RTX 4090 (24GB), ~100GB disk. `/workspace` is a MooseFS
network mount shared across RunPod's cluster (`mfs#...:9421`) — it persists across pod
stop/restart, but it does **not** honor `chmod`; anything needing real Unix permissions
(private keys, etc.) must be staged into `/root` at boot instead. **The pod itself is
disposable** — only `/workspace`'s contents survive a restart, not `/root`.

**Everything of value lives in git.** Model weights and datasets re-download. A lost pod
costs a `git clone` and twenty minutes.

**Session start:**
```bash
cd /workspace
git clone https://github.com/KingCorsair/GPU_Profiling.git   # first time only
cd GPU_Profiling && git pull
uv pip install --system -r requirements.txt
bash scripts/download_checkpoint.sh   # re-fetches LLaVA weights; never committed to git
uv pip sync --system requirements.txt # catches anything download_checkpoint.sh or other
                                       # setup steps pulled outside the pin (see Gotchas)
```

**Session end:** `git add . && git commit && git push`, then **Stop the pod in the console**
(square icon, wait for "Exited"). Closing the browser does not stop billing.

**Never store work in `/root`** — it dies with the pod. Use `/workspace`.

---

## Non-negotiable rules

These silently invalidate everything downstream if broken.

### Benchmarking

1. **CUDA events, never `time.time()`**, when timing GPU operations directly. CUDA is
   asynchronous — a wall clock measures how long Python took to *queue* work, not do it.
   (Timing a whole subprocess is exempt: by the time it exits, all its kernels have
   completed.)
2. **Use `time.perf_counter()`, not `time.time()`**, for durations. Monotonic; can't be
   skewed by a clock adjustment mid-run.
3. **Warm up.** Discard the first ~10 iterations (CUDA context init, memory pools, Triton
   autotuning). For whole-script runs, note that model loading is constant overhead in
   every measurement and report it separately.
4. **Synchronize outside the timing loop, never inside it.** An inner sync drains the
   pipeline and inflates fast kernels.
5. **Open-loop load generation** at fixed arrival rates. Closed-loop causes *coordinated
   omission*: a slow response delays the next request, so load backs off exactly when the
   system is struggling and the queue never forms. Numbers look great and mean nothing.
6. **Report p50/p95/p99, never means.** Latency is right-skewed; users experience the tail.
   A stable p99 needs ~1000+ samples.
7. **Record the git commit and GPU model with every run.** An unattributable benchmark is
   unrepeatable.
8. **Know the noise floor.** Run the baseline repeatedly first. If run-to-run variance is 8%
   and an optimisation shows 5%, nothing has been measured.
9. **One person on the GPU during benchmark runs.** Concurrent work measures contention.
10. **Randomise run order** across settings. A fixed order confounds thermal drift with the
    variable being tested.

### Evaluation

11. **Split dev/test once, at the start. Lock the test set.** Tune only against dev. Looking
    at test results and then changing the method contaminates it, and every number after is
    dishonest — usually without anyone noticing.
12. **Include tasks compression should break:** OCR, counting, spatial reasoning. An eval set
    of only coarse description will conclude compression is free. It isn't.
13. **Validate the scorer.** Exact match marks correct answers wrong ("the car is red" vs
    "red"). Run multiple scorers, measure agreement (Cohen's kappa), hand-check a sample of
    disagreements. An accuracy figure from an unvalidated scorer is not a measurement.
14. **Report per-category accuracy, not just overall.** A method can hold total accuracy
    steady while OCR collapses, and the average hides it.

### Implementation

15. **Real removal, not masking.** Masked tokens still cost compute. Gathering on kept
    indices produces a genuinely smaller tensor. Then fix everything downstream: position
    IDs, attention masks, KV cache sizing. **Current state:** the boolean-mask removal
    itself already exists, vendored from VisPruner in
    `vis_pruner_copy/llava/model/llava_arch.py`. The "fix everything downstream" part —
    the gather/position-id/mask-rebuild fixup — is Amay's active work; live status and plan
    in `AMAY_SPEED_PLAN.md`.
16. **Preserve spatial order after top-k.** `topk` returns score-sorted indices. Feeding
    tokens out of raster order silently degrades quality in ways that are miserable to debug.
17. **Always run a `random` scoring baseline.** If random performs as well as the method,
    the method isn't working.
18. **Visualise the importance map before building on it.** If it looks like noise, the
    scoring is broken and every downstream result is meaningless.
19. **Prototype on something fast before touching something slow.** A bug should cost
    seconds, not a 40-minute rerun.

---

## Key technical concepts

**Prefill vs decode** — the most load-bearing distinction in this project.
- *Prefill* processes the whole prompt at once. Large matrices, high arithmetic intensity.
  **Compute-bound.** This is what pruning shrinks.
- *Decode* generates one token at a time, each depending on the last. Reads the entire
  weight matrix per token. **Memory-bound.** Pruning does not help it.

Consequence: the longer the output, the more decode dominates and the less pruning helps.
**ScienceQA answers are single letters, so it is nearly pure prefill — results there are an
upper bound on the benefit, not the general answer.** Say so explicitly in any writeup.

**Variable-length batching** — the biggest gap between paper claims and serving reality.
Different images keep different token counts. Standard batching pads to the longest, so a
request pruned to 300 tokens in a batch padded to 900 costs full price. Fix with length
bucketing or a packed layout. No paper hits this because papers run one request at a time.

**KV cache** — grows with batch × context. Memory, not compute, caps batch size in practice.
Fewer visual tokens means a smaller cache means more concurrent requests fit — a throughput
win papers don't report.

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

# How to work with each of us

We are all learning, and we will each be asked about our own component verbally, in depth.
Code we don't understand is worth approximately nothing to us.

**For everyone, write freely without asking:**
- Scaffolding, config, Dockerfiles, CI YAML, requirements files
- Boilerplate: API routes, DB schemas, argument parsing
- Plotting, reporting, and analysis code
- Environment and dependency debugging
- Explanations, code review, "why is this slow" reasoning
- Anything in a language or layer that isn't the person's own specialism

**And for everyone:** when asked "why is this slow" or "why is this wrong," don't just fix
it. Walk through diagnosing it — what to measure, what the numbers mean, what the candidate
explanations are. Push back on bad designs. Stop us before we break one of the rules above.

---

## Amay — don't write these for him unprompted

- Batching and scheduling logic
- KV cache management
- Triton kernels
- Core timing and profiling code
- The token-removal implementation

These are the GPU-serving concepts he'll be interviewed on. Default to helping him think:
ask what he's tried, point at the relevant concept, review what he writes and say what's
wrong with it. Don't generate a finished implementation of your own initiative.

If Amay explicitly asks for the finished code outright, confirm once — e.g. "this is core timing/profiling code — the stuff you're supposed to be able to explain from memory. You want me to write it for you instead of walking you through it — sure?" — and if he still says yes, write it. One confirmation, not a renegotiation each time.


---

## Rithvik — don't write these for him

- The load generator (open-loop arrival scheduling in particular)
- Timing and latency measurement code
- Percentile and statistical comparison logic

Coordinated omission is his best interview story — he needs to have hit it himself. Explain
the concept, review his implementation, but let him write it.

Everything else is fair game: the FastAPI server, React dashboard, job queue, database
schema, Docker Compose, CI. Those are standard engineering and he learns them by shipping.

If Rithvik explicitly asks for the finished code outright, confirm once — e.g. "you want the
implementation itself, not another pointer toward it — sure?" — and if he still says yes,
write it. One confirmation, not a renegotiation each time.

---

## Sribhav — don't write these for him

- Scoring logic for free-form answers
- The compression method implementations
- Eval set design (which categories, what splits)

Scoring is the genuinely hard part of his role and the most transferable skill in it —
deciding what counts as correct, and validating that decision, is the work. Help him
think about it; don't hand him a scorer.

Everything else is fair game: the Spring Boot service, JPA entities, REST endpoints,
frontend, tests, deployment.

If Sribhav explicitly asks for the finished code outright, confirm once — e.g. "you want the
implementation itself, not another pointer toward it — sure?" — and if he still says yes,
write it. One confirmation, not a renegotiation each time.

---

## The self-test

Before anything goes on a resume: close the laptop and explain that component's behaviour
out loud, from memory, including the numbers and why they came out that way. If you can't,
it isn't yours yet.

---

## Gotchas already hit — don't repeat these

- `AutoModelForVision2Seq` was renamed `AutoModelForImageTextToText` in recent transformers.
- `pip install pytorch` is wrong; the package is `torch`. Plain `pip install torch` on Linux
  gives a CPU-only build — use the CUDA index URL.
- Ubuntu 22.04 has no `python3.12` package; the deadsnakes PPA provides it.
- `sentencepiece` needs `cmake` and `pkg-config` to build from source. Pin ≥0.2.0 for
  prebuilt Python 3.12 wheels instead.
- `pip freeze` writes editable installs as git URLs and local paths — strip those, and the
  `nvidia-*` lines, before using it in a Dockerfile.
- A container with no long-running process exits immediately and RunPod restarts it in a
  loop. The image needs a `CMD` that keeps it alive.
- Moving a venv breaks it — the path is hardcoded inside. Recreate, don't move.
- Cloning another repo inside this one creates a git submodule and its contents aren't
  tracked. Delete the nested `.git`, or keep reference repos outside the tree.
- `~/.ssh/authorized_keys` edits do not survive a pod restart.
- SSH silently falls back to password auth if key permissions are loose. `chmod 700 ~/.ssh`
  and `chmod 600 ~/.ssh/authorized_keys`.
- `df -h /workspace` reports RunPod's shared cluster, not your quota. Use `du -sh /workspace`
  against the disk size you configured.
- The eval scripts use relative paths — run them from `vis_pruner_copy`, or pass `cwd`.
- `/workspace` doesn't honor `chmod` (MooseFS mount) — `chmod 600` on a file there silently
  reverts to `666`. SSH refuses to load a private key left there directly. Keep the key's
  bytes on `/workspace` for persistence, but stage a copy into `/root` (chmod'd correctly)
  at container boot — see the `CMD` in the Dockerfile for the pattern.