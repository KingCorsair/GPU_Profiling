# Benchmarking Tasks — Rithvik

Eight tasks, in order. Each one teaches a technique you'll need for the real harness.
Do them in sequence — later ones assume the earlier ones.

Your job on this project: **find out how fast it really is, and where it breaks.**
Not making it fast (that's Amay). Not checking correctness (that's Sribhav).

If your harness is wrong, every number in the project is wrong — including theirs.
That's why this matters and why it's worth doing carefully.

---

## Task 1 — Prove Python's clock lies

**Goal:** see for yourself why `time.time()` can't measure GPU work.

Time a large matrix multiply two ways:

```python
import time, torch
x = torch.randn(4000, 4000, device="cuda")

# Wrong
t0 = time.time()
y = x @ x
wrong = time.time() - t0

# Right
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
y = x @ x
end.record()
torch.cuda.synchronize()
right = start.elapsed_time(end)
```

**What you'll see:** the wrong number is absurdly small.

**Why:** CUDA is asynchronous. Python hands work to the GPU and immediately moves to the
next line. `time.time()` measures how long it took to *queue* the work, not do it.
CUDA events are markers placed inside the GPU's own work queue, so they measure the GPU's
actual timeline.

**Deliverable:** a script printing both numbers and the ratio between them.

---

## Task 2 — Warmup and variance

**Goal:** learn why the first measurements are always garbage.

Run the same operation 100 times, recording each duration. Plot them in order.

**What you'll see:** the first few runs are dramatically slower, then it settles.

**Why:** the first CUDA call initialises the context, allocates memory pools, and loads
kernels. Triton (later) runs a full autotuning search on first call. None of that is the
operation's real cost.

Then compute mean, median, min, max, and standard deviation over runs 10–100.

**Question to answer:** which single number best represents "how fast is this"? Write down
your reasoning. (Hint: consider what a user actually experiences.)

**Deliverable:** the plot, plus a `time_gpu(fn, warmup=10, iters=50)` helper you'll reuse
in every task from here.

---

## Task 3 — L2 cache contamination

**Goal:** discover a subtle way benchmarks silently lie.

Time a small operation on the same tensor repeatedly. Then time it with a large dummy
tensor allocated and zeroed between each iteration.

```python
# Between iterations:
dummy = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
dummy.zero_()
del dummy
```

**What you'll see:** the second version is slower — sometimes much slower.

**Why:** the first version leaves data sitting in L2 cache between runs, so you're
measuring cache hits rather than the operation's real cost. In production, that data
wouldn't be cached. Flushing the cache gives you the honest number.

**Deliverable:** both numbers, and a note on when cache flushing matters (small tensors)
versus when it doesn't (tensors much larger than L2).

---

## Task 4 — Closed-loop vs open-loop

**This is the most important task on the list.**

**Goal:** understand coordinated omission — the error that makes most homemade benchmarks
worthless.

Build two load generators against a simple async server:

**Closed-loop:** send a request, wait for the response, send the next. N workers doing
this in parallel.

**Open-loop:** send a request every X milliseconds regardless of whether previous ones
have finished.

Run both at increasing load. Plot p50, p95, p99 latency for each.

**What you'll see:** closed-loop latency stays flat and pleasant. Open-loop latency climbs
sharply past a certain arrival rate.

**Why:** in closed-loop, a slow request *delays the next request*, so the load
automatically backs off exactly when the system is struggling. The queue never forms.
The system's worst behaviour suppresses the evidence of itself. That's coordinated
omission, and it's why closed-loop numbers look great and mean nothing.

Real users don't wait for your server to catch up. They arrive when they arrive.

**Deliverable:** both plots on the same axes. This chart is your single best interview
story — "I found our benchmarks were hiding X% of real queueing latency."

---

## Task 5 — Find the saturation point

**Goal:** locate the knee of the curve.

Using your open-loop generator, sweep arrival rate upward. At each rate record throughput
(completed requests/sec) and p95 latency.

**What you'll see:** throughput rises linearly, then flattens. Latency stays flat, then
goes near-vertical at roughly the same point.

**Why:** below saturation the system keeps up and queueing is minimal. Above it, arrivals
exceed service capacity, the queue grows without bound, and latency is dominated by
waiting rather than working.

**Deliverable:** a two-axis chart with throughput and p95 latency against arrival rate,
and a stated maximum sustainable rate. This is the headline number for any serving system.

---

## Task 6 — Percentiles and tail latency

**Goal:** learn why averages are misleading.

Collect 10,000 request latencies. Compute mean, p50, p90, p95, p99, p99.9, max.
Plot the full distribution as a histogram and as a CDF.

**What you'll see:** the mean sits well below p95. The tail is long.

**Why:** latency distributions are right-skewed — a few very slow requests drag the mean
up but hide inside it. Users experience the tail, not the mean. p99 is the number that
determines whether people think your service is fast.

**Also try:** compute p99 from 100 samples, then from 10,000. Note how unstable the small
sample is. That tells you your minimum sample size.

**Deliverable:** the table, the two plots, and a note on how many samples you need for a
stable p99.

---

## Task 7 — Persist and compare runs

**Goal:** turn one-off measurements into a platform.

Store results in Postgres:
- `runs` — config, git commit hash, GPU model, timestamp
- `measurements` — one row per request: latency, queue time, tokens, timestamp

Build a query that compares two runs and reports the difference in p50/p95/p99.

**Why it matters:** in week 8 you'll need to answer "did Amay's change actually help?"
Without stored history and a commit hash on every run, that question is unanswerable.

**Deliverable:** schema, ingestion code, and a comparison query. Index on `run_id` and
`timestamp` — this table gets large.

---

## Task 8 — Is the difference real?

**Goal:** avoid reporting noise as improvement.

Run the identical benchmark five times without changing anything. Look at the variation
between runs.

Then implement a simple comparison: given two sets of measurements, report whether the
difference exceeds the run-to-run noise. A Mann-Whitney U test is appropriate here
(latency distributions aren't normal, so a t-test is the wrong tool).

**Why:** if run-to-run variance is 8% and Amay's optimisation shows a 5% improvement, you
have measured nothing. Knowing your own noise floor is what separates a benchmark from a
number generator.

**Deliverable:** your measured noise floor as a percentage, and a function that answers
"is this difference significant?"

---

## What you'll have at the end

A benchmarking platform with correct GPU timing, open-loop load generation, percentile
analysis, persistent storage, and statistical comparison.

**Resume line:** *Built a benchmarking platform for GPU inference — open-loop load
generation at controlled arrival rates, CUDA-event timing, percentile analysis over a
time-series store, and statistical significance testing across runs.*

**Interview story:** task 4. Finding that closed-loop measurement was hiding real queueing
latency is a better story than any feature you could build, because it shows you understand
what you're measuring rather than just how to measure.

---

## Rules for all of these

- Warm up, always. Discard the first ~10 iterations.
- Synchronize outside the timing loop, never inside it — an inner sync creates GPU idle
  bubbles and inflates fast operations.
- Report distributions, never bare means.
- Record the git commit hash with every run. A benchmark you can't attribute to a code
  version is unrepeatable.
- Fix GPU clocks if you can (`nvidia-smi -lgc`). Clock speed varies with temperature, and
  a "10% improvement" measured on a cold GPU is not an improvement.
- One person on the GPU at a time. Another process running skews everything.

---

## Before you start

Do tasks 1–3 in one sitting. They're short and they build the helper you'll use everywhere
else. Task 4 is the one worth taking your time over.

Ask Amay for GPU time — don't run benchmarks while he's working on the same pod.