"""
BENCHMARKING - A COMPLETE COURSE
=================================

Ten lessons. Each is a self-contained cell you can run on its own.
Run everything, or one lesson at a time:

    python benchmarking_course.py            # all lessons
    python benchmarking_course.py 4          # just lesson 4
    python benchmarking_course.py 1 2 3      # a few

Lessons 1-6 need a GPU. Lessons 7-10 run anywhere.

Each lesson prints:  what it did -> what you should notice -> why it happens
"""

import sys
import time
import random
import asyncio
import statistics
from contextlib import contextmanager

import torch

HAS_GPU = torch.cuda.is_available()


# ===============================================================
# Shared helpers
# ===============================================================

def header(n, title):
    print("\n" + "=" * 66)
    print(f"LESSON {n} - {title}")
    print("=" * 66)


def observe(text):
    print("\n  >>> WHAT TO NOTICE" + text.rstrip() + "\n")


def percentile(values, p):
    """Simple percentile. values in seconds, returns milliseconds."""
    s = sorted(values)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx] * 1000


def time_gpu(fn, warmup=10, iters=50, flush_cache=False):
    """The correct way to time a GPU operation.

    warmup      - discard early runs (CUDA init, autotuning)
    iters       - average over this many
    flush_cache - clear L2 between runs, for small ops
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if not flush_cache:
        # Fast path: one timed block, sync only at the ends.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    total = 0.0
    for _ in range(iters):
        junk = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
        junk.zero_()
        del junk
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        total += time.perf_counter() - t0
    return total / iters * 1000


# ===============================================================
# LESSON 1 - The asynchronous GPU
# ===============================================================

def lesson_1():
    header(1, "THE GPU DOES NOT WAIT FOR PYTHON")

    x = torch.randn(4000, 4000, device="cuda")

    t0 = time.perf_counter()
    _ = x @ x
    naive = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = x @ x
    torch.cuda.synchronize()
    with_sync = (time.perf_counter() - t0) * 1000

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    _ = x @ x
    end.record()
    torch.cuda.synchronize()
    events = start.elapsed_time(end)

    print(f"""
  time.perf_counter(), no sync : {naive:8.3f} ms
  time.perf_counter(), + sync  : {with_sync:8.3f} ms
  CUDA events                  : {events:8.3f} ms""")

    observe(f"""
  The first number is ~{with_sync/naive:.0f}x too small.

  WHY: `x @ x` does not run the multiply. It QUEUES the multiply on the
  GPU and returns immediately. Python raced ahead and stopped the clock
  while the GPU was still working.

  synchronize() blocks until the GPU drains its queue, which fixes it.
  CUDA events are better still - they are timestamps placed INSIDE the
  GPU's queue, so they measure the GPU's own timeline with no Python
  overhead in the way.

  RULE: CUDA events for GPU work. Never a bare wall clock.""")


# ===============================================================
# LESSON 2 - Warmup
# ===============================================================

def lesson_2():
    header(2, "THE FIRST RUNS ARE ALWAYS WRONG")

    torch.cuda.empty_cache()
    a = torch.randn(2000, 2000, device="cuda")

    timings = []
    for _ in range(40):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = a @ a
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)

    print("\n  Run-by-run (ms):")
    for i in range(0, 40, 8):
        row = "  ".join(f"{t:6.2f}" for t in timings[i:i+8])
        print(f"    {i:>2}-{i+7:>2}: {row}")

    cold = statistics.mean(timings[:3])
    warm = statistics.mean(timings[15:])
    spread = statistics.stdev(timings[15:])

    print(f"""
  First 3 runs   : {cold:6.2f} ms
  Runs 15+       : {warm:6.2f} ms  (+/- {spread:.2f})
  Cold inflation : {cold / warm:6.1f}x""")

    observe("""
  The first few runs are much slower, then it settles into a stable band.

  WHY: the first CUDA call on a process initialises the CUDA context,
  allocates the caching allocator's memory pools, and JIT-loads kernels.
  Later, when you use Triton, the first call also runs a full autotuning
  search over block sizes.

  None of that is the operation's steady-state cost, which is what you
  are trying to measure.

  RULE: discard ~10 warmup iterations before timing anything.""")


# ===============================================================
# LESSON 3 - Cache contamination
# ===============================================================

def lesson_3():
    header(3, "TIGHT LOOPS CHEAT VIA CACHE")

    small = torch.randn(1024, 1024, device="cuda")
    big = torch.randn(8192, 8192, device="cuda")

    small_hot = time_gpu(lambda: small * 2.0, flush_cache=False)
    small_cold = time_gpu(lambda: small * 2.0, flush_cache=True, iters=20)
    big_hot = time_gpu(lambda: big * 2.0, flush_cache=False, iters=20)
    big_cold = time_gpu(lambda: big * 2.0, flush_cache=True, iters=20)

    print(f"""
  {'tensor':>12} | {'hot cache':>10} | {'flushed':>10} | {'difference':>11}
  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*11}
  {'1024x1024':>12} | {small_hot:>9.4f}m | {small_cold:>9.4f}m | {(small_cold/small_hot-1)*100:>10.1f}%
  {'8192x8192':>12} | {big_hot:>9.4f}m | {big_cold:>9.4f}m | {(big_cold/big_hot-1)*100:>10.1f}%""")

    observe("""
  The small tensor is noticeably faster with a hot cache. The large one
  barely changes.

  WHY: 1024x1024 floats is 4MB, which fits comfortably in L2. Run it in
  a loop and the data never leaves cache, so you measure cache hits
  rather than the real operation. The 8192x8192 tensor is 256MB - far
  too big for L2 - so it always comes from VRAM either way.

  In production, data usually arrives cold.

  RULE: flush L2 between runs when benchmarking small operations.""")


# ===============================================================
# LESSON 4 - Where to synchronize
# ===============================================================

def lesson_4():
    header(4, "SYNCHRONIZING IN THE WRONG PLACE")

    x = torch.randn(512, 512, device="cuda")

    # Wrong: sync every iteration. Each sync drains the pipeline,
    # so the GPU idles between launches.
    for _ in range(10):
        _ = x @ x
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        _ = x @ x
        torch.cuda.synchronize()
    inner = (time.perf_counter() - t0) / 200 * 1000

    # Right: sync once at each end.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        _ = x @ x
    torch.cuda.synchronize()
    outer = (time.perf_counter() - t0) / 200 * 1000

    print(f"""
  sync inside the loop  : {inner:8.4f} ms per op
  sync outside the loop : {outer:8.4f} ms per op
  overstated by         : {(inner/outer - 1) * 100:7.1f}%""")

    observe("""
  Syncing every iteration makes the operation look slower than it is.

  WHY: normally the CPU queues many kernels ahead of the GPU, keeping it
  continuously fed. A synchronize() drains that queue completely, so the
  GPU goes idle while Python catches up and launches the next one. You
  end up measuring launch overhead as if it were compute.

  The effect is worst for fast kernels, where launch overhead is a large
  fraction of the total.

  RULE: sync before the loop and after the loop. Not inside it.""")


# ===============================================================
# LESSON 5 - Batching
# ===============================================================

def lesson_5():
    header(5, "BATCHING - THROUGHPUT vs LATENCY")

    w = torch.randn(1024, 1024, device="cuda")

    print(f"\n  {'batch':>6} | {'total ms':>9} | {'per item':>9} | {'items/s':>10} | {'speedup':>8}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-{'-'*8}")

    base = None
    for b in [1, 2, 4, 8, 16, 32, 64]:
        try:
                     data = torch.randn(b, 512, 1024, device="cuda")
                     ms = time_gpu(lambda: data @ w, iters=30)
                     per = ms / b
                     base = base or per
                     print(f"  {b:>6} | {ms:>9.3f} | {per:>9.4f} | {1000/per:>10.0f} | {base/per:>7.1f}x")
                     del data
                     torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  {b:>6} | OUT OF MEMORY - this is your ceiling")
            torch.cuda.empty_cache()
            break

    observe("""
  Total time per batch rises. Time PER ITEM falls sharply.

  WHY: a GPU has thousands of cores. One small request leaves most of
  them idle - you are paying for the whole chip and using a fraction.
  Batching fills the machine.

  But notice the speedup column flattening at larger batches. That is
  saturation: the GPU is now fully occupied and extra work simply
  queues.

  THE TRADE-OFF: batching serves many more users per second, but each
  individual user waits longer. Where you sit on that curve is a product
  decision. Amay's scheduler is what implements it.""")


# ===============================================================
# LESSON 6 - Memory-bound vs compute-bound
# ===============================================================

def lesson_6():
    header(6, "MEMORY-BOUND vs COMPUTE-BOUND")

    n = 4096
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    add_ms = time_gpu(lambda: a + b, iters=30)
    mul_ms = time_gpu(lambda: a * 2.0, iters=30)
    mm_ms = time_gpu(lambda: a @ b, iters=20)

    add_gbps = (3 * a.numel() * 4) / (add_ms / 1000) / 1e9
    mul_gbps = (2 * a.numel() * 4) / (mul_ms / 1000) / 1e9
    mm_tflops = (2 * n ** 3) / (mm_ms / 1000) / 1e12

    print(f"""
  a + b        : {add_ms:8.3f} ms  ->  {add_gbps:7.1f} GB/s   (memory-bound)
  a * 2.0      : {mul_ms:8.3f} ms  ->  {mul_gbps:7.1f} GB/s   (memory-bound)
  a @ b        : {mm_ms:8.3f} ms  ->  {mm_tflops:7.2f} TFLOP/s (compute-bound)

  Arithmetic intensity (flops per byte moved):
    a + b   : {1/12:.3f}
    a @ b   : {(2*n**3)/(3*n*n*4):.1f}""")

    observe("""
  The elementwise ops do almost no arithmetic but move enormous amounts
  of data. They are limited by MEMORY BANDWIDTH - compare that GB/s to
  your card's spec sheet and you will be near the ceiling.

  The matmul reuses each loaded value many times, so it is limited by
  the CORES instead.

  WHY IT MATTERS: the fix differs completely.
    Memory-bound  -> fuse kernels, avoid round-trips to VRAM, lower precision
    Compute-bound -> better algorithms, tensor cores, more parallelism

  Optimising the wrong axis wastes weeks.

  IN AN LLM: prefill (reading the prompt) is compute-bound. Decode
  (generating one token at a time) is memory-bound - it re-reads the
  entire weight matrix to produce a single token. This is why token
  compression helps prefill a lot and decode almost not at all.""")


# ===============================================================
# LESSON 7 - Closed-loop vs open-loop
# ===============================================================

def lesson_7():
    header(7, "CLOSED-LOOP vs OPEN-LOOP  <- most important lesson here")

    class Server:
        """Service time degrades as the queue grows. Like a real server."""
        def __init__(self):
            self.in_flight = 0

        async def handle(self):
            self.in_flight += 1
            depth = self.in_flight
            t0 = time.perf_counter()
            await asyncio.sleep(0.008 * (1 + depth * 0.4) + random.uniform(0, 0.002))
            self.in_flight -= 1
            return time.perf_counter() - t0

    async def closed(workers, duration=1.5):
        srv, lat = Server(), []
        stop = time.perf_counter() + duration

        async def w():
            while time.perf_counter() < stop:
                lat.append(await srv.handle())

        await asyncio.gather(*[w() for _ in range(workers)])
        return lat

    async def open_(rate, duration=1.5):
        srv, lat, tasks = Server(), [], []
        start = time.perf_counter()

        async def fire(at):
            await asyncio.sleep(max(0, at - (time.perf_counter() - start)))
            t0 = time.perf_counter()
            await srv.handle()
            lat.append(time.perf_counter() - t0)

        for i in range(int(rate * duration)):
            tasks.append(asyncio.create_task(fire(i / rate)))
        await asyncio.gather(*tasks)
        return lat

    async def run():
        print(f"\n  {'load':>17} | {'p50':>8} | {'p95':>8} | {'p99':>8}")
        print(f"  {'-'*17}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        print("\n  CLOSED-LOOP  (wait for reply, then send next)")
        for w in [1, 4, 16, 64]:
            l = await closed(w)
            print(f"  {w:>9} workers | {percentile(l,50):>7.1f}m | {percentile(l,95):>7.1f}m | {percentile(l,99):>7.1f}m")

        print("\n  OPEN-LOOP  (fixed arrival rate, regardless of server)")
        for r in [25, 60, 120, 250]:
            l = await open_(r)
            print(f"  {r:>8} req/sec | {percentile(l,50):>7.1f}m | {percentile(l,95):>7.1f}m | {percentile(l,99):>7.1f}m")

    asyncio.run(run())

    observe("""
  Closed-loop latency stays calm even at high worker counts. Open-loop
  latency explodes past a threshold.

  WHY: in closed-loop, a slow response DELAYS THE NEXT REQUEST. The load
  automatically backs off at exactly the moment the server is struggling,
  so a queue never forms. The system's worst behaviour suppresses the
  evidence of its own existence.

  This is COORDINATED OMISSION. Most homemade benchmarks have it, and
  report beautiful numbers that describe a situation no user is in.

  Real users do not wait politely for your server to recover. They arrive
  when they arrive.

  RULE: open-loop at controlled arrival rates. Always.""")


# ===============================================================
# LESSON 8 - Percentiles
# ===============================================================

def lesson_8():
    header(8, "MEANS HIDE THE TAIL")

    random.seed(0)
    lat = [max(1, random.gauss(50, 8)) for _ in range(9700)]
    lat += [max(1, random.gauss(320, 90)) for _ in range(300)]
    s = sorted(lat)

    def p(v):
        return s[min(int(len(s) * v / 100), len(s) - 1)]

    print(f"""
  mean   : {statistics.mean(s):7.1f} ms
  p50    : {p(50):7.1f} ms
  p90    : {p(90):7.1f} ms
  p95    : {p(95):7.1f} ms
  p99    : {p(99):7.1f} ms
  p99.9  : {p(99.9):7.1f} ms
  max    : {max(s):7.1f} ms""")

    print("\n  Distribution:")
    buckets = [0] * 12
    for v in s:
        buckets[min(int(v / 35), 11)] += 1
    for i, c in enumerate(buckets):
        bar = "#" * int(c / len(s) * 200)
        print(f"    {i*35:>4}-{(i+1)*35:>4} ms | {bar}")

    small = sorted(random.sample(s, 100))[98]
    print(f"""
  p99 from    100 samples : {small:7.1f} ms
  p99 from 10,000 samples : {p(99):7.1f} ms""")

    observe(f"""
  The mean says {statistics.mean(s):.0f} ms, but 1 user in 100 waits {p(99):.0f} ms.

  WHY: latency distributions are right-skewed. A small number of very slow
  requests inflate the mean while hiding inside it. Look at the histogram -
  there is a whole second cluster far to the right that the mean averages
  away.

  Users experience the tail. Complaints come from p99, not from the mean.

  Also note the p99 from 100 samples is unstable - you need roughly 1000+
  samples before a p99 means anything.

  RULE: report p50/p95/p99. A bare mean is close to useless.""")


# ===============================================================
# LESSON 9 - Noise floor
# ===============================================================

def lesson_9():
    header(9, "IS THE DIFFERENCE REAL?")

    if HAS_GPU:
        x = torch.randn(2048, 2048, device="cuda")
        runs = [time_gpu(lambda: x @ x, iters=30) for _ in range(6)]
    else:
        random.seed(1)
        runs = [random.gauss(10, 0.35) for _ in range(6)]

    mean = statistics.mean(runs)
    sd = statistics.stdev(runs)
    spread = (max(runs) - min(runs)) / mean * 100

    print("\n  Six identical runs, nothing changed:")
    for i, r in enumerate(runs, 1):
        print(f"    run {i} : {r:8.4f} ms")

    print(f"""
  mean            : {mean:8.4f} ms
  std dev         : {sd:8.4f} ms
  spread          : {spread:7.1f}%""")

    observe(f"""
  Identical code, identical hardware, and the numbers still differ by
  {spread:.1f}%.

  WHY: GPU clock throttling with temperature, other processes, memory
  allocator state, kernel scheduling variation.

  THE CONSEQUENCE: if your noise floor is {spread:.1f}% and someone reports a
  {spread*0.6:.1f}% improvement, they have measured nothing at all. You cannot
  distinguish their change from thermal drift.

  Before reporting any optimisation:
    1. Know your noise floor - run the baseline several times
    2. Require the improvement to clearly exceed it
    3. For close calls use a Mann-Whitney U test (latency is not normal,
       so a t-test is the wrong tool)
    4. Fix GPU clocks if you can: nvidia-smi -lgc <freq>

  RULE: measure your own variance before you trust any comparison.""")


# ===============================================================
# LESSON 10 - Putting it together
# ===============================================================

def lesson_10():
    header(10, "A CORRECT HARNESS")

    print("""
  Everything above, in one function:

    def benchmark(fn, warmup=10, iters=50, flush=False):
        # 1. WARMUP - discard CUDA init and autotuning
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        # 2. CUDA EVENTS - measure the GPU's own timeline
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        # 3. SYNC OUTSIDE THE LOOP - no pipeline bubbles
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end) / iters

  For a serving system, add:

    - OPEN-LOOP arrivals at a fixed rate         (lesson 7)
    - p50/p95/p99, never means                    (lesson 8)
    - repeat runs to establish a noise floor      (lesson 9)
    - record the git commit hash with every run
    - store results so runs are comparable later
    - one person on the GPU at a time

  THE CHECKLIST - before any number leaves your harness:

    [ ] CUDA events, not wall clock
    [ ] warmup discarded
    [ ] sync outside the loop
    [ ] cache flushed if the tensors are small
    [ ] open-loop, not closed-loop
    [ ] percentiles reported, not means
    [ ] enough samples for a stable p99 (~1000+)
    [ ] difference exceeds the measured noise floor
    [ ] git commit recorded

  Every number in this project flows through your harness. If it is
  wrong, everyone's results are wrong. Build it once, carefully, and
  have Amay and Sribhav review it.
""")
    print("=" * 66)


# ===============================================================
# Runner
# ===============================================================

LESSONS = {
    1: (lesson_1, True), 2: (lesson_2, True), 3: (lesson_3, True),
    4: (lesson_4, True), 5: (lesson_5, True), 6: (lesson_6, True),
    7: (lesson_7, False), 8: (lesson_8, False),
    9: (lesson_9, False), 10: (lesson_10, False),
}

if __name__ == "__main__":
    if HAS_GPU:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU found - running lessons 7-10 only (they don't need one).")

    wanted = [int(a) for a in sys.argv[1:]] or sorted(LESSONS)

    for n in wanted:
        fn, needs_gpu = LESSONS[n]
        if needs_gpu and not HAS_GPU:
            print(f"\nSkipping lesson {n} (needs a GPU).")
            continue
        fn()

    print("\nDone.")