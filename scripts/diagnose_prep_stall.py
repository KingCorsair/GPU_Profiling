"""
Throwaway diagnostic (not a reportable benchmark): what is the intermittent ~60 ms
stall inside multimodal_prep, outside the vision tower?

Per full prep call records: wall, thread CPU time, involuntary context switches,
CUDA caching-allocator counters (alloc retries, device mallocs/frees), and any
Python GC collections that overlap the call. Same requests/settings as bench_dev.
"""
import gc
import json
import os
import resource
import sys
import time

import torch

sys.path.insert(0, "/workspace/GPU_Profiling/scripts")
import bench_dev as bd  # noqa: E402

N_THREADS = int(os.environ.get("DIAG_TORCH_THREADS", "0"))  # 0 = torch default
if N_THREADS:
    torch.set_num_threads(N_THREADS)
OUT = ("/workspace/GPU_Profiling/results/timing/bench_dev/"
       f"prep_stall_threads{torch.get_num_threads()}.json")
N_WARMUP = 4
N_MEASURED = 60
CPU_STAT = "/sys/fs/cgroup/cpu/cpu.stat"  # cgroup v1 CFS bandwidth counters


def throttle_stats():
    with open(CPU_STAT) as f:
        d = dict(line.split() for line in f)
    return int(d["nr_throttled"]), int(d["throttled_time"])  # count, ns

gc_events = []
_gc_open = {}


def gc_cb(phase, info):
    if phase == "start":
        _gc_open["t"] = time.perf_counter()
    else:
        gc_events.append((info["generation"], _gc_open.pop("t", None), time.perf_counter()))


gc.callbacks.append(gc_cb)

ALLOC_KEYS = ("num_alloc_retries", "num_device_alloc", "num_device_free", "num_ooms")


def alloc_stats():
    s = torch.cuda.memory_stats()
    return {k: s.get(k, 0) for k in ALLOC_KEYS}


def nivcsw():
    return resource.getrusage(resource.RUSAGE_THREAD).ru_nivcsw


bd.disable_torch_init()
tokenizer, model, image_processor, _ = bd.load_pretrained_model(
    bd.MODEL_PATH, None, "llava-v1.5-7b",
    visual_token_num=576, important_ratio=bd.IMPORTANT_RATIO,
)
model.eval()
items = bd.load_items()

recorder = bd.RegionRecorder()
handles = bd.install_vision_tower_hooks(model, recorder)

original = model.prepare_inputs_labels_for_multimodal
prep_log = []


def diag_prep(*args, **kwargs):
    t0 = time.perf_counter()
    c0 = time.thread_time()
    v0 = nivcsw()
    a0 = alloc_stats()
    n0, ns0 = throttle_stats()
    result = original(*args, **kwargs)
    if len(result) != 7:  # decode-step early return
        return result
    torch.cuda.synchronize()  # generate() syncs right after prep anyway
    t1 = time.perf_counter()
    n1, ns1 = throttle_stats()
    a1 = alloc_stats()
    prep_log.append({
        "t0": t0, "t1": t1,
        "wall_ms": (t1 - t0) * 1000,
        "thr_prep": n1 - n0,
        "thr_prep_ms": (ns1 - ns0) / 1e6,
        "thread_cpu_ms": (time.thread_time() - c0) * 1000,
        "nivcsw": nivcsw() - v0,
        **{k: a1[k] - a0[k] for k in ALLOC_KEYS},
    })
    return result


model.prepare_inputs_labels_for_multimodal = diag_prep
os.environ["LLAVA_TIMING_FILE"] = OUT + ".timing.jsonl"


def request(cfg, item):
    model.visual_token_num = cfg
    n0, _ = throttle_stats()
    input_ids, attention_mask, images, image_sizes = bd.build_request(
        item, tokenizer, model, image_processor)
    n_build, _ = throttle_stats()
    recorder.reset()
    torch.cuda.synchronize()
    with torch.inference_mode():
        model.generate(input_ids, attention_mask=attention_mask, images=images,
                       image_sizes=image_sizes, do_sample=False, use_cache=True,
                       min_new_tokens=bd.FIXED_OUTPUT_TOKENS,
                       max_new_tokens=bd.FIXED_OUTPUT_TOKENS)
    n1, _ = throttle_stats()
    return recorder.read_ms()["vision_tower"][0], n_build - n0, n1 - n0


for cfg in bd.CONFIGS:
    for i in range(N_WARMUP):
        request(cfg, items[i % len(items)])
prep_log.clear()
gc_events.clear()

rows = []
for k in range(N_MEASURED):
    cfg = bd.CONFIGS[k % 2]
    item = items[(k // 2) % len(items)]
    vt, thr_build, thr_request = request(cfg, item)
    p = prep_log[-1]
    overlapping = [g for g, s, e in gc_events if s is not None and s < p["t1"] and e > p["t0"]]
    rows.append({"k": k, "cfg": cfg, "qid": item["question_id"], "vt_ms": vt,
                 "other_ms": p["wall_ms"] - vt, "gc_gens_in_prep": overlapping,
                 "thr_build": thr_build, "thr_request": thr_request,
                 **{key: v for key, v in p.items() if key not in ("t0", "t1")}})

print(f"torch threads: {torch.get_num_threads()}")
print(f"{'k':>3}{'cfg':>5} {'qid':<28}{'wall':>7}{'vt':>6}{'other':>7}{'cpu':>6}"
      f"{'thrP':>5}{'thrPms':>7}{'thrB':>5}{'thrR':>5}{'retry':>6}  gc")
for r in rows:
    print(f"{r['k']:>3}{r['cfg']:>5} {r['qid']:<28}{r['wall_ms']:>7.1f}{r['vt_ms']:>6.1f}"
          f"{r['other_ms']:>7.1f}{r['thread_cpu_ms']:>6.1f}{r['thr_prep']:>5}"
          f"{r['thr_prep_ms']:>7.1f}{r['thr_build']:>5}{r['thr_request']:>5}"
          f"{r['num_alloc_retries']:>6}  {r['gc_gens_in_prep']}")

STALL_MS = 25.0  # other_ms is ~1.5 (576) / ~13 (128) when nothing goes wrong
stalled = [r for r in rows if r["other_ms"] > STALL_MS]
clean = [r for r in rows if r["other_ms"] <= STALL_MS]
print(f"\nstalled prep calls (other > {STALL_MS:.0f} ms): {len(stalled)}/{len(rows)}; "
      f"of those, throttled during prep: {sum(1 for r in stalled if r['thr_prep'])}")
print(f"clean prep calls: {len(clean)}/{len(rows)}; "
      f"of those, throttled during prep: {sum(1 for r in clean if r['thr_prep'])}")
print(f"throttle events: during build_request {sum(r['thr_build'] for r in rows)}, "
      f"whole requests {sum(r['thr_request'] for r in rows)}")
print(f"total GC collections during measured phase: {len(gc_events)} "
      f"(gen2: {sum(1 for g, _, _ in gc_events if g == 2)})")
with open(OUT, "w") as f:
    json.dump({"rows": rows, "gc_events": gc_events}, f, indent=1)
