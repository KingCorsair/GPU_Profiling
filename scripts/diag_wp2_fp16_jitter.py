"""WP2 diagnostic: why does the FP16 vision tower wander, and why do regions it
never touches (prep_other, decode) slow down in the same trials?

Four same-pod bench_dev runs (FP32_1 -> FP16_1 -> FP16_2 -> FP32_2) showed:
  - FP16 vision_tower p50 reproducible (~14 ms) but per-trial 10.5-19.9 ms,
    while FP32 stays within ~0.6 ms of 34 ms;
  - in FP16 runs, vision_tower slows in exact proportion to prep_other at 128
    (slope ~1.1), a launch-bound loop that works as a CPU-speed gauge;
  - decode/prep_other slow-downs also occur in FP32 runs, just less often.

Leading explanations this run separates:
  H1  Host CPU slows in episodes (other tenants / frequency). The FP16 tower is
      short enough to be CPU-launch-bound, so it exposes them; FP32 hides them.
      The FP16 runs simply caught longer episodes.
  H2  Something about the FP16 state itself causes the slow-downs.
  H3  GPU clock / power-state changes.

Design -- one process, one model, one time window, so the host is identical
for both arms:
  - Two inner CLIPVisionModel copies loaded from the same checkpoint, one
    torch_dtype=float32 and one float16. Per trial the CLIPVisionTower wrapper's
    .vision_tower is pointed at one of them. The wrapper reads its dtype from
    the inner model, so its input/output casts follow, and bench_dev's hooks
    (on the wrapper) time it exactly as in the four runs. No model code edited.
  - FP32 and FP16 trials randomly interleaved (fixed seed), vtn=128 only: that's
    where prep_other is a usable CPU gauge, and the tower is config-independent.
  - Per trial, all outside the timed window: a fixed pure-Python CPU probe
    (host single-core speed, no GPU involved), and nvidia-smi SM/mem clocks,
    power, temperature and clock-throttle reasons.
  - Inside the timed window, one extra measurement: perf_counter around the
    tower's forward (no sync), i.e. how long the CPU spent *issuing* the tower's
    kernels. If that ~= the tower's CUDA-event time, the tower is CPU-bound.

Reading the result:
  H1 -> both arms' decode/prep_other track the CPU probe equally; FP16 trials
        are no slower than their FP32 neighbours; FP16 tower ~= its CPU issue
        time and tracks the probe, FP32 tower doesn't.
  H2 -> FP16 trials are slower than adjacent FP32 trials in decode/prep_other
        under the same CPU probe.
  H3 -> slow trials line up with SM/mem clock drops or non-idle throttle
        reasons, not with the CPU probe.

This is a diagnostic, not a reportable benchmark.

  python scripts/diag_wp2_fp16_jitter.py --run-id wp2_diag_1
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

REPO = "/workspace/GPU_Profiling"
sys.path.insert(0, f"{REPO}/vis_pruner_copy")
sys.path.insert(0, f"{REPO}/scripts")

from bench_dev import (  # noqa: E402
    FIXED_OUTPUT_TOKENS,
    IMPORTANT_RATIO,
    TORCH_CPU_THREADS,
    RegionRecorder,
    cgroup_throttle_count,
    derive_region_fields,
    git_dirty,
    git_dirty_files,
    install_mm_projector_hooks,
    install_prep_wrapper,
    install_vision_tower_hooks,
    load_items,
    merge_model_timing,
    other_gpu_processes,
    run_one,
    uninstall_region_timing,
)
from run_wp1_controlled_benchmark import MODEL_PATH, git_commit  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402
from transformers import CLIPVisionModel  # noqa: E402

VISUAL_TOKEN_NUM = 128
ARMS = {"fp32": torch.float32, "fp16": torch.float16}
N_WARMUP_PER_ARM = 10
N_MEASURED_PER_ARM = 60
TRIAL_ORDER_SEED = 20260922
CPU_PROBE_ITERS = 200_000   # ~10 ms of pure-Python work on this host

OUT_DIR = f"{REPO}/results/timing/wp2_diag"


def cpu_probe_ms():
    """Fixed single-threaded CPU work, timed on the same thread that launches
    kernels. No GPU involved, so it measures host core speed only."""
    t0 = time.perf_counter()
    s = 0
    for i in range(CPU_PROBE_ITERS):
        s += i * i
    return (time.perf_counter() - t0) * 1000.0


def gpu_state():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,"
             "clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        sm, mem, power, temp, reasons = [v.strip() for v in out.split(",")]
        return {"sm_clock_mhz": float(sm), "mem_clock_mhz": float(mem),
                "power_w": float(power), "temp_c": float(temp),
                "throttle_reasons": reasons}
    except Exception:
        return None


def install_tower_cpu_timer(tower, sink):
    """perf_counter around the tower's forward, no sync: CPU time spent issuing
    its kernels. Registered after bench_dev's event hooks, so it sits just
    inside them; the offset is a few microseconds."""
    state = {}

    def pre(_m, _a):
        state["t0"] = time.perf_counter()

    def post(_m, _a, _o):
        sink.append((time.perf_counter() - state.pop("t0")) * 1000.0)

    return [tower.register_forward_pre_hook(pre), tower.register_forward_hook(post)]


def load_inner_towers(tower):
    inner = {}
    for arm, dtype in ARMS.items():
        m = CLIPVisionModel.from_pretrained(tower.vision_tower_name, torch_dtype=dtype)
        m.requires_grad_(False)
        inner[arm] = m.to("cuda").eval()
    return inner


def pct(values, q):
    return float(np.percentile(values, q))


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize(trials):
    lines = []
    fields = ["vision_tower_ms", "tower_cpu_issue_ms", "mm_projector_ms", "prep_other_ms",
              "prefill_ms", "decode_per_token_ms", "e2e_ms", "cpu_probe_ms"]
    by_arm = {arm: [t for t in trials if t["arm"] == arm] for arm in ARMS}

    lines.append(f"{'field':22s}" + "".join(
        f"{arm + ' ' + s:>12s}" for arm in ARMS for s in ("p50", "p95", "sd")))
    for f in fields:
        row = f"{f:22s}"
        for arm in ARMS:
            v = [t[f] for t in by_arm[arm]]
            row += f"{pct(v, 50):12.2f}{pct(v, 95):12.2f}{np.std(v, ddof=1):12.3f}"
        lines.append(row)

    lines.append("")
    lines.append("tower CUDA-event time / CPU issue time (p50; ~1.0 => CPU-launch-bound):")
    for arm in ARMS:
        r = [t["vision_tower_ms"] / t["tower_cpu_issue_ms"] for t in by_arm[arm]]
        lines.append(f"  {arm}: {pct(r, 50):.2f}")

    lines.append("")
    lines.append("correlation with the CPU probe (H1 predicts: fp16 tower high, fp32 tower ~0,")
    lines.append("prep_other/decode similar in both arms):")
    for arm in ARMS:
        p = [t["cpu_probe_ms"] for t in by_arm[arm]]
        lines.append(f"  {arm}: " + "  ".join(
            f"{f.replace('_ms', '')} {corr(p, [t[f] for t in by_arm[arm]]):+.2f}"
            for f in ("vision_tower_ms", "prep_other_ms", "decode_per_token_ms", "e2e_ms")))

    lines.append("")
    lines.append("correlation with SM clock (H3 predicts strongly negative):")
    for arm in ARMS:
        s = [t["gpu"]["sm_clock_mhz"] for t in by_arm[arm] if t["gpu"]]
        if len(s) == len(by_arm[arm]):
            lines.append(f"  {arm}: " + "  ".join(
                f"{f.replace('_ms', '')} {corr(s, [t[f] for t in by_arm[arm]]):+.2f}"
                for f in ("vision_tower_ms", "prep_other_ms", "decode_per_token_ms")))
    clocks = [t["gpu"]["sm_clock_mhz"] for t in trials if t["gpu"]]
    reasons = sorted({t["gpu"]["throttle_reasons"] for t in trials if t["gpu"]})
    if clocks:
        lines.append(f"  SM clock pre-trial: min {min(clocks):.0f}  p50 {pct(clocks, 50):.0f}  "
                     f"max {max(clocks):.0f} MHz; throttle reasons seen: {reasons}")

    # H2 test: each fp16 trial vs the fp32 trials within +-3 positions, i.e. the
    # same host conditions. H1 predicts ~0 for every region except the tower.
    lines.append("")
    lines.append("fp16 minus neighbouring fp32 trials (within +-3 positions), median over fp16 trials:")
    by_pos = {t["trial_position"]: t for t in trials}
    for f in ("vision_tower_ms", "prep_other_ms", "decode_per_token_ms", "e2e_ms", "cpu_probe_ms"):
        diffs = []
        for t in by_arm["fp16"]:
            nb = [by_pos[p][f] for p in range(t["trial_position"] - 3, t["trial_position"] + 4)
                  if p in by_pos and by_pos[p]["arm"] == "fp32"]
            if nb:
                diffs.append(t[f] - float(np.mean(nb)))
        lines.append(f"  {f:22s} {pct(diffs, 50):+8.3f}  (n={len(diffs)})")
    return "\n".join(lines)


def run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = f"{OUT_DIR}/{args.run_id}.json"
    out_table = f"{OUT_DIR}/{args.run_id}.txt"
    if os.path.exists(out_json):
        sys.exit(f"{out_json} already exists. Pick a new --run-id.")
    procs = other_gpu_processes()
    if procs:
        sys.exit("Other processes are on the GPU (rule 9):\n  " + "\n  ".join(procs))

    torch.set_num_threads(TORCH_CPU_THREADS)
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, "llava-v1.5-7b",
        visual_token_num=VISUAL_TOKEN_NUM, important_ratio=IMPORTANT_RATIO,
    )
    model.eval()
    model.visual_token_num = VISUAL_TOKEN_NUM
    tower = model.get_model().get_vision_tower()
    inner = load_inner_towers(tower)
    tower.vision_tower = None  # drop the checkpoint-loaded copy; the arms replace it
    torch.cuda.empty_cache()
    for arm, m in inner.items():
        tower.vision_tower = m
        assert tower.dtype == ARMS[arm], (arm, tower.dtype)
    print(f"arms: " + ", ".join(f"{a}={m.dtype}" for a, m in inner.items()))

    items = load_items()
    recorder = RegionRecorder()
    handles = install_vision_tower_hooks(model, recorder)
    handles += install_mm_projector_hooks(model, recorder)
    install_prep_wrapper(model, recorder)
    issue_ms = []
    handles += install_tower_cpu_timer(tower, issue_ms)

    warmup_timing = f"{OUT_DIR}/{args.run_id}_warmup_model_timing.jsonl"
    measured_timing = f"{OUT_DIR}/{args.run_id}_model_timing.jsonl"
    for path in (warmup_timing, measured_timing):
        if os.path.exists(path):
            os.remove(path)

    trials = []
    try:
        os.environ["LLAVA_TIMING_FILE"] = warmup_timing
        for arm in ARMS:
            tower.vision_tower = inner[arm]
            print(f"Warm-up {arm}: {N_WARMUP_PER_ARM} requests, discarded")
            for i in range(N_WARMUP_PER_ARM):
                run_one(items[i % len(items)], tokenizer, model, image_processor,
                        recorder, True, f"warmup {arm} #{i}")
        os.remove(warmup_timing)
        issue_ms.clear()

        specs = [(arm, k) for arm in ARMS for k in range(N_MEASURED_PER_ARM)]
        random.Random(TRIAL_ORDER_SEED).shuffle(specs)
        os.environ["LLAVA_TIMING_FILE"] = measured_timing
        print(f"Measured: {len(specs)} interleaved requests ({N_MEASURED_PER_ARM}/arm, vtn={VISUAL_TOKEN_NUM})")
        for pos, (arm, k) in enumerate(specs):
            tower.vision_tower = inner[arm]
            item = items[k % len(items)]
            gpu = gpu_state()
            probe = cpu_probe_ms()
            throttles_before = cgroup_throttle_count()
            result = run_one(item, tokenizer, model, image_processor, recorder, True,
                             f"trial {pos} {arm}")
            throttles_after = cgroup_throttle_count()
            trials.append({
                "trial_position": pos,
                "arm": arm,
                "repeat_index": k,
                "question_id": item["question_id"],
                "utc": datetime.now(timezone.utc).isoformat(),
                "cpu_probe_ms": probe,
                "gpu": gpu,
                "cgroup_throttles": (None if throttles_before is None
                                     else throttles_after - throttles_before),
                "tower_cpu_issue_ms": issue_ms[-1],
                **result,
            })
            if pos % 20 == 0 or pos == len(specs) - 1:
                print(f"  [{pos + 1}/{len(specs)}] {arm} {item['question_id']}: "
                      f"{result['e2e_ms']:.1f} ms")
    finally:
        uninstall_region_timing(model, handles)

    if len(issue_ms) != len(trials):
        raise RuntimeError(f"{len(issue_ms)} tower CPU timings for {len(trials)} trials")
    merge_model_timing(trials, measured_timing)
    os.remove(measured_timing)
    for t in trials:
        derive_region_fields(t)

    bad_length = [t["trial_position"] for t in trials if t["n_forward_calls"] != FIXED_OUTPUT_TOKENS]
    throttled = [t["trial_position"] for t in trials if t["cgroup_throttles"]]
    dirty_files = git_dirty_files()
    report = {
        "metadata": {
            "run_id": args.run_id,
            "purpose": "WP2 FP16 variability diagnostic -- not a reportable benchmark",
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "git_dirty": git_dirty(dirty_files),
            "git_dirty_files": dirty_files,
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "torch_cpu_threads": torch.get_num_threads(),
            "visual_token_num": VISUAL_TOKEN_NUM,
            "important_ratio": IMPORTANT_RATIO,
            "arms": {a: str(d) for a, d in ARMS.items()},
            "n_warmup_per_arm": N_WARMUP_PER_ARM,
            "n_measured_per_arm": N_MEASURED_PER_ARM,
            "trial_order_seed": TRIAL_ORDER_SEED,
            "cpu_probe_iters": CPU_PROBE_ITERS,
            "fixed_output_tokens": FIXED_OUTPUT_TOKENS,
            "bad_length_trial_positions": bad_length,
            "throttled_trial_positions": throttled,
        },
        "trials": trials,
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=1)

    m = report["metadata"]
    header = (f"diag run '{args.run_id}' -- {m['gpu']}  git={m['git_commit'][:10]}"
              f"{' (DIRTY)' if m['git_dirty'] else ''}  cpu_threads={m['torch_cpu_threads']}\n"
              f"vtn={VISUAL_TOKEN_NUM}  {N_MEASURED_PER_ARM}/arm interleaved  "
              f"bad_length={bad_length}  throttled={throttled}\n")
    table = header + "\n" + summarize(trials) + "\n"
    with open(out_table, "w") as f:
        f.write(table)
    print("\n" + table)
    print(f"Wrote {out_json}\n      {out_table}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
