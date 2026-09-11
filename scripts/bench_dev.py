"""
WP1 dev benchmark -- per-region CUDA-event timing for visual_token_num=576 vs 128,
fixed-length output, trial-interleaved. Spec: AMAY_ENGINEERING_ROADMAP.md §WP1.

What this adds over scripts/run_wp1_controlled_benchmark.py (the accepted
whole-request baseline, left untouched):
  - Component timing for vision_tower, mm_projector and multimodal_prep, which the
    model does not emit on its own. (Its existing multimodal_prep CUDA event lives
    in forward(), which never runs prep -- generate() does prep itself and hands HF
    inputs_embeds -- so that event reads ~0.012 ms. See the WP1 raw jsonl.)
  - Roadmap spec: 20 fixed dev images, 32 output tokens, per-region p50/p95.
  - A --run-id per run, so three baseline runs don't overwrite each other or the
    accepted WP1 artifacts, and a --compare mode that computes the run-to-run
    noise floor across them.

Ownership: the scaffold is Claude's; the three TODO(AMAY) blocks -- the code that
actually creates and records CUDA events -- are Amay's (CLAUDE.md, "core timing
and profiling code"). Until they're filled in, run with --no-regions.

Usage (run from anywhere; paths are absolute):
  python scripts/bench_dev.py --run-id smoke --quick --no-regions   # scaffold only
  python scripts/bench_dev.py --run-id smoke --quick                # after the TODOs
  python scripts/bench_dev.py --run-id baseline_1                   # full run, x3
  python scripts/bench_dev.py --compare results/timing/bench_dev/baseline_{1,2,3}.json

Commit this file before the three baseline runs: the output records whether
tracked files were dirty, and a dirty-tree baseline isn't attributable (rule 7).
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

# Reused from the accepted WP1 script so both benchmarks build requests
# identically (same prompt template, same image preprocessing, same dtype).
from run_wp1_controlled_benchmark import (  # noqa: E402
    DATASET_PATH,
    MODEL_PATH,
    build_request,
    git_commit,
    sample_gpu_util,
)
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402

CONFIGS = [576, 128]          # 576 = unpruned baseline, 128 = VisPruner
IMPORTANT_RATIO = 0.5         # held constant; Sribhav's axis, not swept here
N_IMAGES = 20                 # roadmap §WP1 step 2: same 20 dev images, same order
FIXED_OUTPUT_TOKENS = 32      # roadmap §WP1 step 3: min_new_tokens == max_new_tokens
N_WARMUP_PER_CONFIG = 10      # rule 3, discarded
N_MEASURED_TRIALS_PER_CONFIG = 40  # standing protocol since 2026-09-07 (roadmap says 30)
TRIAL_ORDER_SEED = 20260911   # same seed every run: only system noise varies between runs
VISION_TOWER_SPREAD_GATE_MS = 5.0  # roadmap §WP1 §9

OUT_DIR = f"{REPO}/results/timing/bench_dev"

HOOKED_REGIONS = ("vision_tower", "mm_projector", "multimodal_prep")

# Per-trial fields summarised into p50/p95, in report order.
#   prep_other_ms     = multimodal_prep - vision_tower - mm_projector: VisPruner's
#                       token selection (incl. the diversity while-loop) + splice/pad.
#                       This is the part of prep that changes with visual_token_num.
#   prep_wallclock_ms = the model's own perf_counter+sync prep number, kept as a
#                       cross-check. Wallclock minus CUDA-event prep ~= CPU time the
#                       GPU spent waiting on.
REPORTED_FIELDS = [
    "vision_tower_ms",
    "mm_projector_ms",
    "prep_other_ms",
    "multimodal_prep_ms",
    "prep_wallclock_ms",
    "prefill_ms",
    "decode_ms",
    "decode_per_token_ms",
    "e2e_ms",
]


# ---------------------------------------------------------------------------
# Event storage shared by the three TODO(AMAY) blocks
# ---------------------------------------------------------------------------
class RegionRecorder:
    """Holds the CUDA events recorded during ONE request.

    Contract the TODO(AMAY) code has to meet: for every call of a region, append
    exactly one (start_event, end_event) pair to self.pairs[region_name]. Nothing
    else touches those events until read_ms().
    """

    def __init__(self):
        self.pairs = {region: [] for region in HOOKED_REGIONS}
        # Free scratch space -- e.g. somewhere to keep a start event between the
        # moment it's recorded and the moment its matching end is recorded.
        self.pending = {}

    def reset(self):
        for region in HOOKED_REGIONS:
            self.pairs[region].clear()
        self.pending.clear()

    def read_ms(self):
        # Reading events is only safe here. model.generate() ends with
        # torch.cuda.synchronize() (llava_llama.py:223), so by the time run_one()
        # calls this, every event recorded during the request has completed on
        # the GPU. elapsed_time() on an event that hasn't completed raises.
        return {
            region: [start.elapsed_time(end) for start, end in pairs]
            for region, pairs in self.pairs.items()
        }


# ---------------------------------------------------------------------------
# TODO(AMAY) #1 -- vision_tower CUDA-event timing via forward hooks
#
# What has to happen: time one call of the vision tower module -- from just
# before its forward() starts to just after it returns -- and leave a
# (start, end) event pair in recorder.pairs["vision_tower"].
#
# Available:
#   module   -- the CLIPVisionTower wrapper (clip_encoder.py:7). Hook THIS, not
#               module.vision_tower (the inner HF CLIPVisionModel): the wrapper's
#               forward includes the fp16->fp32 input cast and fp32->fp16 output
#               cast (clip_encoder.py:64-70), which WP2 removes, so they belong in
#               the region WP2 is judged on.
#   recorder -- RegionRecorder above. recorder.pending is yours to use.
#   handles  -- append every handle your registration calls return, so
#               uninstall_region_timing() can remove the hooks afterwards.
#
# Rules: record events only. No elapsed_time() and no torch.cuda.synchronize()
# in here -- see RegionRecorder.read_ms() for where reading happens and why.
# Expected: exactly 1 call per request (single-image path). run_one() checks.
# ---------------------------------------------------------------------------
def install_vision_tower_hooks(model, recorder):
    module = model.get_model().get_vision_tower()
    handles = []
    # --- TODO(AMAY) #1 begins ---
    raise NotImplementedError("TODO(AMAY) #1: vision_tower forward hooks")
    # --- TODO(AMAY) #1 ends ---
    return handles


# ---------------------------------------------------------------------------
# TODO(AMAY) #2 -- mm_projector CUDA-event timing via forward hooks
#
# Same shape as #1, different module, pairs go in recorder.pairs["mm_projector"].
#
# Available:
#   module   -- model.get_model().mm_projector, an nn.Sequential (Linear-GELU-
#               Linear), already fp16. Called once per request at
#               llava_arch.py:185, AFTER the pruning selection -- note it still
#               runs on all 576 tokens at both configs (it projects before the
#               gather happens), so don't expect it to differ between 576 and 128.
#   recorder, handles -- as in #1.
#
# Watch for: if #1 and #2 both use recorder.pending, they need separate keys.
# Expected: exactly 1 call per request. run_one() checks.
# ---------------------------------------------------------------------------
def install_mm_projector_hooks(model, recorder):
    module = model.get_model().mm_projector
    handles = []
    # --- TODO(AMAY) #2 begins ---
    raise NotImplementedError("TODO(AMAY) #2: mm_projector forward hooks")
    # --- TODO(AMAY) #2 ends ---
    return handles


# ---------------------------------------------------------------------------
# TODO(AMAY) #3 -- multimodal_prep CUDA-event timing
#
# prepare_inputs_labels_for_multimodal is a method, not an nn.Module, so there
# are no forward hooks to register. Instead the scaffold below swaps a wrapper in
# on the model INSTANCE: generate() and forward() both call
# self.prepare_inputs_labels_for_multimodal(...), and Python checks the instance's
# __dict__ before the class, so they find timed_prep without any edit to
# llava_arch.py. uninstall_region_timing() deletes the instance attribute, which
# brings the class method back.
#
# What has to happen: inside timed_prep, bracket the call to `original` so that a
# (start, end) pair lands in recorder.pairs["multimodal_prep"], and return
# original's result unchanged (6 values on its early-return path, 7 on the full
# path -- pass it through, don't unpack it).
#
# Careful -- this fires more than once per request:
#   - once from generate() with the full prompt. This is the real work: vision
#     tower + VisPruner selection + projector + splice/pad. Its region contains
#     the #1 and #2 regions, and that nesting is fine.
#   - once more per decode step, from forward(), where input_ids.shape[1] == 1 and
#     it returns immediately (llava_arch.py:194-195).
# You can either record every call or only the real one -- the summary uses the
# FIRST pair of each request and checks that there's at least one.
#
# Available: original (the bound method, captured before the swap, so calling it
# can't recurse into timed_prep), recorder, and *args/**kwargs exactly as the
# caller passed them.
# ---------------------------------------------------------------------------
def install_prep_wrapper(model, recorder):
    original = model.prepare_inputs_labels_for_multimodal

    def timed_prep(*args, **kwargs):
        # --- TODO(AMAY) #3 begins ---
        raise NotImplementedError("TODO(AMAY) #3: multimodal_prep timing")
        # --- TODO(AMAY) #3 ends ---

    model.prepare_inputs_labels_for_multimodal = timed_prep


def uninstall_region_timing(model, handles):
    for handle in handles:
        handle.remove()
    model.__dict__.pop("prepare_inputs_labels_for_multimodal", None)


# ---------------------------------------------------------------------------
# Scaffold: environment checks, request loop, summaries, output
# ---------------------------------------------------------------------------
def git_dirty():
    """True if tracked files differ from HEAD -- the recorded commit then doesn't
    describe the code that ran. Untracked files (earlier results) don't count."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO
        ).decode().strip()
        return bool(out)
    except Exception:
        return None


def other_gpu_processes():
    """Rule 9: one person on the GPU during benchmark runs."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            timeout=5,
        ).decode().strip()
        return [line for line in out.splitlines() if line.strip()]
    except Exception:
        return None


def load_items():
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    return dataset[:N_IMAGES]


def check_region_counts(hooked_ms, desc):
    for region in ("vision_tower", "mm_projector"):
        n = len(hooked_ms[region])
        if n != 1:
            raise RuntimeError(
                f"{desc}: {region} recorded {n} (start, end) pairs, expected 1. "
                "0 means the hook didn't record/append; >1 means a pair per call is "
                "being appended more than once, or the multi-image branch ran."
            )
    if not hooked_ms["multimodal_prep"]:
        raise RuntimeError(f"{desc}: multimodal_prep recorded no (start, end) pairs.")


def run_one(item, tokenizer, model, image_processor, recorder, regions_enabled, desc):
    # Request construction (CPU image preprocessing, tokenisation, H2D copy) is
    # deliberately outside the timed window, same as the WP1 script.
    input_ids, attention_mask, images, image_sizes = build_request(
        item, tokenizer, model, image_processor
    )
    recorder.reset()
    torch.cuda.synchronize()  # previous request's work can't leak into this window
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids, visual_token_num_used = model.generate(
            input_ids, attention_mask=attention_mask, images=images,
            image_sizes=image_sizes, do_sample=False, use_cache=True,
            min_new_tokens=FIXED_OUTPUT_TOKENS, max_new_tokens=FIXED_OUTPUT_TOKENS,
        )
    torch.cuda.synchronize()
    e2e_ms = (time.perf_counter() - t0) * 1000.0

    if visual_token_num_used != model.visual_token_num:
        raise RuntimeError(
            f"{desc}: model kept {visual_token_num_used} visual tokens, "
            f"expected {model.visual_token_num} -- the config flip didn't take effect."
        )

    hooked_ms = recorder.read_ms() if regions_enabled else None
    if regions_enabled:
        check_region_counts(hooked_ms, desc)

    return {
        "e2e_ms": e2e_ms,
        "prompt_len": input_ids.shape[1],
        # HF prepends a BOS id when generate() gets only inputs_embeds, so this is
        # FIXED_OUTPUT_TOKENS + 1. n_forward_calls (merged in later) is the real
        # generated-token count.
        "output_ids_len": output_ids.shape[1],
        "visual_token_num_used": visual_token_num_used,
        "hooked_ms": hooked_ms,
    }


def merge_model_timing(trials, timing_path):
    """Attach the model's own per-generate() timing line (llava_llama.py:229-248)
    to each trial. prefill/decode come from its forward() CUDA events, which do
    bracket real work (unlike its multimodal_prep event)."""
    with open(timing_path) as f:
        lines = [json.loads(line) for line in f]
    if len(lines) != len(trials):
        raise RuntimeError(
            f"expected {len(trials)} model timing lines, got {len(lines)} -- "
            "trials and timing lines are misaligned"
        )
    for trial, line in zip(trials, lines):
        calls = line["forward_calls"]
        trial["n_forward_calls"] = len(calls)
        trial["prefill_ms"] = calls[0]["lm_forward_time"] * 1000.0
        trial["decode_ms"] = sum(c["lm_forward_time"] for c in calls[1:]) * 1000.0
        trial["decode_per_token_ms"] = trial["decode_ms"] / max(len(calls) - 1, 1)
        trial["prep_wallclock_ms"] = line["multimodal_prep_time_generate"] * 1000.0


def derive_region_fields(trial):
    hooked = trial.pop("hooked_ms")
    if hooked is None:
        for field in ("vision_tower_ms", "mm_projector_ms", "multimodal_prep_ms",
                      "prep_other_ms"):
            trial[field] = None
        trial["multimodal_prep_calls"] = None
        return
    trial["vision_tower_ms"] = hooked["vision_tower"][0]
    trial["mm_projector_ms"] = hooked["mm_projector"][0]
    trial["multimodal_prep_ms"] = hooked["multimodal_prep"][0]  # first = generate()'s call
    trial["multimodal_prep_calls"] = len(hooked["multimodal_prep"])
    trial["prep_other_ms"] = (
        trial["multimodal_prep_ms"] - trial["vision_tower_ms"] - trial["mm_projector_ms"]
    )


def summarize(values):
    # p50/p95 only -- with 40 samples p99 is effectively the max, i.e. noise.
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def build_summary(trials):
    summary = {}
    for cfg in CONFIGS:
        rows = [t for t in trials if t["visual_token_num"] == cfg]
        summary[str(cfg)] = {
            field: (summarize([r[field] for r in rows]) if rows[0][field] is not None else None)
            for field in REPORTED_FIELDS
        }
    return summary


def format_table(report):
    meta, summary = report["metadata"], report["summary"]
    lines = [
        f"bench_dev run '{meta['run_id']}' -- {meta['gpu']}  git={meta['git_commit'][:10]}"
        f"{' (DIRTY)' if meta['git_dirty'] else ''}",
        f"n_measured/config={meta['n_measured_trials_per_config']}  "
        f"warmup/config={meta['n_warmup_per_config']}  "
        f"output_tokens={meta['fixed_output_tokens']}  images={meta['n_images']}  "
        f"regions={'on' if meta['regions_enabled'] else 'OFF'}",
        "",
        f"{'region':<22}" + "".join(f"{f'{c} p50':>12}{f'{c} p95':>12}" for c in CONFIGS),
    ]
    for field in REPORTED_FIELDS:
        cells = []
        for cfg in CONFIGS:
            s = summary[str(cfg)][field]
            cells.append(f"{'--':>12}{'--':>12}" if s is None
                         else f"{s['p50_ms']:>12.2f}{s['p95_ms']:>12.2f}")
        lines.append(f"{field:<22}" + "".join(cells))
    if not meta["fixed_length_held"]:
        lines += ["", "WARNING: fixed output length did not hold for every trial -- see raw trials."]
    return "\n".join(lines)


def run_benchmark(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = f"{OUT_DIR}/{args.run_id}.json"
    out_table = f"{OUT_DIR}/{args.run_id}.txt"
    if os.path.exists(out_json) and not args.overwrite:
        sys.exit(f"{out_json} already exists. Pick a new --run-id, or pass --overwrite.")

    procs = other_gpu_processes()
    if procs:
        print("Other processes are on the GPU (rule 9):\n  " + "\n  ".join(procs))
        if not args.allow_shared_gpu:
            sys.exit("Refusing to benchmark on a shared GPU. --allow-shared-gpu overrides.")

    n_warmup = 2 if args.quick else N_WARMUP_PER_CONFIG
    n_measured = 4 if args.quick else args.trials
    regions_enabled = not args.no_regions

    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, "llava-v1.5-7b",
        visual_token_num=CONFIGS[0], important_ratio=IMPORTANT_RATIO,
    )
    model.eval()
    vision_tower_dtype = str(model.get_model().get_vision_tower().dtype)

    items = load_items()
    print(f"{len(items)} fixed dev images, vision_tower dtype={vision_tower_dtype}")

    recorder = RegionRecorder()
    handles = []
    if regions_enabled:
        handles += install_vision_tower_hooks(model, recorder)
        handles += install_mm_projector_hooks(model, recorder)
        install_prep_wrapper(model, recorder)

    warmup_timing = f"{OUT_DIR}/{args.run_id}_warmup_model_timing.jsonl"
    measured_timing = f"{OUT_DIR}/{args.run_id}_model_timing.jsonl"
    for path in (warmup_timing, measured_timing):
        if os.path.exists(path):
            os.remove(path)

    trials = []
    try:
        # Warm-up: per config, discarded. Region counts are still checked, so a
        # broken hook fails on the first request instead of after the full run.
        os.environ["LLAVA_TIMING_FILE"] = warmup_timing
        for cfg in CONFIGS:
            model.visual_token_num = cfg
            print(f"Warm-up vtn={cfg}: {n_warmup} requests, discarded")
            for i in range(n_warmup):
                run_one(items[i % len(items)], tokenizer, model, image_processor,
                        recorder, regions_enabled, f"warmup vtn={cfg} #{i}")
        os.remove(warmup_timing)

        specs = [(cfg, k) for cfg in CONFIGS for k in range(n_measured)]
        random.Random(TRIAL_ORDER_SEED).shuffle(specs)
        os.environ["LLAVA_TIMING_FILE"] = measured_timing
        print(f"Measured: {len(specs)} interleaved requests ({n_measured}/config)")
        for pos, (cfg, k) in enumerate(specs):
            model.visual_token_num = cfg
            item = items[k % len(items)]
            gpu_util = sample_gpu_util()
            result = run_one(item, tokenizer, model, image_processor, recorder,
                             regions_enabled, f"trial {pos} vtn={cfg}")
            trials.append({
                "trial_position": pos,
                "visual_token_num": cfg,
                "repeat_index": k,
                "question_id": item["question_id"],
                "gpu_util_pct_pre_trial": gpu_util,
                **result,
            })
            if pos % 10 == 0 or pos == len(specs) - 1:
                print(f"  [{pos + 1}/{len(specs)}] vtn={cfg} {item['question_id']}: "
                      f"{result['e2e_ms']:.1f} ms")
    finally:
        uninstall_region_timing(model, handles)

    merge_model_timing(trials, measured_timing)
    os.remove(measured_timing)  # merged into the trials below; nothing lost
    for trial in trials:
        derive_region_fields(trial)

    bad_length = [t["trial_position"] for t in trials
                  if t["n_forward_calls"] != FIXED_OUTPUT_TOKENS]
    if bad_length:
        print(f"WARNING: {len(bad_length)} trials didn't produce {FIXED_OUTPUT_TOKENS} "
              f"tokens (trial positions {bad_length}); output length variance is back in.")

    report = {
        "metadata": {
            "run_id": args.run_id,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "model_path": MODEL_PATH,
            "vision_tower_dtype": vision_tower_dtype,
            "important_ratio": IMPORTANT_RATIO,
            "configs": CONFIGS,
            "n_images": len(items),
            "question_ids": [it["question_id"] for it in items],
            "fixed_output_tokens": FIXED_OUTPUT_TOKENS,
            "fixed_length_held": not bad_length,
            "n_warmup_per_config": n_warmup,
            "n_measured_trials_per_config": n_measured,
            "trial_order_seed": TRIAL_ORDER_SEED,
            "regions_enabled": regions_enabled,
            "quick_smoke_test": args.quick,
            "other_gpu_processes_at_start": procs,
        },
        "summary": build_summary(trials),
        "trials": trials,
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    table = format_table(report)
    with open(out_table, "w") as f:
        f.write(table + "\n")
    print("\n" + table + f"\n\nWrote {out_json}\n      {out_table}")


# ---------------------------------------------------------------------------
# --compare: run-to-run noise floor across baseline runs (roadmap §WP1 §8/§9)
# ---------------------------------------------------------------------------
def compare_runs(paths):
    if len(paths) < 2:
        sys.exit("--compare needs at least two run JSONs.")
    runs = []
    for path in paths:
        with open(path) as f:
            runs.append(json.load(f))
    metas = [r["metadata"] for r in runs]
    ids = [m["run_id"] for m in metas]

    warnings = []
    for key in ("git_commit", "gpu", "fixed_output_tokens", "n_measured_trials_per_config",
                "vision_tower_dtype"):
        if len({str(m[key]) for m in metas}) > 1:
            warnings.append(f"runs differ on {key}: {[m[key] for m in metas]}")
    for m in metas:
        if m["git_dirty"]:
            warnings.append(f"{m['run_id']} ran on a dirty tree")
        if m["quick_smoke_test"]:
            warnings.append(f"{m['run_id']} is a --quick smoke run")
        if not m["regions_enabled"]:
            warnings.append(f"{m['run_id']} has no region timing (--no-regions)")

    # Spread = max - min of the per-run p50s: how far a no-change rerun moves
    # the number. A later before/after difference smaller than this isn't a result.
    spreads = {}
    for cfg in map(str, CONFIGS):
        spreads[cfg] = {}
        for field in REPORTED_FIELDS:
            p50s = [r["summary"][cfg][field]["p50_ms"] if r["summary"][cfg][field] else None
                    for r in runs]
            if None in p50s:
                spreads[cfg][field] = None
                continue
            spreads[cfg][field] = {
                "p50s_ms": p50s,
                "spread_ms": max(p50s) - min(p50s),
                "spread_pct_of_median": (max(p50s) - min(p50s)) / float(np.median(p50s)) * 100.0,
            }

    gate = {}
    for cfg in map(str, CONFIGS):
        s = spreads[cfg]["vision_tower_ms"]
        gate[cfg] = None if s is None else s["spread_ms"] <= VISION_TOWER_SPREAD_GATE_MS

    lines = [f"Noise floor across {len(runs)} runs: {', '.join(ids)}", ""]
    lines += [f"WARNING: {w}" for w in warnings]
    if warnings:
        lines.append("")
    header = f"{'region':<22}{'vtn':>6}" + "".join(f"{i[:10]:>12}" for i in ids)
    lines.append(header + f"{'spread':>10}{'%':>7}")
    for field in REPORTED_FIELDS:
        for cfg in map(str, CONFIGS):
            s = spreads[cfg][field]
            if s is None:
                continue
            lines.append(f"{field:<22}{cfg:>6}" + "".join(f"{p:>12.2f}" for p in s["p50s_ms"])
                         + f"{s['spread_ms']:>10.2f}{s['spread_pct_of_median']:>6.1f}%")
    lines.append("")
    for cfg, ok in gate.items():
        verdict = "n/a (no region timing)" if ok is None else ("PASS" if ok else "FAIL")
        lines.append(f"GATE vtn={cfg}: vision_tower p50 spread <= "
                     f"{VISION_TOWER_SPREAD_GATE_MS:.0f} ms -> {verdict}")
    text = "\n".join(lines)
    print(text)

    stem = f"{OUT_DIR}/compare_{'_'.join(ids)}"
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{stem}.json", "w") as f:
        json.dump({"runs": paths, "warnings": warnings, "spreads": spreads,
                   "vision_tower_gate_ms": VISION_TOWER_SPREAD_GATE_MS,
                   "gate_passed": gate}, f, indent=2)
    with open(f"{stem}.txt", "w") as f:
        f.write(text + "\n")
    print(f"\nWrote {stem}.json\n      {stem}.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", help="Output name, e.g. baseline_1. Required unless --compare.")
    parser.add_argument("--trials", type=int, default=N_MEASURED_TRIALS_PER_CONFIG,
                        help=f"Measured trials per config (default {N_MEASURED_TRIALS_PER_CONFIG}).")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 2 warm-up + 4 measured per config.")
    parser.add_argument("--no-regions", action="store_true",
                        help="Skip the TODO(AMAY) region timing; prefill/decode/e2e only.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing an existing run with the same --run-id.")
    parser.add_argument("--allow-shared-gpu", action="store_true",
                        help="Run even if other processes are on the GPU (breaks rule 9).")
    parser.add_argument("--compare", nargs="+", metavar="RUN_JSON",
                        help="Compare finished runs instead of benchmarking.")
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare)
    elif args.run_id:
        run_benchmark(args)
    else:
        parser.error("--run-id is required (or use --compare).")


if __name__ == "__main__":
    main()