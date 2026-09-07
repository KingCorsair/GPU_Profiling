"""
WP1 -- Controlled Benchmark: a trustworthy latency baseline for
visual_token_num=576 vs. 128, fixed-output-length, interleaved trials.

Purpose (see AMAY_ENGINEERING_ROADMAP.md §8/§10 and CLAUDE.md rules 3/6/7/8/10):
establish a noise floor for complete-request latency at both configs so that
later phases (FP16 vision tower, etc.) can judge whether a ~20-30ms change is
real or noise. This is a measurement-only script -- no model/pruning code is
changed.

Differs from scripts/run_noload_ab_comparison.py (Part 1 of the earlier A/B
experiment) in exactly the ways WP1 requires:
  - Fixed output length (min_new_tokens == max_new_tokens) instead of natural
    EOS stopping, so decode-length variance doesn't leak into latency.
  - 576 and 128 trials are interleaved at the individual-trial level (random
    order), not run as two separate blocks.
  - 40 measured trials per config (not 10), warm-up excluded, with an
    explicit noise-floor / minimum-detectable-effect (MDE) calculation at the
    end so "is this stable enough" is answered with a number, not assumed.

WP1's own run used 30 trials/config and was accepted on 2026-09-07 for
measuring ~30ms effects, with the ~20ms claim resting on the MAD-based robust
estimate (raw stdev put the 128 config at a 20.4ms MDE, just over the line).
Per that acceptance, the standing protocol from WP2 onward is 40 trials/config
so the ~20ms claim holds on ORDINARY/RAW variance without leaning primarily on
the robust estimate. The success criterion below was changed to match.

Reuses the model's own CUDA-event timing (llava_llama.py's generate(), see
DEFAULT_TIMING_FILE there) for multimodal-prep/prefill/decode component
timing -- nothing added or changed in model code.

NOT representative of real-world serving latency: fixed-length generation is
a controlled engineering benchmark only, chosen specifically to remove
output-length variance from the comparison. Natural-EOS behavior is measured
elsewhere (ab_noload_comparison.json, the load-sweep scripts).
"""
import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from scipy import stats as scipy_stats

sys.path.insert(0, "vis_pruner_copy")

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from PIL import Image

MODEL_PATH = "/workspace/GPU_Profiling/vis_pruner_copy/checkpoints/llava-v1.5-7b"
DATASET_PATH = "/workspace/GPU_Profiling/vis_pruner_copy/vispruner_eval_dataset/dev.json"
IMAGES_ROOT = "/workspace/GPU_Profiling/vis_pruner_copy/vispruner_eval_dataset"

IMPORTANT_RATIO = 0.5      # held constant across both configs
CONFIGS = [576, 128]       # A: effectively-unpruned baseline, B: VisPruner
N_FIXED_REQUESTS = 10      # same fixed set of requests/images for both configs
FIXED_OUTPUT_TOKENS = 64   # min_new_tokens == max_new_tokens -- forced length,
                           # removes decode-length variance from the timing
N_WARMUP_PER_CONFIG = 10   # rule 3, discarded, run once per config before any
                           # measured (interleaved) trial for that config
N_MEASURED_TRIALS_PER_CONFIG = 40  # standing protocol from WP2 onward (raised
                           # from 30 at WP1 acceptance, 2026-09-07): enough that
                           # the RAW-variance MDE clears 20ms at both configs.
                           # Projected from WP1's observed stdev: 11.9ms (576),
                           # 17.6ms (128), vs. 13.8/20.4ms at n=30.
TRIAL_ORDER_SEED = 20260907  # fixed so the interleave order is reproducible

RESULTS_DIR = "/workspace/GPU_Profiling/results/timing"
OUT_RAW = f"{RESULTS_DIR}/wp1_controlled_benchmark_raw.jsonl"
OUT_SUMMARY = f"{RESULTS_DIR}/wp1_controlled_benchmark_summary.json"
OUT_TABLE = f"{RESULTS_DIR}/wp1_controlled_benchmark_table.txt"
WARMUP_TIMING_FILE = f"{RESULTS_DIR}/wp1_warmup_scratch.jsonl"
MEASURED_TIMING_FILE = f"{RESULTS_DIR}/wp1_measured_timing.jsonl"

# Detectability targets this baseline must be able to speak to (per the
# class of optimization WP2+ expects to test).
MDE_TARGETS_MS = [20.0, 30.0]
ALPHA = 0.05
POWER = 0.80
# Pass/fail target for the criterion below. RAW (ordinary) variance must clear
# this on its own; the robust estimate stays reported as a cross-check.
SUCCESS_CRITERION_TARGET_MS = 20.0


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd="/workspace/GPU_Profiling"
        ).decode().strip()
    except Exception:
        return "unknown"


def sample_gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=2,
        ).decode().strip()
        return float(out)
    except Exception:
        return None


def build_request(item, tokenizer, model, image_processor):
    image = Image.open(f"{IMAGES_ROOT}/{item['image']}").convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]
    images = image_tensor.unsqueeze(0).half().cuda()
    image_sizes = [image.size]

    qs = DEFAULT_IMAGE_TOKEN + "\n" + item["question"]
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    return input_ids, attention_mask, images, image_sizes


def run_one(item, tokenizer, model, image_processor, fixed_output_tokens):
    input_ids, attention_mask, images, image_sizes = build_request(item, tokenizer, model, image_processor)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids, visual_token_num_used = model.generate(
            input_ids, attention_mask=attention_mask, images=images, image_sizes=image_sizes,
            do_sample=False, use_cache=True,
            min_new_tokens=fixed_output_tokens, max_new_tokens=fixed_output_tokens,
        )
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    # model.generate() calls super().generate() with inputs_embeds only (no
    # input_ids -- see llava_llama.py's generate()), so HF returns ONLY the
    # newly generated token ids, not prompt+generated. output_ids.shape[1]
    # is already the new-token count; do not subtract prompt_len here.
    new_tokens = output_ids.shape[1]
    return {
        "wall_latency_s": wall_s,
        "prompt_len": input_ids.shape[1],
        "new_tokens": new_tokens,
        "visual_token_num_used": visual_token_num_used,
    }


def percentile(data, p):
    return float(np.percentile(np.asarray(data, dtype=float), p))


def mad_stdev(data):
    """Median absolute deviation, scaled to be a consistent estimator of
    stdev under near-normal data (factor 1.4826). Robust to the occasional
    single-trial system hiccup (page fault, scheduling jitter) that a raw
    stdev over n~30 samples cannot absorb -- one such outlier can dominate
    statistics.stdev without reflecting genuine run-to-run noise."""
    med = statistics.median(data)
    return 1.4826 * statistics.median([abs(x - med) for x in data])


def flag_outliers(data, threshold=3.5):
    """Iglewicz & Hoaglin modified z-score outlier flag: robust to the very
    outliers it's trying to detect, unlike a mean/stdev-based z-score."""
    med = statistics.median(data)
    mad = statistics.median([abs(x - med) for x in data])
    if mad == 0:
        return []
    return [i for i, x in enumerate(data) if abs(0.6745 * (x - med) / mad) > threshold]


def summarize_latencies(latencies_ms):
    n = len(latencies_ms)
    mean = statistics.mean(latencies_ms)
    stdev = statistics.stdev(latencies_ms) if n > 1 else 0.0
    robust_stdev = mad_stdev(latencies_ms) if n > 1 else 0.0
    return {
        "n": n,
        "mean_ms": mean,
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": percentile(latencies_ms, 95),
        "stdev_ms": stdev,
        "mad_stdev_ms": robust_stdev,
        "sem_ms": stdev / math.sqrt(n) if n > 1 else 0.0,
        "cv_percent": (stdev / mean * 100.0) if mean else None,
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "outlier_indices": flag_outliers(latencies_ms),
    }


def two_sample_mde_ms(stdev_ms, n_per_group, alpha=ALPHA, power=POWER):
    """Minimum detectable difference between two independent samples of
    size n_per_group, each with the observed stdev, at the given
    significance/power -- i.e. the smallest true before/after change this
    benchmark design could reliably tell apart from noise."""
    df = 2 * n_per_group - 2
    t_alpha = scipy_stats.t.ppf(1 - alpha / 2, df)
    t_beta = scipy_stats.t.ppf(power, df)
    return float((t_alpha + t_beta) * stdev_ms * math.sqrt(2.0 / n_per_group))


def required_n_for_mde(stdev_ms, target_mde_ms, alpha=ALPHA, power=POWER):
    """Smallest n_per_group (iterated, since df depends on n) that would
    bring the MDE at/under target_mde_ms."""
    n = 10
    for _ in range(50):
        mde = two_sample_mde_ms(stdev_ms, n, alpha, power)
        if mde <= target_mde_ms:
            return n
        n += 1
        if n > 5000:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                         help="Smoke test: 2 warm-up + 4 measured trials per config instead of the full run.")
    parser.add_argument("--trials", type=int, default=N_MEASURED_TRIALS_PER_CONFIG,
                         help=f"Measured trials per config (default {N_MEASURED_TRIALS_PER_CONFIG}). "
                              "Lowering it below the standing protocol will usually fail the "
                              "raw-variance success criterion -- the run reports required_n either way.")
    args = parser.parse_args()

    n_warmup = 2 if args.quick else N_WARMUP_PER_CONFIG
    n_measured = 4 if args.quick else args.trials

    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, "llava-v1.5-7b",
        visual_token_num=576,  # starting value; flipped per trial below
        important_ratio=IMPORTANT_RATIO,
    )
    model.eval()

    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    items = dataset[:N_FIXED_REQUESTS]
    print(f"Fixed set of {len(items)} requests (same items/order for both configs):")
    for it in items:
        print(f"  {it['question_id']}: {it['question'][:60]!r} [{it['category']}]")

    # --- Warm-up: per config, discarded, before any measured trial runs ---
    os.environ["LLAVA_TIMING_FILE"] = WARMUP_TIMING_FILE
    for cfg in CONFIGS:
        model.visual_token_num = cfg
        print(f"\nWarming up visual_token_num={cfg} ({n_warmup} runs, discarded)...")
        for i in range(n_warmup):
            run_one(items[i % len(items)], tokenizer, model, image_processor, FIXED_OUTPUT_TOKENS)
    torch.cuda.synchronize()
    if os.path.exists(WARMUP_TIMING_FILE):
        os.remove(WARMUP_TIMING_FILE)

    # --- Build the interleaved measured-trial order ---
    trial_specs = [(cfg, k) for cfg in CONFIGS for k in range(n_measured)]
    rng = random.Random(TRIAL_ORDER_SEED)
    rng.shuffle(trial_specs)

    if os.path.exists(MEASURED_TIMING_FILE):
        os.remove(MEASURED_TIMING_FILE)
    os.environ["LLAVA_TIMING_FILE"] = MEASURED_TIMING_FILE

    print(f"\nRunning {len(trial_specs)} interleaved measured trials "
          f"({n_measured} per config, seed={TRIAL_ORDER_SEED})...")
    executed = []
    for pos, (cfg, k) in enumerate(trial_specs):
        model.visual_token_num = cfg
        item = items[k % len(items)]
        gpu_util = sample_gpu_util()
        result = run_one(item, tokenizer, model, image_processor, FIXED_OUTPUT_TOKENS)
        executed.append({
            "trial_position": pos,
            "visual_token_num": cfg,
            "repeat_index": k,
            "question_id": item["question_id"],
            "gpu_util_pct_pre_trial": gpu_util,
            **result,
        })
        if pos % 10 == 0 or pos == len(trial_specs) - 1:
            print(f"  [{pos + 1}/{len(trial_specs)}] cfg={cfg} {item['question_id']}: "
                  f"{result['wall_latency_s'] * 1000:.1f} ms, new_tokens={result['new_tokens']}")

    # --- Read back the model's own per-call CUDA-event timing, in the same order ---
    with open(MEASURED_TIMING_FILE) as f:
        detailed = [json.loads(line) for line in f]
    assert len(detailed) == len(executed), (
        f"expected {len(executed)} timing records, got {len(detailed)} -- "
        "a generate() call didn't write a timing line, results are misaligned"
    )

    for r, d in zip(executed, detailed):
        fc = d["forward_calls"]
        r["multimodal_prep_time_generate_s"] = d["multimodal_prep_time_generate"]
        r["generate_time_s"] = d["generate_time"]
        r["prefill_lm_forward_time_s"] = fc[0]["lm_forward_time"] if fc else None
        r["prefill_prep_time_s"] = fc[0]["multimodal_prep_time"] if fc else None
        r["decode_lm_forward_time_s"] = sum(c["lm_forward_time"] for c in fc[1:]) if len(fc) > 1 else 0.0
        r["decode_steps"] = max(len(fc) - 1, 0)

    # min_new_tokens/max_new_tokens forces a FIXED length, but HF's exact new-
    # token count when generate() is called with inputs_embeds only (no
    # input_ids -- see llava_llama.py's generate()) is offset by a constant
    # from FIXED_OUTPUT_TOKENS (observed: +1). What matters for "output length
    # variance removed from the timing" is that every trial produced the SAME
    # length, not that it matches FIXED_OUTPUT_TOKENS literally -- so check
    # that directly instead of assuming HF's internal counting convention.
    observed_lengths = {r["new_tokens"] for r in executed}
    if len(observed_lengths) != 1:
        counts = {}
        for r in executed:
            counts[r["new_tokens"]] = counts.get(r["new_tokens"], 0) + 1
        majority_length = max(counts, key=counts.get)
        for r in executed:
            if r["new_tokens"] != majority_length:
                print(f"  WARNING: trial_position={r['trial_position']} cfg={r['visual_token_num']} "
                      f"produced {r['new_tokens']} new tokens, expected {majority_length} "
                      "(fixed-length generation did not hold for this trial)")
    else:
        majority_length = next(iter(observed_lengths))
        print(f"\nFixed-length check passed: all {len(executed)} trials produced exactly "
              f"{majority_length} new tokens (requested {FIXED_OUTPUT_TOKENS}).")

    with open(OUT_RAW, "w") as f:
        for r in executed:
            f.write(json.dumps(r) + "\n")
    print(f"\nRaw per-trial results written to {OUT_RAW}")

    # --- Per-config summary ---
    summaries = {}
    for cfg in CONFIGS:
        rows = [r for r in executed if r["visual_token_num"] == cfg]
        lat_ms = [r["wall_latency_s"] * 1000 for r in rows]
        prep_ms = [r["multimodal_prep_time_generate_s"] * 1000 for r in rows]
        prefill_ms = [r["prefill_lm_forward_time_s"] * 1000 for r in rows]
        decode_ms = [r["decode_lm_forward_time_s"] * 1000 for r in rows]
        summary = summarize_latencies(lat_ms)
        summary["multimodal_prep_mean_ms"] = statistics.mean(prep_ms)
        summary["prefill_lm_forward_mean_ms"] = statistics.mean(prefill_ms)
        summary["decode_lm_forward_mean_ms"] = statistics.mean(decode_ms)
        summary["outlier_trial_positions"] = [rows[i]["trial_position"] for i in summary["outlier_indices"]]
        del summary["outlier_indices"]
        summaries[cfg] = summary

    # --- Noise-floor / minimum-detectable-effect analysis ---
    # Answers: could a future before/after comparison (same config, ~n_measured
    # trials each side) actually distinguish a ~20-30ms real change from noise,
    # given the variance observed HERE? Computed two ways: RAW (statistics.stdev,
    # per rule 6's honest tail reporting) and ROBUST (MAD-based), since a single
    # system-hiccup outlier in n~30 trials can inflate raw stdev without
    # reflecting genuine run-to-run noise -- see outlier_trial_positions above.
    mde_analysis = {}
    for cfg in CONFIGS:
        stdev = summaries[cfg]["stdev_ms"]
        robust_stdev = summaries[cfg]["mad_stdev_ms"]
        n = summaries[cfg]["n"]
        mde = two_sample_mde_ms(stdev, n)
        mde_robust = two_sample_mde_ms(robust_stdev, n)
        mde_analysis[str(cfg)] = {
            "observed_stdev_ms": stdev,
            "observed_mad_stdev_ms": robust_stdev,
            "n_per_group_assumed": n,
            "mde_ms_raw": mde,
            "mde_ms_robust": mde_robust,
            "distinguishable_20ms_raw": mde <= 20.0,
            "distinguishable_30ms_raw": mde <= 30.0,
            "distinguishable_20ms_robust": mde_robust <= 20.0,
            "distinguishable_30ms_robust": mde_robust <= 30.0,
            "required_n_per_group_for_20ms_raw": required_n_for_mde(stdev, 20.0),
            "required_n_per_group_for_30ms_raw": required_n_for_mde(stdev, 30.0),
            "required_n_per_group_for_20ms_robust": required_n_for_mde(robust_stdev, 20.0),
            "required_n_per_group_for_30ms_robust": required_n_for_mde(robust_stdev, 30.0),
        }

    # Success criterion uses the RAW (ordinary) stdev at the 20ms target, for
    # both configs. Changed at WP1 acceptance (2026-09-07): WP1 itself passed
    # on the ROBUST estimate because raw stdev at n=30 put the 128 config at a
    # 20.4ms MDE, just over the line. Raising the protocol to 40 trials/config
    # buys enough resolution that raw variance clears 20ms on its own, so the
    # detectability claim no longer depends primarily on down-weighting
    # outliers. The robust estimate is still computed and reported alongside,
    # now as a cross-check rather than as the basis for pass/fail.
    overall_verdict = all(
        mde_analysis[str(cfg)]["mde_ms_raw"] <= SUCCESS_CRITERION_TARGET_MS
        for cfg in CONFIGS
    )
    # Divergence between the two estimates is the signal worth watching: if raw
    # fails while robust passes, outliers are driving it -- per the WP1
    # acceptance note, a climbing outlier rate is a real system-level noise
    # source to chase, not one to discount.
    robust_only_verdict = all(
        mde_analysis[str(cfg)]["mde_ms_robust"] <= SUCCESS_CRITERION_TARGET_MS
        for cfg in CONFIGS
    )

    # --- Gain calculation (VisPruner advantage, per §8) ---
    lat576 = summaries[576]["median_ms"]
    lat128 = summaries[128]["median_ms"]
    latency_gain_percent = (lat576 - lat128) / lat576 * 100 if lat576 else None

    metadata = {
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "model": "llava-v1.5-7b",
        "model_path": MODEL_PATH,
        "important_ratio": IMPORTANT_RATIO,
        "visual_token_settings": CONFIGS,
        "n_fixed_requests": len(items),
        "request_ids_used": [it["question_id"] for it in items],
        "fixed_output_tokens_requested": FIXED_OUTPUT_TOKENS,
        "fixed_output_tokens_observed": majority_length,
        "output_token_policy": (
            "FORCED fixed length: min_new_tokens == max_new_tokens. Controlled "
            "engineering benchmark only -- not representative of real-world "
            "serving behavior (which stops at natural EOS; see "
            "ab_noload_comparison.json for that measurement)."
        ),
        "do_sample": False,
        "use_cache": True,
        "n_warmup_per_config": n_warmup,
        "n_measured_trials_per_config": n_measured,
        "trial_order": "interleaved/randomized at individual-trial level",
        "trial_order_seed": TRIAL_ORDER_SEED,
        "git_commit": git_commit(),
        "dataset_path": DATASET_PATH,
        "quick_smoke_test": args.quick,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    report = {
        "metadata": metadata,
        "summaries": summaries,
        "gain": {
            "latency_median_gain_percent": latency_gain_percent,
            "formula": "(latency_576 - latency_128) / latency_576 * 100",
        },
        "noise_floor_analysis": {
            "method": (
                "Two-independent-sample minimum detectable effect (MDE) at "
                f"alpha={ALPHA}, power={POWER}, using each config's OWN observed "
                "wall-latency stdev at n measured trials -- models the case where "
                "a future WP compares the SAME config before vs. after a change, "
                "with a benchmark of this design and this much noise."
            ),
            "targets_ms": MDE_TARGETS_MS,
            "per_config": mde_analysis,
            "outliers_by_config": {str(cfg): summaries[cfg]["outlier_trial_positions"] for cfg in CONFIGS},
            "success_criterion_met": overall_verdict,
            "success_criterion_met_robust_crosscheck": robust_only_verdict,
            "success_criterion_definition": (
                f"Met if the RAW (ordinary statistics.stdev) MDE is <= "
                f"{SUCCESS_CRITERION_TARGET_MS:.0f}ms for BOTH configs. Changed at WP1 acceptance "
                "(2026-09-07): WP1's own run passed on the ROBUST (MAD-based) estimate because raw "
                "stdev at n=30 put the 128 config at a 20.4ms MDE, marginally over the line. The "
                "standing protocol is now 40 trials/config, which gives raw variance enough "
                "resolution to clear 20ms unaided, so pass/fail no longer depends primarily on "
                "down-weighting outliers. The robust estimate remains reported as a cross-check "
                "(success_criterion_met_robust_crosscheck); raw stdev and p95 stay fully visible."
            ),
            "verdict": (
                f"Benchmark distinguishes a ~{SUCCESS_CRITERION_TARGET_MS:.0f}ms change at both "
                "configs on RAW variance, without relying on the robust estimate."
                if overall_verdict else
                (
                    f"Benchmark does NOT clear the {SUCCESS_CRITERION_TARGET_MS:.0f}ms target on RAW "
                    "variance at both configs -- see required_n_per_group_for_20ms_raw per config for "
                    "how many trials would be needed."
                ) + (
                    " The ROBUST estimate DOES clear it, so the gap is driven by flagged outlier "
                    "trials (see outliers_by_config), not by broad run-to-run spread. Per the WP1 "
                    "acceptance note, treat a climbing outlier rate as a real system-level noise "
                    "source to investigate rather than one to discount."
                    if robust_only_verdict else ""
                )
            ),
        },
    }

    with open(OUT_SUMMARY, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Summary written to {OUT_SUMMARY}")

    # --- Human-readable table ---
    lines = [
        "WP1 -- Controlled Benchmark (fixed-length, interleaved, 576 vs 128 visual tokens)",
        f"n_measured_per_config={n_measured}  n_warmup_per_config={n_warmup}  "
        f"fixed_output_tokens={FIXED_OUTPUT_TOKENS}  gpu={metadata['gpu']}  "
        f"git={metadata['git_commit'][:10]}",
        "",
        f"{'metric':<28}{'576':>14}{'128':>14}",
    ]
    for key, label in [
        ("mean_ms", "latency_mean_ms"), ("median_ms", "latency_median_ms"),
        ("p95_ms", "latency_p95_ms"), ("stdev_ms", "latency_stdev_ms"),
        ("mad_stdev_ms", "latency_mad_stdev_ms (robust)"),
        ("cv_percent", "latency_cv_percent"), ("n", "n_trials"),
        ("multimodal_prep_mean_ms", "multimodal_prep_mean_ms"),
        ("prefill_lm_forward_mean_ms", "prefill_lm_forward_mean_ms"),
        ("decode_lm_forward_mean_ms", "decode_lm_forward_mean_ms"),
    ]:
        v576 = summaries[576].get(key)
        v128 = summaries[128].get(key)
        lines.append(f"{label:<28}{v576:>14.2f}{v128:>14.2f}")
    lines.append("")
    for cfg in CONFIGS:
        outliers = summaries[cfg]["outlier_trial_positions"]
        if outliers:
            lines.append(f"vtn={cfg}: outlier trial_position(s) flagged (modified z>3.5): {outliers} "
                          "-- see raw jsonl; excluded from the robust (MAD) spread estimate below, "
                          "not from the raw stdev/p95 reported above")
    lines.append("")
    lines.append(f"latency_median_gain_percent: {latency_gain_percent:.1f}%")
    lines.append("")
    lines.append("Noise-floor / MDE analysis (can we see a 20-30ms change?):")
    lines.append("  RAW  = statistics.stdev (rule 6, includes tail events as-is)")
    lines.append("  ROBUST = MAD-based stdev (down-weights rare single-trial system hiccups)")
    for cfg in CONFIGS:
        a = mde_analysis[str(cfg)]
        lines.append(
            f"  vtn={cfg} RAW:    stdev={a['observed_stdev_ms']:.1f}ms n={a['n_per_group_assumed']} "
            f"-> MDE={a['mde_ms_raw']:.1f}ms "
            f"(20ms {'OK' if a['distinguishable_20ms_raw'] else 'NO -- need n=' + str(a['required_n_per_group_for_20ms_raw'])}, "
            f"30ms {'OK' if a['distinguishable_30ms_raw'] else 'NO -- need n=' + str(a['required_n_per_group_for_30ms_raw'])})"
        )
        lines.append(
            f"  vtn={cfg} ROBUST: stdev={a['observed_mad_stdev_ms']:.1f}ms n={a['n_per_group_assumed']} "
            f"-> MDE={a['mde_ms_robust']:.1f}ms "
            f"(20ms {'OK' if a['distinguishable_20ms_robust'] else 'NO -- need n=' + str(a['required_n_per_group_for_20ms_robust'])}, "
            f"30ms {'OK' if a['distinguishable_30ms_robust'] else 'NO -- need n=' + str(a['required_n_per_group_for_30ms_robust'])})"
        )
    lines.append("")
    lines.append(
        f"SUCCESS CRITERION (raw variance, <= {SUCCESS_CRITERION_TARGET_MS:.0f}ms MDE at both "
        f"configs): {'PASS' if overall_verdict else 'FAIL'}"
        f"   [robust cross-check: {'pass' if robust_only_verdict else 'fail'}]"
    )
    lines.append(report["noise_floor_analysis"]["verdict"])

    table_text = "\n".join(lines)
    print("\n" + table_text)
    with open(OUT_TABLE, "w") as f:
        f.write(table_text + "\n")
    print(f"\nTable written to {OUT_TABLE}")


if __name__ == "__main__":
    main()
