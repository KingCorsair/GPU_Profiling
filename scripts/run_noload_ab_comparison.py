"""
One-off diagnostic (not part of the tracked timing harness): Part 1 of the
VisPruner A/B experiment -- isolated, no-load single-request comparison of
visual_token_num=576 (effectively unpruned baseline) vs. visual_token_num=128
(VisPruner), same checkpoint, same requests, same warm-up procedure.

Loads the model ONCE and flips model.visual_token_num between passes rather
than reloading the checkpoint twice -- encode_images() reads
self.get_visual_token_num()/self.get_important_ratio() fresh on every call
(llava_arch.py:148-149), so this is equivalent to a fresh load for anything
downstream of that read, without paying two ~40s weight loads.

Reuses the model's own existing CUDA-event-based timing instrumentation
(llava_llama.py's generate(), which already records multimodal_prep_time,
per-forward-call prep/lm_forward times, and generate_time to
LLAVA_TIMING_FILE on every call) instead of re-implementing prefill/decode
timing -- that instrumentation is already CUDA-event based (rule 1), so
there's nothing to add, just point it at a dedicated file per config and
read it back.

Throwaway, per AMAY_SPEED_PLAN.md / AMAY_TIMING_NOTES.md's "Amay measures to
find bottlenecks" split. No optimizations implemented here.
"""
import json
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, "vis_pruner_copy")

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from PIL import Image
import os

MODEL_PATH = "/workspace/GPU_Profiling/vis_pruner_copy/checkpoints/llava-v1.5-7b"
DATASET_PATH = "/workspace/GPU_Profiling/vis_pruner_copy/vispruner_eval_dataset/dev.json"
IMAGES_ROOT = "/workspace/GPU_Profiling/vis_pruner_copy/vispruner_eval_dataset"

IMPORTANT_RATIO = 0.5   # held constant across both configs, per experiment design
CONFIGS = [576, 128]    # A: effectively-unpruned baseline, B: VisPruner
N_REQUESTS = 10         # "multiple representative requests" -- same N items for both configs
N_WARMUP = 10           # rule 3, re-run per config since kernel/shape profile changes with visual_token_num
MAX_NEW_TOKENS = 64     # matches the live server's policy (csnbs/server.py) -- natural EOS
                        # stopping, not forced length, so this is directly comparable to
                        # Part 2's load-sweep numbers

RESULTS_DIR = "/workspace/GPU_Profiling/results/timing"
OUT_JSON = f"{RESULTS_DIR}/ab_noload_comparison.json"
OUT_TABLE = f"{RESULTS_DIR}/ab_noload_comparison_table.txt"


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd="/workspace/GPU_Profiling"
        ).decode().strip()
    except Exception:
        return "unknown"


disable_torch_init()
tokenizer, model, image_processor, _ = load_pretrained_model(
    MODEL_PATH, None, "llava-v1.5-7b",
    visual_token_num=576,  # starting value; flipped per config below
    important_ratio=IMPORTANT_RATIO,
)
model.eval()

with open(DATASET_PATH) as f:
    dataset = json.load(f)
items = dataset[:N_REQUESTS]
print(f"Using {len(items)} fixed representative requests (same set for both configs):")
for it in items:
    print(f"  {it['question_id']}: {it['question'][:60]!r} [{it['category']}]")


def build_request(item):
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
    return input_ids, attention_mask, images, image_sizes, prompt


def run_one(item):
    input_ids, attention_mask, images, image_sizes, prompt = build_request(item)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids, visual_token_num_used = model.generate(
            input_ids, attention_mask=attention_mask, images=images, image_sizes=image_sizes,
            do_sample=False, max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
        )
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return {
        "wall_latency_s": wall_s,
        "output_token_count": output_ids.shape[1],
        "prompt_len": input_ids.shape[1],
        "generated_text": text,
        "visual_token_num_used": visual_token_num_used,
    }


all_results = {}

for cfg in CONFIGS:
    print(f"\n=== Config: visual_token_num={cfg} ===")
    model.visual_token_num = cfg

    timing_path = f"{RESULTS_DIR}/ab_noload_timing_{cfg}.jsonl"
    if os.path.exists(timing_path):
        os.remove(timing_path)
    os.environ["LLAVA_TIMING_FILE"] = timing_path

    # Warm-up: same procedure both configs, discarded (rule 3)
    for i in range(N_WARMUP):
        run_one(items[i % len(items)])
    torch.cuda.synchronize()
    if os.path.exists(timing_path):
        os.remove(timing_path)  # drop warm-up timing lines too

    per_request = []
    gpu_util_samples = []
    for item in items:
        try:
            gpu_util = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                timeout=2,
            ).decode().strip()
            gpu_util_samples.append(float(gpu_util))
        except Exception:
            pass
        result = run_one(item)
        per_request.append(result)
        print(f"  {item['question_id']}: {result['wall_latency_s']*1000:.1f} ms, "
              f"{result['output_token_count']} tokens, {result['generated_text'][:50]!r}")

    # Read back the model's own CUDA-event timing for these N calls
    detailed = []
    with open(timing_path) as f:
        for line in f:
            detailed.append(json.loads(line))
    assert len(detailed) == len(items), f"expected {len(items)} timing records, got {len(detailed)}"

    for r, d in zip(per_request, detailed):
        fc = d["forward_calls"]
        r["multimodal_prep_time_generate_s"] = d["multimodal_prep_time_generate"]
        r["generate_time_s"] = d["generate_time"]
        r["prefill_lm_forward_time_s"] = fc[0]["lm_forward_time"] if fc else None
        r["prefill_prep_time_s"] = fc[0]["multimodal_prep_time"] if fc else None
        r["decode_lm_forward_time_s"] = sum(c["lm_forward_time"] for c in fc[1:]) if len(fc) > 1 else 0.0
        r["decode_steps"] = max(len(fc) - 1, 0)

    latencies = [r["wall_latency_s"] for r in per_request]
    token_counts = [r["output_token_count"] for r in per_request]
    prefill_times = [r["prefill_lm_forward_time_s"] for r in per_request]
    prep_times = [r["multimodal_prep_time_generate_s"] for r in per_request]
    decode_times = [r["decode_lm_forward_time_s"] for r in per_request]

    summary = {
        "visual_token_num": cfg,
        "n_requests": len(items),
        "latency_mean_ms": statistics.mean(latencies) * 1000,
        "latency_median_ms": statistics.median(latencies) * 1000,
        "latency_stdev_ms": statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0.0,
        "latency_min_ms": min(latencies) * 1000,
        "latency_max_ms": max(latencies) * 1000,
        "multimodal_prep_mean_ms": statistics.mean(prep_times) * 1000,
        "prefill_lm_forward_mean_ms": statistics.mean(prefill_times) * 1000,
        "decode_lm_forward_mean_ms": statistics.mean(decode_times) * 1000,
        "output_token_count_mean": statistics.mean(token_counts),
        "output_token_count_min": min(token_counts),
        "output_token_count_max": max(token_counts),
        "gpu_util_pct_mean_sampled": statistics.mean(gpu_util_samples) if gpu_util_samples else None,
        "gpu_util_pct_max_sampled": max(gpu_util_samples) if gpu_util_samples else None,
    }
    print(f"  -> median latency: {summary['latency_median_ms']:.1f} ms  "
          f"(mean {summary['latency_mean_ms']:.1f} ms, stdev {summary['latency_stdev_ms']:.1f} ms)")

    all_results[cfg] = {"summary": summary, "per_request": per_request}

# --- Gain calculation ---
lat576 = all_results[576]["summary"]["latency_median_ms"]
lat128 = all_results[128]["summary"]["latency_median_ms"]
latency_gain_percent = (lat576 - lat128) / lat576 * 100

prep576 = all_results[576]["summary"]["multimodal_prep_mean_ms"]
prep128 = all_results[128]["summary"]["multimodal_prep_mean_ms"]
prefill576 = all_results[576]["summary"]["prefill_lm_forward_mean_ms"]
prefill128 = all_results[128]["summary"]["prefill_lm_forward_mean_ms"]
decode576 = all_results[576]["summary"]["decode_lm_forward_mean_ms"]
decode128 = all_results[128]["summary"]["decode_lm_forward_mean_ms"]

metadata = {
    "gpu": torch.cuda.get_device_name(0),
    "torch_version": torch.__version__,
    "transformers_version": __import__("transformers").__version__,
    "model": "llava-v1.5-7b",
    "model_path": MODEL_PATH,
    "important_ratio": IMPORTANT_RATIO,
    "n_requests": N_REQUESTS,
    "n_warmup": N_WARMUP,
    "max_new_tokens_cap": MAX_NEW_TOKENS,
    "do_sample": False,
    "output_token_policy": "natural EOS stopping, capped at max_new_tokens (matches csnbs/server.py)",
    "git_commit": git_commit(),
    "dataset_path": DATASET_PATH,
    "request_ids_used": [it["question_id"] for it in items],
}

report = {
    "metadata": metadata,
    "configs": all_results,
    "gain": {
        "latency_median_gain_percent": latency_gain_percent,
        "formula": "(latency_576 - latency_128) / latency_576 * 100",
        "multimodal_prep_mean_ms": {"576": prep576, "128": prep128,
                                     "gain_percent": (prep576 - prep128) / prep576 * 100 if prep576 else None},
        "prefill_lm_forward_mean_ms": {"576": prefill576, "128": prefill128,
                                        "gain_percent": (prefill576 - prefill128) / prefill576 * 100 if prefill576 else None},
        "decode_lm_forward_mean_ms": {"576": decode576, "128": decode128,
                                       "gain_percent": (decode576 - decode128) / decode576 * 100 if decode576 else None},
    },
}

with open(OUT_JSON, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nFull results written to {OUT_JSON}")

lines = [
    "PART 1 -- No-load / single-request A/B comparison (576 vs 128 visual tokens)",
    f"n_requests={N_REQUESTS}  important_ratio={IMPORTANT_RATIO}  max_new_tokens_cap={MAX_NEW_TOKENS}  "
    f"gpu={metadata['gpu']}  git={metadata['git_commit'][:10]}",
    "",
    f"{'metric':<32}{'576':>12}{'128':>12}{'gain %':>10}",
]


def row(label, v576, v128, unit=""):
    gain = (v576 - v128) / v576 * 100 if v576 else float("nan")
    lines.append(f"{label:<32}{v576:>12.1f}{v128:>12.1f}{gain:>9.1f}%")


s576, s128 = all_results[576]["summary"], all_results[128]["summary"]
row("latency_median_ms", s576["latency_median_ms"], s128["latency_median_ms"])
row("latency_mean_ms", s576["latency_mean_ms"], s128["latency_mean_ms"])
row("latency_stdev_ms", s576["latency_stdev_ms"], s128["latency_stdev_ms"])
row("multimodal_prep_mean_ms", s576["multimodal_prep_mean_ms"], s128["multimodal_prep_mean_ms"])
row("prefill_lm_forward_mean_ms", s576["prefill_lm_forward_mean_ms"], s128["prefill_lm_forward_mean_ms"])
row("decode_lm_forward_mean_ms", s576["decode_lm_forward_mean_ms"], s128["decode_lm_forward_mean_ms"])
row("output_token_count_mean", s576["output_token_count_mean"], s128["output_token_count_mean"])
if s576["gpu_util_pct_mean_sampled"] is not None:
    row("gpu_util_pct_mean_sampled", s576["gpu_util_pct_mean_sampled"], s128["gpu_util_pct_mean_sampled"])

table_text = "\n".join(lines)
print("\n" + table_text)
with open(OUT_TABLE, "w") as f:
    f.write(table_text + "\n")
print(f"\nTable written to {OUT_TABLE}")
