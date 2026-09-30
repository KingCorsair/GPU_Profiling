"""
One-off diagnostic (not part of the tracked timing harness): confirm whether
`prepare_inputs_labels_for_multimodal` (the pruning fixup path AMAY_SPEED_PLAN.md
targets for a Triton kernel) is actually launch/bandwidth-bound before writing
one. Per AMAY_SPEED_PLAN.md step 1 / AMAY_TIMING_NOTES.md's `torch.profiler`
section. Throwaway — not Rithvik's harness, produces no reportable numbers.

WP2 addition: a second, tower-only profile of the exact call encode_images()
makes (vision_tower(images, output_attentions=True)), with its kernels grouped
by family (FP32 SIMT sgemm vs FP16 tensor-core GEMM, softmax, LayerNorm) and the
softmax/LayerNorm kernels' template dtypes printed. The whole-prep profile
can't answer that on its own: VisPruner's selection loop runs FP16 matmuls in
both states, so FP16 GEMMs appear in the FP32 trace too.

  python scripts/profile_multimodal_prep.py --run-id wp2_profile_fp32

Outputs go to results/timing/wp2_profile/<run-id>_*; refuses to overwrite.
"""
import argparse
import os
import re
import subprocess
import sys
import time
import numpy as np
import torch
from torch.autograd import DeviceType

sys.path.insert(0, "/workspace/GPU_Profiling/vis_pruner_copy")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.utils import disable_torch_init

from PIL import Image

MODEL_PATH = "/workspace/GPU_Profiling/vis_pruner_copy/checkpoints/llava-v1.5-7b"
IMAGE_PATH = "/workspace/GPU_Profiling/vis_pruner_copy/vispruner_eval_dataset/images/textvqa_fe32ee8991d2e59f.jpg"
VISUAL_TOKEN_NUM = 128   # aggressive pruning so the mask/gather path actually does work
IMPORTANT_RATIO = 0.5
BATCH_SIZE = 1           # this code path hardcodes batch=1 for single-image requests
                         # (image_features[index_masks].unsqueeze(0) collapses B into
                         # the token dim) -- there is no batched-request path to profile
                         # here yet; that's the batching work still on the roadmap.
N_WARMUP = 10
N_PROFILED = 10
N_TOWER_TIMED = 50       # CUDA-event timing of the tower alone, outside the profiler
TORCH_CPU_THREADS = 4    # same pin as bench_dev.py; unpinned, torch's 48 threads trip
                         # the container's CPU quota (see bench_dev.py)

parser = argparse.ArgumentParser()
parser.add_argument("--run-id", required=True)
args = parser.parse_args()

OUT_DIR = "/workspace/GPU_Profiling/results/timing/wp2_profile"
os.makedirs(OUT_DIR, exist_ok=True)
OUT = {
    "prep_trace": f"{OUT_DIR}/{args.run_id}_prep_trace.json",
    "prep_summary": f"{OUT_DIR}/{args.run_id}_prep_summary.txt",
    "tower_trace": f"{OUT_DIR}/{args.run_id}_tower_trace.json",
    "tower_kernels": f"{OUT_DIR}/{args.run_id}_tower_kernels.txt",
}
existing = [p for p in OUT.values() if os.path.exists(p)]
if existing:
    sys.exit("Refusing to overwrite:\n  " + "\n  ".join(existing) + "\nPick a new --run-id.")

gpu_procs = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
    capture_output=True, text=True,
).stdout.strip()
if gpu_procs:
    sys.exit(f"Other processes are on the GPU (rule 9):\n  {gpu_procs}")

torch.set_num_threads(TORCH_CPU_THREADS)


def _git(*a):
    return subprocess.run(["git", "-C", "/workspace/GPU_Profiling", *a],
                          capture_output=True, text=True).stdout.strip()


GIT_COMMIT = _git("rev-parse", "HEAD")[:10] or "unknown"
GIT_DIRTY = [l[3:] for l in _git("status", "--porcelain", "--untracked-files=no").splitlines()
             if not l.endswith(".md")]
GPU_NAME = torch.cuda.get_device_name(0)
print(f"git commit: {GIT_COMMIT}{' (DIRTY: ' + ', '.join(GIT_DIRTY) + ')' if GIT_DIRTY else ''}"
      f"   GPU: {GPU_NAME}   cpu_threads: {torch.get_num_threads()}")

disable_torch_init()
tokenizer, model, image_processor, _ = load_pretrained_model(
    MODEL_PATH, None, "llava-v1.5-7b",
    visual_token_num=VISUAL_TOKEN_NUM,
    important_ratio=IMPORTANT_RATIO,
)
model.eval()
vision_tower = model.get_model().get_vision_tower()
VISION_DTYPE = str(vision_tower.dtype)
print(f"vision_tower dtype={VISION_DTYPE}")

image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = process_images([image], image_processor, model.config)[0]
images = image_tensor.unsqueeze(0).half().cuda().repeat(BATCH_SIZE, 1, 1, 1)
image_sizes = [image.size] * BATCH_SIZE

qs = DEFAULT_IMAGE_TOKEN + "\nWhat does the text in this image say?\n" \
     "Answer with the option's letter from the given choices directly."
conv = conv_templates["vicuna_v1"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
input_ids = input_ids.unsqueeze(0).repeat(BATCH_SIZE, 1).cuda()
attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

print(f"batch={BATCH_SIZE} prompt_len={input_ids.shape[1]} visual_token_num={VISUAL_TOKEN_NUM}")


def run_prep():
    with torch.inference_mode():
        return model.prepare_inputs_labels_for_multimodal(
            input_ids, None, attention_mask, None, None,
            images, image_sizes=image_sizes,
        )


def run_generate():
    with torch.inference_mode():
        return model.generate(
            input_ids, images=images, image_sizes=image_sizes,
            do_sample=False, max_new_tokens=1, use_cache=True,
        )


# --- Warm-up (rule 3: CUDA context init, memory pools, kernel autotuning) ---
for _ in range(N_WARMUP):
    run_prep()
torch.cuda.synchronize()

# --- Isolated wall-time read: prep vs. one full prefill+1-token generate ---
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_PROFILED):
    run_prep()
torch.cuda.synchronize()
prep_only_s = (time.perf_counter() - t0) / N_PROFILED

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_PROFILED):
    run_generate()
torch.cuda.synchronize()
generate_1tok_s = (time.perf_counter() - t0) / N_PROFILED

print(f"\nmean prepare_inputs_labels_for_multimodal: {prep_only_s*1000:.2f} ms")
print(f"mean generate(max_new_tokens=1) [prep + prefill fwd]: {generate_1tok_s*1000:.2f} ms")
print(f"prep as % of prefill request: {100*prep_only_s/generate_1tok_s:.1f}%")

# --- torch.profiler trace over the prep region specifically ---
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for i in range(N_PROFILED):
        with torch.profiler.record_function(f"prep_iter_{i}"):
            run_prep()
    torch.cuda.synchronize()

print("\n" + "=" * 100)
print("TOP OPS BY CUDA TIME (prepare_inputs_labels_for_multimodal region only)")
print("=" * 100)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

print("\n" + "=" * 100)
print("TOP OPS BY CALL COUNT (reveals per-item / per-iteration Python loops)")
print("=" * 100)
print(prof.key_averages().table(sort_by="count", row_limit=25))

# Aggregate: how much of the region's GPU time sits in ops with tiny per-call
# duration but high call counts (the launch-overhead signature) vs. a few
# big matmul/conv/attention kernels (the compute-bound signature).
#
# `key_averages()` returns TWO rows for the same GPU work: an aten-level
# operator row (device_type CPU, e.g. `aten::addmm`, whose self_cuda_time_total
# is attributed up from the kernels it launched) and a kernel-level row
# (device_type CUDA, e.g. `ampere_sgemm_128x64_tn`). Summing over both
# double-counts every GPU microsecond -- the earlier version of this script did,
# and printed 739.6 ms where the profiler's own footer said 354.9 ms. Sum
# exactly one level: the kernel rows. That is real GPU busy time, and it is what
# "Self CUDA time total" in the table above reports.
events = prof.key_averages()
kernel_events = [e for e in events if e.device_type == DeviceType.CUDA]
aten_events = [e for e in events if e.device_type == DeviceType.CPU]

gpu_busy_us = sum(e.self_cuda_time_total for e in kernel_events)
aten_attributed_us = sum(e.self_cuda_time_total for e in aten_events)
total_cpu_us = sum(e.self_cpu_time_total for e in events)

# Classify the KERNEL rows, by kernel name. The previous classifier matched a set
# of `aten::` names, which never match a kernel row, so every ampere_sgemm_* /
# sm80_xmma_* row fell through into "everything else" and inflated it to 63.2%.
BIG_KERNEL_PATTERNS = ("gemm", "conv", "cutlass", "xmma", "attention", "flash")


def is_big_kernel(name):
    n = name.lower()
    return any(p in n for p in BIG_KERNEL_PATTERNS)


big_kernel_cuda_us = sum(e.self_cuda_time_total for e in kernel_events if is_big_kernel(e.key))
small_op_cuda_us = gpu_busy_us - big_kernel_cuda_us

# The classifier is a name-substring heuristic, so show what it left unmatched:
# if a large GEMM family shows up here, the pattern list is wrong, not the GPU.
unmatched = sorted(
    (e for e in kernel_events if not is_big_kernel(e.key)),
    key=lambda e: -e.self_cuda_time_total,
)[:5]


def _pct(x):
    return 100 * x / max(gpu_busy_us, 1)


summary_lines = [
    f"Total self CPU time in region:  {total_cpu_us/1000:.2f} ms  (over {N_PROFILED} iters)",
    f"GPU busy in region (kernel rows only): {gpu_busy_us/1000:.2f} ms  (over {N_PROFILED} iters)",
    f"  of which big matmul/attn/conv kernels: {big_kernel_cuda_us/1000:.2f} ms "
    f"({_pct(big_kernel_cuda_us):.1f}%)",
    f"  of which everything else (gather/index/cat/argsort/pad/etc.): "
    f"{small_op_cuda_us/1000:.2f} ms ({_pct(small_op_cuda_us):.1f}%)",
    f"CPU time / GPU busy ratio in region: {total_cpu_us/max(gpu_busy_us,1):.2f}x "
    f"(>>1 means launch/dispatch overhead dominates actual GPU work)",
    "",
    f"[cross-check] same GPU time attributed at the aten level: "
    f"{aten_attributed_us/1000:.2f} ms -- should be close to the kernel-row figure "
    f"above. These are two views of the same work; adding them is the double-count.",
    "[cross-check] largest kernels NOT classified as big matmul/attn/conv:",
]
summary_lines += [
    f"    {e.self_cuda_time_total/1000:8.2f} ms  {e.key[:90]}" for e in unmatched
]

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print("\n".join(summary_lines))
print(f"\nRegion as % of a 1-token prefill request: {100*prep_only_s/generate_1tok_s:.1f}%")

prof.export_chrome_trace(OUT["prep_trace"])
print(f"\nChrome trace written to {OUT['prep_trace']}")

# Short, human-readable summary (not the full trace) -- meant to be opened directly,
# not loaded into a trace viewer. Same `summary_lines` the console printed, so the
# two can't drift apart.
with open(OUT["prep_summary"], "w") as f:
    f.write(f"visual_token_num={VISUAL_TOKEN_NUM} batch={BATCH_SIZE} prompt_len={input_ids.shape[1]}\n")
    f.write(f"git commit: {GIT_COMMIT}{' (DIRTY)' if GIT_DIRTY else ''}   GPU: {GPU_NAME}   "
            f"vision_tower dtype={VISION_DTYPE}   cpu_threads={torch.get_num_threads()}\n\n")
    f.write(f"mean prepare_inputs_labels_for_multimodal: {prep_only_s*1000:.2f} ms\n")
    f.write(f"mean generate(max_new_tokens=1) [prep + prefill fwd]: {generate_1tok_s*1000:.2f} ms\n")
    f.write(f"prep as % of prefill request: {100*prep_only_s/generate_1tok_s:.1f}%\n\n")
    f.write("TOP 15 OPS BY CUDA TIME\n")
    f.write(str(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)))
    f.write("\n\n" + "\n".join(summary_lines) + "\n")
print(f"Short summary written to {OUT['prep_summary']}")


# =============================================================================
# WP2: the vision tower alone -- the exact call encode_images() makes
# (llava_arch.py:142), so its kernels can't be confused with the selection
# loop's FP16 matmuls or mm_projector's.
# =============================================================================
def run_tower():
    with torch.inference_mode():
        return vision_tower(images, output_attentions=True)


for _ in range(N_WARMUP):
    run_tower()
torch.cuda.synchronize()

# Per-call CUDA events, recorded back-to-back with no sync inside the loop
# (rule 4); read once after the closing sync. Comparable to bench_dev's
# vision_tower_ms, except calls here are back-to-back rather than one per request.
pairs = []
for _ in range(N_TOWER_TIMED):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    run_tower()
    e.record()
    pairs.append((s, e))
torch.cuda.synchronize()
tower_ms = np.array([s.elapsed_time(e) for s, e in pairs])

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as tprof:
    for i in range(N_PROFILED):
        with torch.profiler.record_function(f"tower_iter_{i}"):
            run_tower()
    torch.cuda.synchronize()
tprof.export_chrome_trace(OUT["tower_trace"])

# Kernel families, by kernel name. Order matters: the first match wins.
#   sgemm without a tensor-core tag      -> FP32 on CUDA cores (the WP2 "before")
#   h16816 / h1688 / f16f16_f16f16_f16   -> FP16 tensor cores, FP16 ACCUMULATE (unsafe)
#   s16816 / s1688 / f16f16_f16f32 / fp16 tensorop / xmma f16 -> FP16 tensor cores, FP32 accumulate
FAMILIES = [
    ("FP16 tensor-core GEMM, FP16 accumulate (FLAG)",
     lambda n: re.search(r"h16816|h1688|f16f16_f16f16_f16", n)),
    ("TF32 tensor-core GEMM", lambda n: "tf32" in n),
    ("FP16 tensor-core GEMM, FP32 accumulate",
     lambda n: re.search(r"s16816|s1688|f16f16_f16f32|fp16.*gemm|gemm.*f16|tensorop.*f16|xmma.*f16", n)),
    ("FP32 SIMT GEMM (ampere_sgemm_*)", lambda n: "sgemm" in n),
    ("other GEMM", lambda n: "gemm" in n),
    ("softmax", lambda n: "softmax" in n),
    ("LayerNorm", lambda n: "layer_norm" in n),
    ("split-K reduce", lambda n: "splitk" in n or "reduce_kernel" in n and "gemm" in n),
    ("everything else", lambda n: True),
]
tkernels = [e for e in tprof.key_averages() if e.device_type == DeviceType.CUDA]
tower_gpu_us = sum(e.self_cuda_time_total for e in tkernels)
fam_us, fam_calls, fam_names = {}, {}, {}
for e in tkernels:
    n = e.key.lower()
    fam = next(name for name, match in FAMILIES if match(n))
    fam_us[fam] = fam_us.get(fam, 0) + e.self_cuda_time_total
    fam_calls[fam] = fam_calls.get(fam, 0) + e.count
    fam_names.setdefault(fam, []).append(e)


def _t(e):
    return -e.self_cuda_time_total


lines = [
    f"vision tower only -- vision_tower(images, output_attentions=True), batch=1",
    f"git commit: {GIT_COMMIT}{' (DIRTY)' if GIT_DIRTY else ''}   GPU: {GPU_NAME}   "
    f"vision_tower dtype={VISION_DTYPE}   cpu_threads={torch.get_num_threads()}",
    "",
    f"CUDA-event time per call, {N_TOWER_TIMED} back-to-back calls, no profiler:",
    f"  p50 {np.percentile(tower_ms, 50):.2f} ms   p95 {np.percentile(tower_ms, 95):.2f} ms   "
    f"min {tower_ms.min():.2f}   max {tower_ms.max():.2f}",
    f"GPU busy per call under the profiler (kernel rows only): "
    f"{tower_gpu_us / 1000 / N_PROFILED:.2f} ms",
    "  (event time well above GPU busy => the GPU waited on the CPU to issue kernels)",
    "",
    f"KERNEL FAMILIES ({N_PROFILED} calls)",
    f"  {'family':48s}{'ms/call':>10s}{'% GPU':>8s}{'launches/call':>15s}",
]
for name, _ in FAMILIES:
    if name in fam_us:
        lines.append(f"  {name:48s}{fam_us[name] / 1000 / N_PROFILED:10.2f}"
                     f"{100 * fam_us[name] / max(tower_gpu_us, 1):8.1f}"
                     f"{fam_calls[name] / N_PROFILED:15.1f}")
lines.append("")
for fam in ("softmax", "LayerNorm"):
    lines.append(f"{fam.upper()} KERNELS (template args = input, output[, accumulate] dtypes):")
    for e in sorted(fam_names.get(fam, []), key=_t):
        lines.append(f"  {e.self_cuda_time_total / 1000 / N_PROFILED:7.2f} ms/call  {e.key[:160]}")
    lines.append("")
lines.append("ALL GEMM KERNELS:")
for e in sorted((e for e in tkernels if "gemm" in e.key.lower()), key=_t):
    lines.append(f"  {e.self_cuda_time_total / 1000 / N_PROFILED:7.2f} ms/call  x{e.count // N_PROFILED:<4d} {e.key[:140]}")
lines.append("")
lines.append("TOP 15 KERNELS BY GPU TIME:")
for e in sorted(tkernels, key=_t)[:15]:
    lines.append(f"  {e.self_cuda_time_total / 1000 / N_PROFILED:7.2f} ms/call  x{e.count // N_PROFILED:<4d} {e.key[:140]}")

text = "\n".join(lines) + "\n"
with open(OUT["tower_kernels"], "w") as f:
    f.write(text)
print("\n" + "=" * 100 + "\nVISION TOWER ONLY\n" + "=" * 100)
print(text)
print("Wrote:\n  " + "\n  ".join(OUT.values()))
