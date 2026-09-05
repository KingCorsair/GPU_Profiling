"""
One-off diagnostic (not part of the tracked timing harness): confirm whether
`prepare_inputs_labels_for_multimodal` (the pruning fixup path AMAY_SPEED_PLAN.md
targets for a Triton kernel) is actually launch/bandwidth-bound before writing
one. Per AMAY_SPEED_PLAN.md step 1 / AMAY_TIMING_NOTES.md's `torch.profiler`
section. Throwaway — not Rithvik's harness, produces no reportable numbers.
"""
import subprocess
import sys
import time
import torch
from torch.autograd import DeviceType

sys.path.insert(0, "vis_pruner_copy")

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

GIT_COMMIT = subprocess.run(
    ["git", "-C", "/workspace/GPU_Profiling", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"
GPU_NAME = torch.cuda.get_device_name(0)
print(f"git commit: {GIT_COMMIT}   GPU: {GPU_NAME}")

disable_torch_init()
tokenizer, model, image_processor, _ = load_pretrained_model(
    MODEL_PATH, None, "llava-v1.5-7b",
    visual_token_num=VISUAL_TOKEN_NUM,
    important_ratio=IMPORTANT_RATIO,
)
model.eval()

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

prof.export_chrome_trace("/workspace/GPU_Profiling/results/timing/prep_trace.json")
print("\nChrome trace written to results/timing/prep_trace.json")

# Short, human-readable summary (not the full trace) -- meant to be opened directly,
# not loaded into a trace viewer. Same `summary_lines` the console printed, so the
# two can't drift apart.
with open("/workspace/GPU_Profiling/results/timing/prep_summary.txt", "w") as f:
    f.write(f"visual_token_num={VISUAL_TOKEN_NUM} batch={BATCH_SIZE} prompt_len={input_ids.shape[1]}\n")
    f.write(f"git commit: {GIT_COMMIT}   GPU: {GPU_NAME}\n\n")
    f.write(f"mean prepare_inputs_labels_for_multimodal: {prep_only_s*1000:.2f} ms\n")
    f.write(f"mean generate(max_new_tokens=1) [prep + prefill fwd]: {generate_1tok_s*1000:.2f} ms\n")
    f.write(f"prep as % of prefill request: {100*prep_only_s/generate_1tok_s:.1f}%\n\n")
    f.write("TOP 15 OPS BY CUDA TIME\n")
    f.write(str(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)))
    f.write("\n\n" + "\n".join(summary_lines) + "\n")
print("Short summary written to results/timing/prep_summary.txt")
