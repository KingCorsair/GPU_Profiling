"""
One-off diagnostic (not part of the tracked timing harness): capture a short
torch.profiler CPU+CUDA trace of several near-simultaneous requests, to see
directly at the kernel level whether "concurrent" requests actually overlap
on the GPU or serialize -- complementing run_concurrent_load_test.py's
throughput/latency numbers (which come from Rithvik's real HTTP load
generator and server, unmodified) with a kernel-level view of the same
question.

torch.profiler has to run in the same process as the CUDA calls it's
tracing. csnbs/server.py is Rithvik's file (the load generator and server
are his to write, per CLAUDE.md), so this does NOT attach to the live
server -- it's a standalone reproduction of the server's exact request-
handling pattern (csnbs/server.py:_infer_model, same settings: 576 visual
tokens, max_new_tokens=64) run directly against the loaded model, launched
sequentially, back-to-back. This deliberately mirrors what the real
server's /infer handler does per request (a single synchronous
model.generate() call, no batching, one request fully finished before the
next begins) -- the only difference is the transport (no FastAPI/HTTP
layer), which isn't GPU-kernel-relevant.

Using real OS threads (not asyncio) so the experiment could show
kernel-level overlap if the CUDA driver/PyTorch default-stream behavior
permitted any -- asyncio coroutines with no internal `await` around the
blocking generate() call would trivially serialize regardless of what the
GPU could do, which would tell us nothing new.

FINDING (from the first run of this script, kept here rather than silently
dropped): launching N_CONCURRENT=3 fully-simultaneous threads against the
same shared model instance crashed with a CUDA device-side assert --
"../aten/src/ATen/native/cuda/Indexing.cu:1237: indexSelectSmallIndex:
Assertion `srcIndex < srcSelectDimSize` failed" -- an out-of-bounds
index_select inside the vision-tower/token-pruning gather path
(llava_arch.py), consistent with concurrent threads racing on
state tied to the single shared model instance. This is itself real
evidence, not a bug to route around: the current single-model-instance
serving path is not safe for genuine concurrent execution, which is
presumably why csnbs/server.py's single-event-loop design processes
requests strictly one at a time rather than an oversight. See
results/timing/concurrent_profile_run_log.txt (first run) for the full
assertion output.

Given that, this script instead profiles N_CONCURRENT requests run
back-to-back, sequentially, in one process -- which is what the real server
actually does per-request today -- to see inter-request gaps and repeated
per-request patterns safely, without re-triggering the race.

Throwaway diagnostic, no optimizations implemented here.
"""
import functools
import json
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, "vis_pruner_copy")

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from PIL import Image
from transformers.cache_utils import DynamicCache

MODEL_PATH = "/workspace/GPU_Profiling/vis_pruner_copy/checkpoints/llava-v1.5-7b"
DATASET_PATH = "/workspace/GPU_Profiling/vis_pruner_copy/vispruner_eval_dataset/dev.json"
IMAGES_ROOT = "/workspace/GPU_Profiling/vis_pruner_copy/vispruner_eval_dataset"

# Match the live server's actual (hardcoded) settings exactly, so this is
# comparable to run_concurrent_load_test.py's numbers.
VISUAL_TOKEN_NUM = 576
IMPORTANT_RATIO = 0.5
MAX_NEW_TOKENS = 64
MIN_NEW_TOKENS = 16   # force enough decode steps to be visible without forcing
                      # the full 64 on every concurrent "request" (keeps the
                      # trace inspectable, per the ask to keep the window short)

N_CONCURRENT = 3       # a small, representative burst -- not the full 20-30s
                       # load window, so the trace stays inspectable. Run
                       # back-to-back/sequential, not concurrent -- see the
                       # module docstring for why (a real thread-concurrent
                       # attempt crashed the CUDA context).
N_WARMUP = 6

TRACE_PATH = "/workspace/GPU_Profiling/results/timing/concurrent_trace.json"
SUMMARY_CUDA_PATH = "/workspace/GPU_Profiling/results/timing/concurrent_summary_by_cuda_time.txt"
SUMMARY_COUNT_PATH = "/workspace/GPU_Profiling/results/timing/concurrent_summary_by_count.txt"
SETTINGS_PATH = "/workspace/GPU_Profiling/results/timing/concurrent_trace_settings.json"
TIMELINE_PATH = "/workspace/GPU_Profiling/results/timing/concurrent_request_timeline.json"

disable_torch_init()
tokenizer, model, image_processor, _ = load_pretrained_model(
    MODEL_PATH, None, "llava-v1.5-7b",
    visual_token_num=VISUAL_TOKEN_NUM,
    important_ratio=IMPORTANT_RATIO,
)
model.eval()

with open(DATASET_PATH) as f:
    dataset = json.load(f)


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
    return input_ids, attention_mask, images, image_sizes


def run_one_request(item):
    input_ids, attention_mask, images, image_sizes = build_request(item)
    with torch.inference_mode():
        return model.generate(
            input_ids, attention_mask=attention_mask, images=images, image_sizes=image_sizes,
            do_sample=False, max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MIN_NEW_TOKENS,
            use_cache=True,
        )


# --- Warm-up: sequential, unprofiled ---
for i in range(N_WARMUP):
    run_one_request(dataset[i % len(dataset)])
torch.cuda.synchronize()

# --- Instrumentation: same monkey-patch approach as profile_full_request.py,
# extended with a thread-local "which concurrent request slot" label so the
# trace shows whether slots interleave or run strictly back-to-back. ---
orig_forward = model.forward
vision_tower = model.get_vision_tower()
mm_projector = model.get_model().mm_projector
orig_vt_forward = vision_tower.forward
orig_proj_forward = mm_projector.forward
orig_cache_update = DynamicCache.update

_current_slot = {"id": "unknown"}
_fwd_call_idx = {}  # slot_id -> count


@functools.wraps(orig_forward)
def wrapped_forward(*args, **kwargs):
    slot = _current_slot["id"]
    idx = _fwd_call_idx.get(slot, 0)
    _fwd_call_idx[slot] = idx + 1
    label = f"slot{slot}_prefill" if idx == 0 else f"slot{slot}_decode_step_{idx}"
    with record_function(label):
        return orig_forward(*args, **kwargs)


@functools.wraps(orig_vt_forward)
def wrapped_vt_forward(*args, **kwargs):
    with record_function(f"slot{_current_slot['id']}_vision_tower"):
        return orig_vt_forward(*args, **kwargs)


@functools.wraps(orig_proj_forward)
def wrapped_proj_forward(*args, **kwargs):
    with record_function(f"slot{_current_slot['id']}_mm_projector"):
        return orig_proj_forward(*args, **kwargs)


def wrapped_cache_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
    is_bulk = key_states.shape[-2] > 1
    kind = "legacy_rebuild" if is_bulk else "append"
    with record_function(f"slot{_current_slot['id']}_kv_cache_{kind}"):
        return orig_cache_update(self, key_states, value_states, layer_idx, cache_kwargs)


model.forward = wrapped_forward
vision_tower.forward = wrapped_vt_forward
mm_projector.forward = wrapped_proj_forward
DynamicCache.update = wrapped_cache_update

request_timeline = []  # list of {"slot": ..., "event": ..., "t_s": ...}
t_ref = time.perf_counter()


def log_event(slot_id, event):
    request_timeline.append({"slot": slot_id, "event": event, "t_s": round(time.perf_counter() - t_ref, 4)})


def run_slot(slot_id, item):
    _current_slot["id"] = slot_id
    log_event(slot_id, "start")
    with record_function(f"slot{slot_id}_full_request"):
        run_one_request(item)
    torch.cuda.synchronize()  # boundary between requests is real in the live server too
                              # (one response fully completes before the next request's
                              # work begins) -- sync here so slot timings reflect that.
    log_event(slot_id, "done")


items = [dataset[i % len(dataset)] for i in range(N_CONCURRENT)]

try:
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        wall_t0 = time.perf_counter()
        for i in range(N_CONCURRENT):
            run_slot(i, items[i])
        wall_total_s = time.perf_counter() - wall_t0
finally:
    model.forward = orig_forward
    vision_tower.forward = orig_vt_forward
    mm_projector.forward = orig_proj_forward
    DynamicCache.update = orig_cache_update

print(f"\n{N_CONCURRENT} back-to-back sequential requests, wall time: {wall_total_s*1000:.1f} ms")
print("Per-slot start/done timeline (seconds since first request launched):")
for e in sorted(request_timeline, key=lambda x: x["t_s"]):
    print(f"  t={e['t_s']:.4f}s  slot {e['slot']}  {e['event']}")

# Sum of each slot's individually-measured duration vs. the actual total wall
# time -- the gap is dead time between one request finishing and the next
# one's work starting (Python-side request setup, tensor construction,
# generate() call overhead) that a real server would also pay between
# requests when serving them one at a time.
slot_durations = {}
starts = {e["slot"]: e["t_s"] for e in request_timeline if e["event"] == "start"}
dones = {e["slot"]: e["t_s"] for e in request_timeline if e["event"] == "done"}
for slot in starts:
    if slot in dones:
        slot_durations[slot] = dones[slot] - starts[slot]
sum_slot_durations = sum(slot_durations.values())
inter_request_gap_s = wall_total_s - sum_slot_durations
print(f"\nSum of individual slot durations: {sum_slot_durations*1000:.1f} ms")
print(f"Actual wall time for all {N_CONCURRENT} back-to-back requests: {wall_total_s*1000:.1f} ms")
print(f"Inter-request gap (dead time between requests): {inter_request_gap_s*1000:.1f} ms "
      f"({inter_request_gap_s/N_CONCURRENT*1000:.1f} ms avg per boundary)")

with open(TIMELINE_PATH, "w") as f:
    json.dump({
        "n_requests": N_CONCURRENT,
        "execution_mode": "sequential (back-to-back), not concurrent -- see module docstring",
        "wall_total_s": wall_total_s,
        "sum_slot_durations_s": sum_slot_durations,
        "inter_request_gap_s": inter_request_gap_s,
        "events": sorted(request_timeline, key=lambda x: x["t_s"]),
    }, f, indent=2)
print(f"Timeline written to {TIMELINE_PATH}")

settings = {
    "gpu": torch.cuda.get_device_name(0),
    "torch_version": torch.__version__,
    "transformers_version": __import__("transformers").__version__,
    "attn_implementation": getattr(model.config, "_attn_implementation", "unknown"),
    "model": "llava-v1.5-7b",
    "model_path": MODEL_PATH,
    "visual_token_num": VISUAL_TOKEN_NUM,
    "important_ratio": IMPORTANT_RATIO,
    "max_new_tokens": MAX_NEW_TOKENS,
    "min_new_tokens": MIN_NEW_TOKENS,
    "n_requests": N_CONCURRENT,
    "execution_mode": "sequential back-to-back (not concurrent) -- a true multi-thread concurrent "
                       "attempt against the shared model instance crashed with a CUDA device-side "
                       "assert (out-of-bounds index_select in the pruning gather path); see the "
                       "module docstring and results/timing/concurrent_profile_run_log.txt",
    "n_warmup_requests_sequential": N_WARMUP,
    "wall_total_ms": round(wall_total_s * 1000, 2),
    "sum_slot_durations_ms": round(sum_slot_durations * 1000, 2),
    "inter_request_gap_ms": round(inter_request_gap_s * 1000, 2),
}
print("\nSettings:")
print(json.dumps(settings, indent=2))
with open(SETTINGS_PATH, "w") as f:
    json.dump(settings, f, indent=2)

prof.export_chrome_trace(TRACE_PATH)
print(f"\nChrome trace written to {TRACE_PATH}")

events = prof.key_averages()
with open(SUMMARY_CUDA_PATH, "w") as f:
    f.write(f"n_concurrent={N_CONCURRENT} visual_token_num={VISUAL_TOKEN_NUM} max_new_tokens={MAX_NEW_TOKENS}\n\n")
    f.write("TOP OPS BY CUDA TIME TOTAL\n")
    f.write(str(events.table(sort_by="cuda_time_total", row_limit=60)))
print(f"CUDA-time-sorted summary written to {SUMMARY_CUDA_PATH}")

with open(SUMMARY_COUNT_PATH, "w") as f:
    f.write(f"n_concurrent={N_CONCURRENT} visual_token_num={VISUAL_TOKEN_NUM} max_new_tokens={MAX_NEW_TOKENS}\n\n")
    f.write("TOP OPS BY CALL COUNT\n")
    f.write(str(events.table(sort_by="count", row_limit=60)))
print(f"Call-count-sorted summary written to {SUMMARY_COUNT_PATH}")
