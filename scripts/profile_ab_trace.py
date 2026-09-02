"""
One-off diagnostic (not part of the tracked timing harness): Part 5 of the
VisPruner A/B experiment -- torch.profiler CPU+CUDA traces for four
conditions:

    python3.12 scripts/profile_ab_trace.py <visual_token_num> <mode>

    visual_token_num: 576 or 128
    mode: "isolated"       -- one no-load request, full path
          "near_saturation" -- N_BACK_TO_BACK requests run back-to-back in
                                one process, to approximate the per-request
                                kernel behavior at queued/saturated load
                                without re-triggering the thread-safety
                                crash documented in profile_concurrent_load.py
                                (genuine multi-thread concurrent execution
                                against the shared model instance crashed
                                with a CUDA device-side assert -- see that
                                script's docstring). Back-to-back is what the
                                real server does per queued request anyway
                                (strictly serial), so this is a faithful,
                                safe reproduction of its per-request GPU
                                behavior under a queue, not a substitute for
                                true concurrency.

Same instrumentation approach as profile_full_request.py /
profile_concurrent_load.py: runtime monkey-patching of model.forward,
prepare_inputs_labels_for_multimodal, vision tower / projector .forward, and
DynamicCache.update -- no vendored LLaVA/transformers source is edited.

Output-token policy matches the rest of this A/B experiment (Parts 1-4) and
the live server: do_sample=False, max_new_tokens=64, natural EOS stopping --
not the forced-length policy used in the earlier, unrelated
profile_full_request.py diagnostic.

Throwaway, no optimizations implemented here.
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

VISUAL_TOKEN_NUM = int(sys.argv[1]) if len(sys.argv) > 1 else 576
MODE = sys.argv[2] if len(sys.argv) > 2 else "isolated"
assert MODE in ("isolated", "near_saturation")

MODEL_PATH = "/workspace/GPU_Profiling/vis_pruner_copy/checkpoints/llava-v1.5-7b"
DATASET_PATH = "/workspace/GPU_Profiling/vis_pruner_copy/vispruner_eval_dataset/dev.json"
IMAGES_ROOT = "/workspace/GPU_Profiling/vis_pruner_copy/vispruner_eval_dataset"

IMPORTANT_RATIO = 0.5
MAX_NEW_TOKENS = 64
N_BACK_TO_BACK = 5 if MODE == "near_saturation" else 1
N_WARMUP = 10

TAG = f"vtn{VISUAL_TOKEN_NUM}_{MODE}"
RESULTS_DIR = "/workspace/GPU_Profiling/results/timing"
TRACE_PATH = f"{RESULTS_DIR}/ab_trace_{TAG}.json"
SUMMARY_CUDA_PATH = f"{RESULTS_DIR}/ab_trace_{TAG}_summary_by_cuda_time.txt"
SUMMARY_CPU_PATH = f"{RESULTS_DIR}/ab_trace_{TAG}_summary_by_cpu_time.txt"
SUMMARY_COUNT_PATH = f"{RESULTS_DIR}/ab_trace_{TAG}_summary_by_count.txt"
STAGE_SUMMARY_PATH = f"{RESULTS_DIR}/ab_trace_{TAG}_stage_breakdown.txt"
SETTINGS_PATH = f"{RESULTS_DIR}/ab_trace_{TAG}_settings.json"

disable_torch_init()
tokenizer, model, image_processor, _ = load_pretrained_model(
    MODEL_PATH, None, "llava-v1.5-7b",
    visual_token_num=VISUAL_TOKEN_NUM,
    important_ratio=IMPORTANT_RATIO,
)
model.eval()
num_layers = len(model.get_model().layers)
attn_impl = getattr(model.config, "_attn_implementation", "unknown")

with open(DATASET_PATH) as f:
    dataset = json.load(f)
items = dataset[:N_BACK_TO_BACK]


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


def run_one(item):
    input_ids, attention_mask, images, image_sizes = build_request(item)
    with torch.inference_mode():
        return model.generate(
            input_ids, attention_mask=attention_mask, images=images, image_sizes=image_sizes,
            do_sample=False, max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
        )


# --- Warm-up (rule 3), unprofiled ---
for i in range(N_WARMUP):
    run_one(dataset[i % len(dataset)])
torch.cuda.synchronize()

# --- Instrumentation ---
orig_prep = model.prepare_inputs_labels_for_multimodal
orig_forward = model.forward
vision_tower = model.get_vision_tower()
mm_projector = model.get_model().mm_projector
orig_vt_forward = vision_tower.forward
orig_proj_forward = mm_projector.forward
orig_cache_update = DynamicCache.update

_current_slot = {"id": 0}
_fwd_call_idx = {}


@functools.wraps(orig_prep)
def wrapped_prep(*args, **kwargs):
    with record_function(f"slot{_current_slot['id']}_multimodal_prep"):
        return orig_prep(*args, **kwargs)


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


model.prepare_inputs_labels_for_multimodal = wrapped_prep
model.forward = wrapped_forward
vision_tower.forward = wrapped_vt_forward
mm_projector.forward = wrapped_proj_forward
DynamicCache.update = wrapped_cache_update

request_timeline = []
t_ref = time.perf_counter()
output_lens = []

try:
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with record_function("image_preprocessing_first_request"):
            pass  # build_request() below (inside each slot) carries this cost;
                  # kept as a landmark label for trace navigation.

        wall_t0 = time.perf_counter()
        for i, item in enumerate(items):
            _current_slot["id"] = i
            request_timeline.append({"slot": i, "event": "start", "t_s": round(time.perf_counter() - t_ref, 4)})
            with record_function(f"slot{i}_full_request"):
                out_ids, _visual_token_num_used = run_one(item)
            torch.cuda.synchronize()
            output_lens.append(out_ids.shape[1])
            request_timeline.append({"slot": i, "event": "done", "t_s": round(time.perf_counter() - t_ref, 4)})
        wall_total_s = time.perf_counter() - wall_t0
finally:
    model.prepare_inputs_labels_for_multimodal = orig_prep
    model.forward = orig_forward
    vision_tower.forward = orig_vt_forward
    mm_projector.forward = orig_proj_forward
    DynamicCache.update = orig_cache_update

print(f"\n[{TAG}] {N_BACK_TO_BACK} request(s), wall time: {wall_total_s*1000:.1f} ms, "
      f"output token counts: {output_lens}")

prof.export_chrome_trace(TRACE_PATH)
print(f"Chrome trace written to {TRACE_PATH}")

events = prof.key_averages()
with open(SUMMARY_CUDA_PATH, "w") as f:
    f.write(f"visual_token_num={VISUAL_TOKEN_NUM} mode={MODE} n_requests={N_BACK_TO_BACK}\n\n")
    f.write("TOP OPS BY CUDA TIME TOTAL\n")
    f.write(str(events.table(sort_by="cuda_time_total", row_limit=60)))
print(f"CUDA-time summary written to {SUMMARY_CUDA_PATH}")

with open(SUMMARY_CPU_PATH, "w") as f:
    f.write(f"visual_token_num={VISUAL_TOKEN_NUM} mode={MODE} n_requests={N_BACK_TO_BACK}\n\n")
    f.write("TOP OPS BY CPU TIME TOTAL\n")
    f.write(str(events.table(sort_by="cpu_time_total", row_limit=60)))
print(f"CPU-time summary written to {SUMMARY_CPU_PATH}")

with open(SUMMARY_COUNT_PATH, "w") as f:
    f.write(f"visual_token_num={VISUAL_TOKEN_NUM} mode={MODE} n_requests={N_BACK_TO_BACK}\n\n")
    f.write("TOP OPS BY CALL COUNT\n")
    f.write(str(events.table(sort_by="count", row_limit=60)))
print(f"Count summary written to {SUMMARY_COUNT_PATH}")

# --- Stage breakdown for slot 0 (representative single request) ---
by_name = {e.key: e for e in events}
max_decode_steps = _fwd_call_idx.get(0, 1) - 1
stage_names = (
    ["slot0_multimodal_prep", "slot0_vision_tower", "slot0_mm_projector",
     "slot0_kv_cache_legacy_rebuild", "slot0_kv_cache_append", "slot0_prefill",
     "slot0_full_request"]
    + [f"slot0_decode_step_{i}" for i in range(1, max_decode_steps + 1)]
)
lines = [f"visual_token_num={VISUAL_TOKEN_NUM} mode={MODE} n_requests={N_BACK_TO_BACK} "
         f"num_transformer_layers={num_layers} attn_implementation={attn_impl}\n",
         f"{'stage':<28}{'count':>8}{'cpu_ms':>12}{'cuda_ms':>12}"]
decode_cuda_sum = 0.0
for name in stage_names:
    e = by_name.get(name)
    if e is None:
        continue
    cpu_ms = e.cpu_time_total / 1000.0
    cuda_ms = e.cuda_time_total / 1000.0
    lines.append(f"{name:<28}{e.count:>8}{cpu_ms:>12.3f}{cuda_ms:>12.3f}")
    if "decode_step" in name:
        decode_cuda_sum += cuda_ms
lines.append(f"\ndecode phase total (slot0, {max_decode_steps} steps): cuda_ms={decode_cuda_sum:.3f} "
             f"(avg {decode_cuda_sum/max(max_decode_steps,1):.3f} ms/step)")
stage_text = "\n".join(lines)
print("\n" + stage_text)
with open(STAGE_SUMMARY_PATH, "w") as f:
    f.write(stage_text + "\n")
print(f"Stage breakdown written to {STAGE_SUMMARY_PATH}")

settings = {
    "gpu": torch.cuda.get_device_name(0),
    "torch_version": torch.__version__,
    "transformers_version": __import__("transformers").__version__,
    "attn_implementation": attn_impl,
    "model": "llava-v1.5-7b",
    "visual_token_num": VISUAL_TOKEN_NUM,
    "important_ratio": IMPORTANT_RATIO,
    "mode": MODE,
    "n_requests": N_BACK_TO_BACK,
    "max_new_tokens_cap": MAX_NEW_TOKENS,
    "do_sample": False,
    "output_token_counts": output_lens,
    "wall_total_ms": round(wall_total_s * 1000, 2),
    "n_warmup": N_WARMUP,
    "num_transformer_layers": num_layers,
}
with open(SETTINGS_PATH, "w") as f:
    json.dump(settings, f, indent=2)
print(f"Settings written to {SETTINGS_PATH}")
