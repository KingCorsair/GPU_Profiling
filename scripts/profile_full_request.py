"""
One-off diagnostic (not part of the tracked timing harness): profile ONE
representative LLaVA request end-to-end -- image preprocessing, multimodal
prep (vision tower + projector + fixup), prefill, every decode step, and
request completion -- to see where latency actually goes across the whole
path, not just the prepare_inputs_labels_for_multimodal() slice that
profile_multimodal_prep.py already covers.

Throwaway, per AMAY_SPEED_PLAN.md / AMAY_TIMING_NOTES.md / CLAUDE.md's
"Amay measures to find bottlenecks" split -- not Rithvik's harness, produces
no reportable numbers. Diagnostic only: no optimizations are implemented
here.

Everything below is external instrumentation (monkey-patched bound methods
and module .forward attributes, restored after profiling) -- no vendored
LLaVA/transformers source is edited. This keeps the vision tower / projector
/ prefill / decode / KV-cache boundaries visible in the trace without
touching llava_arch.py or llava_llama.py.
"""
import functools
import json
import subprocess
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
IMAGE_PATH = "/workspace/GPU_Profiling/vis_pruner_copy/vispruner_eval_dataset/images/textvqa_fe32ee8991d2e59f.jpg"
VISUAL_TOKEN_NUM = 128   # same setting as profile_multimodal_prep.py, for comparability
IMPORTANT_RATIO = 0.5
BATCH_SIZE = 1           # single representative request; batched path isn't built yet
MAX_NEW_TOKENS = 24      # enough decode steps for the phase to be clearly visible in the trace
MIN_NEW_TOKENS = 24      # force full length -- greedy decode on this prompt hits EOS after
                         # ~3 tokens otherwise, which isn't enough to see the decode phase
N_WARMUP = 10            # rule 3: discard first ~10 iters (CUDA context init, kernel-cache warm-up)

TRACE_PATH = "/workspace/GPU_Profiling/results/timing/full_request_trace.json"
SUMMARY_CUDA_PATH = "/workspace/GPU_Profiling/results/timing/full_request_summary_by_cuda_time.txt"
SUMMARY_COUNT_PATH = "/workspace/GPU_Profiling/results/timing/full_request_summary_by_count.txt"
SUMMARY_CPU_PATH = "/workspace/GPU_Profiling/results/timing/full_request_summary_by_cpu_time.txt"
SETTINGS_PATH = "/workspace/GPU_Profiling/results/timing/full_request_settings.json"
STAGE_SUMMARY_PATH = "/workspace/GPU_Profiling/results/timing/full_request_stage_breakdown.txt"


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
    visual_token_num=VISUAL_TOKEN_NUM,
    important_ratio=IMPORTANT_RATIO,
)
model.eval()

num_layers = len(model.get_model().layers)
attn_impl = getattr(model.config, "_attn_implementation", "unknown")


def build_request():
    """Everything a fresh request needs before the model ever runs:
    image load/resize/normalize + tokenization. Deliberately re-done per
    call (not hoisted out) so the profiled iteration includes it -- this is
    real per-request latency in a serving path, not one-time setup."""
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
    return input_ids, attention_mask, images, image_sizes


def run_generate(input_ids, attention_mask, images, image_sizes):
    with torch.inference_mode():
        return model.generate(
            input_ids, attention_mask=attention_mask, images=images, image_sizes=image_sizes,
            do_sample=False, max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MIN_NEW_TOKENS,
            use_cache=True,
        )


# --- Warm-up (rule 3): full request path, unprofiled, thrown away ---
for _ in range(N_WARMUP):
    warm_input_ids, warm_attention_mask, warm_images, warm_image_sizes = build_request()
    run_generate(warm_input_ids, warm_attention_mask, warm_images, warm_image_sizes)
torch.cuda.synchronize()

# --- Instrumentation: monkey-patch bound methods / module .forward so the
# trace distinguishes each stage. Everything here is restored after the
# profiled call -- no source file is touched. ---
orig_prep = model.prepare_inputs_labels_for_multimodal
orig_forward = model.forward
vision_tower = model.get_vision_tower()
mm_projector = model.get_model().mm_projector
orig_vt_forward = vision_tower.forward
orig_proj_forward = mm_projector.forward
orig_cache_update = DynamicCache.update

_fwd_call_idx = {"n": 0}


@functools.wraps(orig_prep)
def wrapped_prep(*args, **kwargs):
    with record_function("multimodal_prep"):
        return orig_prep(*args, **kwargs)


# functools.wraps sets __wrapped__, which inspect.signature() follows by
# default -- needed because HF's generate() calls
# _validate_model_kwargs(), which inspects self.forward's real parameter
# names to decide which kwargs (e.g. attention_mask) are legal. A plain
# (*args, **kwargs) wrapper would hide those names and make that check
# reject valid arguments.
@functools.wraps(orig_forward)
def wrapped_forward(*args, **kwargs):
    idx = _fwd_call_idx["n"]
    _fwd_call_idx["n"] += 1
    label = "prefill" if idx == 0 else f"decode_step_{idx}"
    with record_function(label):
        return orig_forward(*args, **kwargs)


@functools.wraps(orig_vt_forward)
def wrapped_vt_forward(*args, **kwargs):
    with record_function("vision_tower"):
        return orig_vt_forward(*args, **kwargs)


@functools.wraps(orig_proj_forward)
def wrapped_proj_forward(*args, **kwargs):
    with record_function("mm_projector"):
        return orig_proj_forward(*args, **kwargs)


def wrapped_cache_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
    # Two distinct call sites hit DynamicCache.update() per decode step, not one:
    #   1. LlamaModel.forward -> DynamicCache.from_legacy_cache(past_key_values) rebuilds a
    #      fresh DynamicCache from the legacy tuple format generate() passes between steps
    #      (this codebase's forward() doesn't carry a persisted Cache object across calls),
    #      re-registering the WHOLE existing per-layer K/V tensor -- key_states here has the
    #      full running sequence length, not just the new token.
    #   2. LlamaAttention.forward -> the real incremental update, appending exactly the new
    #      token via torch.cat -- key_states here has seq len 1 on decode steps.
    # Splitting on that shape makes the rebuild's overhead visible on its own in the trace,
    # separate from the real per-token append.
    is_bulk = key_states.shape[-2] > 1
    label = "kv_cache_legacy_rebuild" if is_bulk else "kv_cache_append"
    with record_function(label):
        return orig_cache_update(self, key_states, value_states, layer_idx, cache_kwargs)


model.prepare_inputs_labels_for_multimodal = wrapped_prep
model.forward = wrapped_forward
vision_tower.forward = wrapped_vt_forward
mm_projector.forward = wrapped_proj_forward
DynamicCache.update = wrapped_cache_update

# --- Profile exactly ONE representative request, full path ---
prof_input_ids, prof_attention_mask, prof_images, prof_image_sizes = build_request()
prompt_len = prof_input_ids.shape[1]

try:
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with record_function("image_preprocessing"):
            # Re-run just the preprocessing portion inside the profiled region
            # too (build_request() above, outside the profiler, only produced
            # the tensors used to measure prompt_len before entering the
            # context -- this call is the one whose cost actually lands in
            # the trace).
            req_input_ids, req_attention_mask, req_images, req_image_sizes = build_request()

        wall_t0 = time.perf_counter()
        output = run_generate(req_input_ids, req_attention_mask, req_images, req_image_sizes)
        torch.cuda.synchronize()
        wall_generate_s = time.perf_counter() - wall_t0

        with record_function("request_completion"):
            output_ids, visual_token_num_used = output
            generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
finally:
    # Restore, even on failure -- no lasting side effects on the model.
    model.prepare_inputs_labels_for_multimodal = orig_prep
    model.forward = orig_forward
    vision_tower.forward = orig_vt_forward
    mm_projector.forward = orig_proj_forward
    DynamicCache.update = orig_cache_update

n_decode_steps = _fwd_call_idx["n"] - 1  # forward calls minus the one prefill call
output_token_count = output_ids.shape[1]

print(f"batch={BATCH_SIZE} prompt_len={prompt_len} visual_token_num={VISUAL_TOKEN_NUM} "
      f"important_ratio={IMPORTANT_RATIO}")
print(f"forward() calls: {_fwd_call_idx['n']} (1 prefill + {n_decode_steps} decode steps)")
print(f"output_ids shape: {tuple(output_ids.shape)}  generated_text: {generated_text!r}")
print(f"wall time for generate() (post-warm-up, synced): {wall_generate_s*1000:.2f} ms")

# --- Settings record ---
settings = {
    "gpu": torch.cuda.get_device_name(0),
    "gpu_total_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "transformers_version": __import__("transformers").__version__,
    "attn_implementation": attn_impl,
    "model_path": MODEL_PATH,
    "model": "llava-v1.5-7b",
    "num_transformer_layers": num_layers,
    "visual_token_num": VISUAL_TOKEN_NUM,
    "important_ratio": IMPORTANT_RATIO,
    "visual_token_num_actually_used": visual_token_num_used,
    "batch_size": BATCH_SIZE,
    "prompt_len_tokens": prompt_len,
    "max_new_tokens_requested": MAX_NEW_TOKENS,
    "output_token_count": output_token_count,
    "forward_calls_total": _fwd_call_idx["n"],
    "decode_steps": n_decode_steps,
    "n_warmup_requests": N_WARMUP,
    "profiled_requests": 1,
    "generated_text": generated_text,
    "wall_time_generate_ms_post_warmup": round(wall_generate_s * 1000, 2),
    "git_commit": git_commit(),
    "dtype": "fp16",
    "image_path": IMAGE_PATH,
}
print("\nSettings:")
print(json.dumps(settings, indent=2))
with open(SETTINGS_PATH, "w") as f:
    json.dump(settings, f, indent=2)
print(f"\nSettings written to {SETTINGS_PATH}")

# --- Export Chrome/Perfetto trace ---
prof.export_chrome_trace(TRACE_PATH)
print(f"Chrome trace written to {TRACE_PATH}")

# --- Summary tables ---
events = prof.key_averages()

table_cuda = events.table(sort_by="cuda_time_total", row_limit=60)
table_count = events.table(sort_by="count", row_limit=60)
table_cpu = events.table(sort_by="cpu_time_total", row_limit=60)

with open(SUMMARY_CUDA_PATH, "w") as f:
    f.write(f"visual_token_num={VISUAL_TOKEN_NUM} batch={BATCH_SIZE} prompt_len={prompt_len} "
            f"max_new_tokens={MAX_NEW_TOKENS} decode_steps={n_decode_steps}\n\n")
    f.write("TOP OPS BY CUDA TIME TOTAL (includes nested record_function regions)\n")
    f.write(str(table_cuda))
print(f"CUDA-time-sorted summary written to {SUMMARY_CUDA_PATH}")

with open(SUMMARY_COUNT_PATH, "w") as f:
    f.write(f"visual_token_num={VISUAL_TOKEN_NUM} batch={BATCH_SIZE} prompt_len={prompt_len} "
            f"max_new_tokens={MAX_NEW_TOKENS} decode_steps={n_decode_steps}\n\n")
    f.write("TOP OPS BY CALL COUNT (reveals per-step / per-layer repeated small launches)\n")
    f.write(str(table_count))
print(f"Call-count-sorted summary written to {SUMMARY_COUNT_PATH}")

with open(SUMMARY_CPU_PATH, "w") as f:
    f.write(f"visual_token_num={VISUAL_TOKEN_NUM} batch={BATCH_SIZE} prompt_len={prompt_len} "
            f"max_new_tokens={MAX_NEW_TOKENS} decode_steps={n_decode_steps}\n\n")
    f.write("TOP OPS BY CPU TIME TOTAL (CPU dispatch/launch overhead)\n")
    f.write(str(table_cpu))
print(f"CPU-time-sorted summary written to {SUMMARY_CPU_PATH}")

# --- Stage-level breakdown: pull out exactly the record_function labels we
# installed above, so stage totals (including their nested children) are
# readable without hunting through the full op table. cuda_time_total /
# cpu_time_total on a key_averages() entry for a record_function label
# includes everything nested inside that region, not just its own self time.
stage_names = (
    ["image_preprocessing", "multimodal_prep", "vision_tower", "mm_projector",
     "kv_cache_legacy_rebuild", "kv_cache_append", "prefill", "request_completion"]
    + [f"decode_step_{i}" for i in range(1, n_decode_steps + 1)]
)
by_name = {e.key: e for e in events}

lines = [
    f"visual_token_num={VISUAL_TOKEN_NUM} batch={BATCH_SIZE} prompt_len={prompt_len} "
    f"max_new_tokens={MAX_NEW_TOKENS} decode_steps={n_decode_steps} "
    f"num_transformer_layers={num_layers} attn_implementation={attn_impl}\n",
    f"{'stage':<22}{'count':>8}{'cpu_ms':>12}{'cuda_ms':>12}{'self_cpu_ms':>14}{'self_cuda_ms':>14}",
]
decode_cuda_total_ms = 0.0
decode_cpu_total_ms = 0.0
for name in stage_names:
    e = by_name.get(name)
    if e is None:
        continue
    cpu_ms = e.cpu_time_total / 1000.0
    cuda_ms = e.cuda_time_total / 1000.0
    self_cpu_ms = e.self_cpu_time_total / 1000.0
    self_cuda_ms = e.self_cuda_time_total / 1000.0
    lines.append(f"{name:<22}{e.count:>8}{cpu_ms:>12.3f}{cuda_ms:>12.3f}{self_cpu_ms:>14.3f}{self_cuda_ms:>14.3f}")
    if name.startswith("decode_step_"):
        decode_cuda_total_ms += cuda_ms
        decode_cpu_total_ms += cpu_ms

lines.append("")
lines.append(f"decode phase (sum of {n_decode_steps} decode_step_* regions): "
             f"cuda_ms={decode_cuda_total_ms:.3f} cpu_ms={decode_cpu_total_ms:.3f} "
             f"(avg {decode_cuda_total_ms/max(n_decode_steps,1):.3f} ms cuda / step)")

prefill_e = by_name.get("prefill")
if prefill_e is not None:
    lines.append(f"prefill phase: cuda_ms={prefill_e.cuda_time_total/1000.0:.3f} "
                 f"cpu_ms={prefill_e.cpu_time_total/1000.0:.3f}")

stage_summary_text = "\n".join(lines)
print("\n" + "=" * 100)
print("STAGE BREAKDOWN")
print("=" * 100)
print(stage_summary_text)
with open(STAGE_SUMMARY_PATH, "w") as f:
    f.write(stage_summary_text + "\n")
print(f"\nStage breakdown written to {STAGE_SUMMARY_PATH}")
