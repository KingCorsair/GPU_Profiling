"""
VLM Performance Experiments
===========================

Four experiments that produce your project's baseline numbers.

Run on the GPU pod:
    python experiments.py /root/GPU_Profiling/sribhav/photo.jpg

Read the comments as you go. The numbers matter less than understanding
why they come out the way they do.
"""

import time
import sys
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from torch.profiler import profile, ProfilerActivity

MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "/root/GPU_Profiling/sribhav/photo.jpg"

print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Loading model...\n")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
).to("cuda")
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")


def build_inputs(images, question="Describe this image."):
    """Turn images + a question into model inputs.

    Note we build one message per image - that's what makes it a batch.
    """
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(
        text=[prompt] * len(images), images=images, return_tensors="pt", padding=True
    ).to("cuda")


def timed_generate(inputs, max_new_tokens=30):
    """Generate, timed correctly.

    synchronize() before starting and before stopping is essential.
    CUDA is asynchronous: without it, perf_counter() measures how long
    Python took to QUEUE the work, not how long the GPU took to do it.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


# Warmup. The first generate() includes CUDA context setup, memory pool
# allocation, and kernel loading. It is always slower and never representative.
print("Warming up...")
_ = timed_generate(build_inputs([image]), max_new_tokens=5)
print("Ready.\n")


# =================================================================
# EXPERIMENT 1 - How much of the input is the image?
# =================================================================
#
# This is the premise of the entire project, measured on your own data.

print("=" * 62)
print("1. TOKEN BREAKDOWN - where does the input come from?")
print("=" * 62)

with_image = build_inputs([image])
n_with = with_image["input_ids"].shape[1]

# Same prompt, no image, to isolate the text cost.
text_only = processor(text="Describe this image.", return_tensors="pt").to("cuda")
n_text = text_only["input_ids"].shape[1]

print(f"""
  Text only        : {n_text:>6} tokens
  Text + image     : {n_with:>6} tokens
  The image alone  : {n_with - n_text:>6} tokens  ({(n_with - n_text) / n_with * 100:.1f}% of input)

  Every one of those image tokens costs compute in every layer of the
  model. Cutting them is the whole project.
""")


# =================================================================
# EXPERIMENT 2 - Prefill vs decode
# =================================================================
#
# Two completely different phases with different bottlenecks:
#
#   PREFILL - reads the entire prompt at once. Thousands of tokens
#     processed in parallel. Heavy arithmetic, all cores busy.
#     COMPUTE-BOUND.
#
#   DECODE - generates one token, then the next, each depending on
#     the last. Almost no parallelism. Most time spent reading model
#     weights from memory. MEMORY-BOUND.
#
# We separate them by timing generation of 1 token (mostly prefill)
# vs 50 tokens, then subtracting.

print("=" * 62)
print("2. PREFILL vs DECODE")
print("=" * 62)

inputs = build_inputs([image])

_, t_1 = timed_generate(inputs, max_new_tokens=1)     # ~ prefill only
_, t_50 = timed_generate(inputs, max_new_tokens=50)   # prefill + 49 decodes

prefill_s = t_1
decode_per_token = (t_50 - t_1) / 49

print(f"""
  Prefill (~{n_with} tokens) : {prefill_s * 1000:>8.1f} ms
  Decode, per token         : {decode_per_token * 1000:>8.1f} ms
  50-token response total   : {t_50:>8.2f} s

  Prefill handles {n_with} tokens in {prefill_s*1000:.0f}ms.
  Decode handles ONE token in {decode_per_token*1000:.1f}ms.

  Per token, decode looks catastrophically slower - but that's the point.
  Prefill parallelises across all tokens at once; decode cannot, because
  each token depends on the previous one. Decode spends its time reading
  weights from VRAM, not computing.

  Consequence: token compression mainly speeds up PREFILL. If a response
  is long, decode dominates and compression helps less than you'd hope.
  That is exactly the kind of gap between paper claims and reality that
  this project exists to measure.
""")


# =================================================================
# EXPERIMENT 3 - Batching
# =================================================================
#
# A GPU has thousands of cores. One request leaves most idle.
# Batching fills the machine.
#
# Watch total time rise while per-request time falls. That trade -
# throughput up, individual latency up - is what your scheduler manages.

print("=" * 62)
print("3. BATCHING - the throughput/latency trade")
print("=" * 62)
print(f"\n  {'batch':>6} | {'total s':>9} | {'per req':>9} | {'req/s':>8} | {'VRAM GB':>8}")
print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}")

for batch in [1, 2, 4, 8, 16]:
    try:
        torch.cuda.reset_peak_memory_stats()
        batch_inputs = build_inputs([image] * batch)
        _, elapsed = timed_generate(batch_inputs, max_new_tokens=30)

        per_req = elapsed / batch
        throughput = batch / elapsed
        peak_gb = torch.cuda.max_memory_allocated() / 1e9

        print(f"  {batch:>6} | {elapsed:>9.2f} | {per_req:>9.3f} | "
              f"{throughput:>8.2f} | {peak_gb:>8.2f}")

        del batch_inputs
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        print(f"  {batch:>6} |  OUT OF MEMORY - this is your ceiling")
        torch.cuda.empty_cache()
        break

print("""
  Requests per second climbs. Time per individual request climbs too.

  That is the fundamental serving trade-off: batching serves far more
  users, but each waits a little longer. Where you sit on that curve is
  a product decision, and your scheduler is what implements it.

  Note where throughput stops improving - that's GPU saturation. Beyond
  it you're just building a queue.

  Also note VRAM growth. Memory, not compute, is usually what caps your
  batch size. That's why KV cache management matters.
""")


# =================================================================
# EXPERIMENT 4 - Kernel-level profile
# =================================================================
#
# Which individual GPU operations actually consume the time?
# This is where you stop guessing and start knowing.

print("=" * 62)
print("4. KERNEL PROFILE - what's actually expensive")
print("=" * 62)

inputs = build_inputs([image])

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=20, do_sample=False)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

print("""
  Reading this table:

    Self CUDA    - time in this op alone, excluding children.
                   This is the column that matters.
    CUDA total   - includes nested operations.
    # of Calls   - how often it ran.

  Expect matrix multiplies (addmm, bmm, linear) near the top - that's
  normal and healthy. If you instead see lots of small elementwise ops
  (add, mul, layer_norm) eating significant time, those are fusion
  candidates: memory-bound operations that could be merged into one
  kernel. That's what Triton is for.

  Export a timeline you can open in chrome://tracing with:
      prof.export_chrome_trace("trace.json")
""")

print("=" * 62)
print("""
YOUR BASELINE - write these down:

  1. Image tokens as % of input
  2. Prefill time and per-token decode time
  3. Throughput at each batch size, and where it saturates
  4. Peak VRAM at each batch size, and where OOM hits
  5. Top 3 kernels by CUDA time

Every optimisation from here gets measured against these.

Then terminate the pod.
""")
print("=" * 62)