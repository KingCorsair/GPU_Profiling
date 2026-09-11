"""
Resolution Experiment
=====================

The simplest possible way to cut visual tokens: make the image smaller.

Fewer pixels -> fewer patches -> fewer tokens -> less work.

But watch carefully: token count and total time do NOT fall by the same
amount. Understanding why is the point of this script, and it's the core
insight of the whole project.

Run on the GPU pod:
    python resolution_experiment.py photo.jpg
"""

import sys
import time
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
QUESTION = "Describe this image."

RESOLUTIONS = [768, 512, 384, 256]
RESPONSE_LENGTHS = [10, 100]      # short vs long answers - this matters

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "photo.jpg"


# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------

print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Loading model...\n")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
).to("cuda")
model.eval()

original = Image.open(IMAGE_PATH).convert("RGB")
print(f"Original image: {original.size[0]}x{original.size[1]}\n")


def build_inputs(image):
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": QUESTION}],
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(text=prompt, images=[image], return_tensors="pt").to("cuda")


def run(image, max_new_tokens):
    """Generate and time it. Returns (answer, seconds, input_tokens)."""
    inputs = build_inputs(image)
    n_tokens = inputs["input_ids"].shape[1]

    # synchronize() is essential. CUDA is asynchronous - without it we'd
    # be timing how long Python took to queue the work, not the GPU's work.
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    answer = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()

    return answer, elapsed, n_tokens


# Warmup - first run includes CUDA init and is never representative.
print("Warming up...")
run(original.resize((512, 512)), 5)
print("Ready.\n")


# ---------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------

results = {}

for max_new in RESPONSE_LENGTHS:
    print("=" * 68)
    print(f"RESPONSE LENGTH: {max_new} tokens")
    print("=" * 68)
    print(f"\n  {'res':>6} | {'in tok':>7} | {'tok cut':>8} | "
          f"{'time s':>8} | {'time cut':>9}")
    print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}")

    baseline_tok = None
    baseline_time = None
    rows = []

    for res in RESOLUTIONS:
        img = original.resize((res, res))
        answer, elapsed, n_tok = run(img, max_new)

        if baseline_tok is None:
            baseline_tok, baseline_time = n_tok, elapsed
            tok_cut = time_cut = 0.0
        else:
            tok_cut = (1 - n_tok / baseline_tok) * 100
            time_cut = (1 - elapsed / baseline_time) * 100

        rows.append((res, n_tok, tok_cut, elapsed, time_cut, answer))

        print(f"  {res:>6} | {n_tok:>7} | {tok_cut:>7.1f}% | "
              f"{elapsed:>8.3f} | {time_cut:>8.1f}%")

    results[max_new] = rows
    print()


# ---------------------------------------------------------------
# The lesson
# ---------------------------------------------------------------

print("=" * 68)
print("WHAT JUST HAPPENED")
print("=" * 68)

short = results[RESPONSE_LENGTHS[0]][-1]     # smallest res, short response
long_ = results[RESPONSE_LENGTHS[-1]][-1]    # smallest res, long response

print(f"""
At {RESOLUTIONS[-1]}x{RESOLUTIONS[-1]}, tokens fell by {short[2]:.0f}%.

  With a {RESPONSE_LENGTHS[0]}-token response  : total time fell {short[4]:.0f}%
  With a {RESPONSE_LENGTHS[-1]}-token response : total time fell {long_[4]:.0f}%

Same token saving. Different time saving. Why?

Generation has two phases:

  PREFILL - reads the whole prompt at once. Cost scales with input
    length, so fewer image tokens directly means less prefill work.
    Compute-bound.

  DECODE - writes one token at a time, each depending on the last.
    Cost scales with OUTPUT length, and is dominated by reading model
    weights from VRAM. It barely cares how long your prompt was.
    Memory-bound.

Compression only shrinks prefill. Decode is untouched.

So the longer the response, the more decode dominates, and the less
your compression helps. A paper reporting "70% fewer tokens" is telling
you about prefill. A user waiting for a paragraph of text is mostly
experiencing decode.

THIS IS THE PROJECT. That gap - between what papers measure and what
users experience - is exactly what we're quantifying.
""")

print("=" * 68)
print("ANSWER QUALITY (Sribhav's department, but look anyway)")
print("=" * 68)

for res, n_tok, tok_cut, elapsed, time_cut, answer in results[RESPONSE_LENGTHS[-1]]:
    print(f"\n  --- {res}x{res} ({n_tok} tokens) ---")
    print(f"  {answer[:220]}{'...' if len(answer) > 220 else ''}")

print(f"""

Notice detail disappearing as resolution drops. Small text, object
counts, and fine spatial relationships go first.

That's the trade-off in one screen: speed on the left, accuracy here.
Neither number means anything without the other - which is why this
project needs three people.

NOTE: resizing is the crudest possible compression. It throws away
information uniformly, including from the parts of the image that
actually matter. Real methods (FastV, VisionZip) drop tokens AFTER the
vision encoder, keeping the ones the model is actually attending to.
Same idea, much better trade-off. That's what Sribhav builds next.
""")

print("=" * 68)
print("Done. Terminate the pod.")
print("=" * 68)