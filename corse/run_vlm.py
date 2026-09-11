"""
Run a vision-language model and ask it questions.

Takes Sribhav's original and compressed images, asks the same question
about both, and compares the answers and the timing.

Usage:
    python run_vlm.py original.png compressed.png

Run this on the GPU pod, not your laptop.
"""

import sys
import time
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
QUESTION = "Describe this image in detail."


# ---------------------------------------------------------------
# Load the model
# ---------------------------------------------------------------

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Loading {MODEL_ID} (first run downloads ~4.5GB)...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
).to("cuda")

mem_gb = torch.cuda.memory_allocated() / 1e9
print(f"Loaded. Weights using {mem_gb:.2f} GB of VRAM.\n")


# ---------------------------------------------------------------
# Ask a question about one image
# ---------------------------------------------------------------

def ask(image_path, question=QUESTION, max_new_tokens=100):
    """Returns (answer, seconds, n_input_tokens)."""

    image = Image.open(image_path).convert("RGB")

    # SmolVLM expects a chat-style message with an image placeholder.
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")

    # This is the number that matters for the whole project:
    # how many tokens the model has to process. The image is nearly all of it.
    n_input_tokens = inputs["input_ids"].shape[1]

    # Time it properly. synchronize() forces the GPU to finish before we
    # stop the clock - without it we'd measure how long it took to QUEUE
    # the work, not to do it.
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # generate() returns prompt + answer, so trim the prompt off.
    answer = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()

    return answer, elapsed, n_input_tokens


# ---------------------------------------------------------------
# Compare the two images
# ---------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python run_vlm.py original.png [compressed.png]")
    sys.exit(1)

paths = sys.argv[1:]

# Warmup - the first generate() is always slower (CUDA init, memory pools).
# Never include it in a measurement.
print("Warming up...")
ask(paths[0], max_new_tokens=5)
print("Ready.\n")

results = []

for path in paths:
    print("=" * 60)
    print(f"IMAGE: {path}")
    print("=" * 60)

    answer, elapsed, n_tokens = ask(path)
    results.append((path, elapsed, n_tokens))

    print(f"\nInput tokens : {n_tokens}")
    print(f"Time         : {elapsed:.2f} s")
    print(f"\nQ: {QUESTION}")
    print(f"A: {answer}\n")


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------

if len(results) > 1:
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"\n  {'image':<25} {'tokens':>8} {'seconds':>9}")
    print(f"  {'-'*25} {'-'*8} {'-'*9}")
    for path, elapsed, n_tokens in results:
        print(f"  {path:<25} {n_tokens:>8} {elapsed:>9.2f}")

    base_tok, base_time = results[0][2], results[0][1]
    comp_tok, comp_time = results[1][2], results[1][1]

    print(f"""
  Token change : {(comp_tok - base_tok) / base_tok * 100:+.1f}%
  Time change  : {(comp_time - base_time) / base_time * 100:+.1f}%

  If the token count did NOT drop, that is the expected result and it is
  the whole point of this exercise. Blacking out pixels does not reduce
  tokens - the model still encodes every patch, it just sees black ones.

  Real compression drops tokens AFTER the vision encoder, so they never
  reach the language model at all. That is what we build next.
""")

print("Done. Remember to terminate the pod.")