"""
Phase 0 (AMAY_ENGINEERING_ROADMAP.md) -- minimal, instrumentation-free
reproduction of the concurrent-access CUDA crash against the shared
LLaVA+VisPruner model instance.

WHAT THIS IS FOR
----------------
On 2026-09-01, scripts/profile_concurrent_load.py launched 3 real threads at
one shared model and the process died with:

    ../aten/src/ATen/native/cuda/Indexing.cu:1237: indexSelectSmallIndex:
    Assertion `srcIndex < srcSelectDimSize` failed.

followed by CUBLAS_STATUS_EXECUTION_FAILED and "device-side assert triggered"
in the other two threads (results/timing/concurrent_profile_run_log.txt).
That run is the only evidence we have, and it is confounded twice over:

  (1) It ran under an active torch.profiler.
  (2) It monkey-patched model.forward, vision_tower.forward,
      mm_projector.forward AND DynamicCache.update -- the last of which is a
      CLASS-level patch, global to the process -- plus two plain dicts
      (_current_slot, _fwd_call_idx) mutated from every thread with no lock
      (profile_concurrent_load.py:147-186).

So "the model is unsafe under concurrency" and "the model PLUS that
instrumentation is unsafe under concurrency" are both consistent with the
evidence, and they are different claims with different fixes. This script
removes every bit of (1) and (2) so the question can actually be answered.

A CODE AUDIT PRECEDED THIS SCRIPT (roadmap Phase 0, task 2). Every line of
encode_images() (llava_arch.py:139-183) and
prepare_inputs_labels_for_multimodal() (llava_arch.py:186-394) was read for
non-local mutable state. Findings:

  * The pruning math itself holds NO shared mutable state. Everything is
    call-local; visual_token_num / important_ratio are read-only config read
    fresh per call. This is a real negative finding and it argues against the
    roadmap's assumption that the bug lives in the pruning gather path.
  * Three genuine pieces of shared mutable state DO exist, all in this
    repo's own timing instrumentation, not in vendored VisPruner code:
      H1a  self._timing_forward_events / self._timing_prepare_inputs_calls
           (llava_llama.py:76-77) are PER-REQUEST recording lists stored on
           the SHARED INSTANCE. generate() sets them to [] at :172-173 and
           back to None at :245-246. Two concurrent generate()s: B's `= []`
           discards A's partial list; A's `= None` on exit flips B's
           timing_active to False mid-generation; elapsed_time() then gets
           called on CUDA events recorded by a different request.
      H1b  torch.cuda.synchronize() at llava_llama.py:180 and :206 is
           DEVICE-wide, not thread-scoped. Thread A's "sync outside the
           timing loop" waits on thread B's kernels. This does not corrupt
           anything, but it makes any per-request timing from a concurrent
           run fiction -- which is why this phase reports no latency numbers.
      H1c  The LLAVA_TIMING_FILE append at llava_llama.py:236-240 is
           unlocked, and json.dump() writes in many small chunks, so
           concurrent threads can interleave mid-line. Not fixed here (see
           --timing-file below); recorded so it is not rediscovered later.

HYPOTHESES THIS SCRIPT DISCRIMINATES
------------------------------------
  H1  The instance-level timing state is the race.
      -> killed if the crash persists with --fix threadlocal
  H2  The old harness's global patches/profiler were the race.
      -> killed if the crash persists here at --fix none (this script has
         neither)
  H3  generate() on a shared module is unsafe at the torch/transformers
      level, independent of anything in this repo.
      -> supported if the crash survives both of the above

WHY ONE REPETITION PER PROCESS
------------------------------
A CUDA device-side assert poisons the CUDA context for the WHOLE PROCESS.
Every subsequent CUDA call fails -- which is exactly why the original log
shows CUBLAS_STATUS_EXECUTION_FAILED in threads 2 and 3: collateral damage,
not independent failures. You therefore cannot loop repetitions in-process.
This script runs the experiment EXACTLY ONCE and signals the result through
its exit code; scripts/run_phase0_repro.sh loops fresh subprocesses. That
costs a full ~40s weight load per repetition. There is no way around it for
a non-deterministic CUDA bug.

EXIT CODES (the actual output of this script -- stdout is for humans)
  0  ran to completion, every thread's output matched the sequential baseline
  1  at least one thread raised (crash reproduced)
  2  no crash, but at least one thread's output differed from baseline
     -- a silent cross-request corruption, which is WORSE than a crash
  3  setup problem (missing baseline, bad args); experiment did not run

Throwaway diagnostic per AMAY_TIMING_NOTES.md's "Amay measures to find
bottlenecks" split. Produces no reportable numbers and implements no
optimization.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback

REPO_ROOT = "/workspace/GPU_Profiling"
sys.path.insert(0, os.path.join(REPO_ROOT, "vis_pruner_copy"))

MODEL_PATH = f"{REPO_ROOT}/vis_pruner_copy/checkpoints/llava-v1.5-7b"
DATASET_PATH = f"{REPO_ROOT}/vis_pruner_copy/vispruner_eval_dataset/dev.json"
IMAGES_ROOT = f"{REPO_ROOT}/vis_pruner_copy/vispruner_eval_dataset"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--mode", choices=["baseline", "concurrent"], default="concurrent",
                   help="baseline: run items strictly sequentially and record their outputs. "
                        "concurrent: launch --threads workers simultaneously and compare "
                        "against a previously recorded baseline.")
    p.add_argument("--threads", type=int, default=3,
                   help="Concurrent workers. 3 matches the original crashing run.")
    p.add_argument("--seed", type=int, default=1,
                   help="Selects WHICH items the concurrent run uses, from the baseline "
                        "pool. Varying it across repetitions varies prompt lengths and "
                        "therefore the interleaving, without making runs unreproducible.")
    p.add_argument("--pool-size", type=int, default=12,
                   help="How many dataset items the baseline covers.")
    p.add_argument("--fix", choices=["none", "threadlocal", "lock"], default="none",
                   help="none: reproduce as-is. threadlocal: make the timing state "
                        "thread-local (tests H1). lock: serialize generate() entirely "
                        "(tests whether a lock is even a viable answer).")
    p.add_argument("--visual-token-num", type=int, default=576,
                   help="576 = effectively-unpruned baseline. Matches csnbs/server.py.")
    p.add_argument("--important-ratio", type=float, default=0.5)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--min-new-tokens", type=int, default=16,
                   help="Forces enough decode steps that the threads' generate() calls "
                        "genuinely overlap. With a short natural EOS the first thread can "
                        "finish before the last one starts and the race never opens.")
    p.add_argument("--warmup", type=int, default=6,
                   help="Sequential, discarded (rule 3). A race that only appears while "
                        "the caching allocator is still growing is a different bug from "
                        "one in steady state, and you want to know which you have.")
    p.add_argument("--baseline", default="/tmp/phase0/baseline.json")
    p.add_argument("--out", default=None, help="Where to write the JSON result summary.")
    p.add_argument("--timing-file", default=None,
                   help="LLAVA_TIMING_FILE override. Defaults to a per-PID scratch path so "
                        "this NEVER appends to results/timing/llava_llama_timing.json. "
                        "Note H1c: within one process this file's writes are still "
                        "unlocked and can interleave. That is observed here, not fixed.")
    return p.parse_args()


args = parse_args()

# Point the model's built-in timing writer at scratch BEFORE importing/loading
# anything, so no code path can reach the tracked results file.
os.environ["LLAVA_TIMING_FILE"] = args.timing_file or f"/tmp/phase0/timing_pid{os.getpid()}.jsonl"

import torch  # noqa: E402  (must come after the env var is set)

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402
from PIL import Image  # noqa: E402


def git_commit():
    """Rule 7: an unattributable crash-reproduction rate is as useless as an
    unattributable benchmark."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def gpu_name():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Optional fixes, applied to the loaded model. Each one tests a hypothesis;
# neither is meant to ship as-is.
# ---------------------------------------------------------------------------

def install_threadlocal_timing(model):
    """Tests H1: give each thread its own copy of the timing state.

    The state lives on the instance (llava_llama.py:76-77) and is set/cleared
    per generate() call. We replace it with a data descriptor on the CLASS
    backed by a threading.local().

    Why a class-level property works despite nn.Module's custom __setattr__:
    nn.Module.__setattr__ special-cases Parameters/Modules/Buffers and then
    falls through to object.__setattr__, which DOES honour data descriptors
    on the class. And a data descriptor takes precedence over the instance
    __dict__ for both get and set -- so the stale entries __init__ left behind
    are shadowed. We pop them anyway so that anyone inspecting model.__dict__
    later isn't misled into thinking the old attributes are still live.
    """
    cls = type(model)
    tls = threading.local()

    def make_prop(name):
        def getter(self):
            return getattr(tls, name, None)   # None = "not inside generate()", the
                                              # same default __init__ used

        def setter(self, value):
            setattr(tls, name, value)

        return property(getter, setter)

    for name in ("_timing_forward_events", "_timing_prepare_inputs_calls"):
        model.__dict__.pop(name, None)
        setattr(cls, name, make_prop(name))


def install_generate_lock(model):
    """Tests decision option (b): serialize generate() with one lock.

    Note what this does NOT do: it does not make the model concurrent. It
    makes the crash go away by ensuring there is never more than one request
    in flight, which is what csnbs/server.py already achieves structurally.
    If this is the only thing that works, the honest conclusion is the
    roadmap's own rejection criterion -- "true concurrency requires batching,
    not just thread-safety."
    """
    lock = threading.Lock()
    orig_generate = model.generate

    def locked_generate(*a, **kw):
        with lock:
            return orig_generate(*a, **kw)

    model.generate = locked_generate
    return lock


# ---------------------------------------------------------------------------
# Setup: load once, build every request tensor UP FRONT.
# ---------------------------------------------------------------------------
print(f"[setup] mode={args.mode} threads={args.threads} fix={args.fix} "
      f"seed={args.seed} vtn={args.visual_token_num}", flush=True)

disable_torch_init()
tokenizer, model, image_processor, _ = load_pretrained_model(
    MODEL_PATH, None, "llava-v1.5-7b",
    visual_token_num=args.visual_token_num,
    important_ratio=args.important_ratio,
)
model.eval()

if args.fix == "threadlocal":
    install_threadlocal_timing(model)
elif args.fix == "lock":
    install_generate_lock(model)

with open(DATASET_PATH) as f:
    dataset = json.load(f)
pool = dataset[:args.pool_size]


def build_request(item):
    """PIL decode + preprocessing + H2D copy. Deliberately called on the MAIN
    thread only. If this ran inside a worker, host-side image work and host-to-
    device copies would sit inside the race window, and a crash could not be
    attributed to the model rather than to the copy."""
    image = Image.open(f"{IMAGES_ROOT}/{item['image']}").convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]
    images = image_tensor.unsqueeze(0).half().cuda()
    image_sizes = [image.size]

    qs = DEFAULT_IMAGE_TOKEN + "\n" + item["question"]
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    return input_ids, attention_mask, images, image_sizes


def generate_only(req):
    """The bare model call. No torch.cuda.synchronize() anywhere -- sync is
    device-wide, so a sync inside a worker would drain every OTHER thread's
    pipeline too and change the very interleaving we are trying to observe.
    No decode either: batch_decode forces a device-to-host copy; we hand the
    raw ids back and decode on the main thread after join.

    torch.inference_mode() is thread-local in torch, so every worker needs its
    own -- it cannot be hoisted to the main thread."""
    input_ids, attention_mask, images, image_sizes = req
    with torch.inference_mode():
        output_ids, visual_token_num_used = model.generate(
            input_ids, attention_mask=attention_mask, images=images, image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            use_cache=True,
        )
    return output_ids, visual_token_num_used


print(f"[setup] building {len(pool)} request tensors on the main thread", flush=True)
built = {item["question_id"]: build_request(item) for item in pool}

# Warm-up: sequential, discarded (rule 3).
print(f"[setup] warm-up: {args.warmup} sequential requests (discarded)", flush=True)
for i in range(args.warmup):
    generate_only(built[pool[i % len(pool)]["question_id"]])
torch.cuda.synchronize()

result = {
    "git_commit": git_commit(),
    "gpu": gpu_name(),
    "torch_version": torch.__version__,
    "transformers_version": __import__("transformers").__version__,
    "mode": args.mode,
    "threads": args.threads,
    "fix": args.fix,
    "seed": args.seed,
    "visual_token_num": args.visual_token_num,
    "important_ratio": args.important_ratio,
    "max_new_tokens": args.max_new_tokens,
    "min_new_tokens": args.min_new_tokens,
    "warmup": args.warmup,
    "timing_file": os.environ["LLAVA_TIMING_FILE"],
}


def finish(exit_code, extra):
    result.update(extra)
    result["exit_code"] = exit_code
    if args.out:
        d = os.path.dirname(args.out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[out] {args.out}", flush=True)
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# MODE: baseline -- strictly sequential, records the ground-truth answers.
# ---------------------------------------------------------------------------
if args.mode == "baseline":
    outputs = {}
    for item in pool:
        qid = item["question_id"]
        output_ids, vtn = generate_only(built[qid])
        torch.cuda.synchronize()
        outputs[str(qid)] = {
            "text": tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0],
            "n_output_tokens": int(output_ids.shape[1]),
            "visual_token_num_used": int(vtn),
            "question": item["question"],
            "category": item.get("category"),
        }
        print(f"  [{qid}] {outputs[str(qid)]['text'][:70]!r}", flush=True)

    d = os.path.dirname(args.baseline)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.baseline, "w") as fh:
        json.dump({"config": result, "outputs": outputs}, fh, indent=2)
    print(f"[baseline] wrote {len(outputs)} sequential outputs to {args.baseline}", flush=True)
    finish(0, {"n_baseline_items": len(outputs)})


# ---------------------------------------------------------------------------
# MODE: concurrent
# ---------------------------------------------------------------------------
if not os.path.exists(args.baseline):
    print(f"[fatal] no baseline at {args.baseline}. Run --mode baseline first.", file=sys.stderr)
    finish(3, {"error": "missing baseline"})

with open(args.baseline) as fh:
    baseline = json.load(fh)["outputs"]

# Deterministic per-seed item choice, drawn from the pool the baseline covers.
rng = random.Random(args.seed)
chosen = rng.sample(pool, min(args.threads, len(pool)))
while len(chosen) < args.threads:            # more threads than pool items: allow repeats
    chosen.append(rng.choice(pool))

# One pre-sized slot per worker. Each thread writes ONLY its own index.
# A shared dict here would be introducing our own race and would make the
# experiment uninterpretable.
slots = [None] * args.threads

# The single most important line in this file. Without it, thread 0 typically
# clears prefill before thread 2 is even scheduled, the calls serialize by
# accident, no race window ever opens, and you conclude "no crash" for the
# wrong reason.
barrier = threading.Barrier(args.threads)

t_ref = time.perf_counter()


def worker(slot, item):
    qid = item["question_id"]
    try:
        barrier.wait()                       # inside the try: BrokenBarrierError is a
                                             # result to record, not a stack trace to lose
        t0 = time.perf_counter()
        output_ids, vtn = generate_only(built[qid])
        slots[slot] = {
            "status": "ok", "question_id": qid,
            "t_start_s": round(t0 - t_ref, 4),
            "t_end_s": round(time.perf_counter() - t_ref, 4),
            "output_ids": output_ids,        # decoded on the main thread after join
            "visual_token_num_used": int(vtn),
        }
    except BaseException as exc:             # BaseException: a poisoned CUDA context can
                                             # surface in ways that are not Exception
        slots[slot] = {
            "status": "error", "question_id": qid,
            "t_error_s": round(time.perf_counter() - t_ref, 4),
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc(),
        }


print(f"[run] launching {args.threads} threads simultaneously on qids "
      f"{[i['question_id'] for i in chosen]}", flush=True)

threads = [threading.Thread(target=worker, args=(i, chosen[i]), name=f"w{i}")
           for i in range(args.threads)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Main-thread sync AFTER join. Guarded: if the context is poisoned this raises
# too, and that fact is itself part of the result.
sync_error = None
try:
    torch.cuda.synchronize()
except BaseException as exc:
    sync_error = f"{type(exc).__name__}: {str(exc)[:500]}"

errors = [s for s in slots if s and s["status"] == "error"]
per_slot = []
mismatches = []

for i, s in enumerate(slots):
    if s is None:                            # thread died before writing its slot
        per_slot.append({"slot": i, "status": "no_result"})
        errors.append({"error_type": "NoResult", "error": "worker wrote no slot"})
        continue
    entry = {k: v for k, v in s.items() if k != "output_ids"}
    entry["slot"] = i
    if s["status"] == "ok":
        # Decode here, on the main thread, once everything has landed.
        try:
            text = tokenizer.batch_decode(s["output_ids"], skip_special_tokens=True)[0]
        except BaseException as exc:
            text = None
            entry["decode_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        entry["text"] = text
        expected = baseline.get(str(s["question_id"]), {}).get("text")
        entry["expected"] = expected
        # "Didn't crash" is not "correct". A race that silently hands thread A's
        # answer to thread B is worse than one that crashes, because nothing
        # downstream would ever notice.
        entry["matches_baseline"] = (text is not None and text == expected)
        if not entry["matches_baseline"]:
            mismatches.append(entry)
    per_slot.append(entry)

# Classify the failure. A device-side assert is the specific thing Phase 0 is
# chasing; note that only the FIRST thread to trip it is informative -- the
# others report CUBLAS_STATUS_EXECUTION_FAILED because the context is already
# dead, which is collateral, not independent evidence.
blob = json.dumps([e.get("traceback", "") + e.get("error", "") for e in errors])
crash_kind = None
if "device-side assert" in blob or "Indexing.cu" in blob:
    crash_kind = "cuda_device_side_assert"
elif "CUBLAS_STATUS_EXECUTION_FAILED" in blob:
    crash_kind = "cublas_execution_failed_only"
elif errors:
    crash_kind = "other_exception"

print("\n=== RESULT " + "=" * 50)
for e in per_slot:
    if e["status"] == "ok":
        mark = "OK " if e.get("matches_baseline") else "MISMATCH"
        print(f"  slot {e['slot']} [{mark}] qid={e['question_id']} "
              f"{e['t_start_s']}s->{e['t_end_s']}s  {str(e.get('text'))[:60]!r}")
    else:
        print(f"  slot {e['slot']} [ERROR] {e.get('error_type')}: {str(e.get('error'))[:160]}")
if sync_error:
    print(f"  main-thread sync also failed: {sync_error}")
print(f"  crash_kind={crash_kind}  n_errors={len(errors)}  n_mismatches={len(mismatches)}")
print("=" * 61, flush=True)

extra = {
    "slots": per_slot,
    "n_errors": len(errors),
    "n_mismatches": len(mismatches),
    "crash_kind": crash_kind,
    "main_sync_error": sync_error,
    "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
}

if errors:
    finish(1, extra)
if mismatches:
    finish(2, extra)
finish(0, extra)
