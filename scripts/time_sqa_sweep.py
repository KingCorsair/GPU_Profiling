#!/usr/bin/env python3
"""
Harness for the ScienceQA wall-clock timing sweep (accuracy vs. speed).

This is structure only. It owns: which settings run, in what order, how many
repeats, and where results get logged. It does NOT own: how a single run is
timed. That's `run_and_time()` below -- fill it in yourself, per the design
we worked through:

  - whole-subprocess wall time, monotonic clock (time.perf_counter, not
    time.time -- perf_counter can't be skewed by a clock adjustment mid-run)
  - split out model-load time from generation time. That means
    vis_pruner_copy/llava/eval/model_vqa_science.py needs to emit a
    checkpoint (e.g. a sentinel line to stdout, or a small JSON sidecar
    file) right after the model finishes loading and before the question
    loop starts. That instrumentation is yours to write too -- it's the
    same "core timing code" boundary.

Run order is interleaved/randomized on purpose: if anything drifts over the
session (thermal state, whatever else is on the box), a fixed run order would
confound that drift with the setting being tested.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
VIS_PRUNER_DIR = REPO_ROOT / "vis_pruner_copy"

CKPT_DIR = VIS_PRUNER_DIR / "checkpoints"
CKPT = "llava-v1.5-7b"
MODEL_PATH = CKPT_DIR / CKPT
DATA_DIR = REPO_ROOT / "ScienceQA" / "data"
IMAGE_FOLDER = DATA_DIR / "scienceqa" / "images" / "test"
DEFAULT_QUESTION_FILE = (
    VIS_PRUNER_DIR / "playground/data/eval/scienceqa/llava_test_CQM-A.json"
)
# Per-run answer/timing files, scratch only -- not a results artifact.
RUN_TMP_DIR = REPO_ROOT / "results" / "tmp_runs"

# (n_tokens, important_ratio, repeat_count)
# n=576 is the unpruned baseline (see llava_arch.py -- visual_token_num == N
# means index_masks stays all-True, nothing gets dropped). It gets the most
# repeats because everything else is measured relative to it.
SETTINGS = [
    (576, 0.5, 3),   # baseline
    (64, 0.5, 2),    # smallest run -> most exposed to fixed-magnitude noise
    (128, 0.5, 1),
    (144, 0.5, 1),
    (288, 0.5, 1),
]


@dataclass
class RunResult:
    order_index: int
    n_tokens: int
    ratio: float
    repeat_idx: int
    question_count: int
    total_wall_s: float
    model_load_s: float
    generation_s: float
    time_to_make_questions: float
    # Per-question detail from llava_llama.py (multimodal_prep_time_generate,
    # generate_time, and the nested per-forward-call/per-decode-step timings)
    # -- one dict per question, read back from the subprocess's scratch file.
    questions: list


def build_run_plan(settings=SETTINGS, seed=0) -> list[tuple[int, float, int]]:
    """Expand (n, r, repeats) into individual (n, r, repeat_idx) runs and
    shuffle the order so repeats of the same setting aren't back-to-back
    and settings aren't run in a fixed sequence."""
    plan = []
    for n_tokens, ratio, repeats in settings:
        for repeat_idx in range(repeats):
            plan.append((n_tokens, ratio, repeat_idx))
    random.Random(seed).shuffle(plan)
    print(plan)
    return plan


def run_and_time(n_tokens: int, ratio: float, question_subset: Path | None, limit: int) -> dict:
    """Launch model_vqa_science.py as a subprocess for one (n_tokens, ratio)
    setting, time the whole thing wall-clock, and return the load/gen split
    the subprocess reports via its timing sidecar.

    `limit` is passed straight through to model_vqa_science.py's own
    `--limit` flag, which slices the question list in-memory after loading
    it -- no separate subset file needed. 0 means no limit (full file).

    Whole-subprocess wall time is exempt from the "CUDA events, not
    time.time()" rule (see CLAUDE.md) -- by the time the process exits, all
    its kernels have completed. perf_counter is still used because it's
    monotonic. Only this process's perf_counter clock is used for
    total_wall_s; model_load_s/generation_s come entirely from perf_counter
    deltas taken inside the child, so no cross-process clock comparison
    happens anywhere.
    """
    question_file = question_subset if question_subset is not None else DEFAULT_QUESTION_FILE

    RUN_TMP_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"n{n_tokens}_r{ratio}_{time.perf_counter_ns()}"
    answers_file = RUN_TMP_DIR / f"{run_id}.jsonl"
    timing_file = RUN_TMP_DIR / f"{run_id}.timing.json"
    # Per-question detail from inside llava_llama.py -- llava_llama.py picks
    # this path up via LLAVA_TIMING_FILE instead of its own hardcoded default,
    # so this one run's questions don't get mixed in with any other run's.
    llava_timing_file = RUN_TMP_DIR / f"{run_id}.llava_timing.jsonl"

    cmd = [
        sys.executable, "-m", "llava.eval.model_vqa_science",
        "--model-path", str(MODEL_PATH),
        "--question-file", str(question_file),
        "--image-folder", str(IMAGE_FOLDER),
        "--answers-file", str(answers_file),
        "--visual_token_num", str(n_tokens),
        "--important_ratio", str(ratio),
        "--single-pred-prompt",
        "--temperature", "0",
        "--conv-mode", "vicuna_v1",
        "--timing-file", str(timing_file),
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]

    subprocess_env = {**os.environ, "LLAVA_TIMING_FILE": str(llava_timing_file)}

    #start the timing for the subprocess
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # cwd=VIS_PRUNER_DIR: the eval scripts use relative paths internally.
    subprocess.run(cmd, cwd=VIS_PRUNER_DIR, check=True, env=subprocess_env)

    #end the timing for the subprocess
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    #time the subprocess
    total_wall_s = end_time - start_time

    #open the questions subset file if it exists
    if question_subset is not None:
        with open(question_subset,"r") as f:
            data = json.load(f)
        #count the number of
        question_count = len(data)
    else:
        question_count = 4241
    if limit > 0:
        question_count = min(question_count, limit)

    with open(timing_file,"r+") as tf:
        tf_dict = json.load(tf)
    
    print(json.dumps(tf_dict, indent = 4))

    model_load_s = tf_dict["model_load_s"]
    generation_s = tf_dict["generation_s"]
    time_to_make_questions = tf_dict["time_to_make_questions"]

    # Read back the per-question detail llava_llama.py wrote for this run
    # (one JSON object per line, one line per question) and fold it in.
    questions = []
    if llava_timing_file.exists():
        with open(llava_timing_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))

    final_dict = {
        "question_count": question_count,
        "total_wall_s": total_wall_s,
        "model_load_s": model_load_s,
        "generation_s": generation_s,
        "time_to_make_questions": time_to_make_questions,
        "questions": questions,
    }

    return final_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question-subset",
        type=Path,
        default=None,
        help="Path to a reduced question-file JSON to use instead of the full 4241-question set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help=(
            "Only run the first N questions per setting (sliced in-memory by "
            "model_vqa_science.py, no file needed) -- for fast dev/sanity "
            "sweeps. These numbers are not reportable. Pass --limit 0 for a "
            "real, unlimited run."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Run-order shuffle seed")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "results/scienceqa_timing_sweep.jsonl",
        help=(
            "Where to append timing results -- one JSON object per line, "
            "one line per (n_tokens, ratio, repeat) run, each with a nested "
            "'questions' list holding that run's per-question detail. This "
            "is the only file this script writes."
        ),
    )
    args = parser.parse_args()

    plan = build_run_plan(seed=args.seed)
    print(f"Run plan ({len(plan)} runs, order seed={args.seed}):")
    for i, (n_tokens, ratio, repeat_idx) in enumerate(plan):
        print(f"  [{i}] n={n_tokens} r={ratio} repeat={repeat_idx}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as f:
        for order_index, (n_tokens, ratio, repeat_idx) in enumerate(plan):
            print(f"\n=== run {order_index + 1}/{len(plan)}: n={n_tokens} r={ratio} repeat={repeat_idx} ===")
            timing = run_and_time(n_tokens, ratio, args.question_subset, args.limit)
            result = RunResult(
                order_index=order_index,
                n_tokens=n_tokens,
                ratio=ratio,
                repeat_idx=repeat_idx,
                **timing,
            )
            json.dump(asdict(result), f)
            f.write("\n")
            f.flush()
            print(f"  total={result.total_wall_s:.1f}s  load={result.model_load_s:.1f}s  gen={result.generation_s:.1f}s")


if __name__ == "__main__":
    main()