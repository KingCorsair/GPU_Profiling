#!/usr/bin/env python3
"""
Harness for the heterogeneous-set wall-clock timing sweep. Same shape as
time_sqa_sweep.py -- same SETTINGS idea, same randomized run order, same
whole-subprocess wall-clock measurement -- just pointed at
model_vqa_heterogeneous.py instead of model_vqa_science.py.

Kept as a separate script rather than parameterizing time_sqa_sweep.py: the
two eval scripts take different flags (question-file/image-folder defaults,
--top_p/--num_beams/--max_new_tokens exist here and not there) and dumping
that as a pile of if-eval-script-is-X branches would make the ScienceQA path
harder to read for no benefit -- nothing else uses this sweep logic.

dev.json is 90 questions, vs. ScienceQA's 4241 -- noise matters more here
(see AMAY_TIMING_NOTES.md), hence more repeats on baseline and no --limit
default of 5 for real runs.
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
DEFAULT_QUESTION_FILE = VIS_PRUNER_DIR / "vispruner_eval_dataset" / "dev.json"
# Image paths inside dev.json already start with "images/", so this points
# at the dataset root, not the images/ subdir -- matches
# model_vqa_heterogeneous.py's own --image-folder default.
DEFAULT_IMAGE_FOLDER = VIS_PRUNER_DIR / "vispruner_eval_dataset"
# Per-run answer/timing files, scratch only -- not a results artifact.
RUN_TMP_DIR = REPO_ROOT / "results" / "tmp_runs"

# (n_tokens, important_ratio, repeat_count) -- same n_tokens grid as the
# ScienceQA sweep so the two are comparable. More repeats on baseline than
# there: 90 questions is a much smaller set to average noise over than 4241.
SETTINGS = [
    (576, 0.5, 5),   # baseline
    (64, 0.5, 3),    # smallest run -> most exposed to fixed-magnitude noise
    (128, 0.5, 2),
    (144, 0.5, 2),
    (288, 0.5, 2),
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


def run_and_time(n_tokens: int, ratio: float, question_file: Path, image_folder: Path, limit: int) -> dict:
    """Launch model_vqa_heterogeneous.py as a subprocess for one
    (n_tokens, ratio) setting, time the whole thing wall-clock, and return
    the load/data/generation split the subprocess reports via its timing
    sidecar. Mirrors time_sqa_sweep.py's run_and_time() -- see the docstring
    there for why whole-subprocess perf_counter is exempt from the
    "CUDA events, not time.time()" rule.
    """
    RUN_TMP_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"het_n{n_tokens}_r{ratio}_{time.perf_counter_ns()}"
    answers_file = RUN_TMP_DIR / f"{run_id}.jsonl"
    timing_file = RUN_TMP_DIR / f"{run_id}.timing.json"
    # Per-question detail from inside llava_llama.py -- picked up via
    # LLAVA_TIMING_FILE so this run's questions don't mix with any other
    # run's (or with a directly-run model_vqa_heterogeneous.py's default).
    llava_timing_file = RUN_TMP_DIR / f"{run_id}.llava_timing.jsonl"

    cmd = [
        sys.executable, "-m", "llava.eval.model_vqa_heterogeneous",
        "--model-path", str(MODEL_PATH),
        "--question-file", str(question_file),
        "--image-folder", str(image_folder),
        "--answers-file", str(answers_file),
        "--visual_token_num", str(n_tokens),
        "--important_ratio", str(ratio),
        "--temperature", "0",
        "--conv-mode", "llava_v1",
        "--timing-file", str(timing_file),
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]

    subprocess_env = {**os.environ, "LLAVA_TIMING_FILE": str(llava_timing_file)}

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # cwd=VIS_PRUNER_DIR: the eval scripts use relative paths internally.
    subprocess.run(cmd, cwd=VIS_PRUNER_DIR, check=True, env=subprocess_env)

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    total_wall_s = end_time - start_time

    with open(question_file, "r") as f:
        question_count = len(json.load(f))
    if limit > 0:
        question_count = min(question_count, limit)

    with open(timing_file, "r") as tf:
        tf_dict = json.load(tf)

    print(json.dumps(tf_dict, indent=4))

    questions = []
    if llava_timing_file.exists():
        with open(llava_timing_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))

    return {
        "question_count": question_count,
        "total_wall_s": total_wall_s,
        "model_load_s": tf_dict["model_load_s"],
        "generation_s": tf_dict["generation_s"],
        "time_to_make_questions": tf_dict["time_to_make_questions"],
        "questions": questions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question-file", type=Path, default=DEFAULT_QUESTION_FILE,
        help="Defaults to the 90-question heterogeneous dev set.",
    )
    parser.add_argument(
        "--image-folder", type=Path, default=DEFAULT_IMAGE_FOLDER,
    )
    parser.add_argument(
        "--limit", type=int, default=5,
        help=(
            "Only run the first N questions per setting -- for fast "
            "dev/sanity sweeps. These numbers are not reportable. Pass "
            "--limit 0 for a real, unlimited run."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Run-order shuffle seed")
    parser.add_argument(
        "-o", "--output", type=Path,
        default=REPO_ROOT / "results/heterogeneous_timing_sweep.jsonl",
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
            timing = run_and_time(n_tokens, ratio, args.question_file, args.image_folder, args.limit)
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
