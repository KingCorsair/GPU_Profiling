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
import csv
import json
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


def run_and_time(n_tokens: int, ratio: float, question_subset: Path | None) -> dict:
    """Launch model_vqa_science.py as a subprocess for one (n_tokens, ratio)
    setting, time the whole thing wall-clock, and return the load/gen split
    the subprocess reports via its timing sidecar.

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

    t_start = time.perf_counter()
    # cwd=VIS_PRUNER_DIR: the eval scripts use relative paths internally.
    subprocess.run(cmd, cwd=VIS_PRUNER_DIR, check=True)
    total_wall_s = time.perf_counter() - t_start

    timing = json.loads(timing_file.read_text())

    final_dict = {
        "question_count": timing["question_count"],
        "total_wall_s": total_wall_s,
        "model_load_s": timing["model_load_s"],
        "generation_s": timing["generation_s"],
    }

    with open("output.json","a") as f:
        json.dump(final_dict+"\n",f)

    return final_dict

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question-subset",
        type=Path,
        default=None,
        help=(
            "Path to a reduced question-file JSON to use instead of the full "
            "4241-question set. Undecided as of last discussion -- full set "
            "lets you cross-check accuracy reproduces; a subset is much "
            "cheaper for 7 full runs. Pass explicitly once you've picked."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Run-order shuffle seed")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "results/scienceqa_timing_sweep.csv",
        help="Where to append timing results",
    )
    args = parser.parse_args()

    plan = build_run_plan(seed=args.seed)
    print(f"Run plan ({len(plan)} runs, order seed={args.seed}):")
    for i, (n_tokens, ratio, repeat_idx) in enumerate(plan):
        print(f"  [{i}] n={n_tokens} r={ratio} repeat={repeat_idx}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists()

    with args.output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RunResult.__dataclass_fields__))
        if write_header:
            writer.writeheader()

        for order_index, (n_tokens, ratio, repeat_idx) in enumerate(plan):
            print(f"\n=== run {order_index + 1}/{len(plan)}: n={n_tokens} r={ratio} repeat={repeat_idx} ===")
            timing = run_and_time(n_tokens, ratio, args.question_subset)
            result = RunResult(
                order_index=order_index,
                n_tokens=n_tokens,
                ratio=ratio,
                repeat_idx=repeat_idx,
                **timing,
            )
            writer.writerow(asdict(result))
            f.flush()
            print(f"  total={result.total_wall_s:.1f}s  load={result.model_load_s:.1f}s  gen={result.generation_s:.1f}s")


if __name__ == "__main__":
    main()
