"""WP2 Layer-2 OCR gate: TextVQA accuracy with the CLIP vision tower in FP32 vs
FP16 (AMAY_ENGINEERING_ROADMAP.md, WP2 section 8).

A fixed, seeded subsample of eval/textvqa/llava_textvqa_val_v051_ocr.jsonl,
drawn after removing every question whose image is in the project's locked
test set (rule 11: this gate decides whether the change is kept, so it must
not look at test images). Answered by both towers in one process (two inner CLIPVisionModel copies
swapped into the tower wrapper, as in scripts/diag_wp2_fp16_jitter.py), so
every question is a matched FP32/FP16 pair. Prompting and generation follow
vis_pruner_copy/llava/eval/model_vqa_loader.py as called by
scripts/v1_5/eval/textvqa.sh: vicuna_v1, greedy, max_new_tokens=128. Scoring is
the vendored official scorer, llava/eval/m4c_evaluator.py, unchanged.

The roadmap's gate is the unpruned config (576). 128 is run as well because
Layer 1 showed the pruner selects different tokens under FP16.

Per config it reports both accuracies, the paired difference with a bootstrap
95% interval, and how many answers and scores changed. The answers are also
written in the official format, one file per config and arm, so
`python -m llava.eval.eval_textvqa` can re-score them independently.

Data (not in git): TextVQA_0.5.1_val.json and train_images/ under
/workspace/datasets/textvqa. scripts/download_textvqa_subset.py fetches the
annotations and exactly the images this subsample needs.

  python scripts/wp2_layer2_textvqa.py --run-id wp2_layer2_1
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image

REPO = "/workspace/GPU_Profiling"
sys.path.insert(0, f"{REPO}/vis_pruner_copy")
sys.path.insert(0, f"{REPO}/scripts")

from bench_dev import (  # noqa: E402
    IMPORTANT_RATIO,
    TORCH_CPU_THREADS,
    git_dirty,
    git_dirty_files,
    other_gpu_processes,
)
from diag_wp2_fp16_jitter import ARMS, load_inner_towers  # noqa: E402
from run_wp1_controlled_benchmark import MODEL_PATH, git_commit  # noqa: E402
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.eval.eval_textvqa import prompt_processor  # noqa: E402
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402

QUESTION_FILE = f"{REPO}/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
LOCKED_TEST_SET = f"{REPO}/vis_pruner_copy/vispruner_eval_dataset/test.json"
DATA_DIR = "/workspace/datasets/textvqa"
CONV_MODE = "vicuna_v1"
MAX_NEW_TOKENS = 128
N_BOOTSTRAP = 10_000
DEFAULT_N = 1000
DEFAULT_SEED = 20261002

OUT_DIR = f"{REPO}/results/wp2_correctness"


def choose_questions(n, seed):
    """The fixed subsample: (all questions, eligible indices, chosen indices)."""
    with open(QUESTION_FILE) as f:
        questions = [json.loads(line) for line in f]
    # Only the image names are read from the locked test set, nothing else.
    with open(LOCKED_TEST_SET) as f:
        test_images = {os.path.basename(r["image"]).removeprefix("textvqa_")
                       for r in json.load(f) if r["source_dataset"] == "TEXT_VQA"}
    eligible = [i for i, q in enumerate(questions) if q["image"] not in test_images]
    chosen = sorted(random.Random(seed).sample(eligible, n))
    return questions, eligible, chosen


def build_request(line, image_folder, tokenizer, model, image_processor):
    """Same construction as model_vqa_loader.CustomDataset.__getitem__."""
    qs = DEFAULT_IMAGE_TOKEN + "\n" + line["text"]
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    image = Image.open(os.path.join(image_folder, line["image"])).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]
    input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors="pt")
    return (input_ids.unsqueeze(0).cuda(),
            image_tensor.unsqueeze(0).to(dtype=torch.float16, device="cuda"),
            [image.size])


def answer(model, tokenizer, request):
    input_ids, images, image_sizes = request
    with torch.inference_mode():
        output_ids, _ = model.generate(
            input_ids, images=images, image_sizes=image_sizes,
            do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def per_question_scores(evaluator, texts, gt_answers):
    """The official scorer's per-question step, kept per question so the two
    arms can be compared as pairs. Checked against eval_pred_list below."""
    scores = []
    for text, gt in zip(texts, gt_answers):
        unique_answer_scores = evaluator._compute_answer_scores(gt)
        scores.append(unique_answer_scores.get(evaluator.answer_processor(text), 0.0))
    official = evaluator.eval_pred_list(
        [{"pred_answer": t, "gt_answers": g} for t, g in zip(texts, gt_answers)])
    assert abs(official - sum(scores) / len(scores)) < 1e-9, "per-question scores disagree with the scorer"
    return scores


def paired_bootstrap_ci(diffs, seed):
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(N_BOOTSTRAP, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = f"{OUT_DIR}/{args.run_id}.json"
    out_table = f"{OUT_DIR}/{args.run_id}.txt"
    if os.path.exists(out_json):
        sys.exit(f"{out_json} already exists. Pick a new --run-id.")
    procs = other_gpu_processes()
    if procs:
        sys.exit("Other processes are on the GPU (rule 9):\n  " + "\n  ".join(procs))

    image_folder = f"{DATA_DIR}/train_images"
    with open(f"{DATA_DIR}/TextVQA_0.5.1_val.json") as f:
        annotations = {(a["image_id"], a["question"].lower()): a for a in json.load(f)["data"]}
    questions, eligible, chosen = choose_questions(args.n, args.seed)
    if args.limit:
        chosen = chosen[:args.limit]
    lines = [questions[i] for i in chosen]
    gt_answers = [annotations[(q["question_id"], prompt_processor(q["text"]))]["answers"]
                  for q in lines]
    missing = [q["image"] for q in lines if not os.path.exists(os.path.join(image_folder, q["image"]))]
    if missing:
        sys.exit(f"{len(missing)} images missing from {image_folder}, e.g. {missing[:3]}")

    # The model appends a timing line per generate(); keep it out of the tracked file.
    os.environ["LLAVA_TIMING_FILE"] = "/dev/null"
    torch.set_num_threads(TORCH_CPU_THREADS)
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, "llava-v1.5-7b",
        visual_token_num=args.configs[0], important_ratio=IMPORTANT_RATIO,
    )
    model.eval()
    tower = model.get_model().get_vision_tower()
    inner = load_inner_towers(tower)
    tower.vision_tower = None  # drop the checkpoint-loaded copy; the arms replace it
    torch.cuda.empty_cache()

    answer_files = {(cfg, arm): open(f"{OUT_DIR}/{args.run_id}_textvqa_vtn{cfg}_{arm}.jsonl", "x")
                    for cfg in args.configs for arm in ARMS}
    texts = {key: [] for key in answer_files}
    for n, line in enumerate(lines):
        request = build_request(line, image_folder, tokenizer, model, image_processor)
        for cfg in args.configs:
            model.visual_token_num = cfg
            for arm in ARMS:
                tower.vision_tower = inner[arm]
                text = answer(model, tokenizer, request)
                texts[(cfg, arm)].append(text)
                out = answer_files[(cfg, arm)]
                out.write(json.dumps({"question_id": line["question_id"], "prompt": line["text"],
                                      "text": text, "model_id": "llava-v1.5-7b",
                                      "metadata": {"visual_token_num": cfg, "vision_tower": arm}}) + "\n")
                out.flush()
        if n % 50 == 0 or n == len(lines) - 1:
            print(f"  [{n + 1}/{len(lines)}] {line['question_id']}", flush=True)
    for out in answer_files.values():
        out.close()

    evaluator = TextVQAAccuracyEvaluator()
    summary, table = {}, []
    for cfg in args.configs:
        scores = {arm: per_question_scores(evaluator, texts[(cfg, arm)], gt_answers) for arm in ARMS}
        diffs = [b - a for a, b in zip(scores["fp32"], scores["fp16"])]
        low, high = paired_bootstrap_ci(diffs, args.seed)
        s = {
            "n": len(lines),
            "accuracy_fp32": sum(scores["fp32"]) / len(lines),
            "accuracy_fp16": sum(scores["fp16"]) / len(lines),
            "difference_fp16_minus_fp32": sum(diffs) / len(lines),
            "difference_ci95": [low, high],
            "answers_changed": sum(a != b for a, b in zip(texts[(cfg, "fp32")], texts[(cfg, "fp16")])),
            "scores_changed": sum(d != 0 for d in diffs),
            "fp16_scored_higher": sum(d > 0 for d in diffs),
            "fp16_scored_lower": sum(d < 0 for d in diffs),
            # A regression outside noise: the whole interval sits below zero.
            "regression_outside_noise": high < 0,
        }
        summary[str(cfg)] = s
        table += [
            f"vtn={cfg}: FP32 {100 * s['accuracy_fp32']:.2f}%   FP16 {100 * s['accuracy_fp16']:.2f}%   "
            f"difference {100 * s['difference_fp16_minus_fp32']:+.2f} pp "
            f"(paired bootstrap 95% interval {100 * low:+.2f} to {100 * high:+.2f} pp)",
            f"  answers changed: {s['answers_changed']}/{s['n']}; scores changed: {s['scores_changed']} "
            f"({s['fp16_scored_higher']} higher with FP16, {s['fp16_scored_lower']} lower)",
            f"  {'REGRESSION OUTSIDE NOISE' if s['regression_outside_noise'] else 'no regression outside noise'}",
        ]

    dirty_files = git_dirty_files()
    report = {
        "metadata": {
            "run_id": args.run_id,
            "purpose": "WP2 Layer-2 TextVQA gate, FP32 vs FP16 vision tower",
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "git_dirty": git_dirty(dirty_files),
            "git_dirty_files": dirty_files,
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "arms": {a: str(d) for a, d in ARMS.items()},
            "configs": args.configs,
            "important_ratio": IMPORTANT_RATIO,
            "conv_mode": CONV_MODE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "question_file": QUESTION_FILE,
            "questions_excluded_as_locked_test_images": len(questions) - len(eligible),
            "subsample_seed": args.seed,
            "subsample_size": len(lines),
            "subsample_question_indices": chosen,
            "scorer": "llava/eval/m4c_evaluator.py TextVQAAccuracyEvaluator",
            "n_bootstrap": N_BOOTSTRAP,
        },
        "summary": summary,
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=1)

    m = report["metadata"]
    header = (f"WP2 Layer-2 TextVQA '{args.run_id}' -- {m['gpu']}  git={m['git_commit'][:10]}"
              f"{' (DIRTY)' if m['git_dirty'] else ''}\n"
              f"{len(lines)} questions (seed {args.seed}) of {len(eligible)} eligible "
              f"({len(questions) - len(eligible)} on locked-test images excluded), greedy, "
              f"max_new_tokens={MAX_NEW_TOKENS}, important_ratio={IMPORTANT_RATIO}; "
              f"official m4c scorer; FP32 and FP16 towers in one process\n")
    text = header + "\n" + "\n".join(table) + "\n"
    with open(out_table, "w") as f:
        f.write(text)
    print("\n" + text)
    print(f"Wrote {out_json}\n      {out_table}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="subsample size")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--configs", type=int, nargs="+", default=[576, 128])
    parser.add_argument("--limit", type=int, default=None, help="first N of the subsample (smoke test)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
