"""WP2 Layer-1 correctness: did loading the CLIP vision tower in FP16 change
what the model does? (AMAY_ENGINEERING_ROADMAP.md, WP2 section 8.)

One process, one LLaVA model, two inner CLIPVisionModel copies loaded from the
same checkpoint (float32 and float16), swapped into the tower wrapper per
request -- the same arrangement as scripts/diag_wp2_fp16_jitter.py. The FP32
arm is the pre-WP2 state (e295fd0), the FP16 arm is main (caf19fe).

Checks, over all 90 dev images:

  1. Exact-text match, greedy, natural-length output, at 576 and 128.
     FP32 answer vs FP16 answer.                      Bar: >= 85% identical.
  2. Token-selection equivalence at 128: the index mask encode_images()
     produces, FP32 vs FP16. Also the attention-ranked half on its own, to
     show whether a difference comes from the ranking or the diversity stage.
                                                      Report the match rate.
  3. Cosine similarity of the pre-projector image features the tower wrapper
     returns, FP32 vs FP16.                           Bar: > 0.999.
  4. Determinism: the FP16 arm run twice.             Bar: identical output ids.

Not a benchmark: nothing here is timed.

  python scripts/wp2_layer1_correctness.py --run-id wp2_layer1_1
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch

REPO = "/workspace/GPU_Profiling"
sys.path.insert(0, f"{REPO}/vis_pruner_copy")
sys.path.insert(0, f"{REPO}/scripts")

from bench_dev import (  # noqa: E402
    IMPORTANT_RATIO,
    N_IMAGES,
    TORCH_CPU_THREADS,
    git_dirty,
    git_dirty_files,
    other_gpu_processes,
)
from diag_wp2_fp16_jitter import ARMS, load_inner_towers  # noqa: E402
from run_wp1_controlled_benchmark import (  # noqa: E402
    DATASET_PATH,
    MODEL_PATH,
    build_request,
    git_commit,
)
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402

CONFIGS = [576, 128]
SELECTION_CONFIG = 128      # at 576 nothing is pruned, so there is no selection to compare
MAX_NEW_TOKENS = 64         # same as csnbs/server.py
EXACT_TEXT_BAR = 0.85
COSINE_BAR = 0.999

OUT_DIR = f"{REPO}/results/wp2_correctness"


def generate(model, tokenizer, request):
    input_ids, attention_mask, images, image_sizes = request
    with torch.inference_mode():
        output_ids, _ = model.generate(
            input_ids, attention_mask=attention_mask, images=images,
            image_sizes=image_sizes, do_sample=False, use_cache=True,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return output_ids[0].tolist(), text


def tower_outputs(model, tower, images):
    """What the tower wrapper hands to the rest of the model, plus the mask
    encode_images() derives from it at the current visual_token_num."""
    with torch.inference_mode():
        features, attentions = tower(images, output_attentions=True)
        _, index_mask = model.encode_images(images)
    important = int(model.get_visual_token_num() * model.get_important_ratio())
    top_by_attention = attentions.mean(dim=1).argsort(dim=-1, descending=True)[0, :important]
    return {
        "features": features[0].float().cpu(),
        "mask": index_mask[0].cpu(),
        "top_by_attention": set(top_by_attention.tolist()),
    }


def pct(part, whole):
    return 100.0 * part / whole if whole else float("nan")


def run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = f"{OUT_DIR}/{args.run_id}.json"
    out_table = f"{OUT_DIR}/{args.run_id}.txt"
    if os.path.exists(out_json):
        sys.exit(f"{out_json} already exists. Pick a new --run-id.")
    procs = other_gpu_processes()
    if procs:
        sys.exit("Other processes are on the GPU (rule 9):\n  " + "\n  ".join(procs))

    # The model appends a timing line per generate(); keep it out of the tracked file.
    os.environ["LLAVA_TIMING_FILE"] = "/dev/null"
    torch.set_num_threads(TORCH_CPU_THREADS)
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, "llava-v1.5-7b",
        visual_token_num=CONFIGS[0], important_ratio=IMPORTANT_RATIO,
    )
    model.eval()
    tower = model.get_model().get_vision_tower()
    shipped_dtype = str(tower.dtype)
    inner = load_inner_towers(tower)
    tower.vision_tower = None  # drop the checkpoint-loaded copy; the arms replace it
    torch.cuda.empty_cache()

    with open(DATASET_PATH) as f:
        items = json.load(f)
    if args.limit:
        items = items[:args.limit]

    rows = []
    for n, item in enumerate(items):
        request = build_request(item, tokenizer, model, image_processor)
        images = request[2]
        row = {"question_id": item["question_id"], "category": item["category"],
               "question": item["question"], "reference_answers": item.get("answers"),
               "configs": {}}

        model.visual_token_num = SELECTION_CONFIG
        seen = {}
        for arm in ARMS:
            tower.vision_tower = inner[arm]
            seen[arm] = tower_outputs(model, tower, images)
        a, b = seen["fp32"], seen["fp16"]
        per_token = torch.nn.functional.cosine_similarity(a["features"], b["features"], dim=-1)
        kept = int(a["mask"].sum())
        row["features"] = {
            "cosine_whole_map": float(torch.nn.functional.cosine_similarity(
                a["features"].flatten(), b["features"].flatten(), dim=0)),
            "cosine_per_token_min": float(per_token.min()),
            "cosine_per_token_mean": float(per_token.mean()),
        }
        row["selection"] = {
            "kept": kept,
            "kept_fp16": int(b["mask"].sum()),
            "mask_identical": bool(torch.equal(a["mask"], b["mask"])),
            "mask_overlap": int((a["mask"] & b["mask"]).sum()),
            "attention_half_size": len(a["top_by_attention"]),
            "attention_half_overlap": len(a["top_by_attention"] & b["top_by_attention"]),
        }

        for cfg in CONFIGS:
            model.visual_token_num = cfg
            out = {}
            for label, arm in (("fp32", "fp32"), ("fp16", "fp16"), ("fp16_repeat", "fp16")):
                tower.vision_tower = inner[arm]
                ids, text = generate(model, tokenizer, request)
                out[label] = {"output_ids": ids, "text": text}
            row["configs"][str(cfg)] = {
                "fp32_text": out["fp32"]["text"],
                "fp16_text": out["fp16"]["text"],
                "text_identical": out["fp32"]["text"] == out["fp16"]["text"],
                "ids_identical": out["fp32"]["output_ids"] == out["fp16"]["output_ids"],
                "fp16_deterministic": out["fp16"]["output_ids"] == out["fp16_repeat"]["output_ids"],
                "fp16_output_tokens": len(out["fp16"]["output_ids"]),
            }
        rows.append(row)
        if n % 10 == 0 or n == len(items) - 1:
            print(f"  [{n + 1}/{len(items)}] {item['question_id']}")

    # --- summary ---
    lines = []
    summary = {"exact_text": {}, "determinism": {}}
    categories = sorted({r["category"] for r in rows})
    for cfg in CONFIGS:
        cells = [r["configs"][str(cfg)] for r in rows]
        same = sum(c["text_identical"] for c in cells)
        deterministic = sum(c["fp16_deterministic"] for c in cells)
        by_category = {
            cat: [sum(r["configs"][str(cfg)]["text_identical"] for r in rows if r["category"] == cat),
                  sum(1 for r in rows if r["category"] == cat)]
            for cat in categories
        }
        summary["exact_text"][str(cfg)] = {
            "identical": same, "n": len(cells), "fraction": same / len(cells),
            "pass": same / len(cells) >= EXACT_TEXT_BAR, "by_category": by_category,
        }
        summary["determinism"][str(cfg)] = {
            "identical": deterministic, "n": len(cells), "pass": deterministic == len(cells),
        }
        lines.append(f"1. Exact text, FP32 vs FP16, vtn={cfg}: {same}/{len(cells)} identical "
                     f"({pct(same, len(cells)):.1f}%)  bar >= {EXACT_TEXT_BAR:.0%}  "
                     f"{'PASS' if summary['exact_text'][str(cfg)]['pass'] else 'FAIL'}")
        lines.append("     by category: " + "  ".join(
            f"{cat} {k}/{n}" for cat, (k, n) in by_category.items()))

    def selection_summary(subset):
        kept = sum(r["selection"]["kept"] for r in subset)
        half = sum(r["selection"]["attention_half_size"] for r in subset)
        return {
            "n": len(subset),
            "mask_identical": sum(r["selection"]["mask_identical"] for r in subset),
            "mask_overlap_fraction": sum(r["selection"]["mask_overlap"] for r in subset) / kept,
            "mask_overlap_min": min(r["selection"]["mask_overlap"] for r in subset),
            "attention_half_overlap_fraction":
                sum(r["selection"]["attention_half_overlap"] for r in subset) / half,
        }

    summary["selection"] = {"all": selection_summary(rows),
                            "bench_dev_images": selection_summary(rows[:N_IMAGES])}
    for label, s in (("all dev images", summary["selection"]["all"]),
                     (f"first {N_IMAGES} (the bench_dev set)", summary["selection"]["bench_dev_images"])):
        lines.append(f"2. Token selection at vtn={SELECTION_CONFIG}, {label}: "
                     f"{s['mask_identical']}/{s['n']} images keep the identical token set; "
                     f"{100 * s['mask_overlap_fraction']:.2f}% of kept tokens shared "
                     f"(worst image {s['mask_overlap_min']}/{SELECTION_CONFIG}); "
                     f"attention-ranked half {100 * s['attention_half_overlap_fraction']:.2f}% shared")

    whole = [r["features"]["cosine_whole_map"] for r in rows]
    token_min = [r["features"]["cosine_per_token_min"] for r in rows]
    summary["features"] = {
        "cosine_whole_map_min": min(whole),
        "cosine_whole_map_mean": sum(whole) / len(whole),
        "cosine_per_token_min": min(token_min),
        "images_with_a_token_below_bar": sum(1 for v in token_min if v <= COSINE_BAR),
        "pass": min(whole) > COSINE_BAR,
    }
    f = summary["features"]
    lines.append(f"3. Pre-projector features, cosine FP32 vs FP16, per image over the whole 576-token map: "
                 f"min {f['cosine_whole_map_min']:.6f}, mean {f['cosine_whole_map_mean']:.6f}  "
                 f"bar > {COSINE_BAR}  {'PASS' if f['pass'] else 'FAIL'}")
    lines.append(f"     single worst token across all images: {f['cosine_per_token_min']:.6f} "
                 f"({f['images_with_a_token_below_bar']} images have a token at or below the bar)")
    for cfg in CONFIGS:
        d = summary["determinism"][str(cfg)]
        lines.append(f"4. Determinism, FP16 run twice, vtn={cfg}: {d['identical']}/{d['n']} "
                     f"identical output ids  {'PASS' if d['pass'] else 'FAIL'}")

    lines.append("")
    lines.append("Answers that differ (FP32 -> FP16):")
    for cfg in CONFIGS:
        for r in rows:
            c = r["configs"][str(cfg)]
            if not c["text_identical"]:
                lines.append(f"  vtn={cfg} {r['question_id']} [{r['category']}] "
                             f"{c['fp32_text']!r} -> {c['fp16_text']!r}")

    dirty_files = git_dirty_files()
    report = {
        "metadata": {
            "run_id": args.run_id,
            "purpose": "WP2 Layer-1 correctness, FP32 vs FP16 vision tower",
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "git_dirty": git_dirty(dirty_files),
            "git_dirty_files": dirty_files,
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "shipped_vision_tower_dtype": shipped_dtype,
            "arms": {a: str(d) for a, d in ARMS.items()},
            "configs": CONFIGS,
            "selection_config": SELECTION_CONFIG,
            "important_ratio": IMPORTANT_RATIO,
            "max_new_tokens": MAX_NEW_TOKENS,
            "dataset_path": DATASET_PATH,
            "n_images": len(rows),
        },
        "summary": summary,
        "rows": rows,
    }
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=1)

    m = report["metadata"]
    header = (f"WP2 Layer-1 correctness '{args.run_id}' -- {m['gpu']}  git={m['git_commit'][:10]}"
              f"{' (DIRTY)' if m['git_dirty'] else ''}\n"
              f"{m['n_images']} dev images, greedy, max_new_tokens={MAX_NEW_TOKENS}, "
              f"important_ratio={IMPORTANT_RATIO}; FP32 and FP16 towers in one process\n")
    table = header + "\n" + "\n".join(lines) + "\n"
    with open(out_table, "w") as fh:
        fh.write(table)
    print("\n" + table)
    print(f"Wrote {out_json}\n      {out_table}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=None, help="first N images only (smoke test)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
