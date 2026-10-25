"""
WP4/WP5 correctness and scaling checks for llava/serve/batch_engine.py, no server.

    python scripts/wp5_batch_validation.py --run-id wp5_val_1

All in one process, both visual_token_num configs (switched at runtime, as bench_dev.py does):

  1. Reference: every dev image through csnbs/server.py's exact S0 code path (B=1, no
     attention mask, no queue).
  2. S1 identity: engine.generate_batch at B=1. This is the call S1's worker makes, so a
     match here is the WP4 "byte-identical to S0" check.
  3. Smoke + equivalence: generate_batch at B=2, 4, 8 over the 90 dev images in a seeded
     shuffled order (batches mix categories and prompt lengths). Every answer is compared
     with its reference. Also records whether the kept visual-token set changed, which
     separates "the vision tower/pruning saw different numerics" from "the LLM did".
  4. Contamination: anchor requests batched with the shortest-prompt partner and with the
     longest-output partner. The anchor's answer must not depend on who it shares a batch
     with.
  5. Batched generate() latency vs B at a fixed 64-token output (CUDA events, warm-up,
     interleaved B order). Throwaway scaling curve, not a serving number.

Dev split only (rule 11). Writes results/wp5_validation/<run-id>.{json,txt}.
"""
import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "vis_pruner_copy"))

from PIL import Image  # noqa: E402

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import (get_model_name_from_path, process_images,  # noqa: E402
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.serve.batch_engine import InferenceEngine  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402

DATASET = REPO_ROOT / "vis_pruner_copy" / "vispruner_eval_dataset" / "dev.json"
MODEL_PATH = REPO_ROOT / "vis_pruner_copy" / "checkpoints" / "llava-v1.5-7b"
CONFIGS = (576, 128)
BATCH_SIZES = (2, 4, 8)
MAX_NEW_TOKENS = 64
SCAN_BATCH_SIZES = (1, 2, 4, 8, 16)
SCAN_WARMUP, SCAN_REPS = 5, 10


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def load_items(tokenizer, image_processor, model):
    items = json.loads(DATASET.read_text())
    out = []
    for it in items:
        pil = Image.open(DATASET.parent / it["image"]).convert("RGB")
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + it["question"])
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                    return_tensors="pt")
        image = process_images([pil], image_processor, model.config)[0]
        out.append({"id": it["question_id"], "category": it["category"],
                    "input_ids": ids, "image": image, "size": pil.size})
    return out


class MaskRecorder:
    """Wraps encode_images to keep the kept-token masks of the most recent call."""

    def __init__(self, model):
        self.model, self.inner, self.masks = model, model.encode_images, None

        def wrapped(images):
            feats, masks = self.inner(images)
            self.masks = masks.cpu()
            return feats, masks
        model.encode_images = wrapped


@torch.inference_mode()
def s0_reference(model, tokenizer, item):
    """csnbs/server.py _infer_model, minus the HTTP and PIL parts."""
    output_ids, _ = model.generate(
        item["input_ids"].unsqueeze(0).to(model.device),
        images=item["image"].unsqueeze(0).to(model.device, dtype=torch.float16),
        image_sizes=[item["size"]], do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True)
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def run_batch(engine, rec, batch):
    answers = engine.generate_batch([b["image"] for b in batch],
                                    [b["input_ids"] for b in batch],
                                    [b["size"] for b in batch])
    return answers, rec.masks


def latency_scan(model, tokenizer, items, seed):
    engine = InferenceEngine(model, tokenizer, max_batch_size=max(SCAN_BATCH_SIZES),
                             max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MAX_NEW_TOKENS)
    plan = [b for b in SCAN_BATCH_SIZES for _ in range(SCAN_REPS)]
    random.Random(seed).shuffle(plan)
    for b in SCAN_BATCH_SIZES:                      # warm-up, discarded
        for _ in range(SCAN_WARMUP):
            batch = items[:b]
            engine.generate_batch([x["image"] for x in batch], [x["input_ids"] for x in batch],
                                  [x["size"] for x in batch])
    events = {b: [] for b in SCAN_BATCH_SIZES}
    rng = random.Random(seed + 1)
    for b in plan:
        batch = rng.sample(items, b)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        engine.generate_batch([x["image"] for x in batch], [x["input_ids"] for x in batch],
                              [x["size"] for x in batch])
        end.record()
        events[b].append((start, end))
    torch.cuda.synchronize()
    engine.shutdown()
    result = {}
    for b, evs in events.items():
        t = sorted(s.elapsed_time(e) for s, e in evs)
        p50 = statistics.median(t)
        result[b] = {"n": len(t), "batch_ms_p50": round(p50, 1), "batch_ms_min": round(t[0], 1),
                     "batch_ms_max": round(t[-1], 1),
                     "requests_per_s_at_p50": round(1000 * b / p50, 3)}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--skip-scan", action="store_true")
    args = ap.parse_args()
    out_dir = REPO_ROOT / "results" / "wp5_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / f"{args.run_id}.json").exists():
        raise SystemExit("run id exists")

    torch.set_num_threads(4)
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(MODEL_PATH), None, get_model_name_from_path(str(MODEL_PATH)),
        visual_token_num=576, important_ratio=0.5)
    model.eval()
    rec = MaskRecorder(model)
    items = load_items(tokenizer, image_processor, model)
    engine = InferenceEngine(model, tokenizer, max_batch_size=max(BATCH_SIZES),
                             max_new_tokens=MAX_NEW_TOKENS)

    meta = {"run_id": args.run_id, "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "gpu": torch.cuda.get_device_name(), "n_items": len(items),
            "padding_side": model.config.tokenizer_padding_side, "seed": args.seed}
    results = {"meta": meta, "configs": {}}
    t0 = time.perf_counter()

    for cfg in CONFIGS:
        model.visual_token_num = cfg
        ref, ref_masks = {}, {}
        for it in items:
            ref[it["id"]] = s0_reference(model, tokenizer, it)
            ref_masks[it["id"]] = rec.masks[0]

        cfg_res = {"batch": {}}
        for b in (1,) + BATCH_SIZES:
            order = items[:]
            random.Random(args.seed + b).shuffle(order)
            diffs, mask_changed, shapes_ok = [], 0, True
            for i in range(0, len(order), b):
                batch = order[i:i + b]
                answers, masks = run_batch(engine, rec, batch)
                if masks.shape[0] != len(batch) or len(answers) != len(batch):
                    shapes_ok = False
                for j, (it, ans) in enumerate(zip(batch, answers)):
                    same_mask = torch.equal(masks[j], ref_masks[it["id"]])
                    mask_changed += not same_mask
                    if ans != ref[it["id"]]:
                        diffs.append({"id": it["id"], "category": it["category"],
                                      "batch_index": i // b, "position": j,
                                      "token_set_changed": not same_mask,
                                      "reference": ref[it["id"]], "batched": ans})
            cfg_res["batch"][b] = {
                "identical": len(items) - len(diffs), "n": len(items),
                "token_set_changed": mask_changed, "shapes_ok": shapes_ok, "diffs": diffs}
            print(f"vtn={cfg} B={b}: identical {len(items) - len(diffs)}/{len(items)}, "
                  f"token set changed on {mask_changed}", flush=True)

        # Contamination: anchor answer must not depend on its batch partner.
        by_len = sorted(items, key=lambda x: x["input_ids"].shape[0])
        short_partner = by_len[0]
        long_out_partner = max(items, key=lambda x: len(ref[x["id"]]))
        anchors = random.Random(args.seed).sample(
            [x for x in items if x["id"] not in (short_partner["id"], long_out_partner["id"])], 10)
        contamination = []
        for a in anchors:
            outs = {}
            for name, partner in (("short_prompt", short_partner), ("long_output", long_out_partner)):
                for order_name, batch in ((f"{name}_after", [a, partner]),
                                          (f"{name}_before", [partner, a])):
                    answers, _ = run_batch(engine, rec, batch)
                    outs[order_name] = answers[batch.index(a)]
                    # The partner's answer must not contain the anchor's (or vice versa)
                    # beyond what each produces alone.
                    outs[order_name + "_partner_ok"] = answers[batch.index(partner)] == ref[partner["id"]]
            contamination.append({"id": a["id"], "reference": ref[a["id"]], **outs,
                                  "all_equal_reference": all(
                                      v == ref[a["id"]] for k, v in outs.items()
                                      if not k.endswith("_ok"))})
        cfg_res["contamination"] = {
            "short_partner": short_partner["id"], "long_output_partner": long_out_partner["id"],
            "anchors_all_equal": sum(c["all_equal_reference"] for c in contamination),
            "n_anchors": len(contamination), "cases": contamination}
        print(f"vtn={cfg} contamination: {cfg_res['contamination']['anchors_all_equal']}/10 "
              f"anchors identical to reference in every pairing", flush=True)
        cfg_res["reference_answers"] = ref

        if not args.skip_scan:
            cfg_res["latency_scan"] = latency_scan(model, tokenizer, items, args.seed)
            for b, r in cfg_res["latency_scan"].items():
                print(f"vtn={cfg} scan B={b}: {r}", flush=True)
        results["configs"][cfg] = cfg_res

    engine.shutdown()
    meta["elapsed_s"] = round(time.perf_counter() - t0, 1)
    (out_dir / f"{args.run_id}.json").write_text(json.dumps(results, indent=2))

    lines = [f"WP4/WP5 batch validation -- {args.run_id}",
             f"commit {meta['git_commit']} (dirty={meta['git_dirty']})  GPU {meta['gpu']}  "
             f"{meta['n_items']} dev images  greedy, max_new_tokens={MAX_NEW_TOKENS}  "
             f"padding_side={meta['padding_side']}", ""]
    for cfg, r in results["configs"].items():
        lines.append(f"vtn={cfg}")
        for b, br in r["batch"].items():
            label = "S1 identity (B=1 via engine)" if b == 1 else f"B={b}"
            lines.append(f"  {label:30} identical {br['identical']}/{br['n']}  "
                         f"token-set changed {br['token_set_changed']}  shapes_ok={br['shapes_ok']}")
            for d in br["diffs"]:
                lines.append(f"      {d['id']} [{d['category']}] token_set_changed={d['token_set_changed']}"
                             f"\n        ref: {d['reference']!r}\n        bat: {d['batched']!r}")
        c = r["contamination"]
        lines.append(f"  contamination: {c['anchors_all_equal']}/{c['n_anchors']} anchors identical "
                     f"to reference with every partner/order")
        if "latency_scan" in r:
            lines.append(f"  batched generate(), fixed {MAX_NEW_TOKENS}-token output, "
                         f"{SCAN_REPS} reps after {SCAN_WARMUP} warm-up (CUDA events):")
            base = r["latency_scan"][1]["requests_per_s_at_p50"]
            for b, s in r["latency_scan"].items():
                lines.append(f"    B={b:>2}: batch p50 {s['batch_ms_p50']:>8} ms  "
                             f"[{s['batch_ms_min']}-{s['batch_ms_max']}]  "
                             f"{s['requests_per_s_at_p50']:>6} req/s  "
                             f"({s['requests_per_s_at_p50'] / base:.2f}x B=1)")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (out_dir / f"{args.run_id}.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
