"""
WP5 diagnostic: why does batching change VisPruner's kept-token set at vtn=128?

    python scripts/wp5_selection_diag.py --run-id wp5_seldiag_1

For N dev images, compares B=1 against the same image inside a batch of 8 (seeded random
partners), at three points:
  1. CLIP output (pre-projector features and CLS attention): cosine / max abs diff.
     Nonzero but tiny = batched-GEMM rounding. Large = images are mixing.
  2. Kept-token set: overlap with the B=1 set.
  3. Duplicate test: [img, img, ...] batch. If every copy keeps the same set as each other,
     nothing crosses between batch rows.
Then repeats (2) with the vision tower run one image at a time inside the batch, to check
whether that alone restores the B=1 selection exactly.
Writes results/wp5_validation/<run-id>.{json,txt}.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "vis_pruner_copy"))

from PIL import Image  # noqa: E402

from llava.mm_utils import get_model_name_from_path, process_images  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402

DATASET = REPO_ROOT / "vis_pruner_copy" / "vispruner_eval_dataset" / "dev.json"
MODEL_PATH = REPO_ROOT / "vis_pruner_copy" / "checkpoints" / "llava-v1.5-7b"
B = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261025)
    args = ap.parse_args()

    torch.set_num_threads(4)
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(MODEL_PATH), None, get_model_name_from_path(str(MODEL_PATH)),
        visual_token_num=128, important_ratio=0.5)
    model.eval()
    tower = model.get_vision_tower()
    items = json.loads(DATASET.read_text())
    images = [process_images([Image.open(DATASET.parent / it["image"]).convert("RGB")],
                             image_processor, model.config)[0].to(model.device, torch.float16)
              for it in items]
    rng = random.Random(args.seed)
    picks = rng.sample(range(len(images)), args.n)

    def masks_of(batch):
        with torch.inference_mode():
            return model.encode_images(torch.stack(batch))[1].cpu()

    def tower_out(batch):
        with torch.inference_mode():
            f, a = tower(torch.stack(batch), output_attentions=True)
        return f.float().cpu(), a.float().cpu()

    rows = []
    for i in picks:
        partners = rng.sample([j for j in range(len(images)) if j != i], B - 1)
        batch = [images[i]] + [images[j] for j in partners]
        f1, a1 = tower_out([images[i]])
        fb, ab = tower_out(batch)
        cos = torch.nn.functional.cosine_similarity(f1[0], fb[0], dim=-1)
        m1 = masks_of([images[i]])[0]
        mb = masks_of(batch)[0]
        dup = masks_of([images[i]] * B)
        rows.append({
            "index": i,
            "feature_cos_min": cos.min().item(), "feature_cos_mean": cos.mean().item(),
            "feature_max_abs_diff": (f1[0] - fb[0]).abs().max().item(),
            "attn_max_abs_diff": (a1[0] - ab[0]).abs().max().item(),
            "kept_overlap_batched": (m1 & mb).sum().item() / m1.sum().item(),
            "dup_copies_agree": all(torch.equal(dup[0], d) for d in dup),
            "dup_overlap_with_b1": (m1 & dup[0]).sum().item() / m1.sum().item(),
        })

    # Candidate fix: vision tower one image at a time, rest of encode_images batched.
    original_forward = tower.forward

    def per_image_forward(imgs, output_attentions=False):
        outs = [original_forward(x.unsqueeze(0), output_attentions=output_attentions) for x in imgs]
        if output_attentions:
            return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])
        return torch.cat(outs)

    tower.forward = per_image_forward
    fix_exact = 0
    for r in rows:
        i = r["index"]
        partners = random.Random(args.seed + i).sample(
            [j for j in range(len(images)) if j != i], B - 1)
        mb = masks_of([images[i]] + [images[j] for j in partners])[0]
        tower.forward = original_forward
        m1 = masks_of([images[i]])[0]
        tower.forward = per_image_forward
        r["per_image_tower_exact"] = torch.equal(m1, mb)
        fix_exact += r["per_image_tower_exact"]
    tower.forward = original_forward

    def mean(k):
        return sum(r[k] for r in rows) / len(rows)

    summary = {
        "run_id": args.run_id,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                              text=True).strip(),
        "gpu": torch.cuda.get_device_name(), "n": len(rows), "batch_size": B,
        "feature_cos_min_over_images": min(r["feature_cos_min"] for r in rows),
        "feature_cos_mean": mean("feature_cos_mean"),
        "feature_max_abs_diff_max": max(r["feature_max_abs_diff"] for r in rows),
        "attn_max_abs_diff_max": max(r["attn_max_abs_diff"] for r in rows),
        "kept_overlap_batched_mean": mean("kept_overlap_batched"),
        "kept_identical_batched": sum(r["kept_overlap_batched"] == 1.0 for r in rows),
        "dup_copies_agree": sum(r["dup_copies_agree"] for r in rows),
        "dup_overlap_with_b1_mean": mean("dup_overlap_with_b1"),
        "per_image_tower_exact": fix_exact,
    }
    out = REPO_ROOT / "results" / "wp5_validation" / args.run_id
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    text = "\n".join(f"{k}: {v}" for k, v in summary.items())
    out.with_suffix(".txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
