#!/usr/bin/env python3
"""
Plot ScienceQA accuracy vs. number of visual tokens kept, sweeping over the
n_<TOKEN>/r_<RATIO>_result.json files produced by
vis_pruner_copy/scripts/v1_5/eval/sqa.sh (and sqa_hyperparams.sh).

Usage:
    python3 scripts/plot_sqa_sweep.py [answers_dir] [-o output.png]

Defaults to:
    vis_pruner_copy/playground/data/eval/scienceqa/answers/llava_test_CQM-A/llava-v1.5-7b
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_DIR = (
    Path(__file__).resolve().parent.parent
    / "vis_pruner_copy/playground/data/eval/scienceqa/answers/llava_test_CQM-A/llava-v1.5-7b"
)

# n_128/r_0.5_result.json -> n=128, r=0.5
RESULT_RE = re.compile(r"n_(\d+)/r_([0-9.]+)_result\.json$")


def collect_results(answers_dir: Path) -> list[dict]:
    rows = []
    for result_file in sorted(answers_dir.glob("n_*/r_*_result.json")):
        match = RESULT_RE.search(str(result_file))
        if not match:
            continue
        n_tokens, ratio = int(match.group(1)), float(match.group(2))
        data = json.loads(result_file.read_text())
        rows.append(
            {
                "n_tokens": n_tokens,
                "ratio": ratio,
                "acc": data["acc"],
                "correct": data["correct"],
                "count": data["count"],
            }
        )
    return rows


def report_incomplete(answers_dir: Path, rows: list[dict]) -> None:
    scored = {(r["n_tokens"], r["ratio"]) for r in rows}
    for answers_file in sorted(answers_dir.glob("n_*/r_*.jsonl")):
        if answers_file.name.endswith("_output.jsonl"):
            continue
        match = re.search(r"n_(\d+)/r_([0-9.]+)\.jsonl$", str(answers_file))
        if not match:
            continue
        key = (int(match.group(1)), float(match.group(2)))
        if key not in scored:
            n_lines = sum(1 for _ in answers_file.open())
            print(
                f"  skipping n={key[0]} r={key[1]}: no *_result.json yet "
                f"({n_lines} answers generated so far)"
            )


def plot(rows: list[dict], output_path: Path) -> None:
    rows = sorted(rows, key=lambda r: (r["ratio"], r["n_tokens"]))
    ratios = sorted({r["ratio"] for r in rows})

    # Fixed categorical color order (colorblind-safe qualitative set), assigned
    # by ratio identity rather than cycled/reused.
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, ratio in enumerate(ratios):
        series = [r for r in rows if r["ratio"] == ratio]
        x = [r["n_tokens"] for r in series]
        y = [r["acc"] for r in series]
        color = palette[i % len(palette)]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=7,
            linewidth=2,
            color=color,
            label=f"important_ratio={ratio}",
        )
        for r in series:
            ax.annotate(
                f"{r['acc']:.1f}%",
                (r["n_tokens"], r["acc"]),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=9,
                color="#333333",
            )

    ax.set_xlabel("Visual tokens kept (n)")
    ax.set_ylabel("ScienceQA accuracy (%)")
    ax.set_title("LLaVA-1.5-7B + VisPruner: ScienceQA accuracy vs. visual tokens kept")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if len(ratios) > 1:
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "answers_dir",
        nargs="?",
        default=DEFAULT_DIR,
        type=Path,
        help="Directory containing n_<TOKEN>/r_<RATIO>_result.json files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "results/scienceqa_acc_vs_tokens.png",
        help="Where to save the plot",
    )
    args = parser.parse_args()

    rows = collect_results(args.answers_dir)
    if not rows:
        raise SystemExit(f"No *_result.json files found under {args.answers_dir}")

    print(f"Found {len(rows)} scored run(s) under {args.answers_dir}:")
    for r in rows:
        print(f"  n={r['n_tokens']:>4}  r={r['ratio']}  acc={r['acc']:.2f}%  ({r['correct']}/{r['count']})")

    report_incomplete(args.answers_dir, rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, args.output)


if __name__ == "__main__":
    main()
