"""
WP6 demo plots (DEMO_PLAN.md Plot 1 and Plot 2) from wp6_analyze.py's summary.json.

    python scripts/wp6_plot.py results/timing/wp6_matrix/wp6_matrix_2

Rows: answer policy (natural EOS, fixed 64 tokens). Columns: S0 (csnbs/server.py, one
request at a time) and S2 (batched, B<=8). Each panel: p50 latency vs offered rate,
576 (unpruned) vs 128 (VisPruner). Shared log y-axis per row so panels compare directly.
Repeated noise-floor blocks are averaged into one point. Writes wp6_demo_plots.png and a
capacity bar chart, wp6_capacity.png, into the run directory.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = {576: ("576 tokens (unpruned)", "#2a78d6", "o"),
          128: ("128 tokens (VisPruner)", "#eb6834", "s")}
STATES = {"S0": "Old server: one request at a time (S0)",
          "S2": "Batched server: up to 8 per GPU pass (S2)"}
POLICIES = {"natural": "natural-length answers", "fixed": "fixed 64-token answers"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    root = Path(sys.argv[1])
    data = json.loads((root / "summary.json").read_text())
    rows = defaultdict(list)
    for r in data["rows"]:
        rows[(r["policy"], r["state"], r["visual_token_num"], r["rps_offered"])].append(r)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), facecolor="#fcfcfb")
    for i, policy in enumerate(POLICIES):
        for j, state in enumerate(STATES):
            ax = axes[i][j]
            style(ax)
            for vtn, (label, color, marker) in SERIES.items():
                pts = sorted((rps, statistics.mean(x["p50_ms"] for x in rs) / 1000)
                             for (p, s, v, rps), rs in rows.items()
                             if p == policy and s == state and v == vtn)
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7,
                        markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label)
                ax.annotate(f"{vtn}", (xs[-1], ys[-1]), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=9, color=TEXT)
            cap = {c["visual_token_num"]: c["throughput_rps"] for c in data["capacity"]
                   if c["policy"] == policy and c["state"] == state}
            gain = 100 * (cap[128] / cap[576] - 1)
            ax.set_yscale("log")
            ax.set_title(f"{STATES[state]}\n{POLICIES[policy]}", fontsize=10.5, color=TEXT,
                         loc="left")
            ax.text(0.02, 0.95, f"max throughput 576 → 128: {cap[576]:.2f} → {cap[128]:.2f} req/s"
                    f"  ({gain:+.0f}%)", transform=ax.transAxes, fontsize=9, color=TEXT,
                    va="top", bbox=dict(facecolor="#fcfcfb", edgecolor=GRID, boxstyle="round"))
            ax.set_xlabel("offered load (requests/s, open-loop)", fontsize=9, color=MUTED)
            if j == 0:
                ax.set_ylabel("p50 latency (s, log scale)", fontsize=9, color=MUTED)
        lo = min(a.get_ylim()[0] for a in axes[i])
        hi = max(a.get_ylim()[1] for a in axes[i])
        for a in axes[i]:
            a.set_ylim(lo, hi)
    axes[0][0].legend(frameon=False, fontsize=9, loc="center left", labelcolor=TEXT)
    meta = data["meta"]
    fig.suptitle("Pruning barely helps one request at a time; batched, it pulls away",
                 fontsize=13, color=TEXT, x=0.01, ha="left")
    fig.text(0.01, 0.005, f"{meta['run_id']} · commit {meta['git_commit'][:7]} · {meta['gpu']} · "
             f"csnbs/measure open-loop load · ≥100 samples per point after 10 warm-up · "
             f"p50 noise floor 1–4.5%", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(root / "wp6_demo_plots.png", dpi=150)

    # Capacity bars: one panel per policy, groups = server state, bars = 576 vs 128.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#fcfcfb")
    for ax, policy in zip(axes, POLICIES):
        style(ax)
        ax.grid(True, axis="y", color=GRID)
        ax.grid(False, axis="x")
        states = ["S0", "S1", "S2"]
        width = 0.38
        for k, (vtn, (label, color, _)) in enumerate(SERIES.items()):
            vals = [next(c["throughput_rps"] for c in data["capacity"]
                         if c["policy"] == policy and c["state"] == s and c["visual_token_num"] == vtn)
                    for s in states]
            xs = [x + (k - 0.5) * (width + 0.02) for x in range(len(states))]
            bars = ax.bar(xs, vals, width=width, color=color, label=label)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8.5, color=TEXT)
        ax.set_xticks(range(3), ["S0\none at a time", "S1\nqueue", "S2\nbatched (≤8)"],
                      fontsize=9, color=MUTED)
        ax.set_title(f"Max throughput, {POLICIES[policy]}", fontsize=10.5, color=TEXT, loc="left")
        ax.set_ylabel("requests/s", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=TEXT)
    fig.text(0.01, 0.01, f"{meta['run_id']} · {meta['gpu']} · highest achieved throughput at any "
             f"offered level", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(root / "wp6_capacity.png", dpi=150)
    print(root / "wp6_demo_plots.png", root / "wp6_capacity.png")


if __name__ == "__main__":
    main()
