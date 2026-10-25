"""
WP6 analysis: one table row per (state, vtn, policy, rps) block from wp6_load_matrix.py.

    python scripts/wp6_analyze.py results/timing/wp6_matrix/<run-id>

Latency percentiles are recomputed from csnbs/measure's per-request records after its
10-request warm-up, so the sample count behind each one is explicit. p99 is printed only
where n >= 1000 (rule 6 / roadmap WP6); otherwise the column is blank. Capacity per cell
is the highest achieved throughput at any level.
Writes summary.{json,txt} in the run directory.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WARMUP = 10
P99_MIN_N = 1000


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def block_row(root, b):
    run_path = REPO_ROOT / b["loadgen_run_json"]
    run = json.loads(run_path.read_text())
    records = [json.loads(l) for l in
               (run_path.parent / run["requests"]["file"]).read_text().splitlines() if l.strip()]
    reqs = sorted(records, key=lambda r: r["sequence"])[WARMUP:]
    ok = [r for r in reqs if r["status"] == 200 and r["error"] is None]
    lat = [r["latencyMs"] for r in ok]
    util = []
    for line in (root / b["gpu_csv"]).read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) == 3 and parts[1]:
            util.append(float(parts[1]))
    s = run["summary"]
    return {
        "state": b["state"], "visual_token_num": b["visual_token_num"], "policy": b["policy"],
        "rps_offered": b["rps"], "rps_achieved_arrival": s["achievedArrivalRateRps"],
        "throughput_rps": s["successfulThroughputRps"], "n": len(lat),
        "failed": len(reqs) - len(ok),
        "p50_ms": pct(lat, 0.5), "p95_ms": pct(lat, 0.95),
        "p99_ms": pct(lat, 0.99) if len(lat) >= P99_MIN_N else None,
        "gpu_util_mean_pct": statistics.mean(util) if util else None,
        "batch_size": (b.get("server_configuration") or {}).get("max_batch_size"),
        "block": b["name"],
    }


def fmt(x, nd=0):
    return "" if x is None else f"{x:.{nd}f}"


def main():
    root = Path(sys.argv[1])
    meta = json.loads((root / "run.json").read_text())
    rows = [block_row(root, b) for b in meta["blocks"]]
    rows.sort(key=lambda r: (r["policy"], r["visual_token_num"], r["state"], r["rps_offered"]))

    lines = [f"WP6 load matrix -- {meta['run_id']}",
             f"commit {meta['git_commit']} (dirty={meta['git_dirty']})  GPU {meta['gpu']}  "
             f"torch_threads={meta['args']['torch_threads']}  S2 batch={meta['args']['batch_size']} "
             f"wait={meta['args']['max_wait_ms']}ms",
             f"latency in ms after {WARMUP} warm-up requests; p99 only where n >= {P99_MIN_N}", "",
             f"{'policy':8} {'vtn':>4} {'state':5} {'offered':>7} {'arrival':>7} {'tput':>6} "
             f"{'n':>5} {'fail':>4} {'p50':>8} {'p95':>8} {'p99':>8} {'GPU%':>5}"]
    for r in rows:
        lines.append(
            f"{r['policy']:8} {r['visual_token_num']:>4} {r['state']:5} {r['rps_offered']:>7} "
            f"{fmt(r['rps_achieved_arrival'], 3):>7} {fmt(r['throughput_rps'], 3):>6} {r['n']:>5} "
            f"{r['failed']:>4} {fmt(r['p50_ms']):>8} {fmt(r['p95_ms']):>8} {fmt(r['p99_ms']):>8} "
            f"{fmt(r['gpu_util_mean_pct'], 1):>5}")

    capacity = defaultdict(dict)
    for r in rows:
        key = (r["policy"], r["visual_token_num"], r["state"])
        if r["throughput_rps"] > capacity[key].get("throughput_rps", -1):
            capacity[key] = {"throughput_rps": r["throughput_rps"], "at_offered": r["rps_offered"],
                             "gpu_util_mean_pct": r["gpu_util_mean_pct"]}
    lines += ["", "Capacity (highest achieved throughput at any offered level):",
              f"{'policy':8} {'vtn':>4} {'state':5} {'rps':>7} {'at offered':>10} {'GPU%':>5}"]
    for (policy, vtn, state), c in sorted(capacity.items()):
        lines.append(f"{policy:8} {vtn:>4} {state:5} {c['throughput_rps']:>7.3f} "
                     f"{c['at_offered']:>10} {fmt(c['gpu_util_mean_pct'], 1):>5}")

    text = "\n".join(lines)
    print(text)
    (root / "summary.txt").write_text(text + "\n")
    (root / "summary.json").write_text(json.dumps(
        {"meta": meta, "rows": rows,
         "capacity": [{"policy": k[0], "visual_token_num": k[1], "state": k[2], **v}
                      for k, v in sorted(capacity.items())]}, indent=2))


if __name__ == "__main__":
    main()
