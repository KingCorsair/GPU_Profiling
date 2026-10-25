"""
WP3 analysis: service-time decomposition, event-loop blocking, and the WP4 go/no-go number.

    python scripts/wp3_analyze.py results/timing/wp3_diag/wp3_diag_2

Reads every session written by scripts/wp3_run_diag.py and writes summary.json and
summary.txt next to them. The first WARMUP requests of each session (by arrival) are
dropped (rule 3).

Definitions (all from server-side perf_counter timestamps):
  overlappable CPU  = t_preprocess_done - t_handler_entry   (roadmap WP3 step 4)
  serial non-GPU    = everything in a request's handler span except generate()
  service time      = handler span (t_handler_entry -> t_response). The S0 server is
                      strictly serial, so this is the per-request service time.
  received-after-previous-response: share of requests whose ASGI call only started after
                      the previous request's response was sent. While the loop is blocked,
                      arrivals are not even received, so a queue-based service interval
                      cannot be measured from server-side timestamps.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

WARMUP = 10

STAGES = [
    # (name, start key, end key)
    ("pydantic_parse", "t_body_done", "t_handler_entry"),
    ("b64_decode", "t_handler_entry", "t_b64_done"),
    ("pil_decode", "t_b64_done", "t_pil_done"),
    ("tokenize", "t_pil_done", "t_tokenize_done"),
    ("ids_h2d", "t_tokenize_done", "t_process_images_start"),
    ("process_images", "t_process_images_start", "t_preprocess_done"),
    ("image_h2d", "t_preprocess_done", "t_generate_start"),
    ("generate", "t_generate_start", "t_generate_end"),
    ("decode_respond", "t_generate_end", "t_response"),
]


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def ms(x):
    return None if x is None else round(1000 * x, 2)


def analyze_session(d):
    meta = json.loads((d / "session.json").read_text())
    reqs = [json.loads(l) for l in (d / "requests.jsonl").read_text().splitlines()]
    reqs = [r for r in reqs if r.get("status") == 200 and "t_generate_end" in r]
    reqs.sort(key=lambda r: r["t_received"])
    reqs = reqs[WARMUP:]
    hb = json.loads((d / "heartbeat.json").read_text())

    stage = {name: [r[b] - r[a] for r in reqs] for name, a, b in STAGES}
    overlappable = [r["t_preprocess_done"] - r["t_handler_entry"] for r in reqs]
    handler_span = [r["t_response"] - r["t_handler_entry"] for r in reqs]
    serial_non_gpu = [h - g for h, g in zip(handler_span, stage["generate"])]

    by_entry = sorted(reqs, key=lambda r: r["t_handler_entry"])
    handoff = [b["t_received"] - a["t_response"] for a, b in zip(by_entry, by_entry[1:])]
    service = handler_span
    window_s = by_entry[-1]["t_response"] - by_entry[0]["t_handler_entry"]

    # Heartbeat: total blocked time vs total handler time, and per-request gap match.
    gaps = hb["gaps"]
    window = (by_entry[0]["t_handler_entry"], by_entry[-1]["t_response"])
    gaps_in = [g for g in gaps if window[0] <= g[0] <= window[1]]
    matched = []
    for r in by_entry:
        hits = [late for start, late in gaps_in
                if start <= r["t_generate_start"] and start + late >= r["t_generate_end"]]
        if hits:
            matched.append(max(hits) / (r["t_response"] - r["t_handler_entry"]))

    mon = [l.split(",") for l in (d / "monitor.csv").read_text().splitlines()[1:]]
    util = [float(row[1]) for row in mon if len(row) == 3 and row[1]]

    loadgen = None
    if meta.get("loadgen_run_json"):
        p = Path(__file__).resolve().parents[1] / meta["loadgen_run_json"]
        if p.exists():
            s = json.loads(p.read_text())["summary"]
            loadgen = {
                "throughput_rps": s["successfulThroughputRps"],
                "p50_ms": s["successfulRequestLatencyMs"]["p50"],
                "p95_ms": s["successfulRequestLatencyMs"]["p95"],
                "failed": s["failedRequests"],
            }

    mean_service = statistics.mean(service) if service else None
    return {
        "session": d.name,
        "visual_token_num": meta["visual_token_num"],
        "torch_threads": meta["torch_threads"],
        "n_requests": len(reqs),
        "nr_throttled_delta": meta["nr_throttled_delta"],
        "stage_p50_ms": {k: ms(pct(v, 0.5)) for k, v in stage.items()},
        "stage_mean_ms": {k: ms(statistics.mean(v)) for k, v in stage.items()},
        "overlappable_cpu_mean_ms": ms(statistics.mean(overlappable)),
        "overlappable_cpu_p50_ms": ms(pct(overlappable, 0.5)),
        "serial_non_gpu_mean_ms": ms(statistics.mean(serial_non_gpu)),
        "handler_span_mean_ms": ms(statistics.mean(handler_span)),
        "service_interval_mean_ms": ms(mean_service),
        "service_interval_p50_ms": ms(pct(service, 0.5)),
        "server_busy_fraction": round(sum(handler_span) / window_s, 3),
        "overlappable_pct_of_service": round(100 * statistics.mean(overlappable) / mean_service, 2)
        if mean_service else None,
        "serial_non_gpu_pct_of_service": round(100 * statistics.mean(serial_non_gpu) / mean_service, 2)
        if mean_service else None,
        "received_after_prev_response": {
            "share": round(sum(h >= 0 for h in handoff) / len(handoff), 3),
            "n": len(handoff),
        },
        "heartbeat": {
            "samples": hb["samples"],
            "gaps_over_1ms": len(gaps_in),
            "max_gap_ms": ms(max((g[1] for g in gaps_in), default=None)),
            "blocked_s_in_window": round(sum(g[1] for g in gaps_in), 2),
            "handler_s_in_window": round(sum(handler_span), 2),
            "requests_with_gap_covering_generate": len(matched),
            "gap_over_handler_span_p50": round(pct(matched, 0.5), 3) if matched else None,
        },
        "gpu_util_mean_pct": round(statistics.mean(util), 1) if util else None,
        "loadgen": loadgen,
    }


def main():
    root = Path(sys.argv[1])
    run = json.loads((root / "run.json").read_text())
    sessions = [analyze_session(root / name) for name in run["session_order"]]

    groups = defaultdict(list)
    for s in sessions:
        groups[(s["visual_token_num"], str(s["torch_threads"]))].append(s)

    def pooled(ss, key):
        vals = [s[key] for s in ss if s[key] is not None]
        return round(statistics.mean(vals), 2) if vals else None

    lines = [
        f"WP3 server diagnostic -- {run['run_id']}",
        f"commit {run['git_commit']} (dirty={run['git_dirty']})  GPU {run['gpu']}  "
        f"rps={run['args']['rps']}  {run['args']['duration']}s/session  warm-up dropped={WARMUP}",
        f"session order: {', '.join(run['session_order'])}",
        "",
        "Per session:",
        f"{'session':32} {'n':>4} {'thrtl':>5} {'overlap':>8} {'nonGPU':>7} {'generate':>9} "
        f"{'service':>8} {'ovl%':>6} {'nonGPU%':>7} {'busy':>5} {'GPU%':>5} {'tput':>6} {'p50':>7} {'p95':>7}",
    ]
    for s in sessions:
        lg = s["loadgen"] or {}
        lines.append(
            f"{s['session']:32} {s['n_requests']:>4} {str(s['nr_throttled_delta']):>5} "
            f"{s['overlappable_cpu_mean_ms']:>8} {s['serial_non_gpu_mean_ms']:>7} "
            f"{s['stage_mean_ms']['generate']:>9} {s['service_interval_mean_ms']:>8} "
            f"{s['overlappable_pct_of_service']:>6} {s['serial_non_gpu_pct_of_service']:>7} "
            f"{s['server_busy_fraction']:>5} {s['gpu_util_mean_pct']:>5} {lg.get('throughput_rps', 0):>6.3f} "
            f"{lg.get('p50_ms', 0):>7.0f} {lg.get('p95_ms', 0):>7.0f}")

    lines += ["", "Stage decomposition, mean ms, pooled over repeats:"]
    names = [n for n, _, _ in STAGES]
    lines.append(f"{'config':18} " + " ".join(f"{n[:10]:>10}" for n in names) + f" {'service':>9}")
    summary_groups = {}
    for (vtn, thr), ss in sorted(groups.items()):
        means = {n: round(statistics.mean(s["stage_mean_ms"][n] for s in ss), 2) for n in names}
        svc = pooled(ss, "service_interval_mean_ms")
        lines.append(f"{f'vtn{vtn} thr={thr}':18} " + " ".join(f"{means[n]:>10}" for n in names)
                     + f" {svc:>9}")
        summary_groups[f"vtn{vtn}_thr{thr}"] = {
            "sessions": [s["session"] for s in ss],
            "stage_mean_ms": means,
            "overlappable_cpu_mean_ms": pooled(ss, "overlappable_cpu_mean_ms"),
            "serial_non_gpu_mean_ms": pooled(ss, "serial_non_gpu_mean_ms"),
            "service_interval_mean_ms": svc,
            "overlappable_pct_of_service": pooled(ss, "overlappable_pct_of_service"),
            "serial_non_gpu_pct_of_service": pooled(ss, "serial_non_gpu_pct_of_service"),
            "gpu_util_mean_pct": pooled(ss, "gpu_util_mean_pct"),
            "server_busy_fraction": pooled(ss, "server_busy_fraction"),
            "nr_throttled_delta_total": sum(s["nr_throttled_delta"] or 0 for s in ss),
        }

    lines += ["", "Event loop / handoff checks:"]
    for s in sessions:
        hb, ho = s["heartbeat"], s["received_after_prev_response"]
        lines.append(
            f"{s['session']:32} max_gap={hb['max_gap_ms']}ms blocked={hb['blocked_s_in_window']}s "
            f"handler={hb['handler_s_in_window']}s gap/handler p50={hb['gap_over_handler_span_p50']} "
            f"({hb['requests_with_gap_covering_generate']}/{s['n_requests']})  "
            f"received after previous response: {ho['share']:.1%} of {ho['n']}")

    text = "\n".join(lines)
    print(text)
    (root / "summary.txt").write_text(text + "\n")
    (root / "summary.json").write_text(json.dumps(
        {"run": run, "groups": summary_groups, "sessions": sessions}, indent=2))


if __name__ == "__main__":
    main()
