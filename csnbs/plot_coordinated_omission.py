"""Verify and plot a saved synthetic coordinated-omission diagnostic.

Requires the project's existing matplotlib/numpy environment; installs nothing.
Usage: python csnbs/plot_coordinated_omission.py /path/to/demo.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def load_verified(path: Path):
    demo = json.loads(path.read_text())
    if demo.get("schema") != "coordinated-omission-demo" or demo.get("schemaVersion") != 1:
        raise ValueError("Expected coordinated-omission-demo version 1")
    root = path.parent.resolve()
    protocol_path = root / demo["protocolFile"]
    if sha256(protocol_path.read_bytes()) != demo["protocolSha256"]:
        raise ValueError("Protocol hash mismatch")
    if json.loads(protocol_path.read_text()) != demo["protocol"]:
        raise ValueError("Protocol contents differ")
    expected = {condition["id"] for condition in demo["protocol"]["conditions"]}
    if len(demo["trials"]) != len(expected) or {trial["id"] for trial in demo["trials"]} != expected:
        raise ValueError("Missing or duplicate prescribed condition")
    samples = {}
    for trial in demo["trials"]:
        directory = (root / trial["directory"]).resolve()
        directory.relative_to(root)
        original = json.loads((directory / "trial.json").read_text())
        if {key: value for key, value in trial.items() if key != "directory"} != original:
            raise ValueError("Saved trial and demo index disagree")
        raw_path = (directory / trial["requests"]["file"]).resolve()
        raw_path.relative_to(directory)
        raw = raw_path.read_bytes()
        if sha256(raw) != trial["requests"]["sha256"]:
            raise ValueError("Raw request hash mismatch")
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != trial["requests"]["count"] or len({row["requestId"] for row in rows}) != len(rows):
            raise ValueError("Raw count or request identity mismatch")
        measured = sorted((row for row in rows if row["phase"] == "measurement"), key=lambda row: row["sequence"])
        if [row["sequence"] for row in measured] != list(range(demo["protocol"]["measuredRequestsPerCondition"])):
            raise ValueError("Measured budget not preserved")
        if len(rows) - len(measured) != demo["protocol"]["warmupRequestsPerCondition"]:
            raise ValueError("Warmup budget not preserved")
        successful = [row for row in measured if row["outcome"] == "success"]
        latencies = sorted(row["latencyMs"] for row in successful)
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in latencies):
            raise ValueError("Invalid latency")
        summary = trial["summary"]
        if len(successful) != summary["successfulRequests"] or len(measured) - len(successful) != summary["failedRequests"]:
            raise ValueError("Outcome summary differs from raw records")
        for name, fraction, minimum in (("p50", .5, 1), ("p95", .95, 200), ("p99", .99, 1000)):
            expected_value = latencies[math.ceil(len(latencies) * fraction) - 1] if len(latencies) >= minimum else None
            saved = summary["successfulRequestLatencyMs"][name]
            if expected_value is None and saved is not None or expected_value is not None and (saved is None or not math.isclose(saved, expected_value, rel_tol=1e-12, abs_tol=1e-9)):
                raise ValueError(f"{name} does not match nearest-rank sample-gated raw observations")
        reference_end = trial["timing"]["referencePlannedEndMs"]
        delivered = sum(row["sentAtMs"] is not None and row["sentAtMs"] <= reference_end for row in measured)
        completed = sum(row["outcome"] == "success" and row["completedAtMs"] <= reference_end for row in measured)
        if delivered != trial["referenceHorizon"]["deliveredRequests"] or completed != trial["referenceHorizon"]["successfulCompletions"]:
            raise ValueError("Reference-horizon counts differ from raw records")
        dispatched = [row for row in measured if row["sentAtMs"] is not None]
        if trial["mode"] == "closed-loop" and any(right["sentAtMs"] < left["completedAtMs"] for left, right in zip(dispatched, dispatched[1:])):
            raise ValueError("Closed-loop control has overlapping requests")
        samples[trial["id"]] = latencies
    return demo, samples


def render(path: Path):
    demo, samples = load_verified(path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = path.parent
    colors = {"open-loop": "#2166ac", "closed-loop": "#d6604d"}
    markers = {"open-loop": "o", "closed-loop": "s"}
    count = demo["protocol"]["measuredRequestsPerCondition"]
    rates = sorted({trial["targetRps"] for trial in demo["trials"]})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for mode in colors:
        trials = sorted((trial for trial in demo["trials"] if trial["mode"] == mode), key=lambda trial: trial["targetRps"])
        x = [trial["targetRps"] for trial in trials]
        axes[0].plot(x, [trial["summary"]["achievedArrivalRateRps"] for trial in trials], marker=markers[mode], color=colors[mode], label=f"{mode}: arrivals")
        axes[0].plot(x, [trial["summary"]["successfulThroughputIncludingDrainRps"] for trial in trials], marker=markers[mode], linestyle=":", color=colors[mode], label=f"{mode}: completions")
        for name, style in (("p95", "-"), ("p99", "--")):
            axes[1].plot(x, [trial["summary"]["successfulRequestLatencyMs"][name] for trial in trials], marker=markers[mode], linestyle=style, color=colors[mode], label=f"{mode} {name}")
        axes[2].plot(x, [trial["referenceHorizon"]["deliveredRequests"] / count * 100 for trial in trials], marker=markers[mode], color=colors[mode], label=mode)
    axes[0].plot(rates, rates, color="grey", linestyle="--", linewidth=1, label="requested arrivals")
    axes[0].set(ylabel="Requests/second", title="Delivered load and completion rate")
    axes[1].set(ylabel="Sent-to-complete latency (ms)", yscale="log", title="Low closed-loop tails hide less offered load")
    axes[2].set(ylabel="Delivered within nominal horizon (%)", ylim=(0, 105), title="Nominal horizon = fixed count / target rate")
    for ax in axes:
        ax.set_xlabel("Target/reference arrival rate (RPS)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Synthetic serial service · {count:,} measured requests per condition · one trial each")
    fig.savefig(root / "throughput-tail.png", dpi=160)
    fig.savefig(root / "throughput-tail.svg")
    plt.close(fig)

    selected_rate = rates[-1]
    selected = sorted((trial for trial in demo["trials"] if trial["targetRps"] == selected_rate), key=lambda trial: trial["mode"])
    all_values = [value for trial in selected for value in samples[trial["id"]] if value > 0]
    if not all_values:
        raise ValueError("No positive successful latencies to plot")
    bins = np.geomspace(min(all_values) * .9, max(all_values) * 1.1, 85)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for trial in selected:
        mode = trial["mode"]
        values = samples[trial["id"]]
        label = f"{mode} (n={len(values):,}, failures={trial['summary']['failedRequests']})"
        if not values:
            continue
        axes[0].hist(values, bins=bins, weights=np.ones(len(values)) / len(values), histtype="step", linewidth=1.8, color=colors[mode], label=label)
        axes[1].step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=colors[mode], label=label)
    axes[0].set(ylabel="Fraction per log-spaced bin", title="Successful-request latency histogram")
    axes[1].set(ylabel="Empirical cumulative fraction", title="Successful-request latency ECDF")
    for ax in axes:
        ax.set(xlabel="Sent-to-complete latency (ms; logarithmic)", xscale="log")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Synthetic diagnostic at {selected_rate:g} target RPS · no GPU/capacity inference")
    fig.savefig(root / "latency-distribution.png", dpi=160)
    fig.savefig(root / "latency-distribution.svg")
    plt.close(fig)

    lines = ["# Synthetic coordinated-omission demonstration", "",
        "CPU-only serial-service diagnostic. These observations do not establish model/GPU speed, production capacity, accuracy, or a statistically resolved optimization.", "",
        f"The saved protocol fixed {count:,} measured requests and ten warmups per condition, target rates {', '.join(f'{rate:g}' for rate in rates)} RPS, requested service timer {demo['protocol']['requestedServiceDelayMs']:g} ms, and seed {demo['protocol']['seed']} before collection. Each condition ran once in the saved shuffled order.", "",
        "Open loop uses the canonical TypeScript load generator. The explicit closed-loop control awaits every response. Both preserve the reference schedule; closed-loop lateness exposes arrivals delayed before dispatch. Its actual observation window is recorded separately from the nominal count/rate horizon.", "",
        "| Mode | Target RPS | Actual arrivals RPS | Completion RPS incl. drain | Success / budget | Failures | Reference horizon s | Actual elapsed s | Delivered in reference horizon | p50 ms | p95 ms | p99 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "unavailable" if value is None else f"{value:.3f}"
    for trial in sorted(demo["trials"], key=lambda trial: (trial["targetRps"], trial["mode"])):
        summary = trial["summary"]
        latency = summary["successfulRequestLatencyMs"]
        lines.append(f"| {trial['mode']} | {trial['targetRps']:g} | {fmt(summary['achievedArrivalRateRps'])} | {fmt(summary['successfulThroughputIncludingDrainRps'])} | {summary['successfulRequests']} / {count} | {summary['failedRequests']} | {fmt(trial['timing']['referenceHorizonSeconds'])} | {fmt(trial['timing']['actualElapsedIncludingDrainSeconds'])} | {trial['referenceHorizon']['deliveredRequests']} / {count} | {fmt(latency['p50'])} | {fmt(latency['p95'])} | {fmt(latency['p99'])} |")
    lines += ["", "![Delivered load and tail latency](throughput-tail.png)", "", "![Latency histogram and ECDF](latency-distribution.png)", "",
        "The plots show sent-to-complete latency conditional on success. Nearest-rank p95 requires 200 successes and p99 requires 1,000; small smoke tests retain unavailable tails. This diagnostic keeps the full distribution and never treats 10,000 correlated requests as 10,000 independent trial replications. No confidence interval or significance claim is made.", "",
        "Requested timer delay is not an exact service duration; actual server queue/service intervals are in each raw response. Each service is a separate process on the same host. Connection scheduling and CPU contention remain part of the observed system.", "",
        "`protocol.json` was saved before collection. `demo.json` links each portable trial directory, raw SHA-256 and counts. The plotting verifier checked hashes, budgets, outcomes, closed-loop nonoverlap, reference-horizon counts and raw nearest-rank tails before rendering. Exact source snapshots are under `source/`.", ""]
    (root / "report.md").write_text("\n".join(lines))
    (root / "source/plot_coordinated_omission.py").write_bytes(Path(__file__).read_bytes())
    return {"report": str(root / "report.md"), "verifiedTrials": len(demo["trials"]),
            "verifiedMeasuredRequests": sum(trial["summary"]["totalRequests"] for trial in demo["trials"]),
            "plots": ["throughput-tail.png", "latency-distribution.png"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.demo), indent=2))
