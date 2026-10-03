"""Render saved V2 campaign evidence without inferring speedup or capacity.

Usage: python csnbs/plot_campaign.py --report path/to/report.json [--rate 1]
The report exporter owns metrics and evidence checks; this module only plots
saved summaries and successful-request observations. Output is PNG plus SVG.
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any


def _number(value: Any, label: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")


def report_run_kind(report: dict[str, Any]) -> str | None:
    """Use explicit mode metadata only; a placeholder rate cannot prove arrivals."""
    kinds = {value for value in [report.get("runKind"), *(run.get("runKind") for run in report["runs"])] if value is not None}
    if not kinds:
        return None
    if not kinds <= {"open-loop", "isolated"} or len(kinds) != 1:
        raise ValueError("Report run kinds must consistently identify open-loop or isolated requests")
    return next(iter(kinds))


def validate_report(report: Any) -> dict[str, Any]:
    """Fail visibly on incompatible evidence instead of silently dropping rows."""
    if not isinstance(report, dict) or report.get("schema") != "measurement-report" or report.get("schemaVersion") != 1:
        raise ValueError("Expected measurement-report schema version 1")
    for field in ("campaignId", "purpose"):
        if not isinstance(report.get(field), str) or not report[field].strip():
            raise ValueError(f"Missing report {field}")
    if not isinstance(report.get("runs"), list):
        raise ValueError("Report runs must be an array")
    seen: set[str] = set()
    for run in report["runs"]:
        if not isinstance(run, dict):
            raise ValueError("Each run must be an object")
        for field in ("runId", "trialId", "variant"):
            if not isinstance(run.get(field), str) or not run[field]:
                raise ValueError(f"Missing run {field}")
        if run["runId"] in seen:
            raise ValueError(f"Duplicate runId: {run['runId']}")
        seen.add(run["runId"])
        _number(run.get("rate"), "Configured trial rate")
        if not isinstance(run.get("summary"), dict):
            raise ValueError("Missing saved run summary")
        summary = run["summary"]
        _number(summary.get("successfulThroughputWithinWindowRps"), "Within-window throughput", nullable=True)
        _number(summary.get("successfulThroughputIncludingDrainRps"), "Including-drain throughput", nullable=True)
        for field in ("successfulRequests", "failedRequests"):
            if field in summary:
                _number(summary[field], field)
        tails = summary.get("successfulRequestLatencyMs", {})
        if not isinstance(tails, dict):
            raise ValueError("Latency summary must be an object")
        for percentile in ("p50", "p95", "p99"):
            _number(tails.get(percentile), f"Saved {percentile}", nullable=True)
        if not isinstance(run.get("latenciesMs"), list) or not isinstance(run.get("timeline"), list):
            raise ValueError("Request latencies and timeline must be arrays")
        for latency in run["latenciesMs"]:
            _number(latency, "Request latency")
        for point in run["timeline"]:
            if not isinstance(point, dict):
                raise ValueError("Timeline entries must be objects")
            _number(point.get("scheduledSeconds"), "Timeline time")
            _number(point.get("outstanding"), "Client outstanding", nullable=True)
            _number(point.get("latencyMs"), "Timeline latency", nullable=True)
    report_run_kind(report)
    return report


def select_representatives(report: dict[str, Any], rate: float | None = None) -> tuple[float | None, list[dict[str, Any]]]:
    """Choose an explicitly described individual trial; never pool repetitions."""
    if report_run_kind(report) != "open-loop":
        if rate is not None:
            raise ValueError("--rate requires an explicitly open-loop report; isolated/unknown modes have no offered rate")
        representatives = {}
        for run in report["runs"]:
            representatives.setdefault(run["variant"], run)
        return None, list(representatives.values())
    rates = sorted({run["rate"] for run in report["runs"]})
    if not rates:
        return None, []
    selected = rates[-1] if rate is None else rate
    if selected not in rates:
        raise ValueError(f"Requested rate {selected:g} is not recorded; available rates: {rates}")
    representatives = {}
    for run in report["runs"]:
        if run["rate"] == selected:
            representatives.setdefault(run["variant"], run)
    return selected, list(representatives.values())


def is_integration(report: dict[str, Any]) -> bool:
    markers = ("integration", "smoke", "fake", "synthetic", "test")
    if any(marker in report["purpose"].lower() for marker in markers):
        return True
    for run in report["runs"]:
        quality = run.get("quality") or {}
        reasons = quality.get("reasons", []) if isinstance(quality, dict) else []
        if any(any(marker in str(reason).lower() for marker in markers) for reason in reasons):
            return True
    return False


def render_report(report: dict[str, Any], output_dir: Path, *, rate: float | None = None) -> dict[str, str]:
    validate_report(report)
    run_kind = report_run_kind(report)
    open_loop = run_kind == "open-loop"
    isolated = run_kind == "isolated"
    selected_rate, representatives = select_representatives(report, rate)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    integration = is_integration(report)
    variants = list(dict.fromkeys(run["variant"] for run in report["runs"]))
    palette = plt.get_cmap("tab10")
    colors = {variant: palette(index % 10) for index, variant in enumerate(variants)}
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        throughput, histogram, cdf, outstanding = axes.flat
        fig.subplots_adjust(left=0.075, right=0.965, top=0.80, bottom=0.19, hspace=0.62, wspace=0.27)
        title = "\n".join(textwrap.wrap(f"Campaign evidence · {report['campaignId']}", 105))
        fig.suptitle(title, fontsize=16, y=0.982)
        banner = "INTEGRATION ONLY · Pipeline validation, not a reportable GPU performance result" if integration else "Saved campaign observations · no capacity or statistical significance inferred by this figure"
        fig.text(0.5, 0.935, banner, ha="center", fontsize=10, color="#8f3b12" if integration else "#425066",
                 bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fff3e8" if integration else "#eef2f7", "edgecolor": "none"})
        if open_loop:
            selected_label = f"{selected_rate:g} requests/s" if selected_rate is not None else "no recorded rate"
            selection_note = f"Throughput: every saved trial. Other panels: first recorded trial per variant at {selected_label}."
        else:
            mode_note = "Isolated serial requests: completion-paced, no offered arrival rate." if isolated else "Arrival mode unrecorded: no offered-load interpretation."
            selection_note = f"{mode_note}\nCompletion rate: every saved trial. Other panels: first recorded trial per variant."
        fig.text(0.075, 0.878, selection_note, fontsize=9, color="#425066")

        missing_throughput = 0
        throughput_key = "successfulThroughputWithinWindowRps" if open_loop else "successfulThroughputIncludingDrainRps"
        trial_sequence = {run["runId"]: index + 1 for index, run in enumerate(report["runs"])}
        for variant in variants:
            runs = [run for run in report["runs"] if run["variant"] == variant]
            valid = [run for run in runs if run["summary"].get(throughput_key) is not None]
            missing_throughput += len(runs) - len(valid)
            # Scatter each repetition at its actual rate; connecting trials would imply interpolation.
            for index, run in enumerate(valid):
                throughput.scatter(run["rate"] if open_loop else trial_sequence[run["runId"]], run["summary"][throughput_key],
                                   s=45, color=colors[variant], marker=markers[index % len(markers)],
                                   alpha=0.75, label=f"{variant} · {len(valid)} trial{'s' if len(valid) != 1 else ''}" if index == 0 else None)
        if report["runs"] and open_loop:
            maximum = max(run["rate"] for run in report["runs"])
            throughput.plot([0, maximum], [0, maximum], color="#9ca3af", linestyle=":", linewidth=1, label="Offered rate (reference)")
        if open_loop:
            throughput.set(title="Successful throughput during the arrival window", xlabel="Offered requests / second", ylabel="Successful completions / second")
        else:
            throughput.set(title="Serial successful completion rate" if isolated else "Saved successful completion rate", xlabel="Recorded trial sequence", ylabel="Successful completions / second (including drain)")
            throughput.xaxis.set_major_locator(MaxNLocator(integer=True))
        throughput.set_xlim(left=0)
        throughput.set_ylim(bottom=0)
        if not open_loop:
            throughput.set_xlim(0.5, max(len(report["runs"]), 1) + 0.5)
            maximum_completion_rate = max((run["summary"].get(throughput_key) or 0 for run in report["runs"]), default=0)
            throughput.set_ylim(0, maximum_completion_rate * 1.15 if maximum_completion_rate else 1)
        if throughput.get_legend_handles_labels()[0]:
            throughput.legend(loc="best", fontsize=8)
        if missing_throughput:
            throughput.text(0.02, 0.98, f"{missing_throughput} runs: saved completion rate unavailable", transform=throughput.transAxes, va="top", fontsize=8)

        representative_notes = []
        # Shared bin boundaries make per-run counts comparable; observations remain separate.
        all_latencies = [latency for run in representatives for latency in run["latenciesMs"]]
        histogram_bins = np.histogram_bin_edges(all_latencies, bins="auto") if all_latencies else [0, 1]
        for run in representatives:
            label = f"{run['variant']} · n={len(run['latenciesMs'])}"
            color = colors[run["variant"]]
            latencies = sorted(run["latenciesMs"])
            if latencies:
                histogram.hist(latencies, bins=histogram_bins, histtype="step", linewidth=1.6, color=color, label=label)
                # An empirical distribution of this run's successful observations, not a pooled estimator.
                cdf.step([latencies[0], *latencies], [0, *[(index + 1) / len(latencies) for index in range(len(latencies))]],
                         where="post", color=color, linewidth=1.6, label=label)
            points = sorted((point for point in run["timeline"] if point.get("outstanding") is not None), key=lambda point: point["scheduledSeconds"])
            if points:
                outstanding.scatter([point["scheduledSeconds"] for point in points], [point["outstanding"] for point in points],
                                    color=color, s=11, alpha=0.55, label=run["variant"])
            tails = run["summary"].get("successfulRequestLatencyMs", {})
            suppressed = [key for key in ("p95", "p99") if tails.get(key) is None]
            flags = f"; {', '.join(suppressed)} unavailable" if suppressed else ""
            failures = run["summary"].get("failedRequests")
            failed_note = f"; {failures} failed" if failures is not None else ""
            representative_notes.append(f"{run['variant']}: {run['trialId']} / {run['runId']}{failed_note}{flags}")
        histogram.set(title="Successful-request latency histogram", xlabel="Send-to-response latency (ms)", ylabel="Requests per bin")
        cdf.set(title="Successful-request empirical CDF", xlabel="Send-to-response latency (ms)", ylabel="Fraction of successful requests")
        cdf.set_ylim(0, 1.02)
        cdf.yaxis.set_major_formatter(PercentFormatter(1))
        outstanding.set(title="Client requests still outstanding", xlabel="Time from measurement start (s)", ylabel="Outstanding client requests")
        outstanding.set_ylim(bottom=0)
        outstanding.yaxis.set_major_locator(MaxNLocator(integer=True))
        for axis in (histogram, cdf, outstanding):
            axis.set_xlim(left=0)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(loc="best", fontsize=8)
            else:
                axis.text(0.5, 0.5, "No saved observations available", ha="center", va="center", transform=axis.transAxes, color="#64748b")
        for axis in axes.flat:
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
        limitations = report.get("limitations", [])
        if isinstance(limitations, list):
            representative_notes.extend(f"Limitation: {note}" for note in limitations if isinstance(note, str))
        notes = "\n".join("\n".join(textwrap.wrap(note, 155)) for note in representative_notes)
        if not notes:
            notes = "No individual trials available."
        caution = "Latencies describe successful requests only. Repetitions are not pooled. Outstanding client requests are not a server queue measurement."
        fig.text(0.075, 0.135, caution, fontsize=8, color="#425066", va="top")
        fig.text(0.075, 0.11, notes, fontsize=7.5, color="#425066", va="top")
        paths = {extension: str(output_dir / f"campaign_overview.{extension}") for extension in ("png", "svg")}
        try:
            for extension, filename in paths.items():
                fig.savefig(filename, format=extension, dpi=170, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="Saved measurement-report JSON")
    parser.add_argument("--output-dir", type=Path, help="Default: the report's directory")
    parser.add_argument("--rate", type=float, help="Open-loop reports only: rate for individual-run distributions; default: highest recorded")
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text())
        paths = render_report(report, args.output_dir or args.report.parent, rate=args.rate)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"Unable to plot campaign: {error}\n")
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
