"""Short, standalone CUDA timing calibration; these are not serving results.

Run on a reserved idle GPU, without installing packages or loading a model:
  python csnbs/gpu_calibration.py --output results/calibration --iterations 100

CUDA events bracket each operation on the current stream. Synchronization occurs
only outside the warmup/measurement loops. See PyTorch's timing contract:
https://docs.pytorch.org/docs/2.2/notes/cuda.html#asynchronous-execution
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any, Callable
from uuid import uuid4


def distribution(values: list[float]) -> dict[str, Any]:
    """Nearest-rank descriptive quantiles, without inferring independent trials."""
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Expected nonempty finite nonnegative durations")
    ordered = sorted(values)
    return {
        "count": len(values), "min": ordered[0], "max": ordered[-1],
        **{name: ordered[math.ceil(q * len(ordered)) - 1]
           for name, q in (("p50", .5), ("p95", .95), ("p99", .99))},
        "units": "ms", "definition": "nearest-rank",
        "tailInterpretation": "descriptive calibration only; no serving-tail or uncertainty claim",
    }


def _count(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 10000:
        raise ValueError(f"{name} must be an integer in [{minimum}, 10000]")


def time_gpu(
    operation: Callable[[], Any], *, cuda: Any, warmup: int = 10,
    iters: int = 100, before_each: Callable[[], Any] | None = None,
) -> dict[str, list[float]]:
    """Return per-operation event samples; the callback uses the current stream.

    before_each can enqueue an eviction-buffer write on that same stream. It is
    ordered before the start event and excluded from the event interval. Neither
    callback may synchronize, allocate a new stream, or perform CPU transfers.
    Events are preallocated and initialized before the first operation.
    """
    _count(warmup, "warmup", 0)
    _count(iters, "iters")
    pairs = [(cuda.Event(enable_timing=True), cuda.Event(enable_timing=True))
             for _ in range(warmup + iters)]
    # CUDA events initialize lazily. Initialize all handles before sampling.
    for start, end in pairs:
        start.record()
        end.record()
    cuda.synchronize()

    def enqueue(selected: list[tuple[Any, Any]]) -> None:
        for start, end in selected:
            if before_each is not None:
                before_each()
            start.record()
            operation()
            end.record()

    enqueue(pairs[:warmup])
    cuda.synchronize()  # Warmup fully drains before measured operations begin.
    enqueue(pairs[warmup:])
    cuda.synchronize()  # One final drain; no per-iteration synchronization.
    elapsed = [float(start.elapsed_time(end)) for start, end in pairs]
    if any(not math.isfinite(value) or value < 0 for value in elapsed):
        raise RuntimeError("CUDA returned an invalid event interval")
    return {"warmupGpuMs": elapsed[:warmup], "measuredGpuMs": elapsed[warmup:]}


def time_host_enqueue(operation: Callable[[], Any], *, cuda: Any, iters: int = 100) -> list[float]:
    """Separate demonstration: CPU call duration, NOT completed GPU latency.

    A launch may itself block; this number is not a pure launch-overhead estimate
    and need not always be smaller than an event interval.
    """
    _count(iters, "iters")
    cuda.synchronize()
    values = []
    for _ in range(iters):
        started = time.perf_counter()
        operation()
        values.append((time.perf_counter() - started) * 1000)
    cuda.synchronize()
    return values


def command(args: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.PIPE, timeout=10).strip()


def gpu_processes() -> list[str]:
    text = command(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name", "--format=csv,noheader"])
    return [line.strip() for line in text.splitlines() if line.strip()]


def source_metadata(declared_commit: str | None) -> dict[str, Any]:
    source = Path(__file__).resolve()
    commit = dirty = None
    try:
        commit = command(["git", "rev-parse", "HEAD"], cwd=source.parent)
        dirty = bool(command(["git", "status", "--porcelain=v1", "--untracked-files=normal"], cwd=source.parent))
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "gitCommit": commit or declared_commit,
        "gitCommitSource": "local-git" if commit else ("operator-declared" if declared_commit else "unavailable"),
        "gitDirty": dirty, "sourceFile": source.name,
        "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "note": "The script SHA-256 identifies these exact bytes even in an isolated uncommitted deployment.",
    }


def save_result(root: Path, manifest: dict[str, Any]) -> Path:
    """Write one new immutable artifact directory, never overwrite an old run."""
    destination = root / manifest["runId"]
    destination.mkdir(parents=True, exist_ok=False)
    samples = []
    for trial in manifest["trials"]:
        for field, phase, clock in (("warmupGpuMs", "warmup", "cuda-event"),
                                    ("measuredGpuMs", "measurement", "cuda-event"),
                                    ("hostEnqueueMs", "host-enqueue-demonstration", "perf_counter")):
            for sequence, value in enumerate(trial.get(field, [])):
                samples.append({"trialId": trial["trialId"], "condition": trial["condition"],
                                "phase": phase, "clock": clock, "sequence": sequence, "durationMs": value})
    raw = "".join(json.dumps(sample, allow_nan=False) + "\n" for sample in samples)
    with (destination / "samples.jsonl").open("x") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    manifest["samples"] = {"file": "samples.jsonl", "count": len(samples),
                           "sha256": hashlib.sha256(raw.encode()).hexdigest()}
    with (destination / "calibration.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def plot_result(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Optional standard plots; raw evidence remains usable without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        return {"available": False, "reason": str(error)}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    conditions = list(dict.fromkeys(trial["condition"] for trial in manifest["trials"]))
    for condition in conditions:
        first = next(trial for trial in manifest["trials"] if trial["condition"] == condition)
        values = first["warmupGpuMs"] + first["measuredGpuMs"]
        axes[0].plot(range(len(values)), values, label=condition, linewidth=1)
        pooled = sorted(value for trial in manifest["trials"] if trial["condition"] == condition for value in trial["measuredGpuMs"])
        axes[1].step(pooled, [(i + 1) / len(pooled) for i in range(len(pooled))], label=condition, where="post")
    axes[0].axvline(manifest["config"]["warmup"] - .5, color="black", linestyle="--", linewidth=.8, label="measurement begins")
    axes[0].set(xlabel="Operation index in first trial (warmup shown)", ylabel="CUDA event interval (ms)")
    axes[1].set(xlabel="Measured CUDA event interval (ms)", ylabel="Empirical CDF")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("CUDA operation calibration — not model serving performance")
    fig.savefig(directory / "calibration.png", dpi=150)
    fig.savefig(directory / "calibration.svg")
    plt.close(fig)
    return {"available": True, "files": ["calibration.png", "calibration.svg"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("results/calibration"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--matrix-size", type=int, default=1024)
    parser.add_argument("--elements", type=int, default=1024 * 1024)
    parser.add_argument("--flush-mib", type=int, default=256, help="0 disables eviction-buffer condition")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--source-commit", help="Parent checkout commit for an isolated script deployment")
    args = parser.parse_args()
    _count(args.iterations, "iterations")
    _count(args.warmup, "warmup", 10)
    _count(args.repetitions, "repetitions")
    if args.repetitions > 100 or not 1 <= args.matrix_size <= 4096 or not 1 <= args.elements <= 64 * 1024 * 1024 or not 0 <= args.flush_mib <= 1024:
        raise ValueError("Calibration size is outside the bounded supported range")
    if args.source_commit and (len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit)):
        raise ValueError("--source-commit must be a full lowercase Git commit")
    if os.environ.get("CUDA_LAUNCH_BLOCKING") not in (None, "", "0"):
        raise RuntimeError("Unset CUDA_LAUNCH_BLOCKING for asynchronous calibration")
    before = gpu_processes()
    if before:
        raise RuntimeError("GPU has active compute processes; refusing concurrent calibration: " + "; ".join(before))
    hardware_before = command(["nvidia-smi", "--query-gpu=uuid,name,driver_version,temperature.gpu,power.draw,clocks.sm", "--format=csv"])

    # Import only after the idle check, and permit CPU-only unit tests to import this module.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no synthetic GPU numbers will be generated")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    trials = []
    started = time.perf_counter()
    with torch.inference_mode():
        a = torch.randn((args.matrix_size, args.matrix_size), device=device, dtype=torch.float32)
        b = torch.randn_like(a)
        product = torch.empty_like(a)
        values = torch.randn(args.elements, device=device, dtype=torch.float32)
        scaled = torch.empty_like(values)
        eviction = torch.empty(args.flush_mib * 1024 * 1024, device=device, dtype=torch.uint8) if args.flush_mib else None
        operations = {
            "matmul-reused-input": (lambda: torch.mm(a, b, out=product), None),
            "scale-reused-input": (lambda: torch.mul(values, 2.0, out=scaled), None),
        }
        if eviction is not None:
            operations["scale-eviction-buffer"] = (lambda: torch.mul(values, 2.0, out=scaled), eviction.zero_)
        rng = random.Random(args.seed)
        for repetition in range(args.repetitions):
            order = list(operations)
            rng.shuffle(order)
            for condition in order:
                operation, prepare = operations[condition]
                measured = time_gpu(operation, cuda=torch.cuda, warmup=args.warmup, iters=args.iterations, before_each=prepare)
                trial = {"trialId": f"r{repetition + 1}-{condition}", "repetition": repetition,
                         "condition": condition, **measured,
                         "gpuEventSummaryMs": distribution(measured["measuredGpuMs"])}
                # Separate hot-input demonstration; no cache-flush cost hides in CPU call timing.
                if prepare is None:
                    trial["hostEnqueueMs"] = time_host_enqueue(operation, cuda=torch.cuda, iters=args.iterations)
                    trial["hostEnqueueSummaryMs"] = distribution(trial["hostEnqueueMs"])
                trials.append(trial)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    after = gpu_processes()
    unexpected = [line for line in after if line.split(",", 1)[0].strip() != str(os.getpid())]
    now = datetime.now(timezone.utc)
    manifest = {
        "schema": "cuda-operation-calibration", "schemaVersion": 1,
        "runId": now.strftime("%Y-%m-%dT%H-%M-%SZ_") + uuid4().hex[:8],
        "recordedAtUtc": now.isoformat(), "runKind": "diagnostic-calibration",
        "reportableServingResult": False, "source": source_metadata(args.source_commit),
        "config": {"warmup": args.warmup, "iterations": args.iterations, "repetitions": args.repetitions,
                   "seed": args.seed, "matrixSize": args.matrix_size, "elements": args.elements,
                   "flushMiB": args.flush_mib, "dtype": "float32", "allowTf32": False},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "hardware": {"gpuModel": properties.name, "totalMemoryBytes": properties.total_memory,
                     "nvidiaSmiBefore": hardware_before,
                     "nvidiaSmiAfter": command(["nvidia-smi", "--query-gpu=uuid,name,driver_version,temperature.gpu,power.draw,clocks.sm", "--format=csv"])},
        "exclusivity": {"processesBefore": before, "processesAfter": after, "unexpectedProcessesAfter": unexpected,
                        "observedUncontended": not unexpected, "note": "Boundary checks cannot detect short-lived overlapping work."},
        "calibrationWallSeconds": elapsed, "trials": trials,
        "limitations": [
            "Synthetic operations; no LLaVA, pruning, throughput, or model-accuracy conclusion.",
            "CUDA event intervals include any stream idle gaps between host launches; not isolated kernel profiler timings.",
            "Host enqueue measurements come from separate warmed executions and are not GPU completion latency.",
            "Reused inputs are cache-friendly, not proof of L2 hits. Eviction-buffer writes perturb cache but do not prove fully cold data.",
            "Eviction-buffer work is ordered before and excluded from each timed CUDA event interval.",
            "Warmup samples show early operations after tensor/context setup; process startup and tensor allocation are excluded.",
            "Raw trial samples and descriptive nearest-rank quantiles are retained; no inferential independence or stable p99 claim.",
        ],
    }
    destination = save_result(args.output, manifest)
    plotting = plot_result(destination, manifest)
    print(json.dumps({"outputDirectory": str(destination.resolve()), "runId": manifest["runId"],
                      "trials": len(trials), "calibrationWallSeconds": elapsed,
                      "observedUncontended": not unexpected, "plot": plotting,
                      "summaries": [{"trialId": trial["trialId"], "gpuEventSummaryMs": trial["gpuEventSummaryMs"]} for trial in trials]}, indent=2))


if __name__ == "__main__":
    main()
