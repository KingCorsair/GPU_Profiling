"""
One-off diagnostic (not part of the tracked timing harness): find out what
limits serving performance as concurrency increases, by reusing Rithvik's
existing open-loop load generator (csnbs/measure, the same tool
run_load_sweep.py already drives) and Rithvik's model server
(scripts/start_model_server.sh) completely unmodified -- this script only
orchestrates them and adds read-only GPU/CPU monitoring alongside.

Per CLAUDE.md's ownership rules ("do not write code in someone else's
directory" / load generation and percentile logic are Rithvik's to write),
nothing about arrival scheduling or percentile computation is reimplemented
here -- it comes straight from csnbs/measure's existing run.json output.
This script's only job is: pick a concurrency/RPS progression, run it,
capture GPU/CPU utilization alongside each level, and fold the existing
per-level run.json files into one comparison table.

Throwaway, per AMAY_SPEED_PLAN.md / AMAY_TIMING_NOTES.md's "Amay measures to
find bottlenecks" split -- mine is throwaway, only Rithvik's harness output
(the run.json files this writes) is a reportable number. No optimizations
implemented here.
"""
import glob
import json
import os
import random
import socket
import subprocess
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

REPO_ROOT = "/workspace/GPU_Profiling"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
SERVICE_NAME = "csnbs-llava-server"
INFER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/infer"

# Progression: calibrated against a measured steady-state single-request
# latency of ~0.56s (see results/timing/loadgen_calibration.json) on this
# server's actual settings (visual_token_num=576, max_new_tokens=64,
# hardcoded in csnbs/server.py -- not configurable per-request). The server
# handles one request at a time (csnbs/server.py's /infer handler blocks the
# single asyncio event loop for the full duration of model.generate(), with
# no threading/batching), so its sustainable open-loop throughput ceiling is
# roughly 1 / (mean service time) =~ 1.8 rps for short answers, less for
# longer ones. This progression starts clearly under that ceiling and
# crosses clearly over it.
RPS_LEVELS = [0.5, 1.0, 1.5, 2.0, 3.0]
DURATION_SECONDS = 30
TIMEOUT_MS = 120000
DATASET_PATH = f"{REPO_ROOT}/vis_pruner_copy/vispruner_eval_dataset/dev.json"

RESULTS_DIR = f"{REPO_ROOT}/results/timing"
MONITOR_DIR = f"{REPO_ROOT}/results/timing/loadgen_monitor"
METADATA_PATH = f"{REPO_ROOT}/results/timing/concurrent_load_metadata.json"
SUMMARY_TABLE_PATH = f"{REPO_ROOT}/results/timing/concurrent_load_summary.txt"

os.makedirs(MONITOR_DIR, exist_ok=True)


def port_is_open() -> bool:
    try:
        with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=1):
            return True
    except OSError:
        return False


def read_health() -> dict:
    with urlopen(HEALTH_URL, timeout=2) as response:
        return json.load(response)


def wait_for_server(process, timeout_seconds=300):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited before becoming ready")
        try:
            health = read_health()
            if health.get("model_loaded") and health.get("pid") == process.pid:
                return
        except (URLError, TimeoutError, ConnectionError, OSError):
            pass
        time.sleep(2)
    raise TimeoutError("Server did not become ready within timeout")


def read_proc_stat():
    with open("/proc/stat") as f:
        parts = f.readline().split()
    vals = [int(x) for x in parts[1:]]
    idle = vals[3] + vals[4]  # idle + iowait
    total = sum(vals)
    return idle, total


class ResourceMonitor:
    """Samples nvidia-smi (GPU util/mem) and /proc/stat (CPU busy%) at ~1Hz
    for the duration of one load level. Read-only observation, no
    interaction with the server or load generator -- doesn't touch anyone's
    ownership area."""

    def __init__(self, out_csv_path: str):
        self.out_csv_path = out_csv_path
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        prev_idle, prev_total = read_proc_stat()
        with open(self.out_csv_path, "w") as f:
            f.write("elapsed_s,gpu_util_pct,gpu_mem_used_mib,gpu_mem_total_mib,cpu_busy_pct\n")
            t0 = time.perf_counter()
            while not self._stop.is_set():
                time.sleep(1.0)
                elapsed = time.perf_counter() - t0
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi",
                         "--query-gpu=utilization.gpu,memory.used,memory.total",
                         "--format=csv,noheader,nounits"],
                        timeout=2,
                    ).decode().strip()
                    gpu_util, mem_used, mem_total = [x.strip() for x in out.split(",")]
                except Exception:
                    gpu_util, mem_used, mem_total = "", "", ""

                idle, total = read_proc_stat()
                d_idle, d_total = idle - prev_idle, total - prev_total
                cpu_busy_pct = 100.0 * (1 - d_idle / d_total) if d_total > 0 else float("nan")
                prev_idle, prev_total = idle, total

                f.write(f"{elapsed:.2f},{gpu_util},{mem_used},{mem_total},{cpu_busy_pct:.1f}\n")
                f.flush()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def summarize_monitor_csv(path: str) -> dict:
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 5 or parts[1] == "":
                continue
            rows.append(parts)
    if not rows:
        return {}
    gpu_utils = [float(r[1]) for r in rows]
    mem_used = [float(r[2]) for r in rows]
    cpu_busy = [float(r[4]) for r in rows if r[4] not in ("", "nan")]
    return {
        "samples": len(rows),
        "gpu_util_pct_mean": round(sum(gpu_utils) / len(gpu_utils), 1),
        "gpu_util_pct_max": round(max(gpu_utils), 1),
        "gpu_mem_used_mib_mean": round(sum(mem_used) / len(mem_used), 1),
        "gpu_mem_used_mib_max": round(max(mem_used), 1),
        "cpu_busy_pct_mean": round(sum(cpu_busy) / len(cpu_busy), 1) if cpu_busy else None,
        "cpu_busy_pct_max": round(max(cpu_busy), 1) if cpu_busy else None,
    }


def find_latest_run_json(since_ts: float) -> str:
    candidates = glob.glob(f"{REPO_ROOT}/results/loadgen/*/*/run.json")
    fresh = [p for p in candidates if os.path.getmtime(p) >= since_ts]
    if not fresh:
        raise RuntimeError("No new results/loadgen/*/run.json produced by this run")
    fresh.sort(key=os.path.getmtime)
    return fresh[-1]


def main():
    if port_is_open():
        raise RuntimeError(f"Port {SERVER_PORT} already in use; stop the existing server first")

    server = subprocess.Popen(
        ["bash", f"{REPO_ROOT}/scripts/start_model_server.sh"],
        cwd=REPO_ROOT,
    )
    levels_results = []
    try:
        print("Waiting for model server...")
        wait_for_server(server)
        print("Server ready.")

        # Randomize execution order (rule 10: don't confound thermal drift
        # with the variable under test) but report sorted by RPS afterward.
        order = list(RPS_LEVELS)
        random.Random(2026).shuffle(order)

        for rps in order:
            print(f"\n--- RPS level: {rps} ---")
            health = read_health()
            if health.get("pid") != server.pid or not health.get("model_loaded"):
                raise RuntimeError(f"Unexpected server state before rps={rps}: {health}")

            monitor_csv = f"{MONITOR_DIR}/rps-{rps}.csv"
            monitor = ResourceMonitor(monitor_csv)
            monitor.start()
            run_start_ts = time.time()

            subprocess.run(
                ["npm", "run", "dev", "--",
                 "--endpoint", INFER_URL,
                 "--rps", str(rps),
                 "--duration", str(DURATION_SECONDS),
                 "--timeout", str(TIMEOUT_MS),
                 "--dataset", DATASET_PATH],
                cwd=f"{REPO_ROOT}/csnbs/measure",
                check=True,
            )

            monitor.stop()
            run_json_path = find_latest_run_json(run_start_ts - 1)
            with open(run_json_path) as f:
                run_data = json.load(f)
            resource_summary = summarize_monitor_csv(monitor_csv)

            levels_results.append({
                "rps_target": rps,
                "run_json_path": run_json_path,
                "monitor_csv_path": monitor_csv,
                "run_summary": run_data["summary"],
                "resource_summary": resource_summary,
            })
            print(f"  -> {run_json_path}")
            print(f"  -> p50={run_data['summary']['successfulRequestLatencyMs']['p50']:.0f}ms "
                  f"p95={run_data['summary']['successfulRequestLatencyMs']['p95']:.0f}ms "
                  f"throughput={run_data['summary']['successfulThroughputRps']:.3f}rps "
                  f"gpu_util_mean={resource_summary.get('gpu_util_pct_mean')}% "
                  f"cpu_busy_mean={resource_summary.get('cpu_busy_pct_mean')}%")

            check = read_health()
            if check.get("pid") != server.pid or not check.get("model_loaded"):
                raise RuntimeError(f"Server unhealthy after rps={rps}: {check}")

        print("\nLoad-test sweep completed.")

    finally:
        print("Stopping server...")
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

    # Sort by RPS for reporting even though execution order was randomized.
    levels_results.sort(key=lambda r: r["rps_target"])

    metadata = {
        "rps_levels_target": RPS_LEVELS,
        "execution_order_rps": order,
        "duration_seconds_per_level": DURATION_SECONDS,
        "dataset_path": DATASET_PATH,
        "dataset_record_count": 90,
        "server_settings": {
            "note": "hardcoded in csnbs/server.py, not configurable per-request",
            "visual_token_num": 576,
            "important_ratio": 0.5,
            "max_new_tokens": 64,
            "do_sample": False,
            "use_cache": True,
        },
        "server_concurrency_model": (
            "single uvicorn worker, single asyncio event loop; /infer's model "
            "path (csnbs/server.py:_infer_model) calls model.generate() "
            "synchronously with no await/executor handoff, so it blocks the "
            "event loop for the full request -- requests are serviced strictly "
            "one at a time regardless of arrival concurrency"
        ),
        "levels": levels_results,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written to {METADATA_PATH}")

    lines = [
        f"Concurrent load sweep -- {DURATION_SECONDS}s per level, dataset={DATASET_PATH}",
        "Server settings: visual_token_num=576 important_ratio=0.5 max_new_tokens=64 (hardcoded)",
        "",
        f"{'rps_target':>10}  {'throughput':>10}  {'p50_ms':>8}  {'p95_ms':>8}  {'p99_ms':>8}  "
        f"{'failed':>6}  {'gpu_util%':>9}  {'gpu_mem_MiB':>11}  {'cpu_busy%':>9}",
    ]
    for lvl in levels_results:
        s = lvl["run_summary"]
        r = lvl["resource_summary"]
        lat = s["successfulRequestLatencyMs"]
        lines.append(
            f"{lvl['rps_target']:>10}  {s['successfulThroughputRps']:>10.3f}  "
            f"{lat['p50']:>8.0f}  {lat['p95']:>8.0f}  {lat['p99']:>8.0f}  "
            f"{s['failedRequests']:>6}  {r.get('gpu_util_pct_mean', 'n/a'):>9}  "
            f"{r.get('gpu_mem_used_mib_mean', 'n/a'):>11}  {r.get('cpu_busy_pct_mean', 'n/a'):>9}"
        )
    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    with open(SUMMARY_TABLE_PATH, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary table written to {SUMMARY_TABLE_PATH}")


if __name__ == "__main__":
    main()
