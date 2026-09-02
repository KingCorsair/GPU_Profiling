"""
One-off diagnostic (not part of the tracked timing harness): Part 2 of the
VisPruner A/B experiment -- run ONE visual_token_num configuration through
Rithvik's existing open-loop load generator (csnbs/measure) and model server
(scripts/start_model_server.sh), both unmodified except for the one-line,
backward-compatible env-var parameterization already added to
csnbs/server.py (VISUAL_TOKEN_NUM, defaults to 576 exactly as before).

This is a generalized version of run_concurrent_load_test.py, parameterized
by VISUAL_TOKEN_NUM so the same script and settings run both A/B legs. The
visual_token_num=576 leg was already run in that earlier script (identical
RPS levels, duration, dataset, server startup path -- the only behavioral
difference from then to now is that VISUAL_TOKEN_NUM is now read from an
env var defaulting to the same 576, so that run's results
(results/timing/concurrent_load_summary.txt /
concurrent_load_metadata.json) are reused as-is for Configuration A rather
than re-run here, to avoid burning a redundant ~5 minutes of GPU time.
This script runs Configuration B (128).

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
import sys
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

VISUAL_TOKEN_NUM = int(sys.argv[1]) if len(sys.argv) > 1 else 128
# Same levels as the 576 run for direct comparability; extend if saturation
# isn't clearly visible by the top of this range.
RPS_LEVELS = [0.5, 1.0, 1.5, 2.0, 3.0]
DURATION_SECONDS = 30
TIMEOUT_MS = 120000
DATASET_PATH = f"{REPO_ROOT}/vis_pruner_copy/vispruner_eval_dataset/dev.json"

MONITOR_DIR = f"{REPO_ROOT}/results/timing/loadgen_monitor"
METADATA_PATH = f"{REPO_ROOT}/results/timing/concurrent_load_metadata_vtn{VISUAL_TOKEN_NUM}.json"
SUMMARY_TABLE_PATH = f"{REPO_ROOT}/results/timing/concurrent_load_summary_vtn{VISUAL_TOKEN_NUM}.txt"

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
    idle = vals[3] + vals[4]
    total = sum(vals)
    return idle, total


class ResourceMonitor:
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

    env = dict(os.environ)
    env["VISUAL_TOKEN_NUM"] = str(VISUAL_TOKEN_NUM)
    server = subprocess.Popen(
        ["bash", f"{REPO_ROOT}/scripts/start_model_server.sh"],
        cwd=REPO_ROOT, env=env,
    )
    levels_results = []
    order = list(RPS_LEVELS)
    try:
        print(f"Waiting for model server (visual_token_num={VISUAL_TOKEN_NUM})...")
        wait_for_server(server)
        print("Server ready.")

        random.Random(7).shuffle(order)  # different seed than the 576 run's ordering

        for rps in order:
            print(f"\n--- visual_token_num={VISUAL_TOKEN_NUM}  RPS level: {rps} ---")
            health = read_health()
            if health.get("pid") != server.pid or not health.get("model_loaded"):
                raise RuntimeError(f"Unexpected server state before rps={rps}: {health}")

            monitor_csv = f"{MONITOR_DIR}/vtn{VISUAL_TOKEN_NUM}_rps-{rps}.csv"
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

        print(f"\nLoad-test sweep completed for visual_token_num={VISUAL_TOKEN_NUM}.")

    finally:
        print("Stopping server...")
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

    levels_results.sort(key=lambda r: r["rps_target"])

    metadata = {
        "visual_token_num": VISUAL_TOKEN_NUM,
        "rps_levels_target": RPS_LEVELS,
        "execution_order_rps": order,
        "duration_seconds_per_level": DURATION_SECONDS,
        "dataset_path": DATASET_PATH,
        "dataset_record_count": 90,
        "server_settings": {
            "visual_token_num": VISUAL_TOKEN_NUM,
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
        f"Concurrent load sweep (visual_token_num={VISUAL_TOKEN_NUM}) -- {DURATION_SECONDS}s per level, "
        f"dataset={DATASET_PATH}",
        f"Server settings: visual_token_num={VISUAL_TOKEN_NUM} important_ratio=0.5 max_new_tokens=64",
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
