"""
WP3 driver: run the instrumented server (scripts/wp3_diag_server.py) under Rithvik's
open-loop load generator (csnbs/measure, unmodified) at a fixed rate.

Sessions = {576, 128} x {torch default threads, 4 threads} x repeats, in a seeded random
order (rule 10). The thread arm tests the competing explanation from STATUS.md: the
server sets no thread count, so it may pay the ~55-70 ms CFS-throttling stall
bench_dev.py hit before its 4-thread pin. nr_throttled is read before and after each
load run.

    python scripts/wp3_run_diag.py --run-id wp3_diag_2 [--rps 2.0 --duration 60 --repeats 2]

Writes results/timing/wp3_diag/<run-id>/<session>/{requests.jsonl,heartbeat.json,
monitor.csv,session.json} plus <run-id>/run.json. Analyse with scripts/wp3_analyze.py.
"""
import argparse
import glob
import json
import os
import random
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
HOST, PORT = "127.0.0.1", 8000
BASE_URL = f"http://{HOST}:{PORT}"
DATASET_PATH = REPO_ROOT / "vis_pruner_copy" / "vispruner_eval_dataset" / "dev.json"
CGROUP_STAT_PATHS = ("/sys/fs/cgroup/cpu/cpu.stat", "/sys/fs/cgroup/cpu.stat")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def gpu_name():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()


def gpu_busy():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    return [line for line in out.splitlines() if line.strip()]


def cgroup_cpu_stat():
    for path in CGROUP_STAT_PATHS:
        try:
            with open(path) as f:
                return {k: int(v) for k, v in (line.split() for line in f)}
        except OSError:
            continue
    return None


def port_is_open():
    try:
        with socket.create_connection((HOST, PORT), timeout=1):
            return True
    except OSError:
        return False


def http_json(path, method="GET"):
    req = Request(BASE_URL + path, method=method, data=b"" if method == "POST" else None)
    with urlopen(req, timeout=10) as resp:
        return json.load(resp)


def wait_for_server(proc, timeout_s=600):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("server exited before becoming ready")
        try:
            health = http_json("/health")
            if health.get("model_loaded") and health.get("pid") == proc.pid:
                return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError("server did not become ready")


class GpuMonitor:
    """1 Hz nvidia-smi utilisation/memory samples, same columns as run_ab_load_sweep.py."""

    def __init__(self, path):
        self.path, self._stop, self._thread = path, threading.Event(), None

    def _run(self):
        t0 = time.perf_counter()
        with open(self.path, "w") as f:
            f.write("elapsed_s,gpu_util_pct,gpu_mem_used_mib\n")
            while not self._stop.wait(1.0):
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                         "--format=csv,noheader,nounits"], timeout=2, text=True)
                    util, mem = [x.strip() for x in out.strip().split(",")]
                except Exception:
                    util, mem = "", ""
                f.write(f"{time.perf_counter() - t0:.2f},{util},{mem}\n")
                f.flush()

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


def run_session(session_dir, vtn, threads, args):
    session_dir.mkdir(parents=True)
    env = dict(os.environ, WP3_OUT_DIR=str(session_dir), VISUAL_TOKEN_NUM=str(vtn),
               MODEL_SERVER_PORT=str(PORT))
    env.pop("WP3_TORCH_THREADS", None)
    if threads:
        env["WP3_TORCH_THREADS"] = str(threads)

    log = open(session_dir / "server.log", "w")
    proc = subprocess.Popen([sys.executable, str(REPO_ROOT / "scripts" / "wp3_diag_server.py")],
                            cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_for_server(proc)
        http_json("/wp3/reset", "POST")
        cpu_before = cgroup_cpu_stat()
        started = time.time()
        with GpuMonitor(session_dir / "monitor.csv"):
            subprocess.run(
                ["npm", "run", "dev", "--", "--endpoint", f"{BASE_URL}/infer",
                 "--rps", str(args.rps), "--duration", str(args.duration),
                 "--timeout", "300000", "--dataset", str(DATASET_PATH)],
                cwd=REPO_ROOT / "csnbs" / "measure", check=True,
                stdout=open(session_dir / "loadgen.log", "w"), stderr=subprocess.STDOUT)
        cpu_after = cgroup_cpu_stat()
        flushed = http_json("/wp3/flush", "POST")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()

    fresh = [p for p in glob.glob(str(REPO_ROOT / "results/loadgen/*/*/run.json"))
             if os.path.getmtime(p) >= started - 1]
    run_json = max(fresh, key=os.path.getmtime) if fresh else None
    meta = {
        "visual_token_num": vtn,
        "torch_threads": threads or "default",
        "rps": args.rps,
        "duration_s": args.duration,
        "loadgen_run_json": os.path.relpath(run_json, REPO_ROOT) if run_json else None,
        "server_counts": flushed,
        "cgroup_before": cpu_before,
        "cgroup_after": cpu_after,
        "nr_throttled_delta": (cpu_after["nr_throttled"] - cpu_before["nr_throttled"])
        if cpu_before and cpu_after else None,
    }
    (session_dir / "session.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rps", type=float, default=2.0)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--configs", default="576,128")
    ap.add_argument("--threads", default="0,4", help="0 = torch default")
    ap.add_argument("--seed", type=int, default=20261025)
    args = ap.parse_args()

    out_root = REPO_ROOT / "results" / "timing" / "wp3_diag" / args.run_id
    if out_root.exists():
        raise SystemExit(f"{out_root} exists; pick a fresh --run-id")
    if port_is_open():
        raise SystemExit(f"port {PORT} in use")
    if gpu_busy():
        raise SystemExit("another process is on the GPU (rule 9)")

    sessions = [(vtn, thr, rep)
                for rep in range(args.repeats)
                for vtn in (int(x) for x in args.configs.split(","))
                for thr in (int(x) for x in args.threads.split(","))]
    random.Random(args.seed).shuffle(sessions)

    out_root.mkdir(parents=True)
    run_meta = {
        "run_id": args.run_id,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "gpu": gpu_name(),
        "cpu_quota": open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read().strip()
        if os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") else None,
        "args": vars(args),
        "session_order": [],
    }
    for i, (vtn, thr, rep) in enumerate(sessions):
        name = f"{i:02d}_vtn{vtn}_thr{thr or 'default'}_rep{rep}"
        print(f"[{i + 1}/{len(sessions)}] {name}", flush=True)
        meta = run_session(out_root / name, vtn, thr, args)
        run_meta["session_order"].append(name)
        print(f"    requests={meta['server_counts']['requests']} "
              f"nr_throttled_delta={meta['nr_throttled_delta']}", flush=True)
        (out_root / "run.json").write_text(json.dumps(run_meta, indent=2))


if __name__ == "__main__":
    main()
