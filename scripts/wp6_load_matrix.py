"""
WP6 load matrix driver: {S0, S1, S2} x {576, 128} x {fixed, natural} under Rithvik's
open-loop load generator (csnbs/measure, unmodified).

    python scripts/wp6_load_matrix.py --run-id wp6_matrix_1
    python scripts/wp6_load_matrix.py --run-id wp6_pilot --states S0,S2 --configs 128 \
        --policies natural --levels-s0 2 --levels-s2 4 --min-samples 40

Design (roadmap WP6 section 5):
  * Every (cell, rps level) is one block. Blocks run in a seeded random order, so run
    order is never confounded with a configuration (the old sweeps ran as two blocks).
    A server process holds one cell; it restarts whenever the next block's cell differs.
  * Block duration = max(--min-duration, (min_samples + 10 warm-up) / rps), so every
    level has >= min_samples post-warm-up requests for p95. p99 is only reported by
    wp6_analyze.py where n >= 1000.
  * Server states: S0 = csnbs/server.py via scripts/wp6_s0_server.py, S1/S2 =
    vis_pruner_copy/llava/serve/batch_server.py. Same torch thread count for all.
  * Policies: fixed = min_new_tokens = max_new_tokens = 64; natural = EOS, cap 64.
  * 1 Hz GPU utilisation/memory sampling per block; commit + GPU recorded.

Reportable numbers come from Rithvik running this (CLAUDE.md); runs by Amay are pilots.
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
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
HOST, PORT = "127.0.0.1", 8000
BASE_URL = f"http://{HOST}:{PORT}"
DATASET_PATH = REPO_ROOT / "vis_pruner_copy" / "vispruner_eval_dataset" / "dev.json"
LOADGEN_WARMUP = 10   # csnbs/measure drops its first 10 requests from the stats


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def port_is_open():
    try:
        with socket.create_connection((HOST, PORT), timeout=1):
            return True
    except OSError:
        return False


def server_cmd_env(cell, args):
    state, vtn, policy = cell
    env = dict(os.environ, VISUAL_TOKEN_NUM=str(vtn), MODEL_SERVER_PORT=str(PORT),
               PYTHONPATH=str(REPO_ROOT / "vis_pruner_copy"))
    for key in ("MIN_NEW_TOKENS", "TORCH_THREADS", "MAX_BATCH_SIZE", "MAX_WAIT_MS"):
        env.pop(key, None)
    if policy == "fixed":
        env["MIN_NEW_TOKENS"] = "64"
    if args.torch_threads:
        env["TORCH_THREADS"] = str(args.torch_threads)
    if state == "S0":
        return [sys.executable, str(REPO_ROOT / "scripts" / "wp6_s0_server.py")], env
    env["MAX_BATCH_SIZE"] = "1" if state == "S1" else str(args.batch_size)
    env["MAX_WAIT_MS"] = str(args.max_wait_ms)
    return [sys.executable, "-m", "uvicorn", "llava.serve.batch_server:app", "--host", HOST,
            "--port", str(PORT), "--log-level", "warning"], env


def wait_for_server(proc, timeout_s=900):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("server exited before becoming ready")
        try:
            with urlopen(f"{BASE_URL}/health", timeout=2) as r:
                h = json.load(r)
            if h.get("model_loaded") and h.get("pid") == proc.pid:
                return h
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError("server did not become ready")


class GpuMonitor:
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


class Server:
    def __init__(self, out_root):
        self.out_root, self.proc, self.cell, self.log, self.health = out_root, None, None, None, None
        self.load_times = []

    def ensure(self, cell, args):
        if self.cell == cell and self.proc and self.proc.poll() is None:
            return
        self.stop()
        cmd, env = server_cmd_env(cell, args)
        self.log = open(self.out_root / "server_logs.txt", "a")
        self.log.write(f"\n=== {cell} {time.strftime('%H:%M:%S')} ===\n")
        self.log.flush()
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=self.log,
                                     stderr=subprocess.STDOUT)
        self.health = wait_for_server(self.proc)
        self.load_times.append(round(time.perf_counter() - t0, 1))
        self.cell = cell

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.log.close()
        self.proc, self.cell = None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--states", default="S0,S1,S2")
    ap.add_argument("--configs", default="576,128")
    ap.add_argument("--policies", default="fixed,natural")
    ap.add_argument("--levels-s0", default="0.5,1,1.5,2,2.5,3", help="rps levels for S0 and S1")
    ap.add_argument("--levels-s2", default="0.5,1,2,3,4,6,8")
    ap.add_argument("--min-samples", type=int, default=110)
    ap.add_argument("--min-duration", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-wait-ms", type=float, default=10)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261026)
    args = ap.parse_args()

    out_root = REPO_ROOT / "results" / "timing" / "wp6_matrix" / args.run_id
    if out_root.exists():
        raise SystemExit(f"{out_root} exists; pick a fresh --run-id")
    if port_is_open():
        raise SystemExit(f"port {PORT} in use")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                "--format=csv,noheader"], text=True).strip():
        raise SystemExit("another process is on the GPU (rule 9)")

    blocks = []
    for state in args.states.split(","):
        levels = args.levels_s2 if state == "S2" else args.levels_s0
        for vtn in (int(x) for x in args.configs.split(",")):
            for policy in args.policies.split(","):
                for rps in (float(x) for x in levels.split(",")):
                    blocks.append(((state, vtn, policy), rps))
    random.Random(args.seed).shuffle(blocks)

    out_root.mkdir(parents=True)
    meta = {"run_id": args.run_id, "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name",
                                            "--format=csv,noheader"], text=True).strip(),
            "args": vars(args), "block_order": [], "blocks": []}
    server = Server(out_root)
    try:
        for i, (cell, rps) in enumerate(blocks):
            state, vtn, policy = cell
            duration = max(args.min_duration, int((args.min_samples + LOADGEN_WARMUP) / rps + 0.999))
            name = f"{i:03d}_{state}_vtn{vtn}_{policy}_rps{rps}"
            print(f"[{i + 1}/{len(blocks)}] {name} ({duration}s)", flush=True)
            server.ensure(cell, args)
            started = time.time()
            with GpuMonitor(out_root / f"{name}_gpu.csv"):
                subprocess.run(
                    ["npm", "run", "dev", "--", "--endpoint", f"{BASE_URL}/infer",
                     "--rps", str(rps), "--duration", str(duration), "--timeout", "600000",
                     "--dataset", str(DATASET_PATH)],
                    cwd=REPO_ROOT / "csnbs" / "measure", check=True,
                    stdout=open(out_root / f"{name}_loadgen.log", "w"), stderr=subprocess.STDOUT)
            fresh = [p for p in glob.glob(str(REPO_ROOT / "results/loadgen/*/*/run.json"))
                     if os.path.getmtime(p) >= started - 1]
            run_json = max(fresh, key=os.path.getmtime)
            with urlopen(f"{BASE_URL}/health", timeout=5) as r:
                health_after = json.load(r)
            meta["block_order"].append(name)
            meta["blocks"].append({
                "name": name, "state": state, "visual_token_num": vtn, "policy": policy,
                "rps": rps, "duration_s": duration,
                "loadgen_run_json": os.path.relpath(run_json, REPO_ROOT),
                "gpu_csv": f"{name}_gpu.csv",
                "server_configuration": health_after.get("configuration")})
            meta["server_load_times_s"] = server.load_times
            (out_root / "run.json").write_text(json.dumps(meta, indent=2))
    finally:
        server.stop()
    print(f"done: {out_root}")


if __name__ == "__main__":
    main()
