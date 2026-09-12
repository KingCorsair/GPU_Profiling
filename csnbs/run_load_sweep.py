import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import random

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
SERVICE_NAME = "csnbs-llava-server"


def port_is_open() -> bool:
    try:
        with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=1):
            return True
    except OSError:
        return False


def read_server_health() -> dict:
    with urlopen(HEALTH_URL, timeout=1) as response:
        if response.status != 200:
            raise RuntimeError(
                f"Health check returned HTTP {response.status}"
            )
        return json.load(response)


def require_model_server(expected_pid: int, health: dict) -> None:
    expected = {
        "service": SERVICE_NAME,
        "pid": expected_pid,
        "mode": "model",
        "model_loaded": True,
    }
    actual = {key: health.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"Port {SERVER_PORT} is not the expected model server: "
            f"expected {expected}, got {actual}"
        )


def check_model_server(process, expected_pid=None) -> None:
    if process is not None and process.poll() is not None:
        raise RuntimeError("Model server exited")
    require_model_server(process.pid if process is not None else expected_pid, read_server_health())
    if process is not None and process.poll() is not None:
        raise RuntimeError("Model server exited during health check")


def wait_for_server(process, timeout_seconds=300):
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited before becoming ready")

        try:
            check_model_server(process)
            return
        except HTTPError as exc:
            raise RuntimeError(
                f"Port {SERVER_PORT} answered without the expected health endpoint"
            ) from exc
        except (URLError, TimeoutError):
            time.sleep(1)

    raise TimeoutError("Server did not become ready within 300 seconds")


def main():
    global SERVER_PORT, HEALTH_URL, SERVICE_NAME
    parser = argparse.ArgumentParser(description="Run the existing RPS sweep against LLaVA or Gemma.")
    parser.add_argument("--model", choices=("llava", "gemma"), default="llava")
    parser.add_argument("--reuse-server", action="store_true", help="Use an already-running matching server; do not stop it afterward")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    SERVER_PORT = 8002 if args.model == "gemma" else 8000
    SERVICE_NAME = "csnbs-gemma-server" if args.model == "gemma" else "csnbs-llava-server"
    HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
    load_env = os.environ.copy()
    node_bin = repo_root / "venvs/node24/bin"
    if args.model == "gemma" and (node_bin / "node").is_file():
        load_env["PATH"] = str(node_bin) + os.pathsep + load_env.get("PATH", "")
    if port_is_open() and not args.reuse_server:
        raise RuntimeError(
            f"Port {SERVER_PORT} is already in use; pass --reuse-server or stop it first"
        )

    server = None
    expected_pid = None
    if args.reuse_server:
        health = read_server_health()
        expected_pid = health.get("pid")
        if not isinstance(expected_pid, int) or expected_pid <= 0:
            raise RuntimeError("Server health does not identify a valid process")
        require_model_server(expected_pid, health)
    else:
        launcher = "csnbs/start_gemma_server.sh" if args.model == "gemma" else "scripts/start_model_server.sh"
        server = subprocess.Popen(["bash", str(repo_root / launcher)], cwd=repo_root)
    try:
        print("Waiting for server...")
        if server is not None:
            wait_for_server(server)
        print("Server is ready.")

        rng = random.Random(90)
        rates = [1.0, 1.25, 1.5, 1.75, 2.0]
        rng.shuffle(rates)
        for rps in rates:
            print(f"--- RPS: {rps} ---")
            check_model_server(server, expected_pid)

            subprocess.run(
                [
                    "npm",
                    "run",
                    "dev",
                    "--",
                    "--endpoint",
                    f"http://{SERVER_HOST}:{SERVER_PORT}/infer",
                    "--rps",
                    str(rps),
                    "--duration",
                    "20",
                    "--timeout",
                    "120000",
                    "--dataset",
                    str(repo_root / "vis_pruner_copy/vispruner_eval_dataset/dev.json"),
                ],
                cwd=repo_root / "csnbs/measure",
                env=load_env,
                check=True,
            )
            check_model_server(server, expected_pid)

        print("Load-test sweep completed.")

    finally:
        if server is not None:
            print("Stopping server...")
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
