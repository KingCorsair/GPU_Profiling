import json
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


def require_model_server(process, health: dict) -> None:
    expected = {
        "service": SERVICE_NAME,
        "pid": process.pid,
        "mode": "model",
        "model_loaded": True,
    }
    actual = {key: health.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"Port {SERVER_PORT} is not the expected model server: "
            f"expected {expected}, got {actual}"
        )


def check_model_server(process) -> None:
    if process.poll() is not None:
        raise RuntimeError("Model server exited")
    require_model_server(process, read_server_health())
    if process.poll() is not None:
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
    if port_is_open():
        raise RuntimeError(
            f"Port {SERVER_PORT} is already in use; stop the existing server first"
        )

    server = subprocess.Popen(
        ["bash", "/workspace/GPU_Profiling/scripts/start_model_server.sh"],
        cwd="/workspace/GPU_Profiling",
    )
    try:
        print("Waiting for server...")
        wait_for_server(server)
        print("Server is ready.")

        rng = random.Random(90)
        rates = [1.0, 1.25, 1.5, 1.75, 2.0]
        rng.shuffle(rates)
        for rps in rates:
            print(f"--- RPS: {rps} ---")
            check_model_server(server)

            subprocess.run(
                [
                    "npm",
                    "run",
                    "dev",
                    "--",
                    "--endpoint",
                    "http://127.0.0.1:8000/infer",
                    "--rps",
                    str(rps),
                    "--duration",
                    "20",
                    "--timeout",
                    "120000",
                    "--dataset",
                    "/workspace/GPU_Profiling/vis_pruner_copy/vispruner_eval_dataset/dev.json",
                ],
                cwd="/workspace/GPU_Profiling/csnbs/measure",
                check=True,
            )
            check_model_server(server)

        print("Load-test sweep completed.")

    finally:
        print("Stopping server...")
        server.terminate()

        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == "__main__":
    main()
