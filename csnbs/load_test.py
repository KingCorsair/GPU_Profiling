import socket
import subprocess
import sys
import time


def wait_for_server(process, timeout_seconds=300):
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited before becoming ready")

        try:
            with socket.create_connection(
                ("127.0.0.1", 8000),
                timeout=1,
            ):
                return
        except OSError:
          
            time.sleep(1)

    raise TimeoutError("Server did not become ready within 300 seconds")


def main():
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd="/workspace/GPU_Profiling/csnbs",
    )

    try:
        print("Waiting for server...")
        wait_for_server(server)
        print("Server is ready.")

        for rps in range(55, 101, 5):
            print(f"--- RPS: {rps} ---")

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