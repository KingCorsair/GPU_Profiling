"""Open-loop load generator for POST /infer.

Requests are fired on a fixed schedule (every --interval-ms), independent of
whether earlier requests have completed. This is what lets a queue form when
the server can't keep up - a closed-loop client (wait for reply, then send
the next one) backs off exactly when the server is struggling and never
reports the queueing that a real user would feel. See lesson 7 in bench.py.

    python load_gen.py --url http://127.0.0.1:8000/infer --interval-ms 100 --n-requests 300
"""

import argparse
import asyncio
import base64
import contextlib
import time
from dataclasses import dataclass
from typing import Optional

import httpx

FAKE_IMAGE_B64 = base64.b64encode(b"not a real png, just payload bytes" * 16).decode()


@dataclass
class Result:
    seq: int
    scheduled_at: float
    sent_at: float = 0.0
    done_at: float = 0.0
    status: int = 0
    error: Optional[str] = None

    @property
    def latency_s(self) -> float:
        return self.done_at - self.sent_at

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 300


def percentile(values: list, p: float) -> float:
    if not values:
        raise ValueError("percentile of an empty sequence")
    s = sorted(values)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


async def fire_one(client: httpx.AsyncClient, seq: int, scheduled_at: float,
                    url: str, question: str, timeout_s: float) -> Result:
    r = Result(seq=seq, scheduled_at=scheduled_at)
    r.sent_at = time.perf_counter()
    try:
        resp = await asyncio.wait_for(
            client.post(url, json={"image_b64": FAKE_IMAGE_B64, "question": question}),
            timeout=timeout_s,
        )
        r.done_at = time.perf_counter()
        r.status = resp.status_code
        if resp.status_code >= 400:
            r.error = f"http_{resp.status_code}"
    except asyncio.TimeoutError:
        r.done_at = time.perf_counter()
        r.error = "timeout"
    except httpx.HTTPError as e:
        r.done_at = time.perf_counter()
        r.error = f"transport_error:{type(e).__name__}"
    return r


async def run(url: str, n_requests: int, interval_ms: float, timeout_s: float,
              question: str, client: Optional[httpx.AsyncClient] = None) -> list:
    """Schedule n_requests at a fixed interval and wait for all of them.

    Every request is launched at start + seq * interval, regardless of
    whether earlier requests have finished - that's the open-loop part.
    """
    interval_s = interval_ms / 1000.0
    start = time.perf_counter()

    async with contextlib.AsyncExitStack() as stack:
        if client is None:
            limits = httpx.Limits(max_connections=max(n_requests, 100))
            client = await stack.enter_async_context(httpx.AsyncClient(limits=limits))

        async def scheduled(seq: int) -> Result:
            target = start + seq * interval_s
            now = time.perf_counter()
            if target > now:
                await asyncio.sleep(target - now)
            return await fire_one(client, seq, target, url, question, timeout_s)

        tasks = [asyncio.create_task(scheduled(i)) for i in range(n_requests)]
        return list(await asyncio.gather(*tasks))


def summarize(results: list, n_scheduled: int) -> None:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    latencies_ms = [r.latency_s * 1000 for r in ok]

    by_error: dict = {}
    for r in failed:
        by_error[r.error] = by_error.get(r.error, 0) + 1

    print(f"scheduled : {n_scheduled}")
    print(f"completed : {len(results)}  (ok={len(ok)}, failed={len(failed)})")
    if by_error:
        print("failures  : " + ", ".join(f"{k}={v}" for k, v in sorted(by_error.items())))
    if latencies_ms:
        print(f"p50       : {percentile(latencies_ms, 50):7.1f} ms")
        print(f"p95       : {percentile(latencies_ms, 95):7.1f} ms")
        print(f"p99       : {percentile(latencies_ms, 99):7.1f} ms")
    else:
        print("p50/p95/p99: n/a - no successful responses")


def main() -> None:
    ap = argparse.ArgumentParser(description="Open-loop load generator for POST /infer")
    ap.add_argument("--url", default="http://127.0.0.1:8000/infer")
    ap.add_argument("--interval-ms", type=float, default=100.0)
    ap.add_argument("--n-requests", type=int, default=200)
    ap.add_argument("--timeout-s", type=float, default=5.0)
    ap.add_argument("--question", default="What is in this image?")
    args = ap.parse_args()

    results = asyncio.run(run(
        url=args.url,
        n_requests=args.n_requests,
        interval_ms=args.interval_ms,
        timeout_s=args.timeout_s,
        question=args.question,
    ))
    summarize(results, args.n_requests)


if __name__ == "__main__":
    main()
