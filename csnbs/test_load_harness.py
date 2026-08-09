"""Tests for the open-loop load harness, run against the fake server.

These run in-process against the FastAPI app via httpx's ASGI transport, so
there's no real socket and no port to bind - timing is dominated by the
harness's own asyncio.sleep scheduling, which is what we're actually testing.
"""

import asyncio

import httpx
import pytest

import load_gen
import server as server_module


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(app=server_module.app, base_url="http://testserver")


# --------------------------------------------------------------- percentiles

def test_percentile_known_values():
    values = [float(i) for i in range(1, 101)]  # 1..100
    assert load_gen.percentile(values, 50) == 51
    assert load_gen.percentile(values, 95) == 96
    assert load_gen.percentile(values, 99) == 100


def test_percentile_single_value():
    assert load_gen.percentile([42.0], 50) == 42.0
    assert load_gen.percentile([42.0], 99) == 42.0


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        load_gen.percentile([], 50)


# --------------------------------------------------------------- open-loop overlap

def test_requests_overlap_when_service_time_exceeds_interval(monkeypatch):
    """Fake response time (300ms) >> arrival interval (50ms): requests must
    pile up in flight, proving the harness doesn't wait for replies before
    sending the next one."""
    monkeypatch.setattr(server_module, "_fake_latency_s", lambda: 0.3)

    async def go():
        async with make_client() as client:
            return await load_gen.run(
                url="/infer", n_requests=10, interval_ms=50,
                timeout_s=5.0, question="what is this?", client=client,
            )

    results = asyncio.run(go())

    assert len(results) == 10
    overlaps = sum(1 for a, b in zip(results, results[1:]) if b.sent_at < a.done_at)
    assert overlaps > 0, "requests never overlapped - this looks closed-loop, not open-loop"


def test_closed_loop_reference_never_overlaps(monkeypatch):
    """Sanity check for the assertion above: a client that waits for each
    reply before sending the next never overlaps, even with a slow server."""
    monkeypatch.setattr(server_module, "_fake_latency_s", lambda: 0.05)

    async def go():
        results = []
        async with make_client() as client:
            for i in range(5):
                r = await load_gen.fire_one(client, i, 0.0, "/infer", "q", 5.0)
                results.append(r)
        return results

    results = asyncio.run(go())
    overlaps = sum(1 for a, b in zip(results, results[1:]) if b.sent_at < a.done_at)
    assert overlaps == 0


# --------------------------------------------------------------- completeness / failures

def test_every_scheduled_request_produces_a_result(monkeypatch):
    monkeypatch.setattr(server_module, "_fake_latency_s", lambda: 0.02)

    async def go():
        async with make_client() as client:
            return await load_gen.run(
                url="/infer", n_requests=25, interval_ms=10,
                timeout_s=5.0, question="q", client=client,
            )

    results = asyncio.run(go())

    assert len(results) == 25
    assert sorted(r.seq for r in results) == list(range(25))
    for r in results:
        assert r.sent_at > 0 and r.done_at > 0
        assert r.ok or r.error is not None  # never silently dropped


def test_timeout_is_recorded_not_dropped(monkeypatch):
    async def always_slow(image, question):
        await asyncio.sleep(1.0)
        return "unused"

    monkeypatch.setattr(server_module, "_infer_fake", always_slow)

    async def go():
        async with make_client() as client:
            return await load_gen.run(
                url="/infer", n_requests=3, interval_ms=20,
                timeout_s=0.05, question="q", client=client,  # timeout << server delay
            )

    results = asyncio.run(go())

    assert len(results) == 3
    assert all(r.error == "timeout" for r in results)
    assert all(not r.ok for r in results)


def test_http_error_is_recorded_not_dropped(monkeypatch):
    """A request that gets back a 4xx/5xx must show up as a failed Result,
    not vanish or raise past the harness."""
    async def always_reject(image, question):
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="forced failure")

    monkeypatch.setattr(server_module, "_infer_fake", always_reject)

    async def go():
        async with make_client() as client:
            return await load_gen.fire_one(
                client, seq=0, scheduled_at=0.0, url="/infer",
                question="q", timeout_s=5.0,
            )

    r = asyncio.run(go())
    assert r.status == 500
    assert r.error == "http_500"
    assert not r.ok
