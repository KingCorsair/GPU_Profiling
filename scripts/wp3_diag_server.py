"""
WP3 server diagnostic: serve csnbs/server.py UNMODIFIED, instrumented from outside.

csnbs/server.py is Rithvik's file, so nothing in it changes. This wrapper imports it and
attaches three things before uvicorn starts:

  1. A pure-ASGI middleware that timestamps request arrival, the end of the request body
     read, and the moment the response body is sent.
  2. Timing shims around the stage functions _infer_model calls by name at request time:
     base64.b64decode (handler entry), tokenizer_image_token (marks the end of PIL decode),
     process_images, and model.generate.
  3. A heartbeat coroutine: asyncio.sleep(5 ms) in a loop, recording how late each wake-up
     was. While the event loop is blocked, wake-ups are late by exactly the block length.

Usage (normally launched by scripts/wp3_run_diag.py):
    WP3_OUT_DIR=... VISUAL_TOKEN_NUM=128 [WP3_TORCH_THREADS=4] python scripts/wp3_diag_server.py

Per-request stage timestamps go to $WP3_OUT_DIR/requests.jsonl, heartbeat gaps to
$WP3_OUT_DIR/heartbeat.json. Every timestamp is time.perf_counter() in this process.
Throwaway diagnostic (Amay measures to find bottlenecks); not a reportable number.
"""
import asyncio
import json
import os
import sys
import time
import types
from contextvars import ContextVar
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "vis_pruner_copy"))

OUT_DIR = Path(os.environ["WP3_OUT_DIR"])
OUT_DIR.mkdir(parents=True, exist_ok=True)
HEARTBEAT_PERIOD_S = 0.005
HEARTBEAT_GAP_RECORD_S = 0.001  # only late wake-ups above this are stored individually

os.environ["SERVER_MODE"] = "model"
# generate() appends a timing row per call; keep it out of the tracked results file.
os.environ.setdefault("LLAVA_TIMING_FILE", str(OUT_DIR / "llava_llama_timing.jsonl"))

import torch  # noqa: E402

# Must happen before csnbs.server loads the model at import time.
TORCH_THREADS = os.environ.get("WP3_TORCH_THREADS")
if TORCH_THREADS:
    torch.set_num_threads(int(TORCH_THREADS))

import llava.mm_utils as mm_utils  # noqa: E402
from csnbs import server  # noqa: E402  (loads the model)

_current = ContextVar("wp3_request", default=None)
_requests = []
_heartbeat = {"samples": 0, "gaps": []}


def _mark(key):
    rec = _current.get()
    if rec is not None:
        rec[key] = time.perf_counter()


def _wrap(fn, start_key, end_key):
    def wrapped(*args, **kwargs):
        if start_key:
            _mark(start_key)
        try:
            return fn(*args, **kwargs)
        finally:
            _mark(end_key)
    return wrapped


# --- stage shims -----------------------------------------------------------------------
# infer() calls base64.b64decode first thing, so its start is handler entry.
server.base64 = types.SimpleNamespace(
    b64decode=_wrap(server.base64.b64decode, "t_handler_entry", "t_b64_done"))
# _infer_model does `from llava.mm_utils import ...` per call, so patching the module
# attribute is enough. tokenizer_image_token runs right after the PIL decode.
mm_utils.tokenizer_image_token = _wrap(
    mm_utils.tokenizer_image_token, "t_pil_done", "t_tokenize_done")
mm_utils.process_images = _wrap(
    mm_utils.process_images, "t_process_images_start", "t_preprocess_done")
server.model.generate = _wrap(server.model.generate, "t_generate_start", "t_generate_end")


# --- ASGI middleware -------------------------------------------------------------------
class StageTimer:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/infer":
            return await self.app(scope, receive, send)

        rec = {"t_received": time.perf_counter()}
        token = _current.set(rec)

        async def timed_receive():
            message = await receive()
            if message["type"] == "http.request" and not message.get("more_body", False):
                rec["t_body_done"] = time.perf_counter()
            return message

        async def timed_send(message):
            if message["type"] == "http.response.start":
                rec["status"] = message["status"]
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                rec["t_response"] = time.perf_counter()

        try:
            await self.app(scope, timed_receive, timed_send)
        finally:
            _current.reset(token)
            _requests.append(rec)


server.app.add_middleware(StageTimer)


# --- heartbeat -------------------------------------------------------------------------
async def _heartbeat_loop():
    while True:
        expected = time.perf_counter() + HEARTBEAT_PERIOD_S
        await asyncio.sleep(HEARTBEAT_PERIOD_S)
        late = time.perf_counter() - expected
        _heartbeat["samples"] += 1
        if late > HEARTBEAT_GAP_RECORD_S:
            _heartbeat["gaps"].append([round(expected, 6), round(late, 6)])


def _flush():
    with open(OUT_DIR / "requests.jsonl", "w") as f:
        for rec in _requests:
            f.write(json.dumps(rec) + "\n")
    with open(OUT_DIR / "heartbeat.json", "w") as f:
        json.dump({"period_s": HEARTBEAT_PERIOD_S,
                   "gap_record_threshold_s": HEARTBEAT_GAP_RECORD_S,
                   **_heartbeat}, f)


@server.app.on_event("startup")
async def _start_heartbeat():
    server.app.state.wp3_heartbeat = asyncio.create_task(_heartbeat_loop())


@server.app.on_event("shutdown")
async def _shutdown_flush():
    _flush()


@server.app.post("/wp3/flush")
async def wp3_flush():
    _flush()
    return {"requests": len(_requests), "heartbeat_samples": _heartbeat["samples"]}


@server.app.post("/wp3/reset")
async def wp3_reset():
    _requests.clear()
    _heartbeat["samples"] = 0
    _heartbeat["gaps"].clear()
    return {"ok": True}


app = server.app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("MODEL_SERVER_PORT", "8000")),
                log_level="warning")
