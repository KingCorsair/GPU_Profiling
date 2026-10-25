"""
LLaVA server for server states S1 (queue) and S2 (static batching).

Same HTTP contract as csnbs/server.py (S0), so csnbs/measure and the sweep drivers work
against either one unchanged:
    POST /infer  {"image_b64": str, "question": str} -> {"answer": str}
    GET  /health {"service", "pid", "mode", "model_loaded", "configuration"}

Difference from S0: the handler never blocks the event loop. Base64 decode, PIL decode,
process_images and tokenization run in a thread pool; the CPU tensors then go to the
single inference worker in batch_engine.py, which owns every CUDA call.

Environment:
    VISUAL_TOKEN_NUM   kept visual tokens (576 = unpruned)            default 576
    MAX_BATCH_SIZE     1 = S1, >1 = S2                                default 1
    MAX_WAIT_MS        S2 collection window after the first request    default 10
    MAX_NEW_TOKENS     default 64 (matches S0)
    MIN_NEW_TOKENS     set equal to MAX_NEW_TOKENS for the fixed-length policy; unset = natural EOS
    PREP_THREADS       CPU preprocessing threads                       default 4
    TORCH_THREADS      torch intra-op threads; unset = torch default   default unset
    MAX_QUEUE          bounded queue size; full queue -> HTTP 503      default 256

    PYTHONPATH=vis_pruner_copy python -m uvicorn llava.serve.batch_server:app --port 8000
"""
import asyncio
import base64
import binascii
import os
import queue
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

# generate() appends a timing row per call; keep serving runs out of the tracked file.
os.environ.setdefault("LLAVA_TIMING_FILE", "/tmp/batch_server_llava_timing.jsonl")

import torch  # noqa: E402
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.serve.batch_engine import InferenceEngine
from llava.utils import disable_torch_init

SERVICE_NAME = "amay-llava-batch-server"
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = os.environ.get(
    "MODEL_PATH", str(REPO_ROOT / "vis_pruner_copy" / "checkpoints" / "llava-v1.5-7b"))


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


CONFIG = {
    "model_id": "liuhaotian/llava-v1.5-7b",
    "visual_token_num": _env_int("VISUAL_TOKEN_NUM", 576),
    "important_ratio": 0.5,
    "max_batch_size": _env_int("MAX_BATCH_SIZE", 1),
    "max_wait_ms": float(os.environ.get("MAX_WAIT_MS", "10")),
    "max_new_tokens": _env_int("MAX_NEW_TOKENS", 64),
    "min_new_tokens": _env_int("MIN_NEW_TOKENS", None),
    "prep_threads": _env_int("PREP_THREADS", 4),
    "torch_threads": _env_int("TORCH_THREADS", None),
    "max_queue": _env_int("MAX_QUEUE", 256),
}
CONFIG["server_state"] = "S1" if CONFIG["max_batch_size"] == 1 else "S2"

if CONFIG["torch_threads"]:
    torch.set_num_threads(CONFIG["torch_threads"])

disable_torch_init()
tokenizer, model, image_processor, _ = load_pretrained_model(
    MODEL_PATH, None, get_model_name_from_path(MODEL_PATH),
    visual_token_num=CONFIG["visual_token_num"], important_ratio=CONFIG["important_ratio"])
model.eval()

engine = InferenceEngine(
    model, tokenizer,
    max_batch_size=CONFIG["max_batch_size"], max_wait_ms=CONFIG["max_wait_ms"],
    max_new_tokens=CONFIG["max_new_tokens"], min_new_tokens=CONFIG["min_new_tokens"],
    max_queue=CONFIG["max_queue"])
prep_pool = ThreadPoolExecutor(max_workers=CONFIG["prep_threads"], thread_name_prefix="prep")

IMAGE_TOKEN = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
               if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN)

app = FastAPI()


class InferRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image bytes")
    question: str = Field(..., min_length=1)


class InferResponse(BaseModel):
    answer: str


def build_prompt(question: str) -> str:
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def prepare(image_b64: str, question: str):
    """All per-request CPU work. Pure CPU, no model state, no CUDA. Same steps, in the
    same order, as csnbs/server.py's _infer_model, so outputs match S0."""
    image_bytes = base64.b64decode(image_b64, validate=True)
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    input_ids = tokenizer_image_token(build_prompt(question), tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors="pt")
    image_tensor = process_images([pil_image], image_processor, model.config)[0]
    return image_tensor, input_ids, pil_image.size


@app.get("/health")
async def health():
    return {
        "service": SERVICE_NAME,
        "pid": os.getpid(),
        "mode": "model",
        "model_loaded": True,
        "configuration": {**CONFIG, "queue_depth": engine.queue_depth(),
                          "batches_run": len(engine.batch_sizes)},
    }


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest) -> InferResponse:
    loop = asyncio.get_running_loop()
    try:
        image, input_ids, size = await loop.run_in_executor(
            prep_pool, prepare, req.image_b64, req.question)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_b64 is not valid base64")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="image_b64 is not a decodable image")
    try:
        answer = await engine.infer(image, input_ids, size)
    except queue.Full:
        raise HTTPException(status_code=503, detail="inference queue full")
    return InferResponse(answer=answer)


@app.on_event("shutdown")
def _shutdown():
    engine.shutdown()
    prep_pool.shutdown(wait=False)
