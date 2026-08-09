"""POST /infer — same HTTP contract in fake and model mode.

fake  : await asyncio.sleep(random_duration), no model loaded.
model : later calls the unpruned model. Not implemented yet.

Mode is chosen with SERVER_MODE=fake|model (default fake) so load_gen.py
never has to know which one it's hitting.
"""

import asyncio
import base64
import binascii
import os
import random

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODE = os.environ.get("SERVER_MODE", "fake")

app = FastAPI()


class InferRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image bytes")
    question: str = Field(..., min_length=1)


class InferResponse(BaseModel):
    answer: str


def _fake_latency_s() -> float:
    """Mostly fast, with an occasional slow tail so p99 has something to show."""
    if random.random() < 0.05:
        return random.uniform(0.4, 0.9)
    return random.uniform(0.08, 0.25)


async def _infer_fake(image: bytes, question: str) -> str:
    await asyncio.sleep(_fake_latency_s())
    return f"fake-answer[{len(image)}b]: {question[:40]}"


async def _infer_model(image: bytes, question: str) -> str:
    raise HTTPException(status_code=501, detail="model mode not implemented yet")


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest) -> InferResponse:
    try:
        image = base64.b64decode(req.image_b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_b64 is not valid base64")

    if MODE == "fake":
        answer = await _infer_fake(image, req.question)
    elif MODE == "model":
        answer = await _infer_model(image, req.question)
    else:
        raise HTTPException(status_code=500, detail=f"unknown SERVER_MODE={MODE!r}")

    return InferResponse(answer=answer)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
