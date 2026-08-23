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
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODE = os.environ.get("SERVER_MODE", "fake")

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "vis_pruner_copy"
    / "checkpoints"
    / "llava-v1.5-7b"
)
MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))
)

tokenizer = None
model = None
image_processor = None

if MODE == "model":
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()

    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(MODEL_PATH),
        None,
        get_model_name_from_path(str(MODEL_PATH)),
        visual_token_num=576,
        important_ratio=0.5,
    )
    model.eval()

app = FastAPI()


class InferRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image bytes")
    question: str = Field(..., min_length=1)


class InferResponse(BaseModel):
    answer: str


_fake_slots = asyncio.Semaphore(16)


def _fake_latency_s() -> float:
    """Mostly fast, with an occasional slow tail so p99 has something to show."""
    if random.random() < 0.05:
        return random.uniform(0.4, 0.9)
    return random.uniform(0.08, 0.25)


async def _infer_fake(image: bytes, question: str) -> str:
    async with _fake_slots:
        await asyncio.sleep(_fake_latency_s())
        return f"fake-answer[{len(image)}b]: {question[:40]}"


async def _infer_model(image: bytes, question: str) -> str:
    from io import BytesIO

    import torch
    from PIL import Image

    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    pil_image = Image.open(BytesIO(image)).convert("RGB")

    if model.config.mm_use_im_start_end:
        image_token = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
        )
    else:
        image_token = DEFAULT_IMAGE_TOKEN

    conversation = conv_templates["llava_v1"].copy()
    conversation.append_message(
        conversation.roles[0],
        image_token + "\n" + question,
    )
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    image_tensor = process_images(
        [pil_image],
        image_processor,
        model.config,
    ).to(model.device, dtype=torch.float16)

    with torch.inference_mode():
        output_ids, visual_token_count = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[pil_image.size],
            do_sample=False,
            max_new_tokens=64,
            use_cache=True,
        )

    return tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0].strip()


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
