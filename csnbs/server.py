"""POST /infer — same HTTP contract in fake and model mode.

fake  : await asyncio.sleep(random_duration), no model loaded.
model : calls the unpruned model after loading it once at startup.

Mode is chosen with SERVER_MODE=fake|model (default fake) so load_gen.py
never has to know which one it's hitting.
"""

import asyncio
import base64
import binascii
import os
import random
import time

if __package__ in (None, ""):
    from benchmark_metadata import collect_provenance, checkpoint_provenance
else:
    from csnbs.benchmark_metadata import collect_provenance, checkpoint_provenance
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODE = os.environ.get("SERVER_MODE", "fake")
SERVICE_NAME = "csnbs-llava-server"
VISUAL_TOKEN_NUM = int(os.environ.get("VISUAL_TOKEN_NUM", "576"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
_extended_flag = os.environ.get("EXTENDED_HTTP_OBSERVATIONS", "0")
if _extended_flag not in ("0", "1"):
    raise ValueError("EXTENDED_HTTP_OBSERVATIONS must be 0 or 1")
EXTENDED_HTTP_OBSERVATIONS = _extended_flag == "1"
if not 1 <= VISUAL_TOKEN_NUM <= 576 or MAX_NEW_TOKENS < 1:
    raise ValueError("invalid visual token or output token budget")
PROVENANCE = collect_provenance()

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "vis_pruner_copy"
    / "checkpoints"
    / "llava-v1.5-7b"
)
MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))
)

CHECKPOINT_PROVENANCE = checkpoint_provenance(MODEL_PATH)

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
        # Overridable so A/B load-test runs (e.g. VisPruner 128 vs. unpruned
        # 576) can select this per-process without a code change; unset
        # keeps prior behavior (576, effectively unpruned) exactly.
        visual_token_num=VISUAL_TOKEN_NUM,
        important_ratio=0.5,
    )
    model.eval()

app = FastAPI()


class InferRequest(BaseModel):
    request_id: str | None = None
    image_b64: str = Field(..., description="Base64-encoded image bytes")
    question: str = Field(..., min_length=1)


class InferResponse(BaseModel):
    answer: str
    request_id: str | None = None
    metrics: dict | None = None


class HealthResponse(BaseModel):
    service: str
    pid: int
    mode: str
    model_loaded: bool
    configuration: dict
    source: dict
    hardware: dict
    runtime: dict


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


def _token_observations(prompt_ids, output_ids, visual_count, image_token_index):
    """Observe the vendored batch-one wrapper; unknown contracts fail to null.

    Transformers 4.37.2 starts inputs_embeds-only decoder generation with one
    synthetic BOS. Its returned sequence is BOS + generated IDs, NOT the prompt.
    Keep this contract deliberately narrow when dependencies or wrappers change.
    """
    observed = {
        "token_contract": "llava-inputs-embeds-transformers-4.37.2-bos-v1",
        "prompt_text_tokens": None, "visual_tokens": None,
        "prompt_text_tokens_pretruncation": None, "visual_tokens_pretruncation": None,
        "image_placeholder_tokens": None, "multimodal_prompt_tokens": None,
        "returned_output_ids": None, "output_seed_tokens": None,
        "generated_token_steps": None, "generated_text_tokens": None,
        "generated_eos_tokens": None, "generation_ended_with_eos": None,
        "prompt_token_unavailable_reason": None, "output_token_unavailable_reason": None,
    }

    def require(condition, reason):
        if not condition:
            raise ValueError(reason)

    def integer_ids(value):
        return isinstance(value, list) and all(type(item) is int for item in value)

    try:
        require(integer_ids(prompt_ids), "Formatted prompt IDs are not a one-dimensional integer list")
        require(all(item >= 0 or item == image_token_index for item in prompt_ids), "Unexpected negative prompt token ID")
        placeholders = prompt_ids.count(image_token_index)
        observed["image_placeholder_tokens"] = placeholders
        observed["prompt_text_tokens_pretruncation"] = len(prompt_ids) - placeholders
        require(type(visual_count) is int and 0 < visual_count <= 576, "Wrapper visual feature count is not an observed integer in 1..576")
        observed["visual_tokens_pretruncation"] = visual_count
        require(placeholders == 1, "Visual accounting requires exactly one image placeholder")
        total = len(prompt_ids) - placeholders + visual_count
        limit = getattr(model.config, "tokenizer_model_max_length", None)
        require(limit is None or (type(limit) is int and limit > 0), "Unknown multimodal truncation limit")
        require(limit is None or total <= limit, "Combined prompt exceeds wrapper truncation limit; post-truncation text/visual attribution is unavailable")
        observed.update(prompt_text_tokens=len(prompt_ids) - placeholders,
                        visual_tokens=visual_count, multimodal_prompt_tokens=total)
    except Exception as error:
        observed["prompt_token_unavailable_reason"] = str(error)

    try:
        rows = output_ids.tolist()
        require(isinstance(rows, list) and len(rows) == 1 and integer_ids(rows[0]), "Generation output is not one integer sequence")
        sequence = rows[0]
        observed["returned_output_ids"] = len(sequence)
        require(all(item >= 0 for item in sequence), "Unexpected negative output token ID")
        require(PROVENANCE.get("runtime", {}).get("transformers") == "4.37.2", "Output-ID contract is verified only for Transformers 4.37.2")
        require(getattr(model.config, "is_encoder_decoder", None) is False, "Output-ID contract requires a decoder-only model")
        config = model.generation_config
        require(getattr(config, "num_beams", None) == 1 and getattr(config, "num_beam_groups", None) == 1
                and getattr(config, "num_return_sequences", None) == 1, "Output-ID contract requires one greedy output sequence")
        require(getattr(config, "penalty_alpha", None) is None and getattr(config, "constraints", None) is None
                and getattr(config, "force_words_ids", None) is None, "Unsupported generation constraints or contrastive decoding")
        bos = getattr(config, "bos_token_id", None)
        require(type(bos) is int and len(sequence) >= 2 and sequence[0] == bos, "Expected one leading synthetic BOS and at least one generated ID")
        special = tokenizer.all_special_ids
        require(integer_ids(special) and bos in special, "Tokenizer special IDs do not validate BOS classification")
        eos = getattr(config, "eos_token_id", None)
        eos_ids = [] if eos is None else eos if isinstance(eos, list) else [eos]
        require(integer_ids(eos_ids) and all(item in special for item in eos_ids), "EOS IDs are not verified tokenizer special IDs")
        generated = sequence[1:]
        require(len(generated) <= MAX_NEW_TOKENS, "Returned generation exceeds the requested new-token budget")
        eos_positions = [index for index, token in enumerate(generated) if token in eos_ids]
        require(not eos_positions or eos_positions == [len(generated) - 1], "EOS appears before the final generated ID")
        require(bool(eos_positions) or len(generated) == MAX_NEW_TOKENS, "Generation ended before budget without an observed EOS")
        observed.update(output_seed_tokens=1, generated_token_steps=len(generated),
                        generated_text_tokens=sum(token not in special for token in generated),
                        generated_eos_tokens=len(eos_positions), generation_ended_with_eos=bool(eos_positions))
    except Exception as error:
        observed["output_token_unavailable_reason"] = str(error)
    reasons = [observed[key] for key in ("prompt_token_unavailable_reason", "output_token_unavailable_reason") if observed[key]]
    observed["token_unavailable_reason"] = "; ".join(reasons) or None
    return observed


async def _infer_model(image: bytes, question: str, observations: dict | None = None) -> str:
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

    if observations is not None:
        preprocess_start = time.perf_counter()
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

    prompt_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    # Retain the existing CPU IDs for later observation; do not copy GPU inputs
    # or insert synchronization before generation.
    input_ids = prompt_ids.unsqueeze(0).to(model.device)

    image_tensor = process_images(
        [pil_image],
        image_processor,
        model.config,
    ).to(model.device, dtype=torch.float16)

    if observations is not None:
        generation_start = time.perf_counter()
        observations["preprocess_ms"] = (generation_start - preprocess_start) * 1000
    with torch.inference_mode():
        output_ids, visual_token_count = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[pil_image.size],
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )

    if observations is not None:
        postprocess_start = time.perf_counter()
        observations["generation_wall_ms"] = (postprocess_start - generation_start) * 1000
    answer = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0].strip()
    if observations is not None:
        observation_start = time.perf_counter()
        observations["postprocess_ms"] = (observation_start - postprocess_start) * 1000
        try:
            observations.update(_token_observations(prompt_ids.tolist(), output_ids, visual_token_count, IMAGE_TOKEN_INDEX))
        except Exception as error:
            # Instrumentation must never turn an otherwise decoded answer into
            # an inference failure when an unfamiliar tensor contract appears.
            observations["token_unavailable_reason"] = f"Token observation failed: {type(error).__name__}"
        observations["output_characters"] = len(answer)
        observations["observation_wall_ms"] = (time.perf_counter() - observation_start) * 1000
    return answer


def _model_loaded() -> bool:
    return (
        MODE == "model"
        and tokenizer is not None
        and model is not None
        and image_processor is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        service=SERVICE_NAME,
        pid=os.getpid(),
        mode=MODE,
        model_loaded=_model_loaded(),
        configuration={
            "model_id": (CHECKPOINT_PROVENANCE or {}).get("model_id") if MODE == "model" else "fake",
            "checkpoint": str(MODEL_PATH) if MODE == "model" else None,
            "download_provenance": CHECKPOINT_PROVENANCE,
            "implementation": "vispruner-vendored-blocking-v1" if MODE == "model" else "fake-async-v1",
            "visual_token_num": VISUAL_TOKEN_NUM, "important_ratio": 0.5,
            "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "use_cache": True,
            "eos_policy": "natural", "prompt_template": "llava_v1", "batch_size": 1,
            "dtype": str(next(model.parameters()).dtype) if _model_loaded() else None,
            "instrumentation": "handler-stage-token-observations-v2" if EXTENDED_HTTP_OBSERVATIONS else "handler-service-wall-v1",
            "queue_observation": "unavailable-blocking-handler",
        },
        **PROVENANCE,
    )


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest) -> InferResponse:
    service_start = time.perf_counter()
    observations = {} if EXTENDED_HTTP_OBSERVATIONS else None
    try:
        image = base64.b64decode(req.image_b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_b64 is not valid base64")
    if observations is not None:
        observations["base64_decode_ms"] = (time.perf_counter() - service_start) * 1000

    if MODE == "fake":
        answer = await _infer_fake(image, req.question)
    elif MODE == "model":
        answer = await _infer_model(image, req.question, observations) if observations is not None else await _infer_model(image, req.question)
    else:
        raise HTTPException(status_code=500, detail=f"unknown SERVER_MODE={MODE!r}")

    metrics = {
        "schema_version": 1,
        "service_ms": (time.perf_counter() - service_start) * 1000,
        "queue_ms": None,
        "queue_unavailable_reason": "Handler entry does not observe socket arrival or pre-handler waiting",
        "generation_wall_ms": None, "preprocess_ms": None, "postprocess_ms": None,
        "generated_text_tokens": None, "prompt_text_tokens": None, "visual_tokens": None,
        "token_unavailable_reason": "Actual token counts require a verified generation-wrapper contract",
    }
    if observations is not None:
        metrics.update(observations)
        metrics["schema_version"] = 2
        metrics["output_characters"] = len(answer)
        metrics["model_internal_instrumentation"] = "vendored CUDA events, boundary synchronizations and diagnostic file writes remain enabled" if MODE == "model" else None
        if MODE == "fake":
            metrics["token_unavailable_reason"] = "Fake mode has no model token IDs"
        metrics["service_ms"] = (time.perf_counter() - service_start) * 1000
    return InferResponse(answer=answer, request_id=req.request_id, metrics=metrics)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
