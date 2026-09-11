"""Unpruned Qwen3-VL comparison server using the existing /infer contract.

One process, synchronous batch-one generation, matching the current LLaVA
wrapper's serving policy. This is not a continuous-batching backend.
"""

import base64
import binascii
from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "checkpoints/Qwen3-VL-8B-Instruct"
)


class InferRequest(BaseModel):
    image_b64: str
    question: str = Field(min_length=1)


class InferResponse(BaseModel):
    answer: str


class QwenBackend:
    def __init__(self):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_path = Path(os.environ.get("QWEN_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        if not (model_path / "config.json").is_file():
            raise RuntimeError(f"Qwen checkpoint missing: {model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen benchmark server requires a CUDA GPU")

        self.max_new_tokens = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "64"))
        # Qwen3-VL merges 2x2 patches of 16x16 pixels into one image token.
        self.min_pixels = int(os.environ.get("QWEN_MIN_PIXELS", str(4 * 32 * 32)))
        self.max_pixels = int(os.environ.get("QWEN_MAX_PIXELS", str(1024 * 32 * 32)))
        if self.max_new_tokens < 1 or not 0 < self.min_pixels <= self.max_pixels:
            raise ValueError("Invalid Qwen generation or image-size limits")

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            use_fast=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        ).eval()
        self.metadata = {
            "model_path": str(model_path),
            "model_type": self.model.config.model_type,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "attention_backend": "sdpa",
            "batch_size": 1,
            "continuous_batching": False,
            "pruning": False,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "min_pixels": self.processor.image_processor.size["shortest_edge"],
            "max_pixels": self.processor.image_processor.size["longest_edge"],
        }

    def generate(self, image: Image.Image, question: str) -> str:
        import torch

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        # Qwen returns the prompt followed by newly generated tokens.
        generated = output[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


def create_app(backend_factory=QwenBackend) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.backend = backend_factory()
        yield
        app.state.backend = None

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        backend = getattr(app.state, "backend", None)
        return {
            "service": "csnbs-qwen-server",
            "pid": os.getpid(),
            "mode": "model",
            "model_loaded": backend is not None,
            "configuration": backend.metadata if backend is not None else None,
        }

    @app.post("/infer", response_model=InferResponse)
    async def infer(req: InferRequest):
        try:
            raw = base64.b64decode(req.image_b64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="image_b64 is not valid base64")
        try:
            with Image.open(BytesIO(raw)) as source:
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise HTTPException(status_code=400, detail="image_b64 is not a readable image")
        return InferResponse(answer=app.state.backend.generate(image, req.question))

    return app


app = create_app()
