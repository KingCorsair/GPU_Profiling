"""Unpruned Gemma 3 comparison backend for the existing loadgen /infer API.

One process, synchronous batch-one inference, like the LLaVA/Qwen wrappers.
This backend does not implement continuous batching or visual token pruning.
"""

import base64
import binascii
from contextlib import asynccontextmanager
from io import BytesIO
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "checkpoints/gemma-3-4b-it"


class InferRequest(BaseModel):
    image_b64: str
    question: str = Field(min_length=1)


class InferResponse(BaseModel):
    answer: str


class GemmaBackend:
    def __init__(self):
        import torch
        import transformers
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        model_path = Path(os.environ.get("GEMMA_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        if not (model_path / "config.json").is_file():
            raise RuntimeError(f"Gemma checkpoint missing: {model_path}. Run csnbs/download_gemma.py first.")
        if not torch.cuda.is_available():
            raise RuntimeError("Gemma benchmark server requires a CUDA GPU")
        self.max_new_tokens = int(os.environ.get("GEMMA_MAX_NEW_TOKENS", "64"))
        if self.max_new_tokens < 1:
            raise ValueError("GEMMA_MAX_NEW_TOKENS must be positive")

        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        ).eval()
        provenance_path = model_path / "download-provenance.json"
        provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else None
        self.metadata = {
            "model_id": getattr(self.model.config, "_name_or_path", str(model_path)),
            "model_path": str(model_path),
            "download_provenance": provenance,
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
            "do_pan_and_scan": False,
            "image_size": self.processor.image_processor.size,
        }
        if provenance is not None:
            self.metadata["model_id"] = provenance["model_id"]

    def generate(self, image: Image.Image, question: str) -> str:
        import torch

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            do_pan_and_scan=False,
        ).to(self.model.device, dtype=self.model.dtype)
        prompt_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens, use_cache=True,
            )
        return self.processor.decode(output[0, prompt_length:], skip_special_tokens=True).strip()


def create_app(backend_factory=GemmaBackend) -> FastAPI:
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
            "service": "csnbs-gemma-server",
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
