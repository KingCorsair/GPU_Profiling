"""
Server state S0 for the WP6 matrix: csnbs/server.py, UNMODIFIED, with two knobs applied
from outside so S0 runs under the same conditions as S1/S2 (llava/serve/batch_server.py):

    MIN_NEW_TOKENS  if set, injected into every model.generate() call (fixed-length policy;
                    S0 hard-codes max_new_tokens=64, so MIN_NEW_TOKENS=64 forces exactly 64)
    TORCH_THREADS   torch intra-op threads, set before the model loads

/health gains a "configuration" block so csnbs/measure records the state in run.json.

    VISUAL_TOKEN_NUM=128 MIN_NEW_TOKENS=64 python scripts/wp6_s0_server.py
"""
import functools
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "vis_pruner_copy"))
os.environ["SERVER_MODE"] = "model"
os.environ.setdefault("LLAVA_TIMING_FILE", "/tmp/wp6_s0_llava_timing.jsonl")

import torch  # noqa: E402

TORCH_THREADS = os.environ.get("TORCH_THREADS")
if TORCH_THREADS:
    torch.set_num_threads(int(TORCH_THREADS))

from csnbs import server  # noqa: E402  (loads the model)

MIN_NEW_TOKENS = os.environ.get("MIN_NEW_TOKENS")
if MIN_NEW_TOKENS:
    _generate = server.model.generate

    @functools.wraps(_generate)
    def _fixed_length_generate(*args, **kwargs):
        kwargs["min_new_tokens"] = int(MIN_NEW_TOKENS)
        return _generate(*args, **kwargs)

    server.model.generate = _fixed_length_generate

CONFIGURATION = {
    "model_id": "liuhaotian/llava-v1.5-7b",
    "server_state": "S0",
    "visual_token_num": int(os.environ.get("VISUAL_TOKEN_NUM", "576")),
    "important_ratio": 0.5,
    "max_batch_size": 1,
    "max_new_tokens": 64,
    "min_new_tokens": int(MIN_NEW_TOKENS) if MIN_NEW_TOKENS else None,
    "torch_threads": int(TORCH_THREADS) if TORCH_THREADS else None,
}

# Replace the /health route with one that also reports the configuration.
server.app.router.routes = [r for r in server.app.router.routes
                            if getattr(r, "path", None) != "/health"]


@server.app.get("/health")
async def health():
    return {"service": server.SERVICE_NAME, "pid": os.getpid(), "mode": server.MODE,
            "model_loaded": server._model_loaded(), "configuration": CONFIGURATION}


app = server.app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("MODEL_SERVER_PORT", "8000")),
                log_level="warning")
