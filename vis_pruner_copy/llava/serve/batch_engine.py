"""
Single-worker inference engine: request queue (WP4, server state S1) and static
micro-batching (WP5, server state S2) in one class.

    engine = InferenceEngine(model, tokenizer, max_batch_size=8, max_wait_ms=10)
    future = engine.submit(image_tensor, input_ids, image_size)   # from any thread
    answer = future.result()                                       # or: await engine.infer(...)

Rules this file is built around:

* All CUDA work happens in ONE worker thread. Callers hand over CPU tensors; the H2D copy,
  the vision tower, pruning, prefill and decode all run in the worker. Two threads inside
  model.generate() is the configuration that produced the device-side assert
  (scripts/repro_concurrent_crash.py).
* max_batch_size=1 is S1: the queue only lets CPU preprocessing of request N+1 overlap
  GPU work on request N. max_batch_size>1 is S2: the worker drains up to that many
  queued requests, waiting at most max_wait_ms after the first, and runs one generate().
* The queue is bounded. submit() raises queue.Full instead of letting a throughput
  problem grow into an out-of-memory problem; the server turns that into HTTP 503.
* Batched prompts are left-padded (model.config.tokenizer_padding_side must be "left",
  set in builder.py). Generation is greedy with one shared max/min_new_tokens per engine,
  so every request in a batch is generated under identical parameters.
"""
import asyncio
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class _Request:
    image: torch.Tensor          # (3, H, W) CPU tensor from process_images
    input_ids: torch.Tensor      # (L,) CPU LongTensor from tokenizer_image_token
    image_size: tuple            # PIL (width, height)
    future: Future = field(default_factory=Future)
    t_submit: float = field(default_factory=time.perf_counter)


class InferenceEngine:
    def __init__(self, model, tokenizer, max_batch_size=1, max_wait_ms=0.0,
                 max_new_tokens=64, min_new_tokens=None, max_queue=256):
        if getattr(model.config, "tokenizer_padding_side", "right") != "left" and max_batch_size > 1:
            raise ValueError("batched generation needs model.config.tokenizer_padding_side='left'")
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self._queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self.batch_sizes = []    # size of every batch run, for diagnostics
        self._worker = threading.Thread(target=self._run, name="inference-worker", daemon=True)
        self._worker.start()

    # --- caller side -------------------------------------------------------------------
    def submit(self, image, input_ids, image_size) -> Future:
        """Queue one request. Raises queue.Full if the queue is at capacity."""
        req = _Request(image=image, input_ids=input_ids, image_size=tuple(image_size))
        self._queue.put_nowait(req)
        return req.future

    async def infer(self, image, input_ids, image_size) -> str:
        return await asyncio.wrap_future(self.submit(image, input_ids, image_size))

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def shutdown(self, timeout=30):
        self._stop.set()
        self._worker.join(timeout=timeout)

    # --- worker side -------------------------------------------------------------------
    def _collect(self) -> List[_Request]:
        """Block for the first request, then take more until the batch is full or the
        wait window since the first one has closed."""
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.1)
                break
            except queue.Empty:
                continue
        else:
            return []
        batch = [first]
        deadline = time.perf_counter() + self.max_wait_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            try:
                batch.append(self._queue.get(timeout=remaining) if remaining > 0
                             else self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _run(self):
        while not self._stop.is_set():
            batch = self._collect()
            if not batch:
                continue
            batch = [r for r in batch if r.future.set_running_or_notify_cancel()]
            if not batch:
                continue
            try:
                answers = self.generate_batch(
                    [r.image for r in batch], [r.input_ids for r in batch],
                    [r.image_size for r in batch])
            except BaseException as exc:  # one bad batch must not kill the worker
                for r in batch:
                    r.future.set_exception(exc)
                continue
            self.batch_sizes.append(len(batch))
            for r, answer in zip(batch, answers):
                r.future.set_result(answer)

    @torch.inference_mode()
    def generate_batch(self, images, input_ids_list, image_sizes) -> List[str]:
        """One generate() over B requests. Public so offline equivalence checks run the
        exact code path the server runs, without a queue or threads in between."""
        device = self.model.device
        max_len = max(ids.shape[0] for ids in input_ids_list)
        input_ids = torch.full((len(input_ids_list), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
        for i, ids in enumerate(input_ids_list):
            input_ids[i, max_len - ids.shape[0]:] = ids          # left padding
            attention_mask[i, max_len - ids.shape[0]:] = 1

        gen_kwargs = dict(do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens,
                          use_cache=True, pad_token_id=self.pad_token_id)
        if self.min_new_tokens is not None:
            gen_kwargs["min_new_tokens"] = self.min_new_tokens

        output_ids, _ = self.model.generate(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            images=torch.stack(images).to(device, dtype=torch.float16),
            image_sizes=list(image_sizes),
            **gen_kwargs,
        )
        # generate() is driven by inputs_embeds, so output_ids holds only new tokens.
        # Sequences that hit EOS early are padded with pad_token_id (<unk>, a special
        # token), which skip_special_tokens strips along with the EOS itself.
        return [s.strip() for s in self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)]
