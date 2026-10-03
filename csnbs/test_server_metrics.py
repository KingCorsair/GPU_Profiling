"""Service instrumentation contracts; no model execution or GPU required."""
import asyncio
from contextlib import ExitStack, nullcontext
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from csnbs import server


class MetricsContractTests(unittest.TestCase):
    def test_health_reports_effective_config_and_host_provenance(self):
        health = asyncio.run(server.health())
        self.assertEqual(health.configuration["important_ratio"], 0.5)
        self.assertEqual(health.configuration["batch_size"], 1)
        self.assertEqual(health.hardware["source"], "server-nvidia-smi")
        self.assertIn("gitCommit", health.source)

    def test_metrics_correlate_without_claiming_socket_queue_time(self):
        async def fake(*_):
            await asyncio.sleep(.01)
            return "answer"
        with patch.object(server, "MODE", "fake"), patch.object(server, "_infer_fake", fake), patch.object(server, "EXTENDED_HTTP_OBSERVATIONS", False):
            reply = asyncio.run(server.infer(server.InferRequest(image_b64="aGVsbG8=", question="test", request_id="r1")))
        self.assertEqual(reply.request_id, "r1")
        self.assertEqual(reply.answer, "answer")
        self.assertGreaterEqual(reply.metrics["service_ms"], 8)
        self.assertIsNone(reply.metrics["queue_ms"])
        self.assertIsNone(reply.metrics["generated_text_tokens"])
        self.assertEqual(reply.metrics["schema_version"], 1)
        self.assertNotIn("output_characters", reply.metrics)

    def test_extended_mode_is_visible_in_health_and_fake_does_not_invent_tokens(self):
        async def fake(*_):
            return "A🙂e\u0301"
        with patch.object(server, "MODE", "fake"), patch.object(server, "_infer_fake", fake):
            with patch.object(server, "EXTENDED_HTTP_OBSERVATIONS", False):
                off = asyncio.run(server.health()).configuration
            with patch.object(server, "EXTENDED_HTTP_OBSERVATIONS", True):
                on = asyncio.run(server.health()).configuration
                reply = asyncio.run(server.infer(server.InferRequest(image_b64="aGVsbG8=", question="test")))
        self.assertEqual([key for key in on if on[key] != off[key]], ["instrumentation"])
        self.assertEqual(on["instrumentation"], "handler-stage-token-observations-v2")
        self.assertEqual(reply.metrics["schema_version"], 2)
        self.assertEqual(reply.metrics["output_characters"], 4)  # Unicode code points, not bytes/graphemes.
        self.assertIsNone(reply.metrics["generated_text_tokens"])
        self.assertIsNone(reply.metrics["generation_wall_ms"])
        self.assertGreaterEqual(reply.metrics["base64_decode_ms"], 0)
        self.assertIn("Fake mode", reply.metrics["token_unavailable_reason"])

    def test_invalid_image_stays_a_client_error(self):
        with self.assertRaises(server.HTTPException) as error:
            asyncio.run(server.infer(server.InferRequest(image_b64="!!!", question="test")))
        self.assertEqual(error.exception.status_code, 400)

    def test_model_identity_comes_from_checkpoint_provenance_and_missing_stays_unknown(self):
        cases = [
            (None, None),
            ({}, None),
            ({"revision": "verified-revision"}, None),
            ({"model_id": "liuhaotian/llava-v1.5-7b", "revision": "verified-revision"}, "liuhaotian/llava-v1.5-7b"),
            ({"model_id": "different/model", "revision": "other-revision"}, "different/model"),
        ]
        for provenance, expected in cases:
            with self.subTest(provenance=provenance), patch.object(server, "MODE", "model"), patch.object(server, "CHECKPOINT_PROVENANCE", provenance), patch.object(server, "_model_loaded", return_value=False):
                health = asyncio.run(server.health())
            self.assertEqual(health.configuration["model_id"], expected)
            self.assertEqual(health.configuration["download_provenance"], provenance)

    def test_direct_script_entry_imports_without_repository_pythonpath_or_binding_socket(self):
        # Reproduce a direct-script import environment; intercept only the final
        # uvicorn entry so this check never opens a port or starts a model.
        code = """
import runpy
import sys
from pathlib import Path
from unittest.mock import patch
script = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(script.parent))
with patch('uvicorn.run') as start:
    runpy.run_path(str(script), run_name='__main__')
    assert start.call_count == 1
    assert start.call_args.kwargs['port'] == 8000
print('direct-script-entry-ok')
"""
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        environment["SERVER_MODE"] = "fake"
        environment["EXTENDED_HTTP_OBSERVATIONS"] = "0"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", code, str(Path(server.__file__).resolve())], cwd=directory,
                                    env=environment, text=True, capture_output=True, timeout=35)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("direct-script-entry-ok", result.stdout)


class OutputIds:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return self.rows


def model_fixture():
    return SimpleNamespace(
        config=SimpleNamespace(mm_use_im_start_end=False, is_encoder_decoder=False, tokenizer_model_max_length=None),
        generation_config=SimpleNamespace(num_beams=1, num_beam_groups=1, num_return_sequences=1,
                                         bos_token_id=1, eos_token_id=2, penalty_alpha=None, constraints=None, force_words_ids=None),
        device="mock-gpu",
    )


class TokenObservationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.model = model_fixture()
        self.stack.enter_context(patch.object(server, "model", self.model))
        self.stack.enter_context(patch.object(server, "tokenizer", SimpleNamespace(all_special_ids=[0, 1, 2, 55])))
        self.stack.enter_context(patch.object(server, "PROVENANCE", {"runtime": {"transformers": "4.37.2"}}))
        self.stack.enter_context(patch.object(server, "MAX_NEW_TOKENS", 64))

    def observe(self, prompt=None, output=None, visual=128):
        return server._token_observations([1, 18, -200, 20] if prompt is None else prompt,
                                          OutputIds([[1, 10, 55, 11, 2]] if output is None else output), visual, -200)

    def test_bos_is_not_prompt_and_eos_is_a_step_but_not_text(self):
        result = self.observe()
        self.assertEqual(result["prompt_text_tokens"], 3)
        self.assertEqual(result["visual_tokens"], 128)
        self.assertEqual(result["multimodal_prompt_tokens"], 131)
        self.assertEqual(result["returned_output_ids"], 5)
        self.assertEqual(result["output_seed_tokens"], 1)
        self.assertEqual(result["generated_token_steps"], 4)
        self.assertEqual(result["generated_text_tokens"], 2)
        self.assertEqual(result["generated_eos_tokens"], 1)
        self.assertTrue(result["generation_ended_with_eos"])
        self.assertIsNone(result["token_unavailable_reason"])

    def test_budget_end_and_immediate_eos_are_distinct_valid_outputs(self):
        budget = self.observe(output=[[1] + [10] * 64])
        self.assertEqual(budget["generated_token_steps"], 64)
        self.assertEqual(budget["generated_text_tokens"], 64)
        self.assertFalse(budget["generation_ended_with_eos"])
        immediate = self.observe(output=[[1, 2]])
        self.assertEqual(immediate["generated_token_steps"], 1)
        self.assertEqual(immediate["generated_text_tokens"], 0)

    def test_output_mismatch_fails_null_without_losing_independent_prompt_counts(self):
        for rows, reason in [([[10, 11, 2]], "BOS"), ([[1, 2, 10]], "EOS"),
                             ([[1, 10]], "before budget"), ([[1] + [10] * 65], "budget"),
                             ([[1, 2], [1, 2]], "one integer sequence"), ([[1, -9, 2]], "negative")]:
            with self.subTest(rows=rows):
                result = self.observe(output=rows)
                self.assertIsNone(result["generated_token_steps"])
                self.assertIsNone(result["generated_text_tokens"])
                self.assertEqual(result["prompt_text_tokens"], 3)
                self.assertIn(reason, result["output_token_unavailable_reason"])
        for version in [None, "4.38.0", "4.37.2+unverified"]:
            with self.subTest(version=version), patch.object(server, "PROVENANCE", {"runtime": {"transformers": version}}):
                result = self.observe()
                self.assertIsNone(result["generated_text_tokens"])
                self.assertEqual(result["returned_output_ids"], 5)
                self.assertIn("4.37.2", result["output_token_unavailable_reason"])

    def test_generation_and_special_id_contracts_are_checked(self):
        for attribute, value in [("num_beams", 2), ("num_return_sequences", 2), ("bos_token_id", None),
                                 ("eos_token_id", 999), ("penalty_alpha", .6), ("constraints", [])]:
            with self.subTest(attribute=attribute), patch.object(self.model.generation_config, attribute, value):
                self.assertIsNone(self.observe()["generated_text_tokens"])
        with patch.object(server, "tokenizer", SimpleNamespace(all_special_ids=[2])):
            self.assertIsNone(self.observe()["generated_text_tokens"])
        with patch.object(self.model.config, "is_encoder_decoder", True):
            self.assertIsNone(self.observe()["generated_text_tokens"])

    def test_visual_count_is_actual_before_truncation_and_post_truncation_stays_unknown(self):
        self.model.config.tokenizer_model_max_length = 131
        self.assertEqual(self.observe()["visual_tokens"], 128)
        self.model.config.tokenizer_model_max_length = 130
        result = self.observe()
        self.assertEqual(result["visual_tokens_pretruncation"], 128)
        self.assertEqual(result["prompt_text_tokens_pretruncation"], 3)
        self.assertIsNone(result["visual_tokens"])
        self.assertIsNone(result["prompt_text_tokens"])
        self.assertEqual(result["generated_text_tokens"], 2)
        self.assertIn("truncation", result["prompt_token_unavailable_reason"])
        for prompt in [[1, 18, 20], [1, -200, -200, 20]]:
            with self.subTest(prompt=prompt):
                self.assertIsNone(self.observe(prompt=prompt)["visual_tokens"])
        for visual in [None, True, 0, 577, 128.0]:
            with self.subTest(visual=visual):
                self.assertIsNone(self.observe(visual=visual)["visual_tokens"])


class ModelBoundaryTests(unittest.TestCase):
    def run_model(self, extended, break_observation=False):
        events = []

        class Tensor:
            def __init__(self, name, values):
                self.name, self.values = name, values

            def unsqueeze(self, dimension):
                events.append(("unsqueeze", self.name, dimension))
                return self

            def to(self, *args, **kwargs):
                events.append(("to", self.name, args, kwargs))
                return self

            def tolist(self):
                events.append(("tolist", self.name))
                if break_observation and self.name == "prompt":
                    raise RuntimeError("unfamiliar input tensor")
                return self.values

        class Conversation:
            roles = ["user", "assistant"]

            def copy(self):
                return Conversation()

            def append_message(self, role, content):
                events.append(("message", role, content))

            def get_prompt(self):
                return "formatted conversation"

        prompt = Tensor("prompt", [1, -200, 18, 20])
        pixels = Tensor("pixels", [])
        output = Tensor("output", [[1, 10, 2]])
        model = model_fixture()
        image = SimpleNamespace(size=(336, 336), convert=lambda mode: image)

        def generate(*args, **kwargs):
            events.append(("generate", args, kwargs))
            return output, 128

        def decode(ids, **kwargs):
            events.append(("decode", ids, kwargs))
            ids.tolist()  # Existing decoding already materializes output IDs.
            return ["  Hi🙂  "]

        model.generate = generate
        tokenizer = SimpleNamespace(all_special_ids=[0, 1, 2], batch_decode=decode)
        modules = {
            "torch": SimpleNamespace(float16="float16", inference_mode=lambda: nullcontext()),
            "PIL": SimpleNamespace(Image=SimpleNamespace(open=lambda data: image)),
            "llava.constants": SimpleNamespace(DEFAULT_IMAGE_TOKEN="<image>", DEFAULT_IM_START_TOKEN="<start>",
                                               DEFAULT_IM_END_TOKEN="<end>", IMAGE_TOKEN_INDEX=-200),
            "llava.conversation": SimpleNamespace(conv_templates={"llava_v1": Conversation()}),
            "llava.mm_utils": SimpleNamespace(process_images=lambda *args: pixels, tokenizer_image_token=lambda *args, **kwargs: prompt),
        }
        with patch.dict(sys.modules, modules), patch.object(server, "model", model), patch.object(server, "tokenizer", tokenizer), \
             patch.object(server, "MODE", "model"), patch.object(server, "EXTENDED_HTTP_OBSERVATIONS", extended), \
             patch.object(server, "PROVENANCE", {"runtime": {"transformers": "4.37.2"}}):
            result = asyncio.run(server.infer(server.InferRequest(image_b64="aGVsbG8=", question="test", request_id="r2")))
        call = next(event for event in events if event[0] == "generate")
        self.assertEqual(call[1], (prompt,))
        self.assertEqual(call[2], dict(images=pixels, image_sizes=[(336, 336)], do_sample=False, max_new_tokens=server.MAX_NEW_TOKENS, use_cache=True))
        decoded = next(event for event in events if event[0] == "decode")
        self.assertIs(decoded[1], output)
        self.assertEqual(decoded[2], {"skip_special_tokens": True})
        return result, events

    def test_observations_do_not_change_generation_or_decode_and_off_adds_no_tensor_inspection(self):
        off, off_events = self.run_model(False)
        on, on_events = self.run_model(True)
        self.assertEqual(off.answer, "Hi🙂")
        self.assertEqual(off.answer, on.answer)
        self.assertEqual(off.request_id, on.request_id)
        self.assertEqual(off.metrics["schema_version"], 1)
        self.assertIsNone(off.metrics["generation_wall_ms"])
        self.assertNotIn(("tolist", "prompt"), off_events)
        self.assertEqual(off_events.count(("tolist", "output")), 1)
        self.assertGreater(on_events.index(("tolist", "prompt")), next(i for i, event in enumerate(on_events) if event[0] == "decode"))
        self.assertEqual(on.metrics["output_characters"], 3)
        self.assertEqual(on.metrics["generated_token_steps"], 2)
        self.assertEqual(on.metrics["generated_text_tokens"], 1)
        self.assertEqual(on.metrics["prompt_text_tokens"], 3)
        for field in ["base64_decode_ms", "preprocess_ms", "generation_wall_ms", "postprocess_ms", "observation_wall_ms"]:
            self.assertGreaterEqual(on.metrics[field], 0)
        self.assertGreaterEqual(on.metrics["service_ms"], sum(on.metrics[field] for field in ["base64_decode_ms", "preprocess_ms", "generation_wall_ms", "postprocess_ms", "observation_wall_ms"]))
        self.assertIn("remain enabled", on.metrics["model_internal_instrumentation"])

    def test_observation_failure_preserves_answer_and_null_counts(self):
        reply, _ = self.run_model(True, break_observation=True)
        self.assertEqual(reply.answer, "Hi🙂")
        self.assertEqual(reply.metrics["output_characters"], 3)
        self.assertIsNone(reply.metrics["generated_text_tokens"])
        self.assertIsNone(reply.metrics["visual_tokens"])
        self.assertIn("Token observation failed", reply.metrics["token_unavailable_reason"])

if __name__ == '__main__':
    unittest.main()
