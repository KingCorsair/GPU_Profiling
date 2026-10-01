"""Service instrumentation contracts; no model execution or GPU required."""
import asyncio
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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
        with patch.object(server, "MODE", "fake"), patch.object(server, "_infer_fake", fake):
            reply = asyncio.run(server.infer(server.InferRequest(image_b64="aGVsbG8=", question="test", request_id="r1")))
        self.assertEqual(reply.request_id, "r1")
        self.assertEqual(reply.answer, "answer")
        self.assertGreaterEqual(reply.metrics["service_ms"], 8)
        self.assertIsNone(reply.metrics["queue_ms"])
        self.assertIsNone(reply.metrics["generated_text_tokens"])

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
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", code, str(Path(server.__file__).resolve())], cwd=directory,
                                    env=environment, text=True, capture_output=True, timeout=35)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("direct-script-entry-ok", result.stdout)

if __name__ == '__main__':
    unittest.main()
