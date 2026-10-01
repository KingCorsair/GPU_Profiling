"""Service instrumentation contracts; no model execution or GPU required."""
import asyncio
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

if __name__ == '__main__':
    unittest.main()
