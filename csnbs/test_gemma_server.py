"""Validate the loadgen HTTP contract without model downloads or GPU inference."""

import base64
from io import BytesIO
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from csnbs.gemma_server import create_app


class GemmaHTTPTests(unittest.TestCase):
    def setUp(self):
        self.loads = 0
        self.calls = []

        def factory():
            self.loads += 1
            owner = self

            class Backend:
                metadata = {"model_id": "google/gemma-3-4b-it", "pruning": False}

                def generate(self, image, question):
                    owner.calls.append((image.mode, image.size, question))
                    return "red"

            return Backend()

        self.client = self.enterContext(TestClient(create_app(factory)))

    def test_loadgen_contract_and_single_model_load(self):
        buffer = BytesIO()
        Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(buffer, "PNG")
        payload = {"image_b64": base64.b64encode(buffer.getvalue()).decode(), "question": "Color?"}
        for _ in range(2):
            response = self.client.post("/infer", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"answer": "red"})
        self.assertEqual(self.loads, 1)
        self.assertEqual(self.calls, [("RGB", (64, 64), "Color?")] * 2)
        health = self.client.get("/health").json()
        self.assertEqual(health["service"], "csnbs-gemma-server")
        self.assertTrue(health["model_loaded"])
        self.assertEqual(health["configuration"]["model_id"], "google/gemma-3-4b-it")

    def test_bad_inputs_never_reach_model(self):
        for value in ["!not-base64", base64.b64encode(b"not an image").decode()]:
            self.assertEqual(self.client.post("/infer", json={"image_b64": value, "question": "Color?"}).status_code, 400)
        for payload in [{"question": "Color?"}, {"image_b64": "", "question": ""}]:
            self.assertEqual(self.client.post("/infer", json=payload).status_code, 422)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
