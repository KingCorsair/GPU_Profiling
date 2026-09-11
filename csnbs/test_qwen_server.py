"""HTTP integration checks; these do not load weights or produce benchmark data."""

import base64
from io import BytesIO
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from csnbs.qwen_server import create_app


class QwenHTTPTests(unittest.TestCase):
    def setUp(self):
        self.loads = 0
        self.calls = []

        def factory():
            self.loads += 1
            owner = self

            class Backend:
                metadata = {"model_type": "qwen3_vl", "pruning": False}

                def generate(self, image, question):
                    owner.calls.append((image.mode, image.size, question))
                    return "red"

            return Backend()

        self.client = self.enterContext(TestClient(create_app(factory)))

    def test_existing_infer_contract_and_single_model_load(self):
        buffer = BytesIO()
        Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(buffer, "PNG")
        payload = {
            "image_b64": base64.b64encode(buffer.getvalue()).decode(),
            "question": "What color is the image?",
        }
        for _ in range(2):
            response = self.client.post("/infer", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"answer": "red"})
        self.assertEqual(self.loads, 1)
        self.assertEqual(self.calls, [("RGB", (64, 64), payload["question"])] * 2)
        health = self.client.get("/health").json()
        self.assertTrue(health["model_loaded"])
        self.assertEqual(health["service"], "csnbs-qwen-server")

    def test_invalid_images_do_not_reach_model(self):
        for content in ["!not-base64", base64.b64encode(b"not an image").decode()]:
            response = self.client.post("/infer", json={"image_b64": content, "question": "Color?"})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.calls, [])

    def test_missing_inputs_rejected(self):
        self.assertEqual(self.client.post("/infer", json={"question": "Color?"}).status_code, 422)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
