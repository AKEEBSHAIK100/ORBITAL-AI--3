"""Regression tests for honest BigEarthNet availability handling."""

import io
import sys
import unittest
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.tools.land_cover import BigEarthNetTool


class FakeUnavailableClassifier:
    def classify_image(self, image_bytes, top_k=5, threshold=0.25):
        return {
            "status": "unavailable",
            "top_label": None,
            "confidence": None,
            "labels": [],
            "active_labels": [],
            "model_id": "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
            "available": False,
            "note": "real classifier unavailable",
            "citation": None,
            "error": "weights missing",
        }


class TestLandCoverIntegrity(unittest.TestCase):
    def test_unavailable_classifier_is_not_rewritten_as_success(self):
        tool = BigEarthNetTool(classifier=FakeUnavailableClassifier())
        image = io.BytesIO()
        Image.new("RGB", (32, 32)).save(image, format="PNG")
        result = tool.run({"image": image.getvalue()})

        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["top_label"])
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["labels"], [])
        self.assertEqual(result["active_labels"], [])

    def test_tool_does_not_expose_heuristic_model_identity(self):
        tool = BigEarthNetTool(classifier=FakeUnavailableClassifier())
        image = io.BytesIO()
        Image.new("RGB", (32, 32)).save(image, format="PNG")
        result = tool.run({"image": image.getvalue()})

        self.assertNotEqual(result.get("model_id"), "heuristic-fallback")
        self.assertNotIn("heuristic", str(result).lower())


if __name__ == "__main__":
    unittest.main()
