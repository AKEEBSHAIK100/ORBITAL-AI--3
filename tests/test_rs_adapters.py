"""
tests/test_rs_adapters.py — Unit tests for BLIP LoRA remote sensing adapters,
execution trace integrity, sensor-agnostic validation, and registry specs.
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.rs_adapters import RSAdapterRuntime, SpecialistState
from backend.tools.registry import get_tool, list_all_tools
from backend.agents.router import classify_query_intent, route_query_to_specialist
from backend.agents.validator import validate_input_imagery


class TestRSAdaptersAndArchitecture(unittest.TestCase):
    """Verifies BLIP LoRA adapters, model registries, validators, and router compliance."""

    def setUp(self):
        # Create a dummy test image for inference tests
        self.test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.pil_img = Image.new("RGB", (100, 100), color=(73, 109, 137))
        self.runtime = RSAdapterRuntime.get_instance()

    def test_01_caption_adapter_unavailable_or_loads(self):
        """Test 1: Caption adapter loads or produces structured SPECIALIST_UNAVAILABLE when weights absent."""
        res = self.runtime.caption(self.test_img)
        self.assertIn("status", res)
        self.assertIn(res["status"], ["success", "specialist_unavailable", "error"])
        if res["status"] == "specialist_unavailable":
            self.assertIsNone(res["caption"])
            self.assertIn("unavailable", res["answer"].lower())

    def test_02_vqa_adapter_unavailable_or_loads(self):
        """Test 2: VQA adapter loads or produces structured SPECIALIST_UNAVAILABLE when weights absent."""
        res = self.runtime.vqa(self.test_img, "Is there a river in the image?")
        self.assertIn("status", res)
        self.assertIn(res["status"], ["success", "specialist_unavailable", "error"])
        if res["status"] == "specialist_unavailable":
            self.assertIn("unavailable", res["answer"].lower())

    def test_03_caption_schema_integrity(self):
        """Test 3: Caption inference returns structured schema (status, caption, model_id, provenance, etc.)."""
        res = self.runtime.caption(self.test_img)
        expected_keys = {
            "status", "caption", "answer", "model_id", "base_model",
            "adapter", "device", "inference_time_ms", "confidence",
            "confidence_status", "provenance", "warnings"
        }
        for k in expected_keys:
            self.assertIn(k, res, f"Key '{k}' missing from caption response")
        self.assertEqual(res["model_id"], "rs-caption-adapted-v1")
        self.assertEqual(res["base_model"], "Salesforce/blip-image-captioning-base")
        self.assertIn("BigEarthNet", res["adapter"])

    def test_04_vqa_schema_integrity(self):
        """Test 4: VQA inference returns structured schema (status, answer, model_id, provenance, etc.)."""
        res = self.runtime.vqa(self.test_img, "What land cover is visible?")
        expected_keys = {
            "status", "question", "answer", "model_id", "base_model",
            "adapter", "device", "inference_time_ms", "confidence",
            "confidence_status", "provenance", "warnings"
        }
        for k in expected_keys:
            self.assertIn(k, res, f"Key '{k}' missing from VQA response")
        self.assertEqual(res["model_id"], "rs-vqa-adapted-v1")
        self.assertEqual(res["base_model"], "Salesforce/blip-vqa-base")
        self.assertIn("BigEarthNet", res["adapter"])

    def test_05_unavailable_adapter_returns_honest_null_confidence(self):
        """Test 5: Unavailable adapter returns status='specialist_unavailable' and confidence=None."""
        # Force a simulation of unavailable runtime
        orig_state = self.runtime.caption_state
        try:
            self.runtime.caption_state = SpecialistState.UNAVAILABLE
            self.runtime.caption_unavailable_reason = "Test missing weights"
            res = self.runtime.caption(self.test_img)
            self.assertEqual(res["status"], "specialist_unavailable")
            self.assertIsNone(res["confidence"])
            self.assertEqual(res["confidence_status"], "not_calibrated")
        finally:
            self.runtime.caption_state = orig_state

    def test_06_registry_exposes_adapted_specialists(self):
        """Test 6: Registry exposes both adapted specialists with correct base model and adapter metadata."""
        tools = list_all_tools()
        self.assertIn("rs_caption_adapted", tools)
        self.assertIn("rs_vqa_adapted", tools)

        caption_spec = tools["rs_caption_adapted"]
        self.assertEqual(caption_spec.get("base_model"), "Salesforce/blip-image-captioning-base")
        self.assertIn("BigEarthNet", caption_spec.get("adapter", ""))

        vqa_spec = tools["rs_vqa_adapted"]
        self.assertEqual(vqa_spec.get("base_model"), "Salesforce/blip-vqa-base")
        self.assertIn("BigEarthNet", vqa_spec.get("adapter", ""))

    def test_07_router_routes_caption_and_vqa(self):
        """Test 7: Router routes caption query to rs_caption_adapted and QA query to rs_vqa_adapted."""
        cap_intent = classify_query_intent("Provide a scene overview and detailed description", image_count=1)
        self.assertEqual(cap_intent, "caption")
        cap_tool, cap_info = route_query_to_specialist("Provide a scene overview and detailed description", image_count=1)
        self.assertIn(cap_tool, ["rs_caption_adapted", "captioning"])
        self.assertEqual(cap_info["task"], "caption")

        vqa_intent = classify_query_intent("Is there a runway or road visible?", image_count=1)
        self.assertEqual(vqa_intent, "vqa")
        vqa_tool, vqa_info = route_query_to_specialist("Is there a runway or road visible?", image_count=1)
        self.assertIn(vqa_tool, ["rs_vqa_adapted", "rs_vqa"])
        self.assertEqual(vqa_info["task"], "vqa")

    def test_08_pair_validation_rejects_single_image(self):
        """Test 8: Pair validation rejects single-image change detection."""
        single_img = [np.zeros((64, 64, 3), dtype=np.uint8)]
        res_cd = validate_input_imagery(task_type="change_detection", images=single_img)
        self.assertEqual(res_cd["compatibility"], "error")
        self.assertTrue(any("two co-registered" in err for err in res_cd["errors"]))

        res_fus = validate_input_imagery(task_type="sar_optical_fusion", images=single_img)
        self.assertEqual(res_fus["compatibility"], "error")

    def test_09_paired_metadata_uncertainty_surfaced(self):
        """Test 9: Paired metadata uncertainty is surfaced (compatibility='unknown')."""
        pair = [np.zeros((64, 64, 3), dtype=np.uint8), np.zeros((64, 64, 3), dtype=np.uint8)]
        # Two images provided but without CRS or affine metadata
        res = validate_input_imagery(
            task_type="change_detection",
            images=pair,
            metadata_list=[{}, {}]
        )
        self.assertEqual(res["compatibility"], "unknown")
        self.assertTrue(
            any("Spatial co-registration could not be independently verified" in w for w in res["warnings"]),
            f"Expected warning missing in: {res['warnings']}"
        )

    def test_10_no_hardcoded_fake_confidence(self):
        """Test 10: No hard-coded confidence constant (0.85, 0.94, 0.98) is returned for adapted models."""
        cap_tool = get_tool("rs_caption_adapted")
        self.assertIsNotNone(cap_tool)
        out = cap_tool.run({"image": self.test_img})
        # Confidence must be None / uncalibrated or properly structured, never fake 0.85/0.94/0.98
        self.assertNotIn(out.get("confidence"), [0.85, 0.94, 0.97, 0.98])
        if out.get("confidence") is not None:
            self.assertNotIn(round(float(out["confidence"]), 2), [0.85, 0.94, 0.97, 0.98])


if __name__ == "__main__":
    unittest.main()
