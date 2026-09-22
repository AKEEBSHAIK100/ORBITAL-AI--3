"""
tests/test_rs_adaptllm.py — Test suite for AdaptLLM Remote-Sensing VLM Integration.

Verifies:
1. ModelRegistry metadata for rs_adaptllm (stable ID, provenance, uncalibrated status)
2. Availability detection (strictly offline, False by default without weights)
3. No-auto-download behavior
4. Structured specialist_unavailable response contract
5. Truthful provenance schema with visual token budget
6. Planner routing hierarchy (known specialists preserved; open queries route to rs_adaptllm if available, else rs_generalist)
7. No automatic fallback substitution from AdaptLLM to Qwen generalist
8. Memory safety, token/pixel budget configuration, and downscaling tracking
9. Lifecycle unload hook and memory management
"""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from models.registry import ModelRegistry, SpecialistEntry
from backend.tools.registry import get_tool, list_all_tools, ALL_TOOL_IDS, CANONICAL_TOOL_IDS
from backend.tools.adaptllm import (
    AdaptLLMTool,
    is_adaptllm_available,
    DEFAULT_MIN_PIXELS,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MAX_NEW_TOKENS,
)
from backend.agents.planner import create_query_plan


class TestAdaptLLMIntegration(unittest.TestCase):
    def setUp(self):
        self.registry = ModelRegistry.get_instance()

    # ── 1. Model Registry Metadata ───────────────────────────────────────────
    def test_01_registry_metadata(self):
        spec = self.registry.get_specialist("rs_adaptllm")
        self.assertIsNotNone(spec, "rs_adaptllm must be registered in ModelRegistry")
        self.assertEqual(spec.id, "rs_adaptllm")
        self.assertEqual(spec.model_id, "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct")
        self.assertEqual(spec.model, "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct")
        self.assertEqual(spec.base_model, "Qwen/Qwen2-VL-2B-Instruct")
        self.assertEqual(spec.model_type, "remote_sensing_vlm")
        self.assertTrue(spec.is_remote_sensing_adapted)
        self.assertTrue(spec.is_remote_sensing_expert)
        self.assertEqual(spec.adaptation_scope, "remote_sensing_domain_post_training")
        self.assertIsNone(spec.benchmark_accuracy_claim)
        self.assertIsNone(spec.confidence)
        self.assertEqual(spec.confidence_status, "not_calibrated")

    # ── 2. Tool Registry Integration ─────────────────────────────────────────
    def test_02_tool_registry_registration(self):
        self.assertIn("rs_adaptllm", CANONICAL_TOOL_IDS)
        self.assertIn("rs_adaptllm", ALL_TOOL_IDS)
        tool = get_tool("rs_adaptllm")
        self.assertIsNotNone(tool)
        self.assertIsInstance(tool, AdaptLLMTool)
        self.assertEqual(tool.id, "rs_adaptllm")
        self.assertEqual(tool.model_id, "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct")

    # ── 3. Availability Detection & No Auto-Download ─────────────────────────
    def test_03_default_availability_is_false(self):
        with patch.dict(os.environ, {}, clear=True):
            tool = AdaptLLMTool()
            self.assertFalse(tool.is_available)
            self.assertFalse(is_adaptllm_available())
            self.assertIn("not installed or enabled", tool.unavailable_reason)

    def test_04_availability_with_valid_weights_dir(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            weight_file = Path(tmp_dir) / "model.safetensors"
            weight_file.write_bytes(b"dummy safetensors")
            with patch.dict(os.environ, {"ADAPTLLM_MODEL_PATH": tmp_dir, "ENABLE_ADAPTLLM": "true"}):
                self.assertTrue(is_adaptllm_available())
                tool = AdaptLLMTool()
                self.assertTrue(tool.is_available)

    # ── 4. Structured Unavailable Response ───────────────────────────────────
    def test_05_unavailable_response_contract(self):
        tool = AdaptLLMTool()
        with patch.object(AdaptLLMTool, "is_available", False):
            res = tool.run({"query": "Are there solar panels?"})
            self.assertEqual(res["status"], "specialist_unavailable")
            self.assertIn("AdaptLLM remote-sensing VLM weights are not installed or enabled.", res["warnings"])
            self.assertIsNone(res["confidence"])
            self.assertEqual(res["confidence_status"], "not_calibrated")
            self.assertEqual(res["model"], "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct")
            self.assertEqual(res["base_model"], "Qwen/Qwen2-VL-2B-Instruct")
            self.assertIsNone(res["adapter"])
            self.assertEqual(res["evidence_source"], "adaptllm_vlm")
            # Must NOT be Qwen generalist
            self.assertNotEqual(res["evidence_source"], "generalist_vlm")

    # ── 5. Provenance Schema & Budget ────────────────────────────────────────
    def test_06_provenance_schema(self):
        tool = AdaptLLMTool()
        prov = tool.get_provenance(budget_meta={"downscaled": False, "original_size": [512, 512], "processed_size": [512, 512]})
        self.assertEqual(prov["model"], "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct")
        self.assertEqual(prov["base_model"], "Qwen/Qwen2-VL-2B-Instruct")
        self.assertIsNone(prov["adapter"])
        self.assertEqual(prov["model_type"], "remote_sensing_vlm")
        self.assertTrue(prov["is_remote_sensing_adapted"])
        self.assertTrue(prov["is_remote_sensing_expert"])
        self.assertEqual(prov["adaptation_scope"], "remote_sensing_domain_post_training")
        self.assertIsNone(prov["benchmark_accuracy_claim"])
        self.assertIsNone(prov["confidence"])
        self.assertEqual(prov["confidence_status"], "not_calibrated")
        self.assertIn("visual_token_budget", prov)
        budget = prov["visual_token_budget"]
        self.assertEqual(budget["min_pixels"], DEFAULT_MIN_PIXELS)
        self.assertEqual(budget["max_pixels"], DEFAULT_MAX_PIXELS)
        self.assertEqual(budget["max_new_tokens"], DEFAULT_MAX_NEW_TOKENS)
        self.assertFalse(budget["downscaled"])

    # ── 6. Memory Safety & Downscaling ───────────────────────────────────────
    def test_07_memory_safety_downscaling(self):
        tool = AdaptLLMTool()
        # Large image exceeding max_image_size 1280
        large_arr = np.zeros((1600, 2000, 3), dtype=np.uint8)
        proc_img, meta = tool._preprocess_image(large_arr, max_dim=1280)
        self.assertTrue(meta["downscaled"])
        self.assertEqual(meta["original_size"], [2000, 1600])
        self.assertLessEqual(max(proc_img.size), 1280)

        # Smaller image not downscaled
        small_arr = np.zeros((500, 600, 3), dtype=np.uint8)
        proc_img2, meta2 = tool._preprocess_image(small_arr, max_dim=1280)
        self.assertFalse(meta2["downscaled"])
        self.assertEqual(meta2["original_size"], [600, 500])
        self.assertEqual(meta2["processed_size"], [600, 500])

    def test_08_configurable_token_budget_env(self):
        tool = AdaptLLMTool()
        with patch.dict(os.environ, {
            "SATQUERY_ADAPTLLM_MIN_PIXELS": "100000",
            "SATQUERY_ADAPTLLM_MAX_PIXELS": "500000",
            "SATQUERY_ADAPTLLM_MAX_NEW_TOKENS": "64",
            "SATQUERY_ADAPTLLM_MAX_IMAGE_SIZE": "1000",
        }):
            cfg = tool.get_token_budget_config()
            self.assertEqual(cfg["min_pixels"], 100000)
            self.assertEqual(cfg["max_pixels"], 500000)
            self.assertEqual(cfg["max_new_tokens"], 64)
            self.assertEqual(cfg["max_image_size"], 1000)

    # ── 7. Planner Routing Hierarchy ─────────────────────────────────────────
    def test_09_planner_routes_to_adaptllm_when_available(self):
        with patch("backend.tools.adaptllm.is_adaptllm_available", return_value=True):
            open_queries = [
                "Are there solar panels on the roofs?",
                "Can you see any airplanes parked on the apron?",
                "What unusual structures are visible in this scene?",
                "What kind of infrastructure appears in this image?"
            ]
            for q in open_queries:
                with self.subTest(query=q):
                    plan = create_query_plan(q)
                    self.assertEqual(plan.specialists, ["rs_adaptllm"])
                    self.assertEqual(plan.execution_order, ["rs_adaptllm"])
                    self.assertEqual(plan.intent, "general_vqa")

    def test_10_planner_routes_to_generalist_when_adaptllm_unavailable(self):
        with patch("backend.tools.adaptllm.is_adaptllm_available", return_value=False):
            plan = create_query_plan("Are there solar panels on the roofs?")
            self.assertEqual(plan.specialists, ["rs_generalist"])
            self.assertEqual(plan.execution_order, ["rs_generalist"])

    def test_11_planner_preserves_verified_specialists(self):
        # Known tasks must NEVER be routed to rs_adaptllm
        with patch("backend.tools.adaptllm.is_adaptllm_available", return_value=True):
            plan_bldg = create_query_plan("Are there buildings?")
            self.assertIn("building_detection", plan_bldg.specialists)
            self.assertNotIn("rs_adaptllm", plan_bldg.specialists)

            plan_lc = create_query_plan("Classify the land cover in this scene.")
            self.assertIn("land_cover", plan_lc.specialists)
            self.assertNotIn("rs_adaptllm", plan_lc.specialists)

            plan_cap = create_query_plan("Describe this image in detail.")
            self.assertIn("rs_caption_adapted", plan_cap.specialists)
            self.assertNotIn("rs_adaptllm", plan_cap.specialists)

            plan_vqa = create_query_plan("Is there vegetation present?")
            self.assertIn("rs_vqa_adapted", plan_vqa.specialists)
            self.assertNotIn("rs_adaptllm", plan_vqa.specialists)

            plan_chg = create_query_plan("Compare these two images for change.", image_count=2)
            self.assertIn("change_detection", plan_chg.specialists)
            self.assertNotIn("rs_adaptllm", plan_chg.specialists)

    # ── 8. Lifecycle Unload ───────────────────────────────────────────────────
    def test_12_lifecycle_unload(self):
        tool = AdaptLLMTool()
        tool._model = MagicMock()
        tool._processor = MagicMock()
        tool.unload_model()
        self.assertIsNone(tool._model)
        self.assertIsNone(tool._processor)

    # ── 9. Mocked Inference Execution ────────────────────────────────────────
    def test_13_mocked_inference_execution(self):
        tool = AdaptLLMTool()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_model = MagicMock()
        mock_processor = MagicMock()

        # Mock tokenizer output
        mock_inputs = MagicMock()
        mock_inputs.input_ids = [[1, 2, 3]]
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs
        mock_processor.apply_chat_template.return_value = "<mock_prompt>"
        mock_processor.batch_decode.return_value = ["Several airplanes are visible parked along the terminal."]

        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]

        with patch.object(AdaptLLMTool, "is_available", True), \
             patch.object(tool, "_load_model", return_value=(mock_model, mock_processor)):
            res = tool.run({"query": "Are there airplanes?", "image": dummy_img})

            self.assertEqual(res["status"], "success")
            self.assertEqual(res["answer"], "Several airplanes are visible parked along the terminal.")
            self.assertEqual(res["model"], "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct")
            self.assertIsNone(res["confidence"])
            self.assertEqual(res["confidence_status"], "not_calibrated")
            self.assertEqual(res["evidence_source"], "adaptllm_vlm")
            self.assertGreater(res["inference_time_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
