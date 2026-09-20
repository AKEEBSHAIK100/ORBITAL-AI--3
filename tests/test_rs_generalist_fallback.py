"""
tests/test_rs_generalist_fallback.py — Test suite for Phase 1 Remote-Sensing Generalist Fallback Architecture.

Verifies:
1. Known land-cover query -> land_cover
2. Known VQA query -> rs_vqa_adapted
3. Known caption query -> rs_caption_adapted
4. Temporal query -> change workflow
5. Optical-SAR query -> optical_sar_fusion
6. Unknown remote-sensing visual question -> rs_generalist candidate ("Generalist fallback required")
7. Truly unsupported request -> explicit unsupported state ("No supported visual capability")
8. ModelRegistry state: rs_generalist registered, unavailable by default, no automatic downloads
9. Generalist contract: exposes status, answer, model, base_model, adapter, device,
   inference_time_ms, confidence (None), confidence_status ("not_calibrated"), provenance, warnings
10. Model provenance rules: model_type = "general_multimodal_vlm", no claims of remote-sensing adaptation
11. End-to-end orchestrator execution with unavailable generalist (clean response, no fabricated metrics)
12. End-to-end orchestrator execution with mocked generalist (qualitative preservation, uncalibrated)
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.planner import (
    create_query_plan,
    get_planner_disposition,
    is_query_unsupported,
    DISPOSITION_KNOWN_SPECIALIST,
    DISPOSITION_GENERALIST_FALLBACK,
    DISPOSITION_NO_CAPABILITY,
)
from backend.agents.orchestrator import run_orbital_analysis
from backend.tools.generalist import GeneralistTool
from backend.tools.registry import get_tool, list_all_tools
from models.registry import ModelRegistry


class TestRSGeneralistFallbackArchitecture(unittest.TestCase):

    def setUp(self):
        self.registry = ModelRegistry.get_instance()
        self.opt_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        self.opt_img2 = np.full((100, 100, 3), 140, dtype=np.uint8)

    # ── Test 1: Known land-cover query → land_cover ──────────────────────────
    def test_01_known_land_cover_routes_to_specialist(self):
        plan = create_query_plan("What type of land is present?")
        self.assertEqual(plan.intent, "land_cover")
        self.assertIn("land_cover", plan.specialists)
        self.assertNotIn("rs_generalist", plan.specialists)
        self.assertEqual(plan.planner_disposition, DISPOSITION_KNOWN_SPECIALIST)
        self.assertEqual(get_planner_disposition(plan), "Known specialist available")
        self.assertEqual(plan.task_category, "KNOWN_TASK")

    # ── Test 2: Known VQA query → rs_vqa_adapted ─────────────────────────────
    def test_02_known_vqa_routes_to_specialist(self):
        for q in ["Is there vegetation?", "Is there water?", "Are there buildings?"]:
            with self.subTest(query=q):
                plan = create_query_plan(q)
                self.assertEqual(plan.intent, "vqa")
                self.assertIn("rs_vqa_adapted", plan.specialists)
                self.assertNotIn("rs_generalist", plan.specialists)
                self.assertEqual(plan.planner_disposition, DISPOSITION_KNOWN_SPECIALIST)
                self.assertEqual(get_planner_disposition(plan), "Known specialist available")

    # ── Test 3: Known caption query → rs_caption_adapted ─────────────────────
    def test_03_known_caption_routes_to_specialist(self):
        plan = create_query_plan("Describe this image.")
        self.assertEqual(plan.intent, "caption")
        self.assertEqual(plan.specialists, ["rs_caption_adapted"])
        self.assertNotIn("rs_generalist", plan.specialists)
        self.assertEqual(plan.planner_disposition, DISPOSITION_KNOWN_SPECIALIST)
        self.assertEqual(get_planner_disposition(plan), "Known specialist available")
        self.assertEqual(plan.task_category, "KNOWN_TASK")

    # ── Test 4: Temporal query → change workflow ─────────────────────────────
    def test_04_temporal_query_routes_to_change_workflow(self):
        plan = create_query_plan("What changed between these images?", image_count=2)
        self.assertEqual(plan.intent, "change_vqa")
        self.assertIn("change_detection", plan.specialists)
        self.assertNotIn("rs_generalist", plan.specialists)
        self.assertEqual(plan.planner_disposition, DISPOSITION_KNOWN_SPECIALIST)
        self.assertEqual(get_planner_disposition(plan), "Known specialist available")

    # ── Test 5: Optical-SAR query → optical_sar_fusion ───────────────────────
    def test_05_optical_sar_query_routes_to_fusion(self):
        plan = create_query_plan("Compare optical and SAR.")
        self.assertEqual(plan.intent, "optical_sar_analysis")
        self.assertEqual(plan.specialists, ["optical_sar_fusion"])
        self.assertNotIn("rs_generalist", plan.specialists)
        self.assertEqual(plan.planner_disposition, DISPOSITION_KNOWN_SPECIALIST)
        self.assertEqual(get_planner_disposition(plan), "Known specialist available")
        self.assertEqual(plan.task_category, "KNOWN_TASK")

    # ── Test 6: Unknown remote-sensing visual question → rs_generalist candidate
    def test_06_unknown_visual_question_routes_to_generalist(self):
        open_queries = [
            "Are there solar panels on the roofs?",
            "Can you see any airplanes parked on the apron?",
            "What color are the shipping containers?",
            "Identify any swimming pools in the residential area.",
            "Can you spot any oil storage tanks?",
            "Is there any construction equipment visible near the dock?"
        ]
        for q in open_queries:
            with self.subTest(query=q):
                plan = create_query_plan(q)
                self.assertEqual(plan.intent, "general_vqa")
                self.assertEqual(plan.specialists, ["rs_generalist"])
                self.assertEqual(plan.execution_order, ["rs_generalist"])
                self.assertEqual(plan.planner_disposition, DISPOSITION_GENERALIST_FALLBACK)
                self.assertEqual(get_planner_disposition(plan), "Generalist fallback required")
                self.assertEqual(plan.task_category, "OPEN_REMOTE_SENSING_QUESTION")

    # ── Test 7: Truly unsupported request → explicit unsupported state ────────
    def test_07_unsupported_request_routes_to_unsupported_state(self):
        unsupported_queries = [
            "What will the population be next year?",
            "Who lives in this neighborhood?",
            "What is the weather forecast tomorrow?",
            "Predict the stock market trend for next month."
        ]
        for q in unsupported_queries:
            with self.subTest(query=q):
                self.assertTrue(is_query_unsupported(q))
                plan = create_query_plan(q)
                self.assertEqual(plan.intent, "unsupported")
                self.assertEqual(plan.specialists, [])
                self.assertEqual(plan.planner_disposition, DISPOSITION_NO_CAPABILITY)
                self.assertEqual(get_planner_disposition(plan), "No supported visual capability")
                self.assertEqual(plan.task_category, "UNSUPPORTED")

    # ── Test 8: Model Registry State for rs_generalist ───────────────────────
    def test_08_registry_state_rs_generalist(self):
        spec = self.registry.get_specialist("rs_generalist")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.id, "rs_generalist")
        self.assertEqual(spec.model_id, "Qwen/Qwen2-VL-2B-Instruct")
        self.assertEqual(spec.task, "general_vqa")
        # Unless local weights were explicitly downloaded, marked unavailable
        self.assertFalse(spec.is_available)
        self.assertIsNotNone(spec.unavailable_reason)
        self.assertIn("Automatic downloads are disabled", spec.unavailable_reason)
        self.assertIsNone(spec.checkpoint_location)

    # ── Test 9: Tool Registry & Generalist Contract ──────────────────────────
    def test_09_generalist_tool_contract(self):
        tool = get_tool("rs_generalist")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.id, "rs_generalist")
        self.assertEqual(tool.base_model, "Qwen/Qwen2-VL-2B-Instruct")
        self.assertEqual(tool.model_type, "general_multimodal_vlm")
        self.assertIsNone(tool.adapter)
        self.assertFalse(tool.is_available)

        # Execute unavailable contract
        res = tool.run({"query": "Are there solar panels on the roof?"})
        # Check all required exposed fields
        required_keys = [
            "status", "answer", "model", "base_model", "adapter", "device",
            "inference_time_ms", "confidence", "confidence_status", "provenance", "warnings"
        ]
        for k in required_keys:
            self.assertIn(k, res)

        # Truthful confidence: uncalibrated null, no fabrication
        self.assertIsNone(res["confidence"])
        self.assertEqual(res["confidence_status"], "not_calibrated")
        self.assertEqual(res["status"], "specialist_unavailable")

    # ── Test 10: Model Provenance Rules ──────────────────────────────────────
    def test_10_provenance_rules_no_false_claims(self):
        tool = GeneralistTool()
        prov = tool.provenance

        self.assertEqual(prov["model"], "Qwen/Qwen2-VL-2B-Instruct")
        self.assertEqual(prov["base_model"], "Qwen/Qwen2-VL-2B-Instruct")
        self.assertEqual(prov["model_type"], "general_multimodal_vlm")
        self.assertFalse(prov["is_remote_sensing_adapted"])
        self.assertFalse(prov["is_remote_sensing_expert"])
        self.assertIsNone(prov["benchmark_accuracy_claim"])
        self.assertIsNone(prov["adapter"])

        # Must NOT claim remote-sensing adapted or expert in description or domain_adaptation
        self.assertNotIn("remote-sensing expert", tool.domain_adaptation.lower())
        self.assertIn("not remote-sensing adapted", tool.domain_adaptation.lower())

    # ── Test 11: End-to-End Orchestrator with Unavailable Generalist ─────────
    def test_11_orchestrator_unavailable_generalist_workflow(self):
        query = "Are there solar panels on the roofs?"
        resp = run_orbital_analysis(
            query=query,
            images=[self.opt_img],
            modalities=["optical"]
        )

        self.assertEqual(resp.status, "SPECIALIST_UNAVAILABLE")
        self.assertFalse(resp.success)
        self.assertIsNone(resp.confidence)
        self.assertEqual(resp.confidence_status, "unavailable")
        self.assertIn("SPECIALIST UNAVAILABLE", resp.answer)
        self.assertIn("Qwen/Qwen2-VL-2B-Instruct", resp.answer)
        self.assertIn("No answer has been fabricated", resp.answer)

    # ── Test 12: End-to-End Orchestrator with Mocked Generalist Output ───────
    def test_12_orchestrator_mocked_generalist_preserves_uncalibrated(self):
        mock_output = {
            "status": "success",
            "answer": "Multiple dark rectangular solar arrays are visible on southern rooftop facets.",
            "model": "Qwen/Qwen2-VL-2B-Instruct",
            "base_model": "Qwen/Qwen2-VL-2B-Instruct",
            "adapter": None,
            "model_type": "general_multimodal_vlm",
            "device": "cpu",
            "inference_time_ms": 120.0,
            "confidence": None,
            "confidence_status": "not_calibrated",
            "provenance": GeneralistTool.provenance,
            "warnings": ["Qualitative assessment; uncalibrated."]
        }

        mock_spec_entry = MagicMock()
        mock_spec_entry.id = "rs_generalist"
        mock_spec_entry.name = "Remote-Sensing Generalist Multimodal Fallback Specialist"
        mock_spec_entry.task = "general_vqa"
        mock_spec_entry.model_id = "Qwen/Qwen2-VL-2B-Instruct"
        mock_spec_entry.is_available = True
        mock_spec_entry.unavailable_reason = None

        mock_tool = MagicMock()
        mock_tool.name = "Remote-Sensing Generalist Multimodal Fallback Specialist"
        mock_tool.model_id = "Qwen/Qwen2-VL-2B-Instruct"
        mock_tool.supported_tasks = ["general_vqa"]
        mock_tool.adapter = None
        mock_tool.is_available = True
        mock_tool.provenance = GeneralistTool.provenance
        mock_tool.permitted_parameters = {}
        mock_tool.run.return_value = mock_output

        with patch.object(ModelRegistry, "get_instance") as mock_get_reg, \
             patch("backend.agents.orchestrator.get_tool", return_value=mock_tool):
            mock_reg_inst = MagicMock()
            mock_reg_inst.get_specialist.return_value = mock_spec_entry
            mock_get_reg.return_value = mock_reg_inst

            resp = run_orbital_analysis(
                query="Are there solar panels on the roofs?",
                images=[self.opt_img],
                modalities=["optical"]
            )

            self.assertEqual(resp.status, "SUCCESS")
            # Confidence must remain uncalibrated / None
            self.assertIsNone(resp.confidence)
            self.assertEqual(resp.confidence_status, "not_calibrated")
            # Answer preserves qualitative answer and acknowledges source model
            self.assertIn("solar arrays", resp.answer)
            self.assertIn("Qwen/Qwen2-VL-2B-Instruct", resp.answer)
            self.assertIn("uncalibrated", resp.answer.lower())

    # ── Test 13: Lazy Loading — No model loading on tool initialization ───────
    def test_13_no_eager_model_loading_on_init(self):
        tool = GeneralistTool()
        self.assertIsNone(tool._model)
        self.assertIsNone(tool._processor)

    # ── Test 14: No startup Qwen loading when accessing registry ─────────────
    def test_14_no_qwen_loading_via_registry(self):
        tool = get_tool("rs_generalist")
        self.assertIsNotNone(tool)
        self.assertIsNone(tool._model)
        self.assertIsNone(tool._processor)

    # ── Test 15: Successful local model loading with local_files_only ─────────
    def test_15_local_model_loading_when_weights_exist(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create dummy safetensors file to simulate local weights
            (tmp_path / "model.safetensors").write_bytes(b"dummy_weights")

            tool = GeneralistTool(weights_dir=tmp_path)
            self.assertTrue(tool.is_available)
            self.assertIsNone(tool.unavailable_reason)

            mock_model = MagicMock()
            mock_processor = MagicMock()
            with patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained", return_value=mock_model) as mock_m_from_p, \
                 patch("transformers.AutoProcessor.from_pretrained", return_value=mock_processor) as mock_p_from_p:

                m, p = tool._load_model()
                self.assertIs(m, mock_model)
                self.assertIs(p, mock_processor)
                # Verify local_files_only=True was passed to prevent internet downloads
                mock_m_from_p.assert_called_once()
                self.assertTrue(mock_m_from_p.call_args[1].get("local_files_only"))
                mock_p_from_p.assert_called_once()
                self.assertTrue(mock_p_from_p.call_args[1].get("local_files_only"))

    # ── Test 16: Repeated inference reuses loaded model without reloading ────
    def test_16_repeated_inference_reuses_loaded_model(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "model.safetensors").write_bytes(b"dummy_weights")

            tool = GeneralistTool(weights_dir=tmp_path)
            mock_model = MagicMock()
            mock_processor = MagicMock()
            mock_processor.apply_chat_template.return_value = "prompt"
            mock_inputs = MagicMock()
            mock_inputs.to.return_value = mock_inputs
            mock_inputs.input_ids = [[1, 2, 3]]
            mock_processor.return_value = mock_inputs
            mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
            mock_processor.batch_decode.return_value = ["Observation of objects."]

            with patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained", return_value=mock_model) as mock_m_from_p, \
                 patch("transformers.AutoProcessor.from_pretrained", return_value=mock_processor) as mock_p_from_p:

                res1 = tool.run({"image": self.opt_img, "query": "Query 1"})
                self.assertEqual(res1["status"], "success")
                self.assertEqual(res1["answer"], "Observation of objects.")

                res2 = tool.run({"image": self.opt_img2, "query": "Query 2"})
                self.assertEqual(res2["status"], "success")

                # The loaders should have been called only once across both inferences
                self.assertEqual(mock_m_from_p.call_count, 1)
                self.assertEqual(mock_p_from_p.call_count, 1)

    # ── Test 17: Explicit unload_model frees memory and supports reloading ───
    def test_17_explicit_unload_model_and_rearm(self):
        tool = GeneralistTool()
        tool._model = MagicMock()
        tool._processor = MagicMock()

        tool.unload_model()
        self.assertIsNone(tool._model)
        self.assertIsNone(tool._processor)

    # ── Test 18: Result caching integration for identical queries ────────────
    def test_18_result_caching_integration(self):
        import tempfile
        from backend.services.cache_manager import get_query_result_cache
        get_query_result_cache().clear()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "model.safetensors").write_bytes(b"dummy_weights")

            tool = GeneralistTool(weights_dir=tmp_path)
            mock_model = MagicMock()
            mock_processor = MagicMock()
            mock_processor.apply_chat_template.return_value = "prompt"
            mock_inputs = MagicMock()
            mock_inputs.to.return_value = mock_inputs
            mock_inputs.input_ids = [[1, 2]]
            mock_processor.return_value = mock_inputs
            mock_model.generate.return_value = [[1, 2, 3]]
            mock_processor.batch_decode.return_value = ["Cached observation."]

            with patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained", return_value=mock_model), \
                 patch("transformers.AutoProcessor.from_pretrained", return_value=mock_processor):

                # First run: cache miss
                res1 = tool.run({"image": self.opt_img, "query": "Find cranes"})
                self.assertEqual(res1["status"], "success")
                self.assertFalse(res1["evidence"].get("cached", False))

                # Second run: identical query + image -> cache hit
                res2 = tool.run({"image": self.opt_img, "query": "Find cranes"})
                self.assertEqual(res2["status"], "success")
                self.assertTrue(res2["evidence"].get("cached", False))
                self.assertEqual(res2["answer"], "Cached observation.")


if __name__ == "__main__":
    unittest.main()
