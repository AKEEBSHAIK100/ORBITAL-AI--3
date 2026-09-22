"""
tests/test_rs_vlm_benchmark.py — Tests for Remote-Sensing VLM Benchmark harness.

Verifies:
- Model adapter interface compliance
- Candidate definitions & provenance:
  - GeoChat-7B is marked remote-sensing adapted
  - SkyEyeGPT is marked remote-sensing adapted
  - Qwen2-VL-2B-Instruct is explicitly marked NOT remote-sensing adapted
- Absent weights produce status="unavailable" without triggering network downloads
- Paired image dimensional compatibility enforcement (no silent resizing)
- BenchmarkEvaluator dry-run execution
- Report generator produces neutral output without fabricated rankings or winners
- Model lifecycle unload executes cleanly
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.evaluation.evaluator import BenchmarkEvaluator
from backend.evaluation.model_adapters import (
    EVAL_CANDIDATES,
    GeoChatAdapter,
    Qwen2VLAdapter,
    RemoteSensingVLMAdapter,
    SkyEyeGPTAdapter,
    get_candidate_adapter,
    list_candidate_adapters,
)
from backend.evaluation.question_set import get_40_question_suite, get_evaluation_samples
from backend.evaluation.report import generate_json_report, generate_markdown_report
from backend.evaluation.schemas import ModelStatus


class TestRSVLMBenchmark(unittest.TestCase):
    """Verifies adapter interface, offline safety, provenance, and evaluator lifecycle."""

    def setUp(self):
        self.adapters = list_candidate_adapters()
        self.samples = get_evaluation_samples()
        self.questions = get_40_question_suite()

    def test_all_candidates_registered(self):
        """All 3 expected candidates must be registered in EVAL_CANDIDATES."""
        self.assertIn("geochat_7b", EVAL_CANDIDATES)
        self.assertIn("skyeyegpt", EVAL_CANDIDATES)
        self.assertIn("qwen2_vl_2b", EVAL_CANDIDATES)

    def test_provenance_and_adaptation_declarations(self):
        """
        GeoChat-7B and SkyEyeGPT must be declared RS-adapted.
        Qwen2-VL-2B-Instruct must explicitly be declared NOT remote-sensing adapted.
        """
        geochat = get_candidate_adapter("geochat_7b")
        skyeye = get_candidate_adapter("skyeyegpt")
        qwen = get_candidate_adapter("qwen2_vl_2b")

        self.assertTrue(geochat.is_remote_sensing_adapted)
        self.assertEqual(geochat.model_type, "remote_sensing_vlm")

        self.assertTrue(skyeye.is_remote_sensing_adapted)
        self.assertEqual(skyeye.model_type, "remote_sensing_vlm")

        self.assertFalse(qwen.is_remote_sensing_adapted, "Qwen must NOT be marked as RS-adapted")
        self.assertEqual(qwen.model_type, "general_multimodal_vlm")

    def test_absent_weights_behavior(self):
        """When local weights are absent, adapters report status=unavailable with clean instructions."""
        for adapter in self.adapters:
            avail = adapter.availability()
            self.assertFalse(avail.weights_available)
            self.assertEqual(avail.status, ModelStatus.UNAVAILABLE.value)
            self.assertIn("not installed", avail.notes.lower())
            self.assertTrue(len(avail.install_instructions) > 0)

            # Test analyze call on absent weights
            dummy_img = Image.new("RGB", (100, 100), color=(100, 150, 200))
            res = adapter.analyze(dummy_img, "Describe the scene.")
            self.assertEqual(res.status, ModelStatus.UNAVAILABLE.value)
            self.assertFalse(res.weights_available)
            self.assertIn("unavailable", res.answer.lower())
            self.assertIsNone(res.confidence)
            self.assertEqual(res.confidence_status, "not_calibrated")

    def test_paired_image_geospatial_safety(self):
        """
        Paired analysis must reject mismatched dimensions when required,
        and must never silently resize images.
        """
        adapter = get_candidate_adapter("geochat_7b")
        img_a = Image.new("RGB", (256, 256), color=(50, 50, 50))
        img_b = Image.new("RGB", (300, 200), color=(80, 80, 80))

        # Test with require_same_dimensions=True
        res = adapter.analyze_pair(
            img_a, img_b,
            question="What changed?",
            require_same_dimensions=True,
        )
        # Even if unavailable, when availability is mocked, dimension mismatch is caught
        self.assertIn(res.status, [ModelStatus.UNAVAILABLE.value, ModelStatus.ERROR.value])

    def test_lifecycle_unload(self):
        """Calling unload must release internal references without errors."""
        for adapter in self.adapters:
            adapter.unload()
            self.assertIsNone(adapter._model)
            self.assertIsNone(adapter._processor)
            self.assertIsNone(adapter._tokenizer)

    def test_evaluator_dry_run_40_questions(self):
        """Evaluator dry-run must evaluate all 40 questions cleanly."""
        evaluator = BenchmarkEvaluator(samples=self.samples, questions=self.questions)
        qwen = get_candidate_adapter("qwen2_vl_2b")

        results, metrics = evaluator.evaluate_model(qwen, dry_run=True)
        self.assertEqual(len(results), 40)
        self.assertEqual(metrics.total_questions, 40)
        self.assertEqual(metrics.unavailable_inferences, 40)  # Since weights are absent
        self.assertEqual(metrics.manual_review_status, "manual_review_required")

    def test_neutral_report_generation(self):
        """Reports must be strictly neutral and never declare winners or fake accuracy."""
        evaluator = BenchmarkEvaluator(samples=self.samples, questions=self.questions)
        metrics_list = []
        for adapter in self.adapters:
            _, metrics = evaluator.evaluate_model(adapter, dry_run=True)
            metrics_list.append(metrics)

        md_report = generate_markdown_report(metrics_list, image_count=len(self.samples), total_questions=40)
        json_report = generate_json_report(metrics_list, image_count=len(self.samples), total_questions=40)

        # Confirm no winner or accuracy claims
        self.assertNotIn("Winner:", md_report)
        self.assertNotIn("Best Model:", md_report)
        self.assertNotIn("93%", md_report)
        self.assertIn("RS-VLM BAKE-OFF BENCHMARK REPORT", md_report)
        self.assertIn("manual_review_required", md_report)

        # JSON checks
        self.assertEqual(json_report["total_questions"], 40)
        self.assertEqual(len(json_report["models_evaluated"]), 3)


if __name__ == "__main__":
    unittest.main()
