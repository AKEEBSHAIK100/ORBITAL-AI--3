"""
tests/test_rs_vlm_schemas.py — Tests for evaluation schemas and contracts.

Verifies:
- Normalized EvaluationResult schema fields
- Confidence default is None with confidence_status == "not_calibrated"
- No fabricated confidence scores
- Memory handling (null if not measurable)
- ModelAvailability and ModelMetadata schema completeness
- HumanReviewGrading default review_status == "manual_review_required"
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.evaluation.schemas import (
    ConfidenceStatus,
    EvaluationQuestion,
    EvaluationResult,
    HumanReviewGrading,
    ImageSample,
    ModelAvailability,
    ModelBenchmarkMetrics,
    ModelMetadata,
    ModelStatus,
)


class TestRSVLMSchemas(unittest.TestCase):
    """Verifies schema contracts and non-fabrication constraints."""

    def test_evaluation_result_normalized_fields(self):
        """EvaluationResult must contain all normalized schema fields."""
        res = EvaluationResult(
            question_id="RS-SCN-01",
            category="scene_description",
            model_id="geochat_7b",
            model_name="GeoChat-7B",
            image_id="sample_optical_aerial",
            status=ModelStatus.UNAVAILABLE.value,
            answer="Model weights are not installed locally.",
        )
        d = res.to_dict()
        required_keys = {
            "question_id", "category", "model_id", "model_name", "image_id",
            "status", "answer", "inference_time_ms", "device", "memory_mb",
            "confidence", "confidence_status", "error", "warnings",
            "is_remote_sensing_adapted", "model_type", "source",
            "weights_available", "ground_truth_available", "review_grading"
        }
        for k in required_keys:
            self.assertIn(k, d, f"Missing normalized field '{k}' in EvaluationResult")

    def test_no_fabricated_confidence_defaults(self):
        """Confidence must default to None and status to 'not_calibrated'."""
        res = EvaluationResult(
            question_id="RS-001",
            category="land_cover",
            model_id="qwen2_vl_2b",
            model_name="Qwen2-VL-2B-Instruct",
            image_id="sample_img",
            status=ModelStatus.SUCCESS.value,
            answer="Test output",
        )
        self.assertIsNone(res.confidence, "Confidence must be None unless calibrated")
        self.assertEqual(res.confidence_status, ConfidenceStatus.NOT_CALIBRATED.value)

    def test_null_memory_handling(self):
        """Memory must be None if not measurable, never an invented float."""
        res = EvaluationResult(
            question_id="RS-002",
            category="buildings",
            model_id="skyeyegpt",
            model_name="SkyEyeGPT",
            image_id="sample_img",
            status=ModelStatus.UNAVAILABLE.value,
            answer="Unavailable",
        )
        self.assertIsNone(res.memory_mb)

    def test_human_review_grading_defaults(self):
        """Human review grading must default to 'manual_review_required'."""
        grading = HumanReviewGrading()
        d = grading.to_dict()
        self.assertEqual(d["review_status"], "manual_review_required")
        self.assertIsNone(d["correctness"])
        self.assertFalse(d["hallucination_observed"])
        self.assertFalse(d["unsupported_claim"])

    def test_model_availability_schema(self):
        """ModelAvailability schema must serialize all fields correctly."""
        avail = ModelAvailability(
            model_id="geochat_7b",
            model_name="GeoChat-7B",
            status="unavailable",
            weights_available=False,
            expected_path="/models/geochat",
            install_instructions="Download checkpoint",
            dependencies_available=True,
        )
        d = avail.to_dict()
        self.assertFalse(d["weights_available"])
        self.assertEqual(d["status"], "unavailable")
        self.assertIn("install_instructions", d)


if __name__ == "__main__":
    unittest.main()
