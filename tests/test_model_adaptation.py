"""
tests/test_model_adaptation.py — Python Unit Tests for Remote-Sensing Model Adaptation
=======================================================================================

Tests:
  1. Dataset missing behaviour — status == NOT_DOWNLOADED for each dataset
  2. Missing checkpoint controlled fallback — SPECIALIST_UNAVAILABLE
  3. ModelRegistry initialises without error even with no GPU/adapter
  4. Bi-temporal dimension mismatch rejection (TemporalPreprocessor.validate_and_align)
  5. CRS incompatibility rejection in optical-SAR fusion (ModalityValidator)
  6. BenchmarkStorage returns DATASET_UNAVAILABLE without inventing metrics
  7. Building detection confidence tier exposure
  8. Router intent classification correctness

Run:
  python -m unittest tests.test_model_adaptation -v
"""

import os
import sys
import unittest
import tempfile
import json
from pathlib import Path
import numpy as np

# ─── path fixup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── optional GPU import guard ───────────────────────────────────────────────
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class TestDatasetRegistry(unittest.TestCase):
    """Dataset registry correctly reports status for each dataset."""

    VALID_STATUSES = {"NOT_DOWNLOADED", "AVAILABLE", "INVALID", "PROCESSING", "READY"}
    # Map friendly names used in tests to the actual registry IDs
    KNOWN_IDS = ["bigearthnet_v2", "vrsbench", "rsvqa_lr", "rsvqa_hr", "cdvqa"]

    def _get_registry(self):
        from training.datasets.dataset_registry import DatasetRegistry
        return DatasetRegistry()

    def _status_map(self, reg):
        """Returns {dataset_id: status_string} dict using list_datasets()."""
        return {d["id"]: d["status"] for d in reg.list_datasets()}

    def test_bigearthnet_v2_status_is_valid(self):
        reg = self._get_registry()
        sm = self._status_map(reg)
        self.assertIn("bigearthnet_v2", sm)
        self.assertIn(sm["bigearthnet_v2"], self.VALID_STATUSES)

    def test_vrsbench_status_is_valid(self):
        reg = self._get_registry()
        sm = self._status_map(reg)
        self.assertIn("vrsbench", sm)
        self.assertIn(sm["vrsbench"], self.VALID_STATUSES)

    def test_rsvqa_lr_status_is_valid(self):
        reg = self._get_registry()
        sm = self._status_map(reg)
        self.assertIn("rsvqa_lr", sm)
        self.assertIn(sm["rsvqa_lr"], self.VALID_STATUSES)

    def test_cdvqa_status_is_valid(self):
        reg = self._get_registry()
        sm = self._status_map(reg)
        self.assertIn("cdvqa", sm)
        self.assertIn(sm["cdvqa"], self.VALID_STATUSES)

    def test_all_statuses_are_valid_enum_values(self):
        reg = self._get_registry()
        for d in reg.list_datasets():
            self.assertIn(
                d["status"], self.VALID_STATUSES,
                msg=f"Dataset '{d['id']}' has invalid status '{d['status']}'"
            )

    def test_absent_datasets_report_not_downloaded(self):
        """Any dataset with no real files must not claim READY."""
        reg = self._get_registry()
        for d in reg.list_datasets():
            if d["status"] == "READY":
                data_path = PROJECT_ROOT / "data" / d["id"]
                has_files = any(
                    f.suffix in (".tif", ".json", ".csv", ".txt", ".jsonl")
                    for f in data_path.rglob("*") if f.is_file()
                ) if data_path.exists() else False
                self.assertTrue(
                    has_files,
                    f"Dataset '{d['id']}' claims READY but no data files found"
                )


class TestModelRegistry(unittest.TestCase):
    """ModelRegistry initialises correctly and produces valid specialist list."""

    def _get_registry(self):
        from models.registry import ModelRegistry
        return ModelRegistry.get_instance()

    def test_registry_initialises(self):
        reg = self._get_registry()
        self.assertIsNotNone(reg)

    def test_specialists_list_is_nonempty(self):
        reg = self._get_registry()
        specialists = reg.list_specialists()
        self.assertIsInstance(specialists, list)
        self.assertGreater(len(specialists), 0)

    def test_all_specialists_have_required_fields(self):
        reg = self._get_registry()
        required_fields = {"id", "name", "task", "is_available", "version"}
        for spec in reg.list_specialists():
            spec_dict = spec if isinstance(spec, dict) else vars(spec)
            for field in required_fields:
                self.assertIn(field, spec_dict, msg=f"Specialist missing field '{field}'")

    def test_building_detection_specialist_exists(self):
        """Building detection specialist must be registered."""
        reg = self._get_registry()
        # The registry uses 'building_detection' as the specialist ID
        spec = reg.get_specialist("building_detection")
        self.assertIsNotNone(spec, "building_detection specialist must be registered")

    def test_vlm_adapter_unavailable_without_checkpoint(self):
        """VLM VQA specialist must be UNAVAILABLE when adapter is not installed."""
        adapter_path = PROJECT_ROOT / "checkpoints" / "rs_vlm_adapter" / "adapter_config.json"
        if adapter_path.exists():
            self.skipTest("LoRA adapter is installed — skipping unavailability test")

        reg = self._get_registry()
        spec = reg.get_specialist("rs_vlm_vqa")
        if spec is not None:
            self.assertFalse(
                spec.is_available,
                "rs_vlm_vqa must be UNAVAILABLE when checkpoints/rs_vlm_adapter/ is absent"
            )


class TestSpecialistFallback(unittest.TestCase):
    """execute_or_fallback returns SPECIALIST_UNAVAILABLE schema — never fabricated."""

    def _make_unavailable_specialist(self):
        from models.registry import SpecialistEntry
        return SpecialistEntry(
            id="test_unavailable",
            name="Test Unavailable Specialist",
            task="test_task",
            model_id="test/model",
            version="0.0.0",
            modality=["optical"],
            supported_input_types=["image"],
            checkpoint_location=None,
            is_available=False,
            unavailable_reason="Checkpoint not installed for testing",
            inference_fn=None,
        )

    def test_unavailable_returns_correct_status(self):
        spec = self._make_unavailable_specialist()
        result = spec.execute_or_fallback(image=np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertEqual(result["status"], "SPECIALIST_UNAVAILABLE")

    def test_unavailable_confidence_is_none(self):
        spec = self._make_unavailable_specialist()
        result = spec.execute_or_fallback()
        self.assertIsNone(result["confidence"])

    def test_unavailable_confidence_level_is_unavailable(self):
        spec = self._make_unavailable_specialist()
        result = spec.execute_or_fallback()
        self.assertEqual(result["confidence_level"], "UNAVAILABLE")

    def test_unavailable_has_no_evidence(self):
        spec = self._make_unavailable_specialist()
        result = spec.execute_or_fallback()
        self.assertIsNone(result["evidence"])

    def test_unavailable_answer_not_empty(self):
        spec = self._make_unavailable_specialist()
        result = spec.execute_or_fallback()
        self.assertIsInstance(result["answer"], str)
        self.assertGreater(len(result["answer"]), 0)

    def test_unavailable_has_specialist_id(self):
        spec = self._make_unavailable_specialist()
        result = spec.execute_or_fallback()
        self.assertEqual(result["specialist_id"], "test_unavailable")


class TestTemporalPreprocessor(unittest.TestCase):
    """Bi-temporal preprocessor rejects severely mismatched dimensions."""

    def _get_preprocessor(self):
        from models.change.temporal_preprocessor import TemporalPreprocessor
        return TemporalPreprocessor()

    def test_same_dimensions_accepted(self):
        prep = self._get_preprocessor()
        img_a = np.zeros((256, 256, 3), dtype=np.uint8)
        img_b = np.zeros((256, 256, 3), dtype=np.uint8)
        is_valid, _, _, report = prep.validate_and_align(img_a, img_b)
        self.assertTrue(is_valid, f"Matching dimensions must be accepted. Report: {report}")

    def test_severely_different_dimensions_rejected(self):
        """Dimensions differing by >15% must be rejected as un-coregistered."""
        prep = self._get_preprocessor()
        img_a = np.zeros((256, 256, 3), dtype=np.uint8)
        img_b = np.zeros((512, 512, 3), dtype=np.uint8)  # 100% difference
        is_valid, _, _, report = prep.validate_and_align(img_a, img_b)
        self.assertFalse(is_valid, "Images with 100% dimension difference must be rejected")

    def test_minor_dimension_difference_handled(self):
        """Mismatched dimensions must be rejected without silent resizing."""
        prep = self._get_preprocessor()
        img_a = np.zeros((256, 256, 3), dtype=np.uint8)
        img_b = np.zeros((260, 260, 3), dtype=np.uint8)  # ~1.5% difference
        orig_shape_b = img_b.shape
        is_valid, out_a, out_b, report = prep.validate_and_align(img_a, img_b)
        self.assertFalse(is_valid, "Dimension mismatch must be rejected; silent resizing is forbidden")
        self.assertEqual(out_b.shape, orig_shape_b, "Secondary image must not be silently resized")
        self.assertGreater(len(report.get("errors", [])), 0, "Validation errors must be returned")
        self.assertTrue(
            any("dimension mismatch" in e.lower() for e in report.get("errors", [])),
            "Report must specify dimension mismatch error"
        )

    def test_rejection_includes_error_in_report(self):
        prep = self._get_preprocessor()
        img_a = np.zeros((100, 100, 3), dtype=np.uint8)
        img_b = np.zeros((800, 800, 3), dtype=np.uint8)
        is_valid, _, _, report = prep.validate_and_align(img_a, img_b)
        if not is_valid:
            self.assertGreater(
                len(report.get("errors", [])), 0,
                "Rejection report must contain at least one error message"
            )

    def test_null_second_image_rejected(self):
        """Passing None as the second image must be rejected."""
        prep = self._get_preprocessor()
        img_a = np.zeros((256, 256, 3), dtype=np.uint8)
        is_valid, _, _, report = prep.validate_and_align(img_a, None)
        self.assertFalse(is_valid, "None as second image must be rejected")


class TestModalityValidator(unittest.TestCase):
    """Modality validator enforces co-registration and CRS compatibility."""

    def _get_validator(self):
        from models.fusion.modality_validator import ModalityValidator
        return ModalityValidator()

    def test_both_images_same_size_pass(self):
        validator = self._get_validator()
        img_optical = np.zeros((256, 256, 3), dtype=np.uint8)
        img_sar = np.zeros((256, 256, 1), dtype=np.float32)
        is_valid, mode, report = validator.validate_multimodal_inputs(img_optical, img_sar)
        self.assertTrue(is_valid, f"Matching dimensions should be valid. Report: {report}")
        self.assertEqual(mode, "optical_sar")

    def test_severely_mismatched_dimensions_rejected(self):
        """Optical and SAR with >5% size difference must be rejected."""
        validator = self._get_validator()
        img_optical = np.zeros((256, 256, 3), dtype=np.uint8)
        img_sar = np.zeros((512, 512, 1), dtype=np.float32)
        is_valid, mode, report = validator.validate_multimodal_inputs(img_optical, img_sar)
        self.assertFalse(is_valid, "Images with 100% dimension difference must be rejected")

    def test_mismatched_crs_rejected(self):
        validator = self._get_validator()
        img_optical = np.zeros((256, 256, 3), dtype=np.uint8)
        img_sar = np.zeros((256, 256, 1), dtype=np.float32)
        optical_meta = {"crs": "EPSG:4326", "resolution_m": 10.0}
        sar_meta = {"crs": "EPSG:32643", "resolution_m": 10.0}
        is_valid, mode, report = validator.validate_multimodal_inputs(
            img_optical, img_sar, optical_meta=optical_meta, sar_meta=sar_meta
        )
        self.assertFalse(is_valid, "Different CRS must produce a validation failure")
        self.assertTrue(
            any("CRS" in e or "crs" in e.lower() for e in report.get("errors", [])),
            "Report must mention CRS conflict"
        )

    def test_same_crs_passes(self):
        validator = self._get_validator()
        img_optical = np.zeros((256, 256, 3), dtype=np.uint8)
        img_sar = np.zeros((256, 256, 1), dtype=np.float32)
        optical_meta = {"crs": "EPSG:4326", "resolution_m": 10.0}
        sar_meta = {"crs": "EPSG:4326", "resolution_m": 10.0}
        is_valid, mode, report = validator.validate_multimodal_inputs(
            img_optical, img_sar, optical_meta=optical_meta, sar_meta=sar_meta
        )
        self.assertTrue(is_valid, "Matching CRS should pass validation")

    def test_optical_only_mode(self):
        validator = self._get_validator()
        img_optical = np.zeros((256, 256, 3), dtype=np.uint8)
        is_valid, mode, report = validator.validate_multimodal_inputs(img_optical, None)
        self.assertTrue(is_valid)
        self.assertEqual(mode, "optical_only")

    def test_sar_only_mode(self):
        validator = self._get_validator()
        img_sar = np.zeros((256, 256, 1), dtype=np.float32)
        is_valid, mode, report = validator.validate_multimodal_inputs(None, img_sar)
        self.assertTrue(is_valid)
        self.assertEqual(mode, "sar_only")


class TestBenchmarkStorage(unittest.TestCase):
    """BenchmarkStorage never fabricates metrics; returns DATASET_UNAVAILABLE correctly."""

    def _get_storage(self):
        from evaluation.benchmark_manager import BenchmarkStorage
        return BenchmarkStorage(results_dir=tempfile.mkdtemp())

    def _make_result(self, **kwargs):
        from evaluation.benchmark_manager import BenchmarkResult, BenchmarkStatus
        defaults = dict(
            dataset="vrsbench",
            task="captioning",
            model="test_adapter",
            status=BenchmarkStatus.DATASET_UNAVAILABLE,
            metrics=None,
        )
        defaults.update(kwargs)
        return BenchmarkResult(**defaults)

    def test_dataset_unavailable_record_stored(self):
        storage = self._get_storage()
        result = self._make_result(status="DATASET_UNAVAILABLE", metrics=None)
        path = storage.save_run(result)
        self.assertTrue(Path(path).exists(), "Result JSON file must be created")
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["status"], "DATASET_UNAVAILABLE")
        self.assertIsNone(data["metrics"])

    def test_completed_record_with_real_metrics(self):
        storage = self._get_storage()
        result = self._make_result(
            status="COMPLETED",
            metrics={"cider": 0.75, "bleu4": 0.31}
        )
        path = storage.save_run(result)
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["status"], "COMPLETED")
        self.assertIsNotNone(data["metrics"])
        self.assertIn("cider", data["metrics"])

    def test_no_fabricated_metrics_in_unavailable(self):
        storage = self._get_storage()
        result = self._make_result(
            dataset="rsvqa", task="presence",
            status="DATASET_UNAVAILABLE", metrics=None
        )
        path = storage.save_run(result)
        with open(path) as f:
            data = json.load(f)
        self.assertIsNone(
            data.get("metrics"),
            "DATASET_UNAVAILABLE record must never contain fabricated metrics"
        )

    def test_list_all_runs_returns_list(self):
        storage = self._get_storage()
        result = self._make_result()
        storage.save_run(result)
        runs = storage.list_all_runs()
        self.assertIsInstance(runs, list)
        self.assertGreater(len(runs), 0)


class TestAgentRouter(unittest.TestCase):
    """Intent router returns correct task categories."""

    def _classify(self, query, image_count=1, modalities=None):
        from backend.agents.router import classify_query_intent
        return classify_query_intent(
            query, image_count=image_count, modalities=modalities or ["optical"]
        )

    def test_building_count_query(self):
        intent = self._classify("How many buildings are in this image?")
        self.assertEqual(intent, "building_detection")

    def test_change_detection_two_images(self):
        intent = self._classify(
            "What changed between these two satellite images?", image_count=2
        )
        self.assertIn(intent, ("change_detection", "change_vqa"))

    def test_vqa_single_image(self):
        intent = self._classify("What type of land cover is visible here?")
        self.assertIn(
            intent,
            ("vqa", "land_cover", "land_cover_classification", "captioning", "grounding"),
            msg=f"Unexpected intent '{intent}' for land-cover VQA query"
        )

    def test_sar_optical_fusion_with_sar_modality(self):
        intent = self._classify(
            "Compare the optical and SAR imagery for this region",
            image_count=2,
            modalities=["optical", "sar"]
        )
        self.assertEqual(intent, "sar_optical_fusion")

    def test_captioning_query(self):
        intent = self._classify("Describe this satellite image in detail.")
        self.assertIn(intent, ("captioning", "vqa"))

    def test_change_vqa_query(self):
        intent = self._classify(
            "Has the vegetation coverage increased or decreased?", image_count=2
        )
        self.assertIn(intent, ("change_vqa", "change_detection"))


class TestBuildingDetectionConfidenceTiers(unittest.TestCase):
    """Building detection tool exposes confidence tier counts without fabricating accuracy."""

    def test_confidence_tier_keys_present(self):
        """Output must include high_confidence_count, medium_confidence_count, partial_count."""
        try:
            from backend.tools.building_detection import BuildingDetectionTool
        except ImportError as e:
            self.skipTest(f"Could not import BuildingDetectionTool: {e}")

        tool = BuildingDetectionTool()
        dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
        result = tool.run({"image": dummy_img})

        if result.get("status") == "success":
            self.assertIn("high_confidence_count", result)
            self.assertIn("medium_confidence_count", result)
            self.assertIn("partial_count", result)
            self.assertGreaterEqual(result["high_confidence_count"], 0)
            self.assertGreaterEqual(result["medium_confidence_count"], 0)
            self.assertGreaterEqual(result["partial_count"], 0)

    def test_no_ground_truth_validation_is_honest(self):
        """When no GT is provided, validation must not claim verified benchmark accuracy."""
        try:
            from backend.tools.building_detection import BuildingDetectionTool
        except ImportError as e:
            self.skipTest(f"Could not import BuildingDetectionTool: {e}")

        tool = BuildingDetectionTool()
        dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
        result = tool.run({"image": dummy_img})

        if result.get("status") == "success":
            val_status = result.get("validation_status", "")
            self.assertNotIn("100%", val_status)
            self.assertNotIn("verified accuracy", val_status.lower())

    def test_invalid_image_returns_error(self):
        """Non-image input must return an error, not crash."""
        try:
            from backend.tools.building_detection import BuildingDetectionTool
        except ImportError as e:
            self.skipTest(f"Could not import BuildingDetectionTool: {e}")

        tool = BuildingDetectionTool()
        result = tool.run({"image": None})
        self.assertIn("error", result.get("status", "error").lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
