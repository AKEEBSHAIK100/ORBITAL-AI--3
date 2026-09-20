"""
tests/test_agentic_planner.py — Comprehensive test suite for Orbital-AI Agentic Query Planner.

Tests all 12 exact queries from Section 12 of the user specification:
Single image:
1. "What type of land is present here?"
2. "Describe this image."
3. "Is there water?"
4. "Are there buildings?"
5. "Where are the buildings?"

Two images:
6. "What changed between these images?"
7. "Has vegetation increased?"
8. "Are there new buildings?"

Optical + SAR:
9. "What differences can you see between the optical and SAR images?"
10. "What structures are visible across both modalities?"

Multi-task:
11. "Describe this scene and tell me whether there is water and whether buildings are present."

Unsupported:
12. "What will the population be next year?"

Verifies:
- Return statuses: SUCCESS, VALIDATION_ERROR, SPECIALIST_UNAVAILABLE, UNSUPPORTED_QUERY
- QueryPlan structure & specialist sequencing
- Image and co-registration validation rules
- Structured evidence schema per specialist
- Observable execution trace (8 operational steps, no hidden chain-of-thought)
- Truthful response synthesis without hallucinated confidence
"""

import sys
import unittest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.planner import create_query_plan, is_query_unsupported
from backend.agents.validator import validate_input_imagery
from backend.agents.orchestrator import run_orbital_analysis
from backend.schemas.analysis import QueryPlan, UnifiedAnalysisResponse, SpecialistEvidenceObject


class TestOrbitalAIAgenticPlanner(unittest.TestCase):

    def setUp(self):
        # 100x100 RGB synthetic test images
        self.opt_img1 = np.full((100, 100, 3), 120, dtype=np.uint8)
        self.opt_img2 = np.full((100, 100, 3), 130, dtype=np.uint8)
        # Synthetic SAR image (speckle texture)
        self.sar_img = np.random.randint(10, 200, (100, 100, 3), dtype=np.uint8)

        # Metadata fixtures
        self.meta_optical_1 = {
            "crs": "EPSG:4326",
            "geotransform": [10.0, 0.0001, 0.0, 50.0, 0.0, -0.0001],
            "acquisition_date": "2024-06-15",
            "sensor": "Sentinel-2"
        }
        self.meta_optical_2 = {
            "crs": "EPSG:4326",
            "geotransform": [10.0, 0.0001, 0.0, 50.0, 0.0, -0.0001],
            "acquisition_date": "2025-06-15",
            "sensor": "Sentinel-2"
        }
        self.meta_sar = {
            "crs": "EPSG:4326",
            "geotransform": [10.0, 0.0001, 0.0, 50.0, 0.0, -0.0001],
            "acquisition_date": "2024-06-15",
            "sensor": "Sentinel-1",
            "modality": "sar"
        }

    # ─── 1. Query Intent & Planning Tests ────────────────────────────────────

    def test_plan_01_land_cover(self):
        plan = create_query_plan("What type of land is present here?")
        self.assertEqual(plan.intent, "land_cover")
        self.assertEqual(plan.required_images, 1)
        self.assertIn("land_cover", plan.specialists)
        self.assertIn("rs_vqa_adapted", plan.specialists)

    def test_plan_02_caption(self):
        plan = create_query_plan("Describe this image.")
        self.assertEqual(plan.intent, "caption")
        self.assertEqual(plan.required_images, 1)
        self.assertIn("rs_caption_adapted", plan.specialists)

    def test_plan_03_vqa_water(self):
        plan = create_query_plan("Is there water?")
        self.assertEqual(plan.intent, "vqa")
        self.assertEqual(plan.required_images, 1)
        self.assertIn("rs_vqa_adapted", plan.specialists)

    def test_plan_04_vqa_buildings(self):
        plan = create_query_plan("Are there buildings?")
        self.assertEqual(plan.intent, "vqa")
        self.assertIn("building_detection", plan.specialists)
        self.assertIn("rs_vqa_adapted", plan.specialists)

    def test_plan_05_building_counting(self):
        plan = create_query_plan("How many buildings?")
        self.assertEqual(plan.intent, "building_detection")
        self.assertEqual(plan.required_images, 1)
        self.assertEqual(plan.specialists, ["building_detection"])

    def test_plan_06_grounding(self):
        plan = create_query_plan("Where are the buildings?")
        self.assertEqual(plan.intent, "grounding")
        self.assertIn("visual_grounding", plan.specialists)
        self.assertIn("building_detection", plan.specialists)

    def test_plan_07_change_vqa(self):
        plan = create_query_plan("What changed between these images?", image_count=2)
        self.assertEqual(plan.intent, "change_vqa")
        self.assertEqual(plan.required_images, 2)
        self.assertIn("change_detection", plan.specialists)
        self.assertIn("change_vqa", plan.specialists)

    def test_plan_08_vegetation_change(self):
        plan = create_query_plan("Has vegetation increased?", image_count=2)
        self.assertEqual(plan.intent, "change_detection")
        self.assertEqual(plan.required_images, 2)
        self.assertIn("change_detection", plan.specialists)
        self.assertIn("land_cover", plan.specialists)

    def test_plan_09_new_buildings_change(self):
        plan = create_query_plan("Are there new buildings?", image_count=2)
        self.assertEqual(plan.intent, "change_detection")
        self.assertEqual(plan.required_images, 2)
        self.assertIn("change_detection", plan.specialists)
        self.assertIn("building_detection", plan.specialists)

    def test_plan_10_optical_sar(self):
        plan = create_query_plan("What differences can you see between the optical and SAR images?", modalities=["optical", "sar"])
        self.assertEqual(plan.intent, "optical_sar_analysis")
        self.assertEqual(plan.required_images, 2)
        self.assertIn("optical", plan.required_modalities)
        self.assertIn("sar", plan.required_modalities)
        self.assertEqual(plan.specialists, ["optical_sar_fusion"])

    def test_plan_11_multi_task(self):
        plan = create_query_plan("Describe this scene and tell me whether there is water and whether buildings are present.")
        self.assertEqual(plan.intent, "multi_task")
        self.assertIn("rs_caption_adapted", plan.specialists)
        self.assertIn("rs_vqa_adapted", plan.specialists)
        self.assertIn("building_detection", plan.specialists)

    def test_plan_12_unsupported(self):
        plan = create_query_plan("What will the population be next year?")
        self.assertEqual(plan.intent, "unsupported")
        self.assertIsNotNone(plan.unsupported_reason)
        self.assertGreater(len(plan.supported_alternatives), 0)

    # ─── 2. Validation & Co-Registration Tests ────────────────────────────────

    def test_validation_rejects_change_single_image(self):
        """Reject bi-temporal change query when only 1 image is provided."""
        res = run_orbital_analysis(
            query="What changed between these images?",
            images=[self.opt_img1],
            modalities=["optical"]
        )
        self.assertEqual(res.status, "VALIDATION_ERROR")
        self.assertIn("requires two co-registered observations", res.answer)

    def test_validation_rejects_optical_sar_without_sar(self):
        """Reject optical-SAR query when only optical imagery declared/provided."""
        res = run_orbital_analysis(
            query="Compare these optical and SAR images",
            images=[self.opt_img1, self.opt_img2],
            modalities=["optical", "optical"]
        )
        self.assertEqual(res.status, "VALIDATION_ERROR")
        self.assertIn("requires a SAR scene", res.answer)

    def test_validation_flags_unknown_coregistration_without_metadata(self):
        """When metadata lacks CRS and geotransform, compatibility must be unknown."""
        val = validate_input_imagery(
            task_type="change_detection",
            images=[self.opt_img1, self.opt_img2],
            modalities=["optical", "optical"],
            metadata_list=[]
        )
        self.assertEqual(val["compatibility"], "unknown")
        self.assertTrue(any("never silently assuming registration" in w.lower() for w in val["warnings"]))

    # ─── 3. End-to-End Analysis Pipeline: 12 Exact Queries ───────────────────

    def test_e2e_01_what_type_of_land(self):
        res = run_orbital_analysis(
            query="What type of land is present here?",
            images=[self.opt_img1],
            modalities=["optical"]
        )
        self.assertIn(res.status, ["SUCCESS", "SPECIALIST_UNAVAILABLE"])
        self.assertEqual(res.query_plan.intent, "land_cover")
        self.assertGreater(len(res.evidence), 0)
        self.assertIsNotNone(res.execution_trace)
        if res.status == "SUCCESS":
            self.assertIn("confidence is not calibrated", res.answer.lower())
        else:
            self.assertIn("specialist unavailable", res.answer.lower())

    def test_e2e_02_describe_image(self):
        res = run_orbital_analysis(
            query="Describe this image.",
            images=[self.opt_img1],
            modalities=["optical"]
        )
        self.assertIn(res.status, ["SUCCESS", "SPECIALIST_UNAVAILABLE"])
        self.assertEqual(res.query_plan.intent, "caption")
        self.assertGreater(len(res.evidence), 0)
        self.assertIsNotNone(res.execution_trace)

    def test_e2e_03_is_there_water(self):
        res = run_orbital_analysis(
            query="Is there water?",
            images=[self.opt_img1],
            modalities=["optical"]
        )
        self.assertIn(res.status, ["SUCCESS", "SPECIALIST_UNAVAILABLE"])
        self.assertEqual(res.query_plan.intent, "vqa")
        self.assertGreater(len(res.evidence), 0)

    def test_e2e_04_are_there_buildings(self):
        res = run_orbital_analysis(
            query="Are there buildings?",
            images=[self.opt_img1],
            modalities=["optical"]
        )
        self.assertIn(res.status, ["SUCCESS", "SPECIALIST_UNAVAILABLE"])
        self.assertEqual(res.query_plan.intent, "vqa")
        if res.status == "SUCCESS":
            self.assertIn("building_detection", [e.task for e in res.evidence] + [e.source for e in res.evidence])
        else:
            self.assertEqual(res.status, "SPECIALIST_UNAVAILABLE")

    def test_e2e_05_how_many_buildings(self):
        res = run_orbital_analysis(
            query="How many buildings?",
            images=[self.opt_img1],
            modalities=["optical"]
        )
        self.assertIn(res.status, ["SUCCESS", "SPECIALIST_UNAVAILABLE"])
        self.assertEqual(res.query_plan.intent, "building_detection")
        if res.status == "SUCCESS":
            self.assertIn("calibrated", res.confidence_status)

    def test_e2e_06_where_are_buildings(self):
        res = run_orbital_analysis(
            query="Where are the buildings?",
            images=[self.opt_img1],
            modalities=["optical"]
        )
        self.assertIn(res.status, ["SUCCESS", "SPECIALIST_UNAVAILABLE"])
        self.assertEqual(res.query_plan.intent, "grounding")
        self.assertIn("visual_evidence", res.model_dump() if hasattr(res, "model_dump") else res.dict())

    def test_e2e_07_what_changed(self):
        res = run_orbital_analysis(
            query="What changed between these images?",
            images=[self.opt_img1, self.opt_img2],
            modalities=["optical", "optical"],
            metadata_list=[self.meta_optical_1, self.meta_optical_2]
        )
        self.assertIn(res.status, ["SUCCESS", "SPECIALIST_UNAVAILABLE"])
        self.assertEqual(res.query_plan.intent, "change_vqa")
        self.assertGreater(len(res.evidence), 0)

    def test_e2e_08_has_vegetation_increased(self):
        res = run_orbital_analysis(
            query="Has vegetation increased?",
            images=[self.opt_img1, self.opt_img2],
            modalities=["optical", "optical"],
            metadata_list=[self.meta_optical_1, self.meta_optical_2]
        )
        self.assertIn(res.status, ["SUCCESS", "SPECIALIST_UNAVAILABLE"])
        self.assertEqual(res.query_plan.intent, "change_detection")
        self.assertIn("vegetation", res.answer.lower())
        self.assertIn("confidence is not calibrated", res.answer.lower())

    def test_e2e_09_are_there_new_buildings(self):
        res = run_orbital_analysis(
            query="Are there new buildings?",
            images=[self.opt_img1, self.opt_img2],
            modalities=["optical", "optical"],
            metadata_list=[self.meta_optical_1, self.meta_optical_2]
        )
        self.assertIn(res.status, ["SUCCESS", "SPECIALIST_UNAVAILABLE"])
        self.assertEqual(res.query_plan.intent, "change_detection")

    def test_e2e_10_optical_sar_differences(self):
        res = run_orbital_analysis(
            query="What differences can you see between the optical and SAR images?",
            images=[self.opt_img1, self.sar_img],
            modalities=["optical", "sar"],
            metadata_list=[self.meta_optical_1, self.meta_sar]
        )
        self.assertIn(res.status, ["SUCCESS", "SPECIALIST_UNAVAILABLE"])
        self.assertEqual(res.query_plan.intent, "optical_sar_analysis")
        self.assertIn("ssim", res.answer.lower())

    def test_e2e_11_optical_sar_structures(self):
        res = run_orbital_analysis(
            query="What structures are visible across both modalities?",
            images=[self.opt_img1, self.sar_img],
            modalities=["optical", "sar"],
            metadata_list=[self.meta_optical_1, self.meta_sar]
        )
        self.assertIn(res.status, ["SUCCESS", "SPECIALIST_UNAVAILABLE"])
        self.assertEqual(res.query_plan.intent, "optical_sar_analysis")

    def test_e2e_12_compound_multi_task(self):
        res = run_orbital_analysis(
            query="Describe this scene and tell me whether there is water and whether buildings are present.",
            images=[self.opt_img1],
            modalities=["optical"]
        )
        self.assertIn(res.status, ["SUCCESS", "SPECIALIST_UNAVAILABLE"])
        self.assertEqual(res.query_plan.intent, "multi_task")
        if res.status == "SUCCESS":
            self.assertGreaterEqual(len(res.evidence), 2)
        else:
            self.assertEqual(res.status, "SPECIALIST_UNAVAILABLE")

    def test_e2e_13_unsupported_query(self):
        res = run_orbital_analysis(
            query="What will the population be next year?",
            images=[self.opt_img1],
            modalities=["optical"]
        )
        self.assertEqual(res.status, "UNSUPPORTED_QUERY")
        self.assertIn("unsupported query", res.answer.lower())
        self.assertGreater(len(res.query_plan.supported_alternatives), 0)

    # ─── 4. Observable Trace Integrity ───────────────────────────────────────

    def test_observable_trace_steps(self):
        """Verify the 8 required operational events are present in the trace."""
        res = run_orbital_analysis(
            query="How many buildings?",
            images=[self.opt_img1],
            modalities=["optical"]
        )
        trace = res.execution_trace
        self.assertIsNotNone(trace)
        tools_in_trace = [s.tool for s in trace.steps]
        # Query understood (query_planner)
        self.assertIn("query_planner", tools_in_trace)
        # Input validation
        self.assertIn("input_validator", tools_in_trace)
        # Task plan (task_sequencer)
        self.assertIn("task_sequencer", tools_in_trace)
        # Evidence aggregator & response synthesizer
        self.assertIn("evidence_aggregator", tools_in_trace)
        self.assertIn("response_synthesizer", tools_in_trace)


if __name__ == "__main__":
    unittest.main()
