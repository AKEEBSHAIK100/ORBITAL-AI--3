"""
Tests for Checkpoint 2: Shared Evidence Context and Multi-Task Deduplication.
Covers:
TEST 1: Standalone caption still works (with supporting inference).
TEST 2: Standalone VQA still works.
TEST 3: Standalone land-cover still works.
TEST 4: Compound caption + land-cover reuses evidence when provided (no double execution).
TEST 5: Compound land-cover + VQA does not execute BigEarthNet twice.
TEST 6: Compound building + VQA does not execute building detection twice.
TEST 7: If supporting evidence is absent, the tool still performs its existing supporting inference.
TEST 8: Failed supporting evidence does not incorrectly get reused.
TEST 9: Shared context is request-scoped and cannot leak between requests.
TEST 10: Existing planner routing remains unchanged.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from backend.agents.planner import create_query_plan
from backend.agents.orchestrator import run_orbital_analysis
from backend.tools.caption import CaptionTool
from backend.tools.vqa import VQATool
from backend.tools.land_cover import BigEarthNetTool


class TestSharedEvidenceDeduplication(unittest.TestCase):
    def setUp(self):
        # 100x100 3-channel dummy image
        self.dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_test1_standalone_caption_still_works(self):
        """TEST 1: Standalone caption still works and gathers supporting land-cover if absent."""
        mock_runtime = MagicMock()
        mock_runtime.caption.return_value = {
            "status": "success",
            "caption": "An aerial image of agricultural fields.",
            "answer": "An aerial image of agricultural fields.",
            "inference_time_ms": 12.0,
            "device": "cpu"
        }
        mock_runtime.device = "cpu"

        mock_ben = MagicMock()
        mock_ben.run.return_value = {
            "status": "success",
            "top_label": "Arable land",
            "active_labels": ["Arable land"]
        }

        tool = CaptionTool(runtime=mock_runtime, ben_tool=mock_ben)
        res = tool.run({"image": self.dummy_img})

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["caption"], "An aerial image of agricultural fields.")
        self.assertIn("supporting_land_cover", res["evidence"])
        self.assertEqual(res["evidence"]["supporting_land_cover"]["top_label"], "Arable land")
        self.assertEqual(mock_ben.run.call_count, 1)

    def test_test2_standalone_vqa_still_works(self):
        """TEST 2: Standalone VQA still works with default supporting checks."""
        mock_runtime = MagicMock()
        mock_runtime.vqa.return_value = {
            "status": "success",
            "answer": "yes",
            "inference_time_ms": 15.0,
            "device": "cpu"
        }
        mock_runtime.device = "cpu"

        mock_ben = MagicMock()
        mock_ben.run.return_value = {
            "status": "success",
            "top_label": "Broad-leaved forest",
            "active_labels": ["Broad-leaved forest"]
        }

        tool = VQATool(runtime=mock_runtime, ben_tool=mock_ben)
        res = tool.run({"image": self.dummy_img, "query": "Is there vegetation in this image?"})

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["answer"], "yes")
        self.assertIn("supporting_land_cover", res["evidence"])
        self.assertEqual(res["evidence"]["supporting_land_cover"]["top_label"], "Broad-leaved forest")
        self.assertEqual(mock_ben.run.call_count, 1)

    def test_test3_standalone_land_cover_still_works(self):
        """TEST 3: Standalone land-cover still works."""
        tool = BigEarthNetTool()
        with patch.object(tool.classifier, "classify_image") as mock_classify:
            mock_classify.return_value = {
                "top_label": "Urban fabric",
                "confidence": 92.0,
                "labels": [{"name": "Urban fabric", "probability": 0.92}],
                "active_labels": [{"name": "Urban fabric", "probability": 0.92}]
            }
            res = tool.run({"image": self.dummy_img})
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["top_label"], "Urban fabric")
            mock_classify.assert_called_once()

    def test_test4_compound_caption_land_cover_reuses_evidence(self):
        """TEST 4: When caption tool receives existing valid supporting land cover, it does NOT execute BigEarthNet again."""
        mock_runtime = MagicMock()
        mock_runtime.caption.return_value = {
            "status": "success",
            "caption": "Scene overview",
            "answer": "Scene overview",
            "inference_time_ms": 10.0,
            "device": "cpu"
        }
        mock_runtime.device = "cpu"

        mock_ben = MagicMock()
        tool = CaptionTool(runtime=mock_runtime, ben_tool=mock_ben)

        shared_lc = {
            "status": "success",
            "top_label": "Industrial or commercial units",
            "active_labels": [{"name": "Industrial or commercial units"}]
        }

        res = tool.run(
            {"image": self.dummy_img},
            parameters={"supporting_land_cover_result": shared_lc}
        )

        self.assertEqual(res["status"], "success")
        # Ensure BigEarthNet was NOT called
        mock_ben.run.assert_not_called()
        # Verify evidence matches shared input and is marked as reused
        self.assertEqual(res["evidence"]["supporting_land_cover"]["top_label"], "Industrial or commercial units")
        self.assertTrue(res["evidence"]["supporting_land_cover"].get("reused"))

    def test_test5_compound_land_cover_vqa_deduplication(self):
        """TEST 5: Orchestrator passes land_cover evidence to VQA so BigEarthNet is not run twice."""
        # Planner: "What type of land is present and is there vegetation?" -> land_cover, rs_vqa_adapted
        mock_ben_result = {
            "status": "success",
            "top_label": "Pastures",
            "confidence": 88.0,
            "labels": [{"name": "Pastures", "probability": 0.88}],
            "active_labels": [{"name": "Pastures", "probability": 0.88}],
            "model_id": "ben-resnet50"
        }

        mock_vqa_result = {
            "status": "success",
            "answer": "yes",
            "model_id": "vqa-blip",
            "evidence": {
                "supporting_land_cover": {"top_label": "Pastures", "reused": True}
            }
        }

        mock_ben_tool = MagicMock()
        mock_ben_tool.run.return_value = mock_ben_result
        mock_ben_tool.supported_tasks = ["land_cover"]
        mock_ben_tool.model_id = "ben-tool"

        mock_vqa_tool = MagicMock()
        mock_vqa_tool.run.return_value = mock_vqa_result
        mock_vqa_tool.supported_tasks = ["vqa"]
        mock_vqa_tool.model_id = "vqa-tool"

        def get_tool_side_effect(tid):
            if tid == "land_cover":
                return mock_ben_tool
            if tid in ["rs_vqa_adapted", "vqa"]:
                return mock_vqa_tool
            return None

        with patch("backend.agents.orchestrator.get_tool", side_effect=get_tool_side_effect):
            query = "What type of land is present and is there vegetation?"
            resp = run_orbital_analysis(query=query, images=[self.dummy_img])

            self.assertEqual(resp.status, "SUCCESS")
            # Verify VQA received the supporting_land_cover_result in parameters
            vqa_call = mock_vqa_tool.run.call_args
            # tool_obj.run is called as (curr_tool_inputs, spec_params)
            vqa_params = vqa_call[0][1] if len(vqa_call[0]) > 1 else vqa_call[1]
            self.assertIn("supporting_land_cover_result", vqa_params)
            self.assertEqual(vqa_params["supporting_land_cover_result"]["top_label"], "Pastures")

            # Verify deduplicator trace step was emitted
            trace_steps = resp.execution_trace.steps if resp.execution_trace else []
            dedup_steps = [s for s in trace_steps if s.tool == "evidence_deduplicator"]
            self.assertGreaterEqual(len(dedup_steps), 1)
            self.assertIn("land_cover", dedup_steps[0].output_summary)

    def test_test6_compound_building_vqa_deduplication(self):
        """TEST 6: Compound building + VQA does not execute building detection twice."""
        mock_bldg_result = {
            "status": "success",
            "building_count": 14,
            "high_confidence_count": 12,
            "confidence_level": "High",
            "detections": [],
            "image_dimensions": {"width": 100, "height": 100}
        }

        mock_vqa_result = {
            "status": "success",
            "answer": "yes, there are buildings",
            "model_id": "vqa-blip",
            "evidence": {
                "supporting_building_detection": {"building_count": 14, "reused": True}
            }
        }

        mock_bldg_tool = MagicMock()
        mock_bldg_tool.run.return_value = mock_bldg_result
        mock_bldg_tool.supported_tasks = ["building_detection"]
        mock_bldg_tool.model_id = "building-tool"

        mock_vqa_tool = MagicMock()
        mock_vqa_tool.run.return_value = mock_vqa_result
        mock_vqa_tool.supported_tasks = ["vqa"]
        mock_vqa_tool.model_id = "vqa-tool"

        mock_runtime = MagicMock()
        mock_runtime.vqa.return_value = {"status": "success", "answer": "yes"}
        mock_runtime.device = "cpu"

        vqa_tool = VQATool(runtime=mock_runtime, building_tool=mock_bldg_tool)
        res = vqa_tool.run(
            {"image": self.dummy_img, "query": "Are there buildings?"},
            parameters={"supporting_building_result": mock_bldg_result}
        )

        self.assertEqual(res["status"], "success")
        mock_bldg_tool.run.assert_not_called()
        self.assertEqual(res["evidence"]["supporting_building_detection"]["building_count"], 14)
        self.assertTrue(res["evidence"]["supporting_building_detection"].get("reused"))

    def test_test7_absent_supporting_evidence_falls_back_to_execution(self):
        """TEST 7: If supporting evidence is absent, tool still runs its internal supporting specialist."""
        mock_runtime = MagicMock()
        mock_runtime.vqa.return_value = {"status": "success", "answer": "yes"}
        mock_runtime.device = "cpu"

        mock_bldg_tool = MagicMock()
        mock_bldg_tool.run.return_value = {
            "status": "success",
            "building_count": 5,
            "high_confidence_count": 4,
            "confidence_level": "High"
        }

        vqa_tool = VQATool(runtime=mock_runtime, building_tool=mock_bldg_tool)
        res = vqa_tool.run(
            {"image": self.dummy_img, "query": "How many structures?"},
            parameters={}  # No supporting_building_result
        )

        self.assertEqual(res["status"], "success")
        self.assertEqual(mock_bldg_tool.run.call_count, 1)
        self.assertEqual(res["evidence"]["supporting_building_detection"]["building_count"], 5)
        self.assertFalse(res["evidence"]["supporting_building_detection"].get("reused", False))

    def test_test8_failed_supporting_evidence_not_reused(self):
        """TEST 8: Failed or invalid supporting evidence is NOT reused and falls back to normal execution."""
        mock_runtime = MagicMock()
        mock_runtime.vqa.return_value = {"status": "success", "answer": "yes"}
        mock_runtime.device = "cpu"

        mock_ben_tool = MagicMock()
        mock_ben_tool.run.return_value = {
            "status": "success",
            "top_label": "Water bodies",
            "active_labels": ["Water bodies"]
        }

        vqa_tool = VQATool(runtime=mock_runtime, ben_tool=mock_ben_tool)
        # Pass an error object as supporting_land_cover_result
        invalid_lc = {"status": "error", "error": "Disk read failure"}

        res = vqa_tool.run(
            {"image": self.dummy_img, "query": "Is there vegetation?"},
            parameters={"supporting_land_cover_result": invalid_lc}
        )

        # The invalid result should have been ignored, and mock_ben_tool.run invoked instead
        self.assertEqual(mock_ben_tool.run.call_count, 1)
        self.assertEqual(res["evidence"]["supporting_land_cover"]["top_label"], "Water bodies")
        self.assertFalse(res["evidence"]["supporting_land_cover"].get("reused", False))

    def test_test9_shared_context_is_request_scoped(self):
        """TEST 9: Shared context does NOT leak between consecutive requests."""
        mock_ben_result = {
            "status": "success",
            "top_label": "Coniferous forest",
            "confidence": 85.0,
            "active_labels": [{"name": "Coniferous forest"}],
            "model_id": "ben"
        }
        mock_caption_result = {
            "status": "success",
            "caption": "Forest area",
            "model_id": "blip"
        }

        mock_ben_tool = MagicMock()
        mock_ben_tool.run.return_value = mock_ben_result
        mock_ben_tool.supported_tasks = ["land_cover"]
        mock_ben_tool.model_id = "ben"

        mock_vqa_tool = MagicMock()
        mock_vqa_tool.run.return_value = {"status": "success", "answer": "forest detected"}
        mock_vqa_tool.supported_tasks = ["vqa"]
        mock_vqa_tool.model_id = "vqa"

        mock_caption_tool = MagicMock()
        mock_caption_tool.run.return_value = mock_caption_result
        mock_caption_tool.supported_tasks = ["caption"]
        mock_caption_tool.model_id = "blip"

        def get_tool_side_effect(tid):
            if tid == "land_cover":
                return mock_ben_tool
            if tid in ["rs_vqa_adapted", "vqa"]:
                return mock_vqa_tool
            if tid in ["rs_caption_adapted", "caption"]:
                return mock_caption_tool
            return None

        with patch("backend.agents.orchestrator.get_tool", side_effect=get_tool_side_effect):
            # Request 1: land_cover query (executes land_cover then rs_vqa_adapted)
            resp1 = run_orbital_analysis(query="What type of land is present?", images=[self.dummy_img])
            self.assertEqual(resp1.status, "SUCCESS")

            # Request 2: standalone caption query ("Describe this image.") -> execution_order: ["rs_caption_adapted"]
            resp2 = run_orbital_analysis(query="Describe this image.", images=[self.dummy_img])
            self.assertEqual(resp2.status, "SUCCESS")

            # Request 2's caption call should NOT have received land_cover from Request 1!
            caption_call = mock_caption_tool.run.call_args
            caption_params = caption_call[0][1] if len(caption_call[0]) > 1 else caption_call[1]
            self.assertNotIn("supporting_land_cover_result", caption_params)

    def test_test10_planner_routing_unchanged(self):
        """TEST 10: Ensure planner deterministic routing remains exact and unaffected."""
        plan1 = create_query_plan("Describe this image.")
        self.assertEqual(plan1.intent, "caption")
        self.assertEqual(plan1.specialists, ["rs_caption_adapted"])

        plan2 = create_query_plan("What type of land is present here?")
        self.assertEqual(plan2.intent, "land_cover")
        self.assertEqual(plan2.specialists, ["land_cover", "rs_vqa_adapted"])

        plan3 = create_query_plan("Describe the scene, identify the main land cover, and tell me whether vegetation is present.")
        self.assertEqual(plan3.intent, "multi_task")
        self.assertEqual(plan3.specialists, ["rs_caption_adapted", "land_cover", "rs_vqa_adapted"])

        plan4 = create_query_plan("Are there solar panels visible?")
        self.assertEqual(plan4.intent, "general_vqa")
        self.assertEqual(plan4.specialists, ["rs_generalist"])


if __name__ == "__main__":
    unittest.main()
