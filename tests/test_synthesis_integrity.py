"""Regression tests for the no-fabrication synthesis contract."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.synthesizer import synthesize_response
from backend.schemas.analysis import QueryPlan, SpecialistEvidenceObject


class TestSynthesisIntegrity(unittest.TestCase):
    def plan(self, intent, order):
        return QueryPlan(
            intent=intent,
            execution_order=order,
            required_modalities=["optical"],
            required_images=1,
            evidence_requirements=[],
        )

    def test_caption_missing_evidence_is_unavailable(self):
        ans, conf, status, _ = synthesize_response(
            "Describe this image.",
            self.plan("caption", ["caption"]),
            [],
            {"compatibility": "compatible"},
            status="SPECIALIST_UNAVAILABLE",
        )
        self.assertEqual(status, "unavailable")
        self.assertIsNone(conf)
        self.assertIn("no scene description has been fabricated", ans.lower())

    def test_change_missing_required_evidence_is_unavailable(self):
        ev = SpecialistEvidenceObject(
            task="change_detection",
            result="",
            evidence={"change_percentage": None, "change_clusters": None},
            source="change_detection",
            model="classical-cv-change",
            confidence=None,
            confidence_status="not_calibrated",
        )
        ans, conf, status, _ = synthesize_response(
            "What changed?",
            self.plan("change_detection", ["change_detection"]),
            [ev],
            {"compatibility": "compatible"},
            status="SUCCESS",
        )
        self.assertEqual(status, "unavailable")
        self.assertIsNone(conf)
        self.assertIn("no change result has been fabricated", ans.lower())

    def test_grounding_does_not_invent_target(self):
        ev = SpecialistEvidenceObject(
            task="grounding",
            result="",
            evidence={},
            source="grounding",
            model="grounding",
            confidence=None,
            confidence_status="not_calibrated",
        )
        ans, conf, status, _ = synthesize_response(
            "Find it.",
            self.plan("grounding", ["grounding"]),
            [ev],
            {"compatibility": "compatible"},
            status="SPECIALIST_UNAVAILABLE",
        )
        self.assertIsNone(conf)
        self.assertEqual(status, "unavailable")
        self.assertNotIn("building", ans.lower())


    def test_grounding_incomplete_geometry_is_unavailable(self):
        ev = SpecialistEvidenceObject(
            task="grounding",
            result="",
            evidence={"target": "water body", "regions": [{"region": {"x_percent": 10.0, "y_percent": None, "w_percent": 20.0, "h_percent": 20.0}}]},
            source="grounding",
            model="grounding",
            confidence=None,
            confidence_status="not_calibrated",
        )
        ans, conf, status, _ = synthesize_response(
            "Where is the water?",
            self.plan("grounding", ["grounding"]),
            [ev],
            {"compatibility": "compatible"},
            status="SUCCESS",
        )
        self.assertEqual(status, "unavailable")
        self.assertIsNone(conf)
        self.assertIn("incomplete region geometry", ans.lower())

    def test_building_confidence_is_not_invented(self):
        ev = SpecialistEvidenceObject(
            task="building_detection",
            result="",
            evidence={"building_count": 7},
            source="building_detection",
            model="building-model",
            confidence=None,
            confidence_status="not_calibrated",
        )
        ans, conf, status, _ = synthesize_response(
            "How many buildings are visible?",
            self.plan("building_detection", ["building_detection"]),
            [ev],
            {"compatibility": "compatible"},
            status="SUCCESS",
        )
        self.assertIn("7", ans)
        self.assertNotIn("high certainty", ans.lower())
        self.assertIsNone(conf)


    def test_building_change_does_not_claim_new_additions_without_pairwise_evidence(self):
        ev = SpecialistEvidenceObject(
            task="change_detection",
            result="",
            evidence={"change_percentage": 12.0},
            source="change_detection",
            model="classical-cv-change",
            confidence=None,
            confidence_status="not_calibrated",
        )
        bldg = SpecialistEvidenceObject(
            task="building_detection",
            result="",
            evidence={"building_count": 14},
            source="building_detection",
            model="building-model",
            confidence=None,
            confidence_status="not_calibrated",
        )
        ans, conf, status, _ = synthesize_response(
            "Are there new buildings?",
            self.plan("change_detection", ["change_detection", "building_detection"]),
            [ev, bldg],
            {"compatibility": "compatible"},
            status="SUCCESS",
        )
        self.assertEqual(status, "not_calibrated")
        self.assertIsNone(conf)
        self.assertIn("does not by itself establish", ans.lower())
        self.assertNotIn("new structural additions are identified", ans.lower())


if __name__ == "__main__":
    unittest.main()
