"""
tests/test_multitask_decomposition.py — Regression test suite for Step 14A:
Multi-task compound query decomposition and focused VQA question routing.

Query under test:
    "Describe the scene, identify the main land cover, and tell me whether
     vegetation is present."

Verifies:
1. Intent is 'multi_task'
2. required_tasks contains 'caption', 'land_cover', 'vqa'
3. Execution order / specialists list is ['rs_caption_adapted', 'land_cover', 'rs_vqa_adapted']
4. vqa_question == "Is there vegetation in this image?"
5. vqa_target == "vegetation"
6. E2E: specialists are executed and focused VQA question appears in trace
7. Final synthesized response explicitly covers scene description, main land cover,
   vegetation presence — and does NOT report "Water assessment" for a vegetation query.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.planner import create_query_plan, extract_vqa_sub_question
from backend.agents.orchestrator import run_orbital_analysis


EXACT_QUERY = (
    "Describe the scene, identify the main land cover, and tell me whether "
    "vegetation is present."
)


class TestMultiTaskDecompositionPlanner(unittest.TestCase):
    """Pure planner-level tests (no model execution, runs fast)."""

    def setUp(self):
        self.query = EXACT_QUERY

    # ── 1. Intent ─────────────────────────────────────────────────────────────

    def test_01_intent_is_multi_task(self):
        """Compound query must be classified as multi_task."""
        plan = create_query_plan(self.query, image_count=1)
        self.assertEqual(
            plan.intent, "multi_task",
            f"Expected intent='multi_task', got '{plan.intent}'"
        )

    # ── 2. Task decomposition ────────────────────────────────────────────────

    def test_02_required_tasks_contain_caption(self):
        plan = create_query_plan(self.query, image_count=1)
        self.assertIn(
            "caption", plan.required_tasks,
            f"required_tasks must contain 'caption'; got {plan.required_tasks}"
        )

    def test_03_required_tasks_contain_land_cover(self):
        plan = create_query_plan(self.query, image_count=1)
        self.assertIn(
            "land_cover", plan.required_tasks,
            f"required_tasks must contain 'land_cover'; got {plan.required_tasks}"
        )

    def test_04_required_tasks_contain_vqa(self):
        plan = create_query_plan(self.query, image_count=1)
        self.assertIn(
            "vqa", plan.required_tasks,
            f"required_tasks must contain 'vqa'; got {plan.required_tasks}"
        )

    # ── 3. Specialist / execution order ──────────────────────────────────────

    def test_05_specialist_execution_order(self):
        """Execution order must be caption -> land_cover -> vqa."""
        plan = create_query_plan(self.query, image_count=1)
        self.assertEqual(
            plan.specialists,
            ["rs_caption_adapted", "land_cover", "rs_vqa_adapted"],
            f"Specialist order wrong: {plan.specialists}"
        )

    def test_06_execution_order_field(self):
        plan = create_query_plan(self.query, image_count=1)
        self.assertEqual(
            plan.execution_order,
            ["rs_caption_adapted", "land_cover", "rs_vqa_adapted"],
            f"execution_order wrong: {plan.execution_order}"
        )

    # ── 4. VQA focused question ──────────────────────────────────────────────

    def test_07_vqa_focused_question(self):
        """VQA specialist must receive the focused vegetation question."""
        plan = create_query_plan(self.query, image_count=1)
        self.assertEqual(
            plan.vqa_question,
            "Is there vegetation in this image?",
            f"vqa_question wrong: '{plan.vqa_question}'"
        )

    def test_08_vqa_target_is_vegetation(self):
        plan = create_query_plan(self.query, image_count=1)
        self.assertEqual(
            plan.vqa_target, "vegetation",
            f"vqa_target wrong: '{plan.vqa_target}'"
        )

    # ── 5. extract_vqa_sub_question helper ───────────────────────────────────

    def test_09_extract_vqa_vegetation(self):
        """Helper correctly extracts vegetation focus."""
        q, target = extract_vqa_sub_question("tell me whether vegetation is present")
        self.assertEqual(q, "Is there vegetation in this image?")
        self.assertEqual(target, "vegetation")

    def test_10_extract_vqa_water(self):
        """Helper correctly extracts water focus."""
        q, target = extract_vqa_sub_question("is there water in the image?")
        self.assertEqual(q, "Is there water in this image?")
        self.assertEqual(target, "water")

    def test_11_extract_vqa_buildings(self):
        """Helper correctly extracts buildings focus."""
        q, target = extract_vqa_sub_question("are there buildings present?")
        self.assertEqual(q, "Are there buildings in this image?")
        self.assertEqual(target, "buildings")

    def test_12_extract_vqa_generic_presence_pattern(self):
        """Generic 'whether X is present' pattern is captured."""
        q, target = extract_vqa_sub_question("tell me whether snow is present")
        self.assertIsNotNone(q, "Generic presence question must yield a focused VQA question")
        self.assertIn("snow", q.lower())

    # ── 6. Single-task routing is preserved ──────────────────────────────────

    def test_13_describe_only_is_caption(self):
        """Pure describe query still routes to caption intent, not multi_task."""
        plan = create_query_plan("Describe this image.", image_count=1)
        self.assertEqual(plan.intent, "caption")

    def test_14_land_cover_only_query(self):
        """Pure land cover query routes to land_cover intent."""
        plan = create_query_plan("What type of land is present here?", image_count=1)
        self.assertEqual(plan.intent, "land_cover")

    def test_15_vegetation_change_with_two_images(self):
        """Bi-temporal vegetation query routes to change_detection, not multi_task."""
        plan = create_query_plan("Has vegetation increased?", image_count=2)
        self.assertEqual(plan.intent, "change_detection")


class TestMultiTaskDecompositionE2E(unittest.TestCase):
    """End-to-end orchestration tests (model execution required)."""

    def setUp(self):
        self.query = EXACT_QUERY
        self.test_img = np.full((120, 120, 3), 128, dtype=np.uint8)

    def _run(self):
        return run_orbital_analysis(
            query=self.query,
            images=[self.test_img],
            modalities=["optical"],
            metadata_list=[{"sensor": "Sentinel-2"}],
        )

    def test_e2e_01_status_is_success_or_unavailable(self):
        """Pipeline must not error -- only SUCCESS or SPECIALIST_UNAVAILABLE."""
        res = self._run()
        self.assertIn(
            res.status, ["SUCCESS", "SPECIALIST_UNAVAILABLE"],
            f"Unexpected status: {res.status} -- {res.answer}"
        )

    def test_e2e_02_query_plan_intent(self):
        """Query plan must carry multi_task intent regardless of model availability."""
        res = self._run()
        self.assertEqual(res.query_plan.intent, "multi_task")

    def test_e2e_03_query_plan_tasks(self):
        """Query plan required_tasks must contain caption, land_cover, vqa."""
        res = self._run()
        for task in ("caption", "land_cover", "vqa"):
            self.assertIn(
                task, res.query_plan.required_tasks,
                f"required_tasks missing '{task}': {res.query_plan.required_tasks}"
            )

    def test_e2e_04_query_plan_specialists(self):
        """Query plan specialists must match expected execution order."""
        res = self._run()
        self.assertEqual(
            res.query_plan.specialists,
            ["rs_caption_adapted", "land_cover", "rs_vqa_adapted"],
        )

    def test_e2e_05_query_plan_vqa_question(self):
        """Query plan vqa_question must be the focused vegetation question."""
        res = self._run()
        self.assertEqual(
            res.query_plan.vqa_question,
            "Is there vegetation in this image?"
        )

    def test_e2e_06_vqa_receives_focused_question(self):
        """
        The VQA specialist trace step must show the focused question
        in its input_summary, NOT the full compound query.
        """
        res = self._run()
        if res.status == "SPECIALIST_UNAVAILABLE":
            self.skipTest("Specialists unavailable -- skipping VQA input trace check")

        # TraceStep.tool holds the specialist name string; search broadly
        vqa_steps = [
            step for step in res.execution_trace.steps
            if "vqa" in step.tool.lower() and step.input_summary
        ]

        if vqa_steps:
            vqa_input = vqa_steps[0].input_summary
            self.assertIn(
                "Is there vegetation in this image?",
                vqa_input,
                f"VQA trace input_summary must contain focused question; got: '{vqa_input}'"
            )
            self.assertNotIn(
                "Describe the scene",
                vqa_input,
                "VQA must NOT receive the full compound query"
            )

    def test_e2e_07_response_covers_scene_description(self):
        """Final response must include scene description when SUCCESS."""
        res = self._run()
        if res.status == "SPECIALIST_UNAVAILABLE":
            self.skipTest("Specialists unavailable")
        self.assertIn(
            "Scene overview:", res.answer,
            f"Response must contain 'Scene overview:'; got:\n{res.answer}"
        )

    def test_e2e_08_response_covers_land_cover(self):
        """Final response must include main land cover when SUCCESS."""
        res = self._run()
        if res.status == "SPECIALIST_UNAVAILABLE":
            self.skipTest("Specialists unavailable")
        self.assertIn(
            "Main land cover:", res.answer,
            f"Response must contain 'Main land cover:'; got:\n{res.answer}"
        )

    def test_e2e_09_response_covers_vegetation_presence(self):
        """Final response must include vegetation presence when SUCCESS."""
        res = self._run()
        if res.status == "SPECIALIST_UNAVAILABLE":
            self.skipTest("Specialists unavailable")
        self.assertIn(
            "Vegetation presence:", res.answer,
            f"Response must contain 'Vegetation presence:'; got:\n{res.answer}"
        )

    def test_e2e_10_response_does_not_report_water_for_vegetation_query(self):
        """Response must NOT report 'Water assessment' for a vegetation query."""
        res = self._run()
        if res.status == "SPECIALIST_UNAVAILABLE":
            self.skipTest("Specialists unavailable")
        self.assertNotIn(
            "Water assessment:", res.answer,
            "Response must NOT report 'Water assessment' for a vegetation query"
        )

    def test_e2e_11_trace_contains_required_steps(self):
        """Observable trace must contain query_planner, input_validator, task_sequencer."""
        res = self._run()
        tools_in_trace = [s.tool for s in res.execution_trace.steps]
        for required_tool in ("query_planner", "input_validator", "task_sequencer"):
            self.assertIn(
                required_tool, tools_in_trace,
                f"Trace missing '{required_tool}'; found: {tools_in_trace}"
            )

    def test_e2e_12_evidence_sources_include_all_specialists(self):
        """When SUCCESS, evidence must include outputs from all three specialists."""
        res = self._run()
        if res.status == "SPECIALIST_UNAVAILABLE":
            self.skipTest("Specialists unavailable")
        sources = {ev.source for ev in res.evidence}
        for expected in ("rs_caption_adapted", "land_cover", "rs_vqa_adapted"):
            self.assertIn(
                expected, sources,
                f"Evidence missing source '{expected}'; found: {sources}"
            )


if __name__ == "__main__":
    unittest.main()
