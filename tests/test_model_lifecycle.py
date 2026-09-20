"""
Tests for Checkpoint 4: Model Lifecycle, On-Demand Tool Loading & Memory Management.

Validates:
1. Per-tool on-demand lazy initialization in ToolRegistry.
2. Tools instantiate instantaneously without eagerly loading heavy deep learning models.
3. Lazy properties in BigEarthNetTool and BuildingDetectionTool.
4. Tool aliases resolve correctly to canonical singletons.
5. list_all_tools() produces all specifications cleanly.
6. unload_model() / unload_models() memory release hooks operate safely.
7. TOOLS proxy mapping preserves dictionary-like access semantics.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from backend.tools.registry import (
    get_tool,
    list_all_tools,
    reset_registry,
    _instances,
    TOOLS,
    ALL_TOOL_IDS,
    CANONICAL_TOOL_IDS,
)
from backend.tools.land_cover import BigEarthNetTool
from backend.tools.building_detection import BuildingDetectionTool
from backend.tools.optical_sar import OpticalSARTool
from backend.tools.base import BaseTool
from backend.services.building_detector import BuildingDetector
from backend.services.ben_classifier import BENClassifier
from backend.services.rs_adapters import RSAdapterRuntime


class TestToolRegistryLifecycle(unittest.TestCase):
    def setUp(self):
        reset_registry()

    def tearDown(self):
        reset_registry()

    def test_per_tool_lazy_instantiation_isolation(self):
        """Verifies requesting one tool does NOT eagerly construct all other tools."""
        self.assertEqual(len(_instances), 0)
        tool = get_tool("optical_sar")
        self.assertIsNotNone(tool)
        self.assertIsInstance(tool, OpticalSARTool)
        # Only optical_sar should be in _instances
        self.assertIn("optical_sar", _instances)
        self.assertNotIn("land_cover", _instances)
        self.assertNotIn("building_detection", _instances)
        self.assertNotIn("vqa", _instances)

    def test_alias_resolution(self):
        """Verifies aliases map to the same canonical singleton instance."""
        vqa_alias = get_tool("rs_vqa_adapted")
        vqa_canon = get_tool("vqa")
        self.assertIsNotNone(vqa_alias)
        self.assertIs(vqa_alias, vqa_canon)

        caption_alias = get_tool("rs_caption_adapted")
        caption_canon = get_tool("caption")
        self.assertIsNotNone(caption_alias)
        self.assertIs(caption_alias, caption_canon)

        grounding_alias = get_tool("visual_grounding")
        grounding_canon = get_tool("grounding")
        self.assertIsNotNone(grounding_alias)
        self.assertIs(grounding_alias, grounding_canon)

        fusion_alias = get_tool("optical_sar_fusion")
        fusion_canon = get_tool("optical_sar")
        self.assertIsNotNone(fusion_alias)
        self.assertIs(fusion_alias, fusion_canon)

        change_alias = get_tool("change_vqa")
        change_canon = get_tool("change_detection")
        self.assertIsNotNone(change_alias)
        self.assertIs(change_alias, change_canon)

    def test_unknown_tool_returns_none(self):
        self.assertIsNone(get_tool("nonexistent_tool_xyz"))

    def test_list_all_tools_structure(self):
        """Verifies list_all_tools returns complete specs for all expected tools."""
        specs = list_all_tools()
        self.assertIsInstance(specs, dict)
        for tid in ALL_TOOL_IDS:
            self.assertIn(tid, specs, f"Expected {tid} in list_all_tools()")
            spec = specs[tid]
            self.assertIn("id", spec)
            self.assertIn("name", spec)
            self.assertIn("description", spec)
            self.assertIn("supported_tasks", spec)
            self.assertIn("permitted_parameters", spec)

    def test_tools_proxy_mapping(self):
        """Verifies TOOLS dict-like proxy works seamlessly."""
        self.assertIn("land_cover", TOOLS)
        self.assertIn("rs_vqa_adapted", TOOLS)
        self.assertNotIn("unknown_spec_xyz", TOOLS)

        tool = TOOLS["land_cover"]
        self.assertIsInstance(tool, BigEarthNetTool)

        # .get() test
        t_get = TOOLS.get("building_detection")
        self.assertIsInstance(t_get, BuildingDetectionTool)
        self.assertIsNone(TOOLS.get("missing", None))

        # keys, values, items test
        self.assertTrue(len(list(TOOLS.keys())) >= len(ALL_TOOL_IDS))
        self.assertTrue(len(list(TOOLS.values())) >= len(ALL_TOOL_IDS))
        self.assertTrue(len(list(TOOLS.items())) >= len(ALL_TOOL_IDS))

        with self.assertRaises(KeyError):
            _ = TOOLS["nonexistent_key_123"]


class TestLazyToolProperties(unittest.TestCase):
    def test_land_cover_lazy_classifier(self):
        """BigEarthNetTool must not instantiate classifier until .classifier is accessed."""
        tool = BigEarthNetTool()
        self.assertIsNone(tool._classifier)

        # Setting mock classifier
        mock_cls = MagicMock()
        tool.classifier = mock_cls
        self.assertIs(tool.classifier, mock_cls)

    def test_building_detection_lazy_detector(self):
        """BuildingDetectionTool must not instantiate detector until .detector is accessed."""
        tool = BuildingDetectionTool()
        self.assertIsNone(tool._detector)

        # Setting mock detector
        mock_det = MagicMock()
        tool.detector = mock_det
        self.assertIs(tool.detector, mock_det)

    def test_dependency_injection_in_constructors(self):
        """Tools allow injection of mock services for test isolation."""
        mock_cls = MagicMock()
        tool1 = BigEarthNetTool(classifier=mock_cls)
        self.assertIs(tool1.classifier, mock_cls)

        mock_det = MagicMock()
        tool2 = BuildingDetectionTool(detector=mock_det)
        self.assertIs(tool2.detector, mock_det)


class TestMemoryUnloadHooks(unittest.TestCase):
    def test_building_detector_unload(self):
        """BuildingDetector unload_model must set _model to None and is_available to False."""
        detector = BuildingDetector.get_instance()
        # Save previous state
        orig_model = detector._model
        orig_avail = detector.is_available

        detector.unload_model()
        self.assertIsNone(detector._model)
        self.assertFalse(detector.is_available)

        # Restore original state so as not to affect other tests
        detector._model = orig_model
        detector.is_available = orig_avail

    def test_ben_classifier_unload(self):
        """BENClassifier unload_model must set _model to None and _available to False."""
        classifier = BENClassifier.get_instance()
        orig_model = classifier._model
        orig_avail = classifier._available

        classifier.unload_model()
        self.assertIsNone(classifier._model)
        self.assertFalse(classifier._available)

        # Restore original state
        classifier._model = orig_model
        classifier._available = orig_avail

    def test_rs_adapter_runtime_unload(self):
        """RSAdapterRuntime unload_models must clear model caches without error."""
        runtime = RSAdapterRuntime.get_instance()
        runtime.unload_models()
        self.assertIsNone(runtime._caption_model)
        self.assertIsNone(runtime._caption_processor)
        self.assertIsNone(runtime._vqa_model)
        self.assertIsNone(runtime._vqa_processor)


if __name__ == "__main__":
    unittest.main()
