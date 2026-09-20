"""
tests/test_change_detection_safety.py

Focused tests for the bi-temporal dimension-mismatch rejection in
backend/tools/change_detection.py.

Covers:
- matching-size pair        -> success (no resize, algorithm runs)
- mismatched-size pair      -> error (explicit rejection, no resize)
- missing T2                -> error (original behaviour preserved)
- missing T1                -> error (original behaviour preserved)
- no cv2.resize in tool src -> confirmed by source inspection
"""
import sys
import inspect
import unittest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.tools.change_detection import ChangeDetectionTool


class TestChangeDetectionSafety(unittest.TestCase):

    def setUp(self):
        self.tool = ChangeDetectionTool()
        self.t1_100 = np.full((100, 100, 3), 50,  dtype=np.uint8)
        self.t2_100 = np.full((100, 100, 3), 120, dtype=np.uint8)
        self.t2_200x150 = np.full((150, 200, 3), 80, dtype=np.uint8)
        self.t2_101x100 = np.full((100, 101, 3), 80, dtype=np.uint8)

    def test_matching_dimensions_accepted(self):
        result = self.tool.run({"image": self.t1_100, "secondary_image": self.t2_100})
        self.assertEqual(result.get("status"), "success",
                         f"Expected success for matching pair, got: {result}")
        self.assertIn("change_percentage", result)
        self.assertIn("method", result)
        self.assertIn("geospatial_note", result)

    def test_identical_images_zero_change(self):
        t = np.full((80, 80, 3), 100, dtype=np.uint8)
        result = self.tool.run({"image": t, "secondary_image": t.copy()})
        self.assertEqual(result.get("status"), "success")
        self.assertLessEqual(result.get("change_percentage", 999), 5.0)

    def test_mismatched_large_rejected(self):
        result = self.tool.run({"image": self.t1_100, "secondary_image": self.t2_200x150})
        self.assertEqual(result.get("status"), "error",
                         "Mismatched dimensions must produce status=error")
        self.assertIn("mismatch", result.get("error", "").lower())
        self.assertIn("t1_dimensions", result)
        self.assertIn("t2_dimensions", result)
        self.assertEqual(result["t1_dimensions"], {"width": 100, "height": 100})
        self.assertEqual(result["t2_dimensions"], {"width": 200, "height": 150})

    def test_mismatched_one_pixel_rejected(self):
        result = self.tool.run({"image": self.t1_100, "secondary_image": self.t2_101x100})
        self.assertEqual(result.get("status"), "error",
                         "Even a 1-pixel dimension difference must be rejected")
        self.assertIn("mismatch", result.get("error", "").lower())
        self.assertEqual(result["t2_dimensions"]["width"], 101)

    def test_missing_t2_returns_error(self):
        result = self.tool.run({"image": self.t1_100})
        self.assertEqual(result.get("status"), "error")
        self.assertIn("secondary image", result.get("error", "").lower())

    def test_missing_t1_returns_error(self):
        result = self.tool.run({"secondary_image": self.t2_100})
        self.assertEqual(result.get("status"), "error")
        self.assertIn("t1", result.get("error", "").lower())

    def test_no_resize_in_source(self):
        source = inspect.getsource(self.tool.run)
        self.assertNotIn(
            "cv2.resize", source,
            "cv2.resize must not appear in ChangeDetectionTool.run(). "
            "Mismatched pairs must be rejected, never silently resampled."
        )

    def test_success_confidence_null(self):
        result = self.tool.run({"image": self.t1_100, "secondary_image": self.t2_100})
        self.assertIsNone(result.get("confidence"))
        self.assertEqual(result.get("confidence_level"), "UNAVAILABLE")
        self.assertEqual(result.get("confidence_source"), "classical_cv_differencing")

    def test_mismatch_result_does_not_modify_inputs(self):
        t2_copy = self.t2_200x150.copy()
        self.tool.run({"image": self.t1_100, "secondary_image": self.t2_200x150})
        np.testing.assert_array_equal(
            self.t2_200x150, t2_copy,
            err_msg="T2 array must not be modified on dimension-mismatch rejection"
        )

    def test_metadata_unavailable_reported(self):
        result = self.tool.run({"image": self.t1_100, "secondary_image": self.t2_100})
        self.assertEqual(result.get("status"), "success")
        self.assertEqual(result.get("geospatial_compatibility"), "unverified")
        self.assertIn("could not be independently verified", result.get("geospatial_note", ""))

    def test_metadata_compatible_crs_accepted(self):
        inputs = {
            "image": self.t1_100,
            "secondary_image": self.t2_100,
            "metadata": {"crs": "EPSG:32632"},
            "secondary_metadata": {"crs": "EPSG:32632"},
        }
        result = self.tool.run(inputs)
        self.assertEqual(result.get("status"), "success")
        self.assertEqual(result.get("geospatial_compatibility"), "verified")
        self.assertIn("EPSG:32632", result.get("geospatial_note", ""))

    def test_metadata_incompatible_crs_rejected(self):
        inputs = {
            "image": self.t1_100,
            "secondary_image": self.t2_100,
            "metadata": {"crs": "EPSG:4326"},
            "secondary_metadata": {"crs": "EPSG:32632"},
        }
        result = self.tool.run(inputs)
        self.assertEqual(result.get("status"), "error")
        self.assertIn("spatial reference", result.get("error", "").lower())
        self.assertEqual(result.get("t1_crs"), "EPSG:4326")
        self.assertEqual(result.get("t2_crs"), "EPSG:32632")

    def test_metadata_incompatible_resolution_rejected(self):
        inputs = {
            "image": self.t1_100,
            "secondary_image": self.t2_100,
            "metadata": {"geotransform": [500000.0, 10.0, 0.0, 4500000.0, 0.0, -10.0]},
            "secondary_metadata": {"geotransform": [500000.0, 20.0, 0.0, 4500000.0, 0.0, -20.0]},
        }
        result = self.tool.run(inputs)
        self.assertEqual(result.get("status"), "error")
        self.assertIn("resolution", result.get("error", "").lower())


if __name__ == "__main__":
    unittest.main()

