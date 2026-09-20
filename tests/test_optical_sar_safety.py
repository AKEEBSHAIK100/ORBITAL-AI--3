"""
tests/test_optical_sar_safety.py

Focused tests for Optical-SAR correctness hardening:
1. CHECK 1: No silent pixel alignment (no resize/warp/crop for SSIM or cross-correlation without verified registration).
2. CHECK 2: Truthful SAR terminology (raw amplitude signal level, not radiometrically calibrated sigma-nought backscatter).
"""
import sys
import unittest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.fusion_service import (
    analyze_fusion_pair,
    compute_cross_modal_metrics,
    extract_optical_features,
    extract_sar_features,
    ensure_same_size,
)
from backend.tools.optical_sar import OpticalSARTool
from backend.agents.synthesizer import synthesize_response
from backend.schemas.analysis import (
    QueryPlan,
    SpecialistEvidenceObject,
)


class TestOpticalSARSafety(unittest.TestCase):

    def setUp(self):
        self.tool = OpticalSARTool()
        # Simulated optical (679x536x3) and SAR (531x518) like the real test pair
        self.opt_679x536 = np.full((679, 536, 3), 120, dtype=np.uint8)
        self.sar_531x518 = np.full((531, 518), 70, dtype=np.uint8)

        # Same dimension test pair
        self.opt_256 = np.full((256, 256, 3), 100, dtype=np.uint8)
        self.sar_256 = np.full((256, 256), 60, dtype=np.uint8)

    # -------------------------------------------------------------
    # CHECK 1: NO SILENT PIXEL ALIGNMENT
    # -------------------------------------------------------------

    def test_ensure_same_size_does_not_silently_resize(self):
        """ensure_same_size must NOT resize either image."""
        orig_sar_shape = self.sar_531x518.shape
        a, b = ensure_same_size(self.opt_679x536, self.sar_531x518)
        self.assertEqual(b.shape, orig_sar_shape, "ensure_same_size must not resize img_b")
        self.assertEqual(a.shape, self.opt_679x536.shape)

    def test_mismatched_dimensions_rejects_pixel_alignment(self):
        """Mismatched dimension pair must NOT claim pixel-level similarity."""
        metrics = compute_cross_modal_metrics(self.opt_679x536, self.sar_531x518, is_coregistered=True)
        self.assertEqual(metrics["alignment_status"], "unavailable")
        self.assertFalse(metrics["registration_verified"])
        self.assertIsNone(metrics["structural_similarity"])
        self.assertIsNone(metrics["cross_correlation"])
        self.assertEqual(metrics["fusion_confidence"], "unavailable")
        self.assertIn("Silent pixel alignment is disabled", metrics["message"])

    def test_unverified_registration_same_dimensions_rejects_pixel_alignment(self):
        """Even with matching dimensions, unverified registration must NOT claim pixel-level SSIM."""
        metrics = compute_cross_modal_metrics(self.opt_256, self.sar_256, is_coregistered=False)
        self.assertEqual(metrics["alignment_status"], "unavailable")
        self.assertFalse(metrics["registration_verified"])
        self.assertIsNone(metrics["structural_similarity"])
        self.assertIsNone(metrics["cross_correlation"])

    def test_verified_registration_same_dimensions_computes_metrics(self):
        """When explicitly verified and dimensions match, pixel-level metrics are computed."""
        metrics = compute_cross_modal_metrics(self.opt_256, self.sar_256, is_coregistered=True)
        self.assertEqual(metrics["alignment_status"], "verified")
        self.assertTrue(metrics["registration_verified"])
        self.assertIsNotNone(metrics["structural_similarity"])
        self.assertIsNotNone(metrics["cross_correlation"])
        self.assertIsInstance(metrics["structural_similarity"], float)
        self.assertIsInstance(metrics["cross_correlation"], float)

    def test_independent_modality_statistics_preserved_for_differing_grids(self):
        """Differing grids must still compute independent optical and SAR statistics."""
        fusion = analyze_fusion_pair(self.opt_679x536, self.sar_531x518, is_coregistered=False)
        # Optical statistics exist
        opt = fusion["optical"]
        self.assertIn("vegetation_fraction", opt)
        self.assertIn("water_fraction", opt)
        self.assertIn("built_up_fraction", opt)
        self.assertIn("texture_entropy", opt)

        # SAR statistics exist
        sar = fusion["sar"]
        self.assertIn("mean_signal_level_db", sar)
        self.assertIn("speckle_index", sar)
        self.assertIn("edge_density", sar)

        # Cross-modal alignment is safely unavailable
        cm = fusion["cross_modal"]
        self.assertIsNone(cm["structural_similarity"])
        self.assertIsNone(cm["cross_correlation"])

    # -------------------------------------------------------------
    # CHECK 2: SAR TERMINOLOGY
    # -------------------------------------------------------------

    def test_sar_features_uses_raw_amplitude_signal_level(self):
        """extract_sar_features must declare uncalibrated raw amplitude signal level."""
        sar_res = extract_sar_features(self.sar_531x518)
        self.assertIn("mean_signal_level_db", sar_res)
        self.assertEqual(sar_res["calibration_status"], "uncalibrated_raw_amplitude")
        self.assertIn("not radiometrically calibrated", sar_res["description"].lower())

    def test_tool_interpretation_truthful_terminology(self):
        """OpticalSARTool interpretation must not claim calibrated backscatter or unverified SSIM."""
        result = self.tool.run({
            "optical_image": self.opt_679x536,
            "sar_image": self.sar_531x518,
            "metadata": {"crs": "EPSG:32631"},
            "secondary_metadata": {}
        })
        self.assertEqual(result["status"], "success")
        interpretation = result["interpretation"]

        # Truthful SAR terminology
        self.assertIn("SAR mean signal level derived from raw amplitude", interpretation)
        self.assertIn("uncalibrated to sigma-nought backscatter", interpretation)
        self.assertNotIn("SAR microwave backscatter measures", interpretation)

        # Cross-modal alignment safely reported as unavailable
        self.assertIn("Cross-modal pixel alignment is unavailable because spatial co-registration is unverified", interpretation)
        self.assertIn("silent pixel alignment is disabled", interpretation)
        self.assertNotIn("Cross-modal structural alignment: SSIM =", interpretation)

    def test_synthesizer_truthful_synthesis_unregistered_pair(self):
        """Synthesizer must report pixel alignment unavailable and raw amplitude signal level."""
        tool_res = self.tool.run({
            "optical_image": self.opt_679x536,
            "sar_image": self.sar_531x518,
        })
        ev = SpecialistEvidenceObject(
            task="optical_sar_analysis",
            result=tool_res["interpretation"],
            evidence={"metrics": tool_res["metrics"]},
            source="optical_sar_fusion",
            model="classical-cv-fusion-engine-v2",
        )
        plan = QueryPlan(
            intent="optical_sar_analysis",
            required_modalities=["optical", "sar"],
            execution_order=["optical_sar_fusion"],
            evidence_requirements=["optical_features", "sar_features"],
        )
        val = {
            "is_valid": True,
            "compatibility": "warning",
            "warnings": ["Dimension disparity"],
            "notes": [],
            "errors": [],
            "declared_modality": "optical,sar",
            "normalized_modalities": ["optical", "sar"],
            "image_count": 2,
            "has_crs": False,
            "has_geotransforms": False,
            "task_type": "sar_optical_fusion",
            "dimensions": [],
        }

        ans, conf, conf_status, warnings = synthesize_response(
            "Compare optical and SAR observations.",
            plan,
            [ev],
            val,
            status="SUCCESS",
        )

        self.assertIn("SAR mean signal level derived from raw amplitude", ans)
        self.assertIn("uncalibrated to sigma-nought backscatter", ans)
        self.assertIn("Cross-modal pixel alignment is unavailable", ans)
        self.assertNotIn("SAR mean backscatter is", ans)
        self.assertNotIn("Structural similarity (SSIM) between sensors is", ans)


if __name__ == "__main__":
    unittest.main()
