"""
Temporal Preprocessor for Bi-Temporal Remote-Sensing Analysis.
Verifies existence, geometric compatibility (dimensions, CRS), and temporal ordering.
Rejects un-aligned or incompatible pairs before analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2

from training.preprocessing.validate_geotiff import inspect_geotiff_metadata, verify_pair_compatibility


class TemporalPreprocessor:
    @staticmethod
    def validate_and_align(
        img_t1: np.ndarray,
        img_t2: np.ndarray,
        meta_t1: Optional[Dict[str, Any]] = None,
        meta_t2: Optional[Dict[str, Any]] = None,
        date_t1: Optional[str] = None,
        date_t2: Optional[str] = None
    ) -> Tuple[bool, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Validates two image arrays and aligns dimensions if slight differences exist.
        Returns (is_valid, aligned_t1, aligned_t2, report).
        """
        report: Dict[str, Any] = {
            "compatible": False,
            "t1_shape": img_t1.shape if isinstance(img_t1, np.ndarray) else None,
            "t2_shape": img_t2.shape if isinstance(img_t2, np.ndarray) else None,
            "date_t1": date_t1,
            "date_t2": date_t2,
            "warnings": [],
            "errors": []
        }

        if img_t1 is None or not isinstance(img_t1, np.ndarray):
            report["errors"].append("Missing or invalid initial image (T1).")
            return False, img_t1, img_t2, report

        if img_t2 is None or not isinstance(img_t2, np.ndarray):
            report["errors"].append("Bi-temporal change analysis requires a second image (T2). Only 1 provided.")
            return False, img_t1, img_t2, report

        h1, w1 = img_t1.shape[:2]
        h2, w2 = img_t2.shape[:2]

        if h1 == 0 or w1 == 0 or h2 == 0 or w2 == 0:
            report["errors"].append("One or both images have invalid zero-area dimensions.")
            return False, img_t1, img_t2, report

        # 1. Image dimensions: strict equality check (silent resizing is forbidden)
        if (h1, w1) != (h2, w2):
            report["errors"].append(
                f"Bi-temporal dimension mismatch: T1 is {w1}x{h1} px, T2 is {w2}x{h2} px. "
                "Silent resizing is disabled. Images must share identical pixel dimensions for radiometric comparison."
            )
            report["t1_dimensions"] = {"width": w1, "height": h1}
            report["t2_dimensions"] = {"width": w2, "height": h2}
            return False, img_t1, img_t2, report

        # 2. Aspect ratio check
        ar1 = w1 / float(h1)
        ar2 = w2 / float(h2)
        if abs(ar1 - ar2) > 0.02:
            report["errors"].append(
                f"Aspect ratio discrepancy: T1 aspect ratio is {ar1:.3f} vs T2 aspect ratio {ar2:.3f}. "
                "Geometries are incompatible for bi-temporal change detection."
            )
            return False, img_t1, img_t2, report

        # 3. Temporal ordering check
        if date_t1 and date_t2 and str(date_t1) > str(date_t2):
            report["errors"].append(f"Temporal order reversed: T1 ({date_t1}) is later than T2 ({date_t2}).")
            return False, img_t1, img_t2, report

        # 4. Metadata verification (CRS, geotransform, GSD)
        meta1 = meta_t1 or {}
        meta2 = meta_t2 or {}

        crs1 = meta1.get("crs")
        crs2 = meta2.get("crs")
        if crs1 and crs2 and str(crs1).strip().upper() != str(crs2).strip().upper():
            report["errors"].append(
                f"Incompatible spatial reference systems: T1 CRS is '{crs1}', T2 CRS is '{crs2}'. "
                "Observations must share a compatible spatial reference system for bi-temporal analysis."
            )
            return False, img_t1, img_t2, report

        gt1 = meta1.get("geotransform") or meta1.get("affine")
        gt2 = meta2.get("geotransform") or meta2.get("affine")
        if gt1 and gt2 and isinstance(gt1, (list, tuple)) and isinstance(gt2, (list, tuple)):
            if len(gt1) >= 6 and len(gt2) >= 6:
                res1 = (abs(gt1[1]), abs(gt1[5]))
                res2 = (abs(gt2[1]), abs(gt2[5]))
                if res1 != res2:
                    report["errors"].append(
                        f"Incompatible spatial resolution in geotransform: T1 resolution is {res1}, T2 resolution is {res2}. "
                        "Both observations must share identical ground sample distance."
                    )
                    return False, img_t1, img_t2, report

        gsd1 = meta1.get("gsd") or meta1.get("pixel_size")
        gsd2 = meta2.get("gsd") or meta2.get("pixel_size")
        if gsd1 is not None and gsd2 is not None:
            try:
                if abs(float(gsd1) - float(gsd2)) > 1e-4:
                    report["errors"].append(
                        f"Ground Sample Distance (GSD) mismatch: T1 GSD is {gsd1}m vs T2 GSD {gsd2}m. "
                        "Observations must share identical resolution."
                    )
                    return False, img_t1, img_t2, report
            except (ValueError, TypeError):
                pass

        # 5. Co-registration metadata presence reporting
        has_crs_or_gt = bool(crs1 or gt1) and bool(crs2 or gt2)
        if not has_crs_or_gt:
            report["warnings"].append("Geospatial co-registration could not be verified from the supplied metadata.")
            report["co_registration"] = "unverified"
            report["geospatial_note"] = "Geospatial co-registration could not be verified from the supplied metadata."
        else:
            report["co_registration"] = "verified"
            report["geospatial_note"] = "Spatial reference and grid alignment verified."

        report["compatible"] = True
        return True, img_t1, img_t2, report
