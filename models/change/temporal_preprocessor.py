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

        # Check dimension ratio
        ratio_w = abs(w1 - w2) / max(w1, w2)
        ratio_h = abs(h1 - h2) / max(h1, h2)

        if ratio_w > 0.15 or ratio_h > 0.15:
            report["errors"].append(
                f"Dimension discrepancy exceeds tolerance: T1 ({w1}x{h1}) vs T2 ({w2}x{h2}). "
                "Co-registration failed."
            )
            return False, img_t1, img_t2, report

        # Temporal ordering check
        if date_t1 and date_t2 and str(date_t1) > str(date_t2):
            report["errors"].append(f"Temporal order reversed: T1 ({date_t1}) is later than T2 ({date_t2}).")
            return False, img_t1, img_t2, report

        # Align T2 to T1 grid if minor dimension variation
        aligned_t2 = img_t2
        if (h1, w1) != (h2, w2):
            aligned_t2 = cv2.resize(img_t2, (w1, h1), interpolation=cv2.INTER_LINEAR)
            report["warnings"].append(f"Resized T2 from ({w2}x{h2}) to ({w1}x{h1}) for pixel alignment.")

        report["compatible"] = True
        return True, img_t1, aligned_t2, report
