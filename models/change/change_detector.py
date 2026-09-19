"""
Bi-Temporal Change Detector Specialist.
Performs pixel-level alteration extraction, connected-component clustering,
and directional spectral variance calculation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from .temporal_preprocessor import TemporalPreprocessor


class ChangeDetector:
    def __init__(self, change_threshold: int = 32, min_area_px: int = 50):
        self.change_threshold = change_threshold
        self.min_area_px = min_area_px

    def detect_changes(
        self,
        img_t1: np.ndarray,
        img_t2: np.ndarray,
        meta_t1: Optional[Dict[str, Any]] = None,
        meta_t2: Optional[Dict[str, Any]] = None,
        date_t1: Optional[str] = None,
        date_t2: Optional[str] = None
    ) -> Dict[str, Any]:
        t0 = time.time()

        valid, t1, t2, prep_report = TemporalPreprocessor.validate_and_align(
            img_t1, img_t2, meta_t1=meta_t1, meta_t2=meta_t2, date_t1=date_t1, date_t2=date_t2
        )
        if not valid:
            return {
                "success": False,
                "error": "; ".join(prep_report.get("errors", ["Incompatible temporal pair"])),
                "temporal_report": prep_report
            }

        h, w = t1.shape[:2]
        g1 = cv2.cvtColor(t1, cv2.COLOR_BGR2GRAY) if len(t1.shape) == 3 else t1
        g2 = cv2.cvtColor(t2, cv2.COLOR_BGR2GRAY) if len(t2.shape) == 3 else t2

        diff = cv2.absdiff(g1, g2)
        _, mask = cv2.threshold(diff, self.change_threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        total_px = w * h
        changed_px = int(np.count_nonzero(cleaned))
        change_pct = round((changed_px / total_px) * 100.0, 2) if total_px > 0 else 0.0

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clusters: List[Dict[str, Any]] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_area_px:
                rx, ry, rw, rh = cv2.boundingRect(c)
                clusters.append({
                    "bbox_pct": [
                        round(rx / w * 100, 2),
                        round(ry / h * 100, 2),
                        round(rw / w * 100, 2),
                        round(rh / h * 100, 2)
                    ],
                    "area_px": float(area)
                })

        # Vegetation delta (Excess Green Index)
        exg1 = float((2.0 * t1[:, :, 1] - t1[:, :, 2] - t1[:, :, 0]).mean()) if len(t1.shape) == 3 else 0.0
        exg2 = float((2.0 * t2[:, :, 1] - t2[:, :, 2] - t2[:, :, 0]).mean()) if len(t2.shape) == 3 else 0.0
        veg_delta_pct = round(((exg2 - exg1) / (abs(exg1) + 1e-5)) * 100.0, 1)

        duration_ms = (time.time() - t0) * 1000

        return {
            "success": True,
            "change_percentage": change_pct,
            "change_clusters_count": len(clusters),
            "clusters": clusters[:20],
            "vegetation_delta_pct": veg_delta_pct,
            "duration_ms": duration_ms,
            "temporal_validation": prep_report
        }
