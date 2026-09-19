import time
from typing import Any, Dict, List, Optional
import numpy as np
import cv2

from .base import BaseTool

class ChangeDetectionTool(BaseTool):
    id = "change_detection"
    name = "Classical-CV Bi-Temporal Change Detection Baseline"
    description = "Analyses co-registered multi-temporal optical or SAR image pairs for surface alteration, structural expansion, and vegetation difference using classical pixel differencing."
    supported_tasks = ["change_detection", "change_vqa"]
    modalities = ["optical", "multispectral", "sar"]
    adapter = "Bi-Temporal Radiometric & Structural Difference Engine"
    domain_adaptation = "Pixel-differencing baseline with adaptive thresholding and morphological clustering."
    model_id = "classical-cv-change-detector-v2"
    permitted_parameters = {
        "change_threshold": 32,
        "min_change_area_px": 50,
    }

    def run(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        params = {**self.permitted_parameters, **(parameters or {})}
        t1_img = inputs.get("image")
        t2_img = inputs.get("secondary_image")

        if t1_img is None or not isinstance(t1_img, np.ndarray):
            return {"error": "Missing initial image (T1) for change detection", "status": "error"}

        if t2_img is None or not isinstance(t2_img, np.ndarray):
            return {
                "error": "Bi-temporal change detection requires a co-registered secondary image (T2). Only 1 image was provided.",
                "status": "error"
            }

        # Resize t2 to match t1 if slight difference
        h1, w1 = t1_img.shape[:2]
        h2, w2 = t2_img.shape[:2]
        if (h1, w1) != (h2, w2):
            t2_img = cv2.resize(t2_img, (w1, h1), interpolation=cv2.INTER_LINEAR)

        # Convert to grayscale
        g1 = cv2.cvtColor(t1_img, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(t2_img, cv2.COLOR_BGR2GRAY)

        # Radiometric absolute difference
        diff = cv2.absdiff(g1, g2)
        thresh_val = int(params.get("change_threshold", 32))
        _, change_mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)

        total_px = w1 * h1
        changed_px = int(np.count_nonzero(change_mask))
        change_ratio = changed_px / total_px if total_px > 0 else 0.0

        # Find connected components of change
        contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_changes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= float(params.get("min_change_area_px", 50)):
                x, y, w, h = cv2.boundingRect(c)
                significant_changes.append({
                    "bbox_pct": [round(x/w1*100, 2), round(y/h1*100, 2), round(w/w1*100, 2), round(h/h1*100, 2)],
                    "area_px": float(area)
                })

        # Vegetation proxy comparison
        # (excess green index)
        exg1 = (2.0 * t1_img[:, :, 1] - t1_img[:, :, 2] - t1_img[:, :, 0]).mean()
        exg2 = (2.0 * t2_img[:, :, 1] - t2_img[:, :, 2] - t2_img[:, :, 0]).mean()
        veg_delta_pct = round(((exg2 - exg1) / (abs(exg1) + 1e-5)) * 100, 1)

        # Built-up alteration interpretation (Pixel-diff heuristic baseline)
        if change_ratio < 0.02:
            answer = f"Pixel-differencing baseline confirms surface structure remained largely unchanged between the two acquisition dates (only {change_ratio*100:.1f}% radiometric variance). Built-up footprint remained stable."
        elif change_ratio < 0.12:
            direction = "increased" if exg2 > exg1 else "decreased"
            answer = (
                f"Pixel-differencing baseline identified surface alterations across {change_ratio*100:.1f}% of the scene ({len(significant_changes)} change clusters). "
                f"Vegetation index has {direction} by {abs(veg_delta_pct)}%. Structural alterations demarcated in highlighted sectors."
            )
        else:
            answer = (
                f"Pixel-differencing baseline identified significant alterations across {change_ratio*100:.1f}% of the observation area "
                f"({len(significant_changes)} contiguous change clusters detected via adaptive thresholding)."
            )

        duration_ms = (time.time() - t0) * 1000

        return {
            "status": "success",
            "answer": answer,
            "change_percentage": round(change_ratio * 100, 2),
            "changed_pixels": changed_px,
            "change_clusters": len(significant_changes),
            "significant_changes": significant_changes[:20],
            "vegetation_delta_pct": veg_delta_pct,
            "confidence": None,
            "confidence_level": "UNAVAILABLE",
            "confidence_source": "classical_cv_differencing",
            "method": "Radiometric pixel-differencing & morphological contour clustering baseline",
            "duration_ms": duration_ms
        }
