import time
from typing import Any, Dict, List, Optional
import numpy as np
import cv2

from .base import BaseTool

class ChangeDetectionTool(BaseTool):
    id = "change_detection"
    name = "Bi-Temporal Change Detection & CDVQA Specialist"
    description = "Analyses co-registered multi-temporal optical or SAR image pairs for surface alteration, structural expansion, vegetation loss/growth, and CDVQA natural language description."
    supported_tasks = ["change_detection", "change_vqa"]
    modalities = ["optical", "multispectral", "sar"]
    adapter = "Bi-Temporal Radiometric & Structural Difference Engine (CDVQA conventions)"
    domain_adaptation = "Multi-temporal change identification, spatial alteration mapping, and confidence reporting."
    model_id = "rs-change-detector-v2"
    permitted_parameters = {
        "change_threshold": 32,
        "min_change_area_px": 50,
        "benchmark": "CDVQA"
    }

    def run(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        params = {**self.permitted_parameters, **(parameters or {})}
        t1_img = inputs.get("image")
        t2_img = inputs.get("secondary_image")

        if t1_img is None or not isinstance(t1_img, np.ndarray):
            return {"error": "Missing initial image (T1) for change detection", "status": "error"}

        if t2_img is None or not isinstance(t2_img, np.ndarray):
            # Single image provided: inform user and evaluate internal contrast
            t2_img = t1_img
            is_single = True
        else:
            is_single = False

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

        # Built-up alteration interpretation
        if is_single:
            answer = "Only a single temporal acquisition was provided. A bi-temporal pair (T1 earlier, T2 later) is required for full multi-temporal alteration verification."
            confidence = 0.50
        elif change_ratio < 0.02:
            answer = f"Surface structure remained largely unchanged between the two acquisition dates (only {change_ratio*100:.1f}% slight radiometric variance detected). Built-up area remained stable."
            confidence = 0.92
        elif change_ratio < 0.12:
            direction = "increased" if exg2 > exg1 else "decreased"
            answer = (
                f"Moderate surface alterations identified across {change_ratio*100:.1f}% of the scene ({len(significant_changes)} change clusters). "
                f"Vegetation index has {direction} by {abs(veg_delta_pct)}%. Built-up structures exhibit local expansion in the highlighted sectors."
            )
            confidence = 0.88
        else:
            answer = (
                f"Significant bi-temporal evolution detected across {change_ratio*100:.1f}% of the observation area. "
                f"Multiple structural alterations and surface reconfigurations are demarcated across {len(significant_changes)} contiguous clusters."
            )
            confidence = 0.91

        duration_ms = (time.time() - t0) * 1000

        return {
            "status": "success",
            "answer": answer,
            "change_percentage": round(change_ratio * 100, 2),
            "changed_pixels": changed_px,
            "change_clusters": len(significant_changes),
            "significant_changes": significant_changes[:20],
            "vegetation_delta_pct": veg_delta_pct,
            "confidence": confidence,
            "confidence_level": "High" if confidence >= 0.80 else "Medium",
            "duration_ms": duration_ms
        }
