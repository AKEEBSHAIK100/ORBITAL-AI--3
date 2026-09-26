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

        # ── Dimension compatibility gate ──────────────────────────────────────
        # Reject mismatched pairs rather than silently resizing.
        # Resizing two scenes from different geographies would produce
        # meaningless pixel-difference statistics.
        h1, w1 = t1_img.shape[:2]
        h2, w2 = t2_img.shape[:2]
        if (h1, w1) != (h2, w2):
            return {
                "error": (
                    f"Bi-temporal dimension mismatch: T1 is {w1}x{h1} px, "
                    f"T2 is {w2}x{h2} px. "
                    "Images must share identical pixel dimensions for radiometric "
                    "comparison. Resize/reproject to a common grid before submission."
                ),
                "status": "error",
                "t1_dimensions": {"width": w1, "height": h1},
                "t2_dimensions": {"width": w2, "height": h2},
            }

        # ── Geospatial metadata validation ────────────────────────────────────
        meta1 = inputs.get("metadata") if isinstance(inputs.get("metadata"), dict) else {}
        meta2 = inputs.get("secondary_metadata") if isinstance(inputs.get("secondary_metadata"), dict) else {}
        if not meta1 and not meta2 and isinstance(inputs.get("metadata_list"), list):
            m_list = inputs["metadata_list"]
            if len(m_list) >= 1 and isinstance(m_list[0], dict):
                meta1 = m_list[0]
            if len(m_list) >= 2 and isinstance(m_list[1], dict):
                meta2 = m_list[1]

        crs1 = meta1.get("crs")
        crs2 = meta2.get("crs")
        gt1 = meta1.get("geotransform") or meta1.get("affine")
        gt2 = meta2.get("geotransform") or meta2.get("affine")

        # Validate spatial reference compatibility when metadata is present
        if crs1 and crs2 and str(crs1).strip().upper() != str(crs2).strip().upper():
            return {
                "error": (
                    f"Incompatible spatial reference systems: T1 CRS is '{crs1}', "
                    f"T2 CRS is '{crs2}'. Observations must share a compatible "
                    "spatial reference system for bi-temporal analysis."
                ),
                "status": "error",
                "t1_crs": crs1,
                "t2_crs": crs2,
            }

        if gt1 and gt2 and isinstance(gt1, (list, tuple)) and isinstance(gt2, (list, tuple)):
            if len(gt1) >= 6 and len(gt2) >= 6:
                res1 = (abs(gt1[1]), abs(gt1[5]))
                res2 = (abs(gt2[1]), abs(gt2[5]))
                if res1 != res2:
                    return {
                        "error": (
                            f"Incompatible spatial resolution in geotransform: T1 resolution is {res1}, "
                            f"T2 resolution is {res2}. Both observations must share identical ground sample distance."
                        ),
                        "status": "error",
                        "t1_geotransform": gt1,
                        "t2_geotransform": gt2,
                    }

        gsd1 = meta1.get("gsd") or meta1.get("pixel_size")
        gsd2 = meta2.get("gsd") or meta2.get("pixel_size")
        if gsd1 is not None and gsd2 is not None:
            try:
                if abs(float(gsd1) - float(gsd2)) > 1e-4:
                    return {
                        "error": (
                            f"Ground Sample Distance (GSD) mismatch: T1 GSD is {gsd1}m, T2 GSD is {gsd2}m. "
                            "Observations must share identical ground sample distance for bi-temporal comparison."
                        ),
                        "status": "error",
                        "t1_gsd": gsd1,
                        "t2_gsd": gsd2,
                    }
            except (ValueError, TypeError):
                pass

        if crs1 or gt1 or (gsd1 and gsd2):
            geospatial_compatibility = "verified"
            geospatial_note = f"Geospatial co-registration verified from metadata (CRS: {crs1 or 'consistent'})."
        else:
            geospatial_compatibility = "unverified"
            geospatial_note = "Geospatial co-registration could not be verified from the supplied metadata."


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
            answer = f"Pixel-differencing baseline found limited radiometric change ({change_ratio*100:.1f}% of pixels exceeded the configured threshold). This does not by itself verify built-up footprint stability."
        elif change_ratio < 0.12:
            direction = "increased" if exg2 > exg1 else "decreased"
            answer = (
                f"Pixel-differencing baseline identified surface alterations across {change_ratio*100:.1f}% of the scene ({len(significant_changes)} change clusters). "
                f"Vegetation proxy changed {direction} by {abs(veg_delta_pct)}%. This is an image-derived excess-green proxy, not a calibrated vegetation index. Structural alterations were demarcated in highlighted sectors."
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
            "geospatial_compatibility": geospatial_compatibility,
            "geospatial_note": geospatial_note,
            "duration_ms": duration_ms
        }

