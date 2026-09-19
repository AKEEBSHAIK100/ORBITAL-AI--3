import time
from typing import Any, Dict, List, Optional
import numpy as np
import cv2

from .base import BaseTool

class GroundingTool(BaseTool):
    id = "grounding"
    name = "Classical-CV Spatial Grounding Baseline"
    description = "Localizes user-specified geographical targets (water bodies, runways, built-up clusters, vegetation corridors) and generates normalized bounding box coordinates using classical spectral thresholding."
    supported_tasks = ["grounding"]
    modalities = ["optical", "multispectral"]
    adapter = "Spectral Thresholding & Connected Component Grounding Baseline"
    domain_adaptation = "Classical spectral indexing and connected component contour demarcation."
    model_id = "classical-cv-grounding-baseline-v2"
    permitted_parameters = {
        "coordinate_system": "normalized_percentage",
        "bbox_format": "[x,y,w,h]",
    }

    def run(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        query = inputs.get("query", "").lower()
        img_bgr = inputs.get("image")

        if img_bgr is None or not isinstance(img_bgr, np.ndarray):
            return {"error": "Missing image for grounding", "status": "error"}

        h, w = img_bgr.shape[:2]
        b, g, r = cv2.split(img_bgr.astype(np.float32))

        # Determine target category from query
        if "water" in query or "lake" in query or "river" in query or "basin" in query or "reservoir" in query:
            target = "water body"
            # Water mask: high blue, lower red/green, or low overall luminance
            mask = ((b > r * 1.08) & (b > g * 0.95)) | ((b > 70) & (r < 65) & (g < 90))
        elif "vegetation" in query or "forest" in query or "tree" in query or "crop" in query or "field" in query:
            target = "vegetation canopy"
            # Green dominance
            mask = (g > r * 1.1) & (g > b * 1.05) & (g > 35)
        elif "road" in query or "highway" in query or "runway" in query:
            target = "transportation corridor / road"
            # Linear high-contrast grayscale structures
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            mask = edges > 0
        else:
            target = "built-up cluster / target structure"
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            mask = (gray > 120) & (gray < 235)

        mask_uint8 = (mask.astype(np.uint8)) * 255
        # Clean small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        # Extract connected components
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_regions = []

        for c in contours:
            area = cv2.contourArea(c)
            if area >= (w * h * 0.005):  # At least 0.5% of the scene
                rx, ry, rw, rh = cv2.boundingRect(c)
                valid_regions.append({
                    "target": target,
                    "region": {
                        "x_percent": round(rx / w * 100, 2),
                        "y_percent": round(ry / h * 100, 2),
                        "w_percent": round(rw / w * 100, 2),
                        "h_percent": round(rh / h * 100, 2),
                    },
                    "confidence": round(min(0.95, 0.70 + (area / (w * h)) * 0.5), 3),
                    "label": f"Demarcated {target.title()}"
                })

        # Sort by area (largest first)
        valid_regions.sort(key=lambda r: r["region"]["w_percent"] * r["region"]["h_percent"], reverse=True)
        top_regions = valid_regions[:3]

        duration_ms = (time.time() - t0) * 1000

        return {
            "status": "success",
            "target": target,
            "regions": top_regions,
            "count": len(top_regions),
            "primary_region": top_regions[0]["region"] if top_regions else None,
            "detected": len(top_regions) > 0,
            "confidence_source": "heuristic",
            "method": "Spectral thresholding & connected component contour extraction",
            "duration_ms": duration_ms
        }
