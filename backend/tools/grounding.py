import time
from typing import Any, Dict, List, Optional
import numpy as np
import cv2

from .base import BaseTool


# Supported classical target definitions
SUPPORTED_TARGET_SPECS = {
    "water": {
        "label": "water body",
        "target_key": "water",
        "aliases": ["water", "water body", "river", "lake", "ocean", "pond", "basin", "reservoir", "stream", "sea", "canal"],
    },
    "vegetation": {
        "label": "vegetation canopy",
        "target_key": "vegetation",
        "aliases": ["vegetation", "canopy", "forest", "tree", "trees", "crop", "crops", "field", "fields", "greenery", "grass"],
    },
    "road": {
        "label": "transportation corridor / road",
        "target_key": "road",
        "aliases": ["road", "roads", "highway", "highways", "runway", "runways", "street", "streets", "transportation corridor"],
    },
    "building": {
        "label": "built-up cluster / target structure",
        "target_key": "building",
        "aliases": ["building", "buildings", "house", "houses", "structure", "structures", "rooftop", "rooftops", "built-up"],
    }
}

UNSUPPORTED_GROUNDING_KEYWORDS = [
    "airplane", "airplanes", "aircraft", "car", "cars", "vehicle", "vehicles",
    "truck", "trucks", "ship", "ships", "boat", "boats", "vessel", "vessels",
    "solar panel", "solar panels", "swimming pool", "swimming pools", "pool",
    "crane", "cranes", "helicopter", "helicopters", "tank", "tanks", "oil tank",
    "storage tank", "stadium", "tennis court", "golf course", "person", "people",
    "alien", "fence", "power line", "wind turbine"
]


class GroundingTool(BaseTool):
    id = "grounding"
    name = "Classical-CV Spatial Grounding Baseline"
    description = (
        "Localizes user-specified geographical targets (water bodies, vegetation corridors, "
        "transportation corridors/roads, built-up clusters) and generates normalized bounding box "
        "coordinates using classical spectral thresholding and connected component contour demarcation."
    )
    supported_tasks = ["grounding"]
    modalities = ["optical", "multispectral"]
    adapter = "Spectral Thresholding & Connected Component Grounding Baseline"
    domain_adaptation = "Classical spectral indexing and connected component contour demarcation."
    model_id = "classical-cv-grounding-baseline-v2"
    permitted_parameters = {
        "coordinate_system": "normalized_percentage",
        "bbox_format": "[x,y,w,h]",
    }

    def _extract_target_mask(self, target_key: str, img_bgr: np.ndarray, b: np.ndarray, g: np.ndarray, r: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        if target_key == "water":
            # Water mask: high blue, lower red/green, or low overall luminance
            mask = ((b > r * 1.08) & (b > g * 0.95)) | ((b > 70) & (r < 65) & (g < 90))
        elif target_key == "vegetation":
            # Green dominance
            mask = (g > r * 1.1) & (g > b * 1.05) & (g > 35)
        elif target_key == "road":
            # Linear high-contrast grayscale structures
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            mask = edges > 0
        else:  # building / built-up
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            mask = (gray > 120) & (gray < 235)

        mask_uint8 = (mask.astype(np.uint8)) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        return cleaned

    def run(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        query = inputs.get("query", "").lower()
        img_bgr = inputs.get("image")

        if img_bgr is None or not isinstance(img_bgr, np.ndarray):
            return {"error": "Missing image for grounding", "status": "error"}

        h, w = img_bgr.shape[:2]
        b, g, r = cv2.split(img_bgr.astype(np.float32))

        # 1. Identify supported targets requested in query (ordered by appearance)
        detected_target_keys: List[str] = []
        for cat_key, cat_spec in SUPPORTED_TARGET_SPECS.items():
            for alias in cat_spec["aliases"]:
                if alias in query:
                    if cat_key not in detected_target_keys:
                        detected_target_keys.append(cat_key)
                    break

        # 2. Identify unsupported targets requested in query
        detected_unsupported: List[str] = []
        for unsupp_kw in UNSUPPORTED_GROUNDING_KEYWORDS:
            if unsupp_kw in query and unsupp_kw not in detected_unsupported:
                detected_unsupported.append(unsupp_kw)

        # No target means the planner/tool contract is incomplete; never invent a target.
        if not detected_target_keys and not detected_unsupported:
            return {"status": "unsupported", "error": "No supported grounding target was identified in the query.", "regions": [], "targets": []}

        targets_output: List[Dict[str, Any]] = []
        all_regions: List[Dict[str, Any]] = []

        # 3. Process each supported target distinctly
        for target_key in detected_target_keys:
            spec = SUPPORTED_TARGET_SPECS[target_key]
            label = spec["label"]
            cleaned_mask = self._extract_target_mask(target_key, img_bgr, b, g, r)

            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_regions: List[Dict[str, Any]] = []

            for c in contours:
                area = cv2.contourArea(c)
                if area >= (w * h * 0.005):  # At least 0.5% of the scene
                    rx, ry, rw, rh = cv2.boundingRect(c)
                    valid_regions.append({
                        "target": label,
                        "region": {
                            "x_percent": round(rx / w * 100, 2),
                            "y_percent": round(ry / h * 100, 2),
                            "w_percent": round(rw / w * 100, 2),
                            "h_percent": round(rh / h * 100, 2),
                        },
                        "confidence": None,
                        "label": f"Demarcated {label.title()}"
                    })

            valid_regions.sort(key=lambda item: item["region"]["w_percent"] * item["region"]["h_percent"], reverse=True)
            top_for_target = valid_regions[:3]
            all_regions.extend(top_for_target)

            targets_output.append({
                "label": label,
                "target": target_key,
                "regions": top_for_target,
                "count": len(top_for_target),
                "detected": len(top_for_target) > 0,
                "status": "success",
            })

        # 4. Record unsupported targets explicitly without inventing regions
        for unsupp in detected_unsupported:
            targets_output.append({
                "label": unsupp,
                "target": unsupp,
                "regions": [],
                "count": 0,
                "detected": False,
                "status": "unsupported",
                "warning": (
                    f"Target entity '{unsupp}' is not supported by classical spectral grounding. "
                    "Classical grounding supports: water bodies, vegetation canopy, transportation corridors/roads, and built-up structures."
                )
            })

        duration_ms = (time.time() - t0) * 1000

        # Sort all aggregated regions by area
        all_regions.sort(key=lambda item: item["region"]["w_percent"] * item["region"]["h_percent"], reverse=True)
        primary_target_label = targets_output[0]["label"] if targets_output else "target"

        return {
            "status": "success",
            "target": primary_target_label if len(detected_target_keys) <= 1 else ", ".join(t["label"] for t in targets_output if t["status"] == "success"),
            "targets": targets_output,
            "regions": all_regions,
            "count": len(all_regions),
            "primary_region": all_regions[0]["region"] if all_regions else None,
            "detected": len(all_regions) > 0,
            "unsupported_targets": detected_unsupported,
            "confidence_source": "heuristic",
            "method": "Spectral thresholding & connected component contour extraction",
            "duration_ms": duration_ms
        }

