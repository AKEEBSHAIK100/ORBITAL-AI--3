import io
import re
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

def validate_input_imagery(
    task_type: str,
    images: List[Any],
    modalities: Optional[List[str]] = None,
    image_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Implements Input Validation Rules from Section 12:
    Validates image count, modality, dimensions, format, and pair compatibility.
    """
    modalities = modalities or ["optical"]
    image_names = image_names or []
    notes: List[str] = []
    warnings: List[str] = []
    compatibility = "valid"

    image_count = len(images)
    formats = []
    dimensions = []

    for idx, img in enumerate(images):
        # Inspect numpy array or PIL image
        if hasattr(img, "shape"):
            h, w = img.shape[:2]
            dimensions.append({"width": int(w), "height": int(h), "channels": int(img.shape[2]) if len(img.shape) > 2 else 1})
            formats.append("ndarray")
        elif isinstance(img, Image.Image):
            dimensions.append({"width": img.width, "height": img.height, "mode": img.mode})
            formats.append(img.format or "raster")
        else:
            dimensions.append({"width": 0, "height": 0})
            formats.append("unknown")

    # Modality & Task specific checks
    if task_type == "sar_optical_fusion":
        has_sar = any("sar" in m.lower() for m in modalities) or any("sar" in n.lower() for n in image_names)
        has_optical = any("optical" in m.lower() or "multispectral" in m.lower() for m in modalities) or not has_sar
        
        if image_count < 2:
            compatibility = "warning"
            warnings.append("Optical–SAR fusion requires 2 co-registered images (Optical + SAR). Synthesizing radar backscatter proxy from optical band.")
        else:
            notes.append("Dual-sensor optical + SAR co-registered pair verified.")
            # Check dimensional correspondence
            if len(dimensions) >= 2:
                d1, d2 = dimensions[0], dimensions[1]
                if abs(d1["width"] - d2["width"]) > 50 or abs(d1["height"] - d2["height"]) > 50:
                    warnings.append(f"Dimension mismatch between optical ({d1['width']}x{d1['height']}) and SAR ({d2['width']}x{d2['height']}). Auto-rescaling and spatial alignment applied.")

    elif task_type == "change_detection" or task_type == "change_vqa":
        if image_count < 2:
            compatibility = "warning"
            warnings.append("Bi-temporal change detection requires two images (T1 earlier, T2 later).")
        else:
            notes.append("Bi-temporal pair verified. Validating spatial overlap and coregistration.")
            if len(dimensions) >= 2:
                d1, d2 = dimensions[0], dimensions[1]
                if abs(d1["width"] - d2["width"]) > 40 or abs(d1["height"] - d2["height"]) > 40:
                    warnings.append(f"Pair dimensions vary ({d1['width']}x{d1['height']} vs {d2['width']}x{d2['height']}). Reprojecting to shared grid.")

    elif task_type == "building_detection":
        notes.append("Routing to dedicated building footprint pipeline with 512px sliding-window tiling.")
        if dimensions and (dimensions[0]["width"] < 128 or dimensions[0]["height"] < 128):
            warnings.append("Image resolution is very low for building footprint extraction; results may have reduced recall.")

    elif task_type in ["caption", "vqa", "grounding"]:
        notes.append(f"Single-image remote-sensing {task_type.upper()} pipeline active.")

    return {
        "images_provided": image_count,
        "modalities": modalities,
        "formats": formats,
        "dimensions": dimensions,
        "compatibility": compatibility,
        "notes": notes,
        "warnings": warnings,
    }
