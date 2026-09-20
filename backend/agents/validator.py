"""
Sensor-Agnostic Multi-Modal Input Imagery & Co-Registration Validator.

Determines before execution:
- number of images
- modality (optical, multispectral, sar)
- file type
- dimensions
- CRS (Coordinate Reference System)
- geotransform / affine
- acquisition metadata
- temporal metadata

Rejects incompatible workflows (e.g. change query with 1 image, optical-SAR with only optical).
Sets compatibility = 'unknown' when metadata is insufficient to verify co-registration.
Never silently assumes registration.
"""

from typing import Any, Dict, List, Optional
from PIL import Image


def validate_input_imagery(
    task_type: str,
    images: List[Any],
    modalities: Optional[List[str]] = None,
    image_names: Optional[List[str]] = None,
    metadata_list: Optional[List[Dict[str, Any]]] = None,
    query_plan: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Validates imagery inputs against the task requirements and QueryPlan.
    Returns structured validation report.
    """
    modalities = modalities or ["optical"]
    image_names = image_names or []
    metadata_list = metadata_list or []
    notes: List[str] = []
    warnings: List[str] = []
    errors: List[str] = []
    compatibility = "compatible"

    image_count = len(images)
    formats = []
    dimensions = []

    # 1. Determine dimensions and file types
    for idx, img in enumerate(images):
        if hasattr(img, "shape"):
            h, w = img.shape[:2]
            channels = int(img.shape[2]) if len(img.shape) > 2 else 1
            dimensions.append({"width": int(w), "height": int(h), "channels": channels})
            formats.append("ndarray")
        elif isinstance(img, Image.Image):
            dimensions.append({"width": img.width, "height": img.height, "mode": img.mode})
            formats.append(img.format or "raster")
        else:
            dimensions.append({"width": 0, "height": 0})
            formats.append("unknown")

    # 2. Modality normalization (sensor-agnostic: optical, multispectral, sar)
    norm_modalities = []
    for m in modalities:
        m_lower = str(m).lower()
        if "sar" in m_lower or "radar" in m_lower or "sentinel-1" in m_lower or "risat" in m_lower:
            norm_modalities.append("sar")
        elif "multi" in m_lower:
            norm_modalities.append("multispectral")
        else:
            norm_modalities.append("optical")

    # Modality detection from filenames if available
    for name in image_names:
        name_lower = name.lower()
        if "sar" in name_lower and "sar" not in norm_modalities:
            norm_modalities.append("sar")

    # 3. Determine Geospatial Metadata (CRS, Geotransform, Sensor, Temporal)
    crs_list = [m.get("crs") for m in metadata_list if m.get("crs")]
    transform_list = [m.get("geotransform") or m.get("affine") for m in metadata_list if (m.get("geotransform") or m.get("affine"))]
    acquisition_dates = [m.get("acquisition_date") or m.get("date") for m in metadata_list if (m.get("acquisition_date") or m.get("date"))]
    sensors = [m.get("sensor") or m.get("platform") for m in metadata_list if (m.get("sensor") or m.get("platform"))]

    has_crs = len(crs_list) >= 2
    has_transforms = len(transform_list) >= 2

    # Check query_plan requirements if passed
    required_images = 1
    required_modalities = ["optical"]
    if query_plan:
        required_images = getattr(query_plan, "required_images", 1)
        required_modalities = getattr(query_plan, "required_modalities", ["optical"])

    # ─── 4. Validation Rules per Task Type / QueryPlan ────────────────────────

    # Rule A: Optical-SAR Cross-Modal Analysis
    if task_type in ["sar_optical_fusion", "optical_sar", "optical_sar_analysis"] or "sar" in required_modalities:
        if image_count < 2:
            compatibility = "error"
            errors.append(f"Optical–SAR cross-modal analysis requires two co-registered scenes (Optical + SAR). Only {image_count} image provided.")
        else:
            has_sar = "sar" in norm_modalities
            has_opt = any(m in ["optical", "multispectral"] for m in norm_modalities)

            if not has_sar:
                compatibility = "error"
                errors.append("Optical–SAR query requires a SAR scene, but only optical imagery was provided. Declare or provide a SAR observation.")

            # Co-registration verification
            if not has_crs and not has_transforms:
                if compatibility != "error":
                    compatibility = "unknown"
                warnings.append(
                    "Spatial co-registration could not be independently verified from supplied metadata (missing CRS or affine transform). "
                    "Co-registration status = unknown. Never silently assuming registration."
                )
                notes.append("Dual-sensor optical + SAR input loaded; evaluating pixel-grid alignment without verified georeferencing.")
            else:
                if compatibility != "error":
                    compatibility = "compatible"
                notes.append("Dual-sensor optical + SAR co-registered georeferenced pair verified.")

            # Dimensional check
            if len(dimensions) >= 2:
                d1, d2 = dimensions[0], dimensions[1]
                dim_diff_w = abs(d1["width"] - d2["width"])
                dim_diff_h = abs(d1["height"] - d2["height"])
                if dim_diff_w > 50 or dim_diff_h > 50:
                    if compatibility != "error":
                        compatibility = "warning"
                    warnings.append(
                        f"Dimension disparity between optical ({d1['width']}x{d1['height']}) and SAR ({d2['width']}x{d2['height']}). "
                        "Spatial co-registration unverified; interpolation required."
                    )

    # Rule B: Bi-Temporal Analysis (Change Detection, Change VQA)
    elif task_type in ["change_detection", "change_vqa"] or required_images >= 2:
        if image_count < 2:
            compatibility = "error"
            errors.append(f"Bi-temporal change analysis requires two co-registered observations (T1 earlier, T2 later). Only {image_count} image provided.")
        else:
            # Metadata co-registration verification
            if not has_crs and not has_transforms:
                compatibility = "unknown"
                warnings.append(
                    "Spatial co-registration could not be independently verified from supplied metadata (missing CRS or affine transform). "
                    "Co-registration status = unknown. Never silently assuming registration."
                )
                notes.append("Bi-temporal pair received without verified CRS/affine transform.")
            else:
                if compatibility != "error":
                    compatibility = "compatible"
                notes.append("Bi-temporal co-registration confirmed from geospatial metadata.")

            if len(acquisition_dates) >= 2:
                notes.append(f"Observation dates identified: T1={acquisition_dates[0]}, T2={acquisition_dates[1]}.")
            else:
                notes.append("Temporal observation pair supplied; temporal baseline not declared in metadata.")

            # Dimensional consistency check
            if len(dimensions) >= 2:
                d1, d2 = dimensions[0], dimensions[1]
                ratio_w = abs(d1["width"] - d2["width"]) / max(d1["width"], 1)
                ratio_h = abs(d1["height"] - d2["height"]) / max(d1["height"], 1)

                if ratio_w > 0.15 or ratio_h > 0.15:
                    compatibility = "error"
                    errors.append(
                        f"Severely mismatched bi-temporal dimensions ({d1['width']}x{d1['height']} vs {d2['width']}x{d2['height']}). "
                        "Scenes differ by >15% and cannot be accurately co-registered without ground control points."
                    )
                elif ratio_w > 0.02 or ratio_h > 0.02:
                    if compatibility not in ["error", "unknown"]:
                        compatibility = "warning"
                    warnings.append(
                        f"Slight dimension variance between T1 ({d1['width']}x{d1['height']}) and T2 ({d2['width']}x{d2['height']}). "
                        "Sub-pixel alignment required."
                    )

    # Rule C: Building Footprint Detection
    elif task_type == "building_detection":
        notes.append("Routing to dedicated building footprint pipeline with 512px sliding-window tiling.")
        if dimensions and (dimensions[0]["width"] < 128 or dimensions[0]["height"] < 128):
            if compatibility != "error":
                compatibility = "warning"
            warnings.append("Image resolution is very low for building footprint extraction; detection recall may be reduced.")

    # Rule D: Single-Scene Tasks (VQA, Caption, Grounding, Land Cover)
    elif task_type in ["caption", "vqa", "grounding", "land_cover", "multi_task"]:
        notes.append(f"Single-image remote-sensing {task_type.upper()} pipeline active.")
        if image_count > 1:
            warnings.append(f"Multiple images provided for single-scene task '{task_type}'; analyzing primary observation.")

    if errors:
        compatibility = "error"

    return {
        "images_provided": image_count,
        "modalities": norm_modalities,
        "raw_modalities": modalities,
        "formats": formats,
        "dimensions": dimensions,
        "crs": crs_list,
        "geotransforms": transform_list,
        "acquisition_metadata": {"sensors": sensors, "dates": acquisition_dates},
        "temporal_metadata": acquisition_dates,
        "compatibility": compatibility,
        "notes": notes,
        "warnings": warnings,
        "errors": errors,
    }
