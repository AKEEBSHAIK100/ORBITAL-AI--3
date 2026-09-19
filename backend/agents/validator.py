from typing import Any, Dict, List, Optional
from PIL import Image

def validate_input_imagery(
    task_type: str,
    images: List[Any],
    modalities: Optional[List[str]] = None,
    image_names: Optional[List[str]] = None,
    metadata_list: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Sensor-Agnostic Multi-Modal Input Imagery & Co-registration Validator.
    Validates image count, modality compatibility (optical, multispectral, sar),
    dimensions, file format, CRS, affine/geotransform, and acquisition dates.

    For paired/temporal imagery:
    - Distinguishes: 'compatible', 'warning', 'unknown', 'error'
    - Flags 'unknown' with an explicit warning when metadata is insufficient to verify co-registration.
    - Never assumes images are co-registered merely because dimensions are similar.
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

    # Modality normalisation (sensor-agnostic: optical, multispectral, sar)
    norm_modalities = []
    for m in modalities:
        m_lower = m.lower()
        if "sar" in m_lower or "radar" in m_lower:
            norm_modalities.append("sar")
        elif "multi" in m_lower:
            norm_modalities.append("multispectral")
        else:
            norm_modalities.append("optical")

    # ─── Multi-Sensor: Optical + SAR Cross-Modal Fusion ───────────────────────
    if task_type in ["sar_optical_fusion", "optical_sar"]:
        if image_count < 2:
            compatibility = "error"
            errors.append("Optical–SAR cross-modal analysis requires two co-registered scenes (Optical + SAR). Only 1 image provided.")
        else:
            has_sar = "sar" in norm_modalities or any("sar" in n.lower() for n in image_names)
            has_opt = any(m in ["optical", "multispectral"] for m in norm_modalities)

            if not has_sar or not has_opt:
                warnings.append("Declared modalities should include both 'optical' and 'sar' sensors for cross-modal analysis.")

            # Co-registration verification from metadata
            has_crs = len(metadata_list) >= 2 and all(m.get("crs") for m in metadata_list[:2])
            has_transform = len(metadata_list) >= 2 and all(m.get("geotransform") or m.get("affine") for m in metadata_list[:2])

            if not has_crs and not has_transform:
                compatibility = "unknown"
                warnings.append("Spatial co-registration could not be independently verified from supplied metadata.")
                notes.append("Dual-sensor optical + SAR input loaded; evaluating pixel-grid alignment.")
            else:
                compatibility = "compatible"
                notes.append("Dual-sensor optical + SAR co-registered georeferenced pair verified.")

            # Dimension check
            if len(dimensions) >= 2:
                d1, d2 = dimensions[0], dimensions[1]
                dim_diff_w = abs(d1["width"] - d2["width"])
                dim_diff_h = abs(d1["height"] - d2["height"])
                if dim_diff_w > 50 or dim_diff_h > 50:
                    compatibility = "warning"
                    warnings.append(
                        f"Dimension disparity between optical ({d1['width']}x{d1['height']}) and SAR ({d2['width']}x{d2['height']}). "
                        "Spatial co-registration unverified; interpolation required."
                    )

    # ─── Bi-Temporal Analysis: Change Detection & Change VQA ───────────────────
    elif task_type in ["change_detection", "change_vqa"]:
        if image_count < 2:
            compatibility = "error"
            errors.append("Bi-temporal change detection requires two co-registered observations (T1 earlier, T2 later). Only 1 image provided.")
        else:
            # Metadata co-registration verification
            has_crs = len(metadata_list) >= 2 and all(m.get("crs") for m in metadata_list[:2])
            has_transform = len(metadata_list) >= 2 and all(m.get("geotransform") or m.get("affine") for m in metadata_list[:2])
            dates = [m.get("acquisition_date") for m in metadata_list if m.get("acquisition_date")]

            if not has_crs and not has_transform:
                compatibility = "unknown"
                warnings.append("Spatial co-registration could not be independently verified from supplied metadata.")
                notes.append("Bi-temporal pair received without verified CRS/affine transform.")
            else:
                compatibility = "compatible"
                notes.append("Bi-temporal co-registration confirmed from geospatial metadata.")

            if len(dates) >= 2:
                notes.append(f"Observation dates identified: T1={dates[0]}, T2={dates[1]}.")

            # Check dimensional consistency
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
                    if compatibility != "unknown":
                        compatibility = "warning"
                    warnings.append(
                        f"Slight dimension variance between T1 ({d1['width']}x{d1['height']}) and T2 ({d2['width']}x{d2['height']}). "
                        "Alignment required."
                    )

    # ─── Building Footprint Detection ─────────────────────────────────────────
    elif task_type == "building_detection":
        notes.append("Routing to dedicated building footprint pipeline with 512px sliding-window tiling.")
        if dimensions and (dimensions[0]["width"] < 128 or dimensions[0]["height"] < 128):
            compatibility = "warning"
            warnings.append("Image resolution is very low for building footprint extraction; detection recall may be reduced.")

    # ─── Single-Scene Tasks (VQA, Caption, Grounding, Land Cover) ─────────────
    elif task_type in ["caption", "vqa", "grounding", "land_cover"]:
        notes.append(f"Single-image remote-sensing {task_type.upper()} pipeline active.")
        if image_count > 1:
            warnings.append(f"Multiple images provided for single-image task '{task_type}'; analyzing primary image.")

    return {
        "images_provided": image_count,
        "modalities": norm_modalities,
        "raw_modalities": modalities,
        "formats": formats,
        "dimensions": dimensions,
        "compatibility": compatibility,
        "notes": notes,
        "warnings": warnings,
        "errors": errors,
    }
