import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

def filter_and_postprocess_detections(
    detections: List[Dict[str, Any]],
    img_width: int,
    img_height: int,
    min_area: float = 60.0,
    max_area_ratio: float = 0.35,
    max_aspect_ratio: float = 4.0,
    min_confidence: float = 0.15
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Filter raw detections to remove false positives (roads, cars, trees, shadows, artifacts)
    and tag edge buildings according to the 50% boundary visibility rule.
    """
    total_image_area = img_width * img_height
    max_allowed_area = total_image_area * max_area_ratio

    filtered: List[Dict[str, Any]] = []
    high_conf_count = 0
    med_conf_count = 0
    low_conf_count = 0
    partial_count = 0

    confidences: List[float] = []

    for d in detections:
        conf = float(d["confidence"])
        if conf < min_confidence:
            continue

        box = d["bbox"]
        gx1, gy1, gx2, gy2 = box
        bw = max(1.0, gx2 - gx1)
        bh = max(1.0, gy2 - gy1)
        area = bw * bh

        # Filter microscopic artifacts or vehicle-sized noise
        if area < min_area:
            continue

        # Filter huge areas (e.g. road networks or whole fields mistakenly grouped)
        if area > max_allowed_area:
            continue

        # Aspect ratio filter: buildings are roughly compact polygons, not extremely long lines (roads/fences)
        aspect = max(bw / bh, bh / bw)
        if aspect > max_aspect_ratio:
            continue

        # Calculate polygon area if available
        poly = d.get("polygon", [])
        actual_area = area
        if len(poly) >= 3:
            pts = np.array(poly, dtype=np.float32)
            contour_area = cv2.contourArea(pts)
            if contour_area > 10.0:
                actual_area = contour_area

        # Edge object rule (Task 6):
        # Determine if detection touches or cuts image boundary
        border_margin = 3.0
        touches_left = gx1 <= border_margin
        touches_top = gy1 <= border_margin
        touches_right = gx2 >= (img_width - border_margin)
        touches_bottom = gy2 >= (img_height - border_margin)
        touches_border = touches_left or touches_top or touches_right or touches_bottom

        is_partial = False
        if touches_border:
            # Check if estimated center is inside and more than ~50% of footprint is visible
            cx, cy = d["centroid"]
            # If centroid is too close to border or outside, marked partial
            if cx < (bw * 0.35) or cx > (img_width - bw * 0.35) or cy < (bh * 0.35) or cy > (img_height - bh * 0.35):
                is_partial = True
                partial_count += 1

        # Confidence tier is a model-score bucket, not a validated accuracy estimate.
        # For aerial building models on CPU, 0.45+ is high, 0.28–0.45 is medium
        if conf >= 0.45:
            conf_tier = "high"
            high_conf_count += 1
        elif conf >= 0.28:
            conf_tier = "medium"
            med_conf_count += 1
        else:
            conf_tier = "low"
            low_conf_count += 1

        confidences.append(conf)

        filtered.append({
            **d,
            "area": round(float(actual_area), 1),
            "aspect_ratio": round(float(aspect), 2),
            "confidence_tier": conf_tier,
            "is_partial": is_partial,
            "touches_border": touches_border,
        })

    # Sort final detections top-to-bottom, left-to-right for consistent numbering
    filtered.sort(key=lambda x: (x["centroid"][1], x["centroid"][0]))

    # Assign unique IDs (B001, B002, ...)
    for idx, item in enumerate(filtered):
        item["id"] = f"B{idx + 1:03d}"

    # Overall score summarizes model confidence only; it is not external validation.
    overall_score = float(np.mean(confidences)) if confidences else None
    if overall_score is None:
        overall_level = None
    elif overall_score >= 0.45:
        overall_level = "High"
    elif overall_score >= 0.28:
        overall_level = "Medium"
    else:
        overall_level = "Low"

    stats = {
        "building_count": len(filtered),
        "high_confidence_count": high_conf_count,
        "medium_confidence_count": med_conf_count,
        "low_confidence_count": low_conf_count,
        "partial_count": partial_count,
        "confidence": round(overall_score, 2) if overall_score is not None else None,
        "confidence_level": overall_level,
        "validation_status": "Model confidence distribution (unverified against external ground truth)",
    }

    return filtered, stats
