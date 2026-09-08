import numpy as np
import cv2
from typing import List, Dict, Any

def compute_box_iou(boxA: List[float], boxB: List[float]) -> float:
    """Compute Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    if unionArea <= 0.0:
        return 0.0
    return interArea / unionArea

def compute_mask_iou(polyA: List[List[float]], polyB: List[List[float]], boxA: List[float], boxB: List[float]) -> float:
    """
    Compute fast mask IoU by rasterizing local bounding box region of the two polygons.
    """
    min_x = int(min(boxA[0], boxB[0]))
    min_y = int(min(boxA[1], boxB[1]))
    max_x = int(max(boxA[2], boxB[2])) + 1
    max_y = int(max(boxA[3], boxB[3])) + 1

    rw = max_x - min_x
    rh = max_y - min_y
    if rw <= 0 or rh <= 0 or rw > 2000 or rh > 2000:
        return compute_box_iou(boxA, boxB)

    maskA = np.zeros((rh, rw), dtype=np.uint8)
    maskB = np.zeros((rh, rw), dtype=np.uint8)

    ptsA = np.array([[pt[0] - min_x, pt[1] - min_y] for pt in polyA], dtype=np.int32)
    ptsB = np.array([[pt[0] - min_x, pt[1] - min_y] for pt in polyB], dtype=np.int32)

    if len(ptsA) >= 3:
        cv2.fillPoly(maskA, [ptsA], 1)
    if len(ptsB) >= 3:
        cv2.fillPoly(maskB, [ptsB], 1)

    intersection = np.logical_and(maskA, maskB).sum()
    union = np.logical_or(maskA, maskB).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)

def merge_duplicate_detections(
    detections: List[Dict[str, Any]],
    box_iou_threshold: float = 0.45,
    mask_iou_threshold: float = 0.35,
    centroid_dist_ratio: float = 0.40
) -> List[Dict[str, Any]]:
    """
    Merge duplicate building detections from overlapping tiles using confidence ranking,
    centroid proximity, box IoU, and mask IoU.
    """
    if len(detections) <= 1:
        return detections

    # Sort detections by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept_dets: List[Dict[str, Any]] = []

    for cand in sorted_dets:
        cand_box = cand["bbox"]
        cand_poly = cand.get("polygon", [])
        cand_cx, cand_cy = cand["centroid"]
        cand_w = max(1.0, cand_box[2] - cand_box[0])
        cand_h = max(1.0, cand_box[3] - cand_box[1])
        cand_diag = (cand_w**2 + cand_h**2) ** 0.5

        is_duplicate = False
        for kept in kept_dets:
            kept_box = kept["bbox"]
            kept_cx, kept_cy = kept["centroid"]

            # Centroid distance
            dist = ((cand_cx - kept_cx)**2 + (cand_cy - kept_cy)**2) ** 0.5
            box_iou = compute_box_iou(cand_box, kept_box)

            # Quick reject if centroids are far and no box overlap
            if dist > cand_diag and box_iou < 0.10:
                continue

            # Check mask IoU if available
            mask_iou = 0.0
            if len(cand_poly) >= 3 and len(kept.get("polygon", [])) >= 3:
                mask_iou = compute_mask_iou(cand_poly, kept["polygon"], cand_box, kept_box)
            else:
                mask_iou = box_iou

            # Duplicate criteria:
            # 1. Significant mask overlap
            # 2. Significant bounding box IoU
            # 3. Very close centroids with moderate overlap
            if mask_iou >= mask_iou_threshold or box_iou >= box_iou_threshold:
                is_duplicate = True
                break
            if dist < (cand_diag * centroid_dist_ratio) and (box_iou > 0.25 or mask_iou > 0.20):
                is_duplicate = True
                break

        if not is_duplicate:
            kept_dets.append(cand)

    return kept_dets
