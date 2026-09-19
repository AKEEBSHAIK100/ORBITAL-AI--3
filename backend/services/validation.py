from typing import Dict, Any, Optional, List

def evaluate_accuracy(
    predicted_count: int,
    ground_truth_count: Optional[int] = None,
    ground_truth_bboxes: Optional[List[List[float]]] = None,
    predicted_bboxes: Optional[List[List[float]]] = None
) -> Dict[str, Any]:
    """
    Compute accuracy metrics against ground truth data if available.
    If no ground truth data is supplied, returns transparent disclaimer.
    """
    if ground_truth_count is None:
        return {
            "has_ground_truth": False,
            "status": "Model confidence distribution (unverified against external ground truth)",
            "message": "No ground truth annotations provided. Confidence represents model certainty, not verified real-world accuracy.",
            "predicted_count": predicted_count,
            "ground_truth_count": None,
            "absolute_error": None,
            "percentage_error": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
        }

    abs_error = abs(predicted_count - ground_truth_count)
    pct_error = (abs_error / ground_truth_count * 100.0) if ground_truth_count > 0 else 0.0

    # If bounding boxes are provided, calculate precision, recall, and F1
    precision = None
    recall = None
    f1 = None

    if ground_truth_bboxes and predicted_bboxes:
        tp = 0
        matched_gt = set()
        for pbox in predicted_bboxes:
            for gi, gbox in enumerate(ground_truth_bboxes):
                if gi in matched_gt:
                    continue
                # Compute IoU
                xA = max(pbox[0], gbox[0])
                yA = max(pbox[1], gbox[1])
                xB = min(pbox[2], gbox[2])
                yB = min(pbox[3], gbox[3])
                inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                areaP = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1])
                areaG = (gbox[2] - gbox[0]) * (gbox[3] - gbox[1])
                union = areaP + areaG - inter
                iou = inter / union if union > 0 else 0
                if iou >= 0.50:
                    tp += 1
                    matched_gt.add(gi)
                    break

        fp = len(predicted_bboxes) - tp
        fn = len(ground_truth_bboxes) - tp
        precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
        f1 = round((2 * precision * recall) / (precision + recall), 3) if (precision + recall) > 0 else 0.0

    return {
        "has_ground_truth": True,
        "status": "Validated against ground truth",
        "predicted_count": predicted_count,
        "ground_truth_count": ground_truth_count,
        "absolute_error": abs_error,
        "percentage_error": round(pct_error, 2),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
