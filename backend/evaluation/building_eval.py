from typing import Any, Dict, List, Optional
from ..services.validation import evaluate_accuracy

def run_building_evaluation(
    predicted_detections: List[Dict[str, Any]],
    ground_truth_bboxes: Optional[List[List[float]]] = None,
    ground_truth_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Implements Section 16 & 21: Reproducible Evaluation & Accuracy Reporting.
    Calculates Precision, Recall, F1, IoU, and count error against ground truth.
    Never invents benchmark scores.
    """
    pred_boxes = [d["bbox"] for d in predicted_detections] if predicted_detections else []
    pred_count = len(pred_boxes)

    if ground_truth_count is None and ground_truth_bboxes is not None:
        ground_truth_count = len(ground_truth_bboxes)

    results = evaluate_accuracy(
        predicted_count=pred_count,
        ground_truth_count=ground_truth_count,
        ground_truth_bboxes=ground_truth_bboxes,
        predicted_bboxes=pred_boxes
    )

    return {
        "task": "building_footprint_extraction",
        "benchmark_dataset": "SpaceNet / Held-Out Building Ground Truth",
        "metrics": results,
        "evaluation_disclaimer": (
            "Actual measured metrics against uploaded/held-out ground truth."
            if results.get("has_ground_truth")
            else "No ground-truth annotations provided; metrics pending ground-truth comparison."
        )
    }
