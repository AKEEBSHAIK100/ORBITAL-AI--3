"""
VRSBench Visual Grounding Evaluation Script.
Evaluates model-predicted bounding boxes against ground truth coordinates.
Outputs genuine mIoU@0.5 and Acc@0.5.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.datasets.vrsbench import VRSBenchAdapter
from evaluation.vrsbench.metrics import compute_grounding_metrics


def evaluate_grounding(
    dataset_path: str = "data/vrsbench",
    split: str = "test",
    model_name: str = "satquery-grounding-v1",
    model_version: str = "1.0.0"
) -> dict:
    adapter = VRSBenchAdapter(dataset_path)

    if not adapter.is_available():
        return {
            "dataset": "VRSBench",
            "task": "visual_grounding",
            "split": split,
            "status": "dataset_not_available",
            "message": f"VRSBench dataset not found at '{dataset_path}'. Download the benchmark to run evaluation.",
            "metrics": None
        }

    samples = adapter.load_grounding_samples(split)
    if not samples:
        return {
            "dataset": "VRSBench",
            "task": "visual_grounding",
            "split": split,
            "status": "dataset_empty_or_invalid",
            "message": "No visual grounding annotations found in dataset.",
            "metrics": None
        }

    preds: list[list[float]] = []
    targets: list[list[float]] = []

    for item in samples:
        gt_box = item.get("bbox") or item.get("target_box")
        pred_box = item.get("predicted_bbox")
        if gt_box and pred_box:
            preds.append(pred_box)
            targets.append(gt_box)

    if not preds:
        return {
            "dataset": "VRSBench",
            "task": "visual_grounding",
            "split": split,
            "status": "pending_predictions",
            "message": f"Loaded {len(samples)} ground truth boxes. Predicted boxes required to calculate metrics.",
            "sample_count": len(samples),
            "metrics": None
        }

    metrics = compute_grounding_metrics(preds, targets, threshold=0.5)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    result = {
        "dataset": "VRSBench",
        "dataset_version": "1.0.0",
        "split": split,
        "task": "visual_grounding",
        "model": model_name,
        "model_version": model_version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "sample_count": len(preds),
        "status": "COMPLETED"
    }

    results_dir = Path("evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"vrsbench_grounding_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run VRSBench Grounding Evaluation")
    parser.add_argument("--data_dir", type=str, default="data/vrsbench")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    res = evaluate_grounding(args.data_dir, args.split)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
