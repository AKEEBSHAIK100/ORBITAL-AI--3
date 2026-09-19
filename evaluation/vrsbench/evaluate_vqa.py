"""
VRSBench Visual Question Answering Evaluation Script.
Computes Exact Match (EM) and Macro F1 over remote-sensing VQA queries.
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
from evaluation.vrsbench.metrics import compute_vqa_accuracy


def evaluate_vqa(
    dataset_path: str = "data/vrsbench",
    split: str = "test",
    model_name: str = "satquery-vlm-v1",
    model_version: str = "1.0.0"
) -> dict:
    adapter = VRSBenchAdapter(dataset_path)

    if not adapter.is_available():
        return {
            "dataset": "VRSBench",
            "task": "vqa",
            "split": split,
            "status": "dataset_not_available",
            "message": f"VRSBench dataset not found at '{dataset_path}'. Download benchmark files to evaluate.",
            "metrics": None
        }

    samples = adapter.load_vqa_samples(split)
    if not samples:
        return {
            "dataset": "VRSBench",
            "task": "vqa",
            "split": split,
            "status": "dataset_empty_or_invalid",
            "message": "No VQA samples found.",
            "metrics": None
        }

    preds: list[str] = []
    targets: list[str] = []

    for item in samples:
        gt = item.get("answer") or item.get("reference_answer")
        pred = item.get("prediction")
        if gt and pred:
            preds.append(str(pred))
            targets.append(str(gt))

    if not preds:
        return {
            "dataset": "VRSBench",
            "task": "vqa",
            "split": split,
            "status": "pending_predictions",
            "message": f"Loaded {len(samples)} ground truth Q&A pairs. Predictions required for evaluation.",
            "sample_count": len(samples),
            "metrics": None
        }

    metrics = compute_vqa_accuracy(preds, targets)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    result = {
        "dataset": "VRSBench",
        "dataset_version": "1.0.0",
        "split": split,
        "task": "vqa",
        "model": model_name,
        "model_version": model_version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "sample_count": len(preds),
        "status": "COMPLETED"
    }

    results_dir = Path("evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"vrsbench_vqa_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run VRSBench VQA Evaluation")
    parser.add_argument("--data_dir", type=str, default="data/vrsbench")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    res = evaluate_vqa(args.data_dir, args.split)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
