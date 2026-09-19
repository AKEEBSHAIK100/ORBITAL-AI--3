"""
CDVQA Evaluation Runner.
Evaluates bi-temporal change answering against reference answers.
Strict compliance: Disclaims when dataset is missing and never produces fake metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.datasets.cdvqa import CDVQAAdapter
from evaluation.cdvqa.metrics import compute_cdvqa_metrics


def evaluate_cdvqa(
    data_dir: str = "data/cdvqa",
    split: str = "test",
    model_name: str = "satquery-change-vqa",
    model_version: str = "2.0.0"
) -> dict:
    adapter = CDVQAAdapter(data_dir=data_dir)

    if not adapter.is_available():
        return {
            "dataset": "CDVQA",
            "task": "change_vqa",
            "split": split,
            "status": "dataset_not_available",
            "message": f"CDVQA dataset not found at '{data_dir}'. Download the dataset to run evaluation.",
            "metrics": None
        }

    samples = adapter.load_samples(split)
    if not samples:
        return {
            "dataset": "CDVQA",
            "task": "change_vqa",
            "split": split,
            "status": "dataset_empty_or_invalid",
            "message": "No CDVQA samples found in directory.",
            "metrics": None
        }

    evaluable = [
        {"prediction": s.get("prediction"), "reference": s.get("answer")}
        for s in samples if s.get("prediction") is not None
    ]

    if not evaluable:
        return {
            "dataset": "CDVQA",
            "task": "change_vqa",
            "split": split,
            "status": "pending_predictions",
            "message": f"Loaded {len(samples)} bi-temporal test items. Model predictions required to compute metrics.",
            "sample_count": len(samples),
            "metrics": None
        }

    metrics = compute_cdvqa_metrics(evaluable)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    result = {
        "dataset": "CDVQA",
        "dataset_version": "1.0.0",
        "split": split,
        "task": "change_vqa",
        "model": model_name,
        "model_version": model_version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "sample_count": len(evaluable),
        "status": "COMPLETED"
    }

    results_dir = Path("evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"cdvqa_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run CDVQA Evaluation")
    parser.add_argument("--data_dir", type=str, default="data/cdvqa")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    res = evaluate_cdvqa(args.data_dir, args.split)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
