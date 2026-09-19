"""
RSVQA Benchmark Evaluation Runner.
Loads RSVQA LR/HR splits, runs evaluation across question categories,
and records genuine verification results to evaluation/results/rsvqa_<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.datasets.rsvqa import RSVQAAdapter
from evaluation.rsvqa.adapter import format_rsvqa_batch
from evaluation.rsvqa.metrics import compute_rsvqa_metrics


def evaluate_rsvqa(
    data_dir: str = "data/rsvqa_lr",
    variant: str = "lr",
    split: str = "test",
    model_name: str = "satquery-rsvqa-v1",
    model_version: str = "1.0.0"
) -> dict:
    adapter = RSVQAAdapter(data_dir=data_dir, variant=variant)

    if not adapter.is_available():
        return {
            "dataset": f"RSVQA_{variant.upper()}",
            "task": "vqa",
            "split": split,
            "status": "dataset_not_available",
            "message": f"RSVQA dataset not found at '{data_dir}'. Download the dataset to run evaluation.",
            "metrics": None
        }

    raw_samples = adapter.load_samples(split)
    if not raw_samples:
        return {
            "dataset": f"RSVQA_{variant.upper()}",
            "task": "vqa",
            "split": split,
            "status": "dataset_empty_or_invalid",
            "message": f"No questions found in '{data_dir}'.",
            "metrics": None
        }

    formatted = format_rsvqa_batch(raw_samples)
    evaluable = [r for r in formatted if r.get("prediction") is not None]

    if not evaluable:
        return {
            "dataset": f"RSVQA_{variant.upper()}",
            "task": "vqa",
            "split": split,
            "status": "pending_predictions",
            "message": f"Loaded {len(formatted)} reference questions. Model predictions required for evaluation.",
            "sample_count": len(formatted),
            "metrics": None
        }

    metrics = compute_rsvqa_metrics(evaluable)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    result = {
        "dataset": f"RSVQA_{variant.upper()}",
        "dataset_version": "1.0.0",
        "split": split,
        "task": "vqa",
        "model": model_name,
        "model_version": model_version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "sample_count": len(evaluable),
        "status": "COMPLETED"
    }

    results_dir = Path("evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"rsvqa_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run RSVQA Evaluation")
    parser.add_argument("--data_dir", type=str, default="data/rsvqa_lr")
    parser.add_argument("--variant", type=str, default="lr", choices=["lr", "hr"])
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    res = evaluate_rsvqa(args.data_dir, args.variant, args.split)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
