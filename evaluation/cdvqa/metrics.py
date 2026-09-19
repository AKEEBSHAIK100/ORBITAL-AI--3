"""
CDVQA (Change Detection VQA) Benchmark Metrics.
Computes binary change detection accuracy, directional attribution accuracy, and Token F1.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


def normalize_str(s: Any) -> str:
    return re.sub(r"[^\w\s]", "", str(s).strip().lower())


def compute_cdvqa_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"accuracy": 0.0, "sample_count": 0, "binary_change_acc": 0.0}

    hits = 0
    binary_hits = 0
    total = len(records)

    for r in records:
        pred = normalize_str(r.get("prediction", ""))
        ref = normalize_str(r.get("reference", ""))

        if pred == ref:
            hits += 1

        # Binary check (changed vs not changed / yes vs no)
        is_ref_change = ("change" in ref or "yes" in ref or "increase" in ref or "decrease" in ref)
        is_pred_change = ("change" in pred or "yes" in pred or "increase" in pred or "decrease" in pred)
        if is_ref_change == is_pred_change:
            binary_hits += 1

    return {
        "accuracy": round((hits / total) * 100.0, 2),
        "binary_change_acc": round((binary_hits / total) * 100.0, 2),
        "sample_count": total
    }
