"""
RSVQA Evaluation Metrics Calculator.
Breaks down evaluation metrics by question taxonomy:
- Presence accuracy
- Comparison accuracy
- Counting accuracy (exact match and MAE)
- Area / land-cover accuracy
- Overall Top-1 accuracy
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List


def normalize_answer(ans: Any) -> str:
    clean = re.sub(r"[^\w\s]", "", str(ans).strip().lower())
    return clean


def compute_rsvqa_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"overall_accuracy": 0.0, "sample_count": 0, "breakdown": {}}

    categories = defaultdict(list)
    overall_hits = 0
    count_errors: List[float] = []

    for r in records:
        cat = r.get("category", "general")
        ref = normalize_answer(r.get("reference_answer", ""))
        pred = normalize_answer(r.get("prediction", ""))

        hit = 1 if (ref and pred and ref == pred) else 0
        overall_hits += hit
        categories[cat].append(hit)

        if cat == "count":
            try:
                ref_num = float(ref)
                pred_num = float(pred)
                count_errors.append(abs(ref_num - pred_num))
            except Exception:
                pass

    total = len(records)
    breakdown = {}
    for cat, hits in categories.items():
        breakdown[cat] = {
            "accuracy": round((sum(hits) / len(hits)) * 100.0, 2),
            "samples": len(hits)
        }

    if count_errors:
        breakdown["count"]["mean_absolute_error"] = round(sum(count_errors) / len(count_errors), 2)

    return {
        "overall_accuracy": round((overall_hits / total) * 100.0, 2),
        "sample_count": total,
        "category_breakdown": breakdown
    }
