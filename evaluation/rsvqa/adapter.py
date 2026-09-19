"""
RSVQA Evaluation Adapter.
Formats RSVQA questions and maps them into question categories:
- presence ("yes" / "no")
- count (numerical count integer)
- comparison ("yes" / "no" or category comparison)
- area / land-cover ("more", "less", or Corine label)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from training.datasets.rsvqa import RSVQAAdapter


def categorize_rsvqa_question(question: str) -> str:
    q = question.lower()
    if q.startswith("how many") or "count" in q or "number of" in q:
        return "count"
    elif "more" in q or "less" in q or "greater" in q or "fewer" in q or "compare" in q:
        return "comparison"
    elif q.startswith("is there") or q.startswith("are there") or "present" in q:
        return "presence"
    elif "area" in q or "land cover" in q or "type" in q:
        return "area"
    return "general"


def format_rsvqa_batch(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted = []
    for s in samples:
        q = s.get("question", "")
        formatted.append({
            "id": s.get("id"),
            "img_id": s.get("img_id") or s.get("image"),
            "question": q,
            "category": s.get("category") or categorize_rsvqa_question(q),
            "reference_answer": s.get("answer"),
            "prediction": s.get("prediction")
        })
    return formatted
