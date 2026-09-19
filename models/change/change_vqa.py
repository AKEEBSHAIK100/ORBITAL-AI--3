"""
Change Visual Question Answering (CDVQA) Specialist.
Interprets natural language inquiries across bi-temporal satellite image pairs.
Answers queries regarding expansion, vegetation trends, and structural changes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from .change_detector import ChangeDetector


class ChangeVQASpecialist:
    def __init__(self):
        self.detector = ChangeDetector()

    def answer_query(
        self,
        query: str,
        img_t1: np.ndarray,
        img_t2: np.ndarray,
        meta_t1: Optional[Dict[str, Any]] = None,
        meta_t2: Optional[Dict[str, Any]] = None,
        date_t1: Optional[str] = None,
        date_t2: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes bi-temporal change detection and answers the user query.
        Returns standard schema with answer, confidence, evidence, warnings.
        """
        q_lower = query.lower()
        det_result = self.detector.detect_changes(
            img_t1, img_t2, meta_t1=meta_t1, meta_t2=meta_t2, date_t1=date_t1, date_t2=date_t2
        )

        if not det_result.get("success"):
            return {
                "answer": f"Unable to perform change reasoning: {det_result.get('error')}",
                "confidence": None,
                "confidence_level": "UNAVAILABLE",
                "evidence": None,
                "warnings": det_result.get("temporal_report", {}).get("errors", []),
                "model": "Bi-Temporal Radiometric Change Engine",
                "model_version": "2.0.0",
                "task": "change_vqa"
            }

        change_pct = det_result["change_percentage"]
        clusters_count = det_result["change_clusters_count"]
        veg_delta = det_result["vegetation_delta_pct"]

        # Formulate answer based on query semantics
        if "vegetation" in q_lower or "green" in q_lower or "forest" in q_lower:
            direction = "increased" if veg_delta > 0 else "decreased"
            answer = (
                f"Bi-temporal spectral comparison indicates vegetation has {direction} by {abs(veg_delta)}% "
                f"between the observation captures. Total surface change across the scene is {change_pct}%."
            )
            confidence = 0.86
        elif "building" in q_lower or "built" in q_lower or "urban" in q_lower or "expansion" in q_lower:
            if change_pct > 3.0:
                answer = (
                    f"Structural alteration audit detected {clusters_count} discrete alteration sectors "
                    f"spanning {change_pct}% of the surveyed scene, indicating urban and ground expansion."
                )
            else:
                answer = (
                    f"Structural footprint remained stable between the two observation dates. "
                    f"Only {change_pct}% surface variation was observed."
                )
            confidence = 0.84
        else:
            if change_pct < 1.5:
                answer = (
                    f"Surface features remained largely stable between the two acquisition dates. "
                    f"Measured alteration is {change_pct}% across {clusters_count} localized areas."
                )
            else:
                answer = (
                    f"Detectable alterations occurred across {change_pct}% of the surface area, "
                    f"distributed across {clusters_count} distinct spatial clusters."
                )
            confidence = 0.85

        return {
            "answer": answer,
            "confidence": confidence,
            "confidence_level": "High" if confidence >= 0.80 else "Medium",
            "evidence": {
                "change_percentage": change_pct,
                "clusters_count": clusters_count,
                "vegetation_delta_pct": veg_delta
            },
            "warnings": det_result.get("temporal_validation", {}).get("warnings", []),
            "model": "Bi-Temporal Radiometric & Spectral Change Engine",
            "model_version": "2.0.0",
            "task": "change_vqa"
        }
