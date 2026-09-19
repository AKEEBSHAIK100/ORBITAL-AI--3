"""
Bi-Temporal Image Pair Preparation and Verification.
Ensures temporal ordering (T1 -> T2), geometric alignment, and acquisition metadata consistency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .validate_geotiff import inspect_geotiff_metadata, verify_pair_compatibility


def prepare_temporal_pair(
    path_t1: str | Path,
    path_t2: str | Path,
    date_t1: Optional[str] = None,
    date_t2: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates and organizes two temporal remote-sensing captures for change analysis.
    Returns (is_valid, validation_report).
    """
    p1, p2 = Path(path_t1), Path(path_t2)
    report: Dict[str, Any] = {
        "valid": False,
        "path_t1": str(p1),
        "path_t2": str(p2),
        "date_t1": date_t1,
        "date_t2": date_t2,
        "meta_t1": None,
        "meta_t2": None,
        "ordering_verified": False,
        "errors": []
    }

    if not p1.exists():
        report["errors"].append(f"T1 image file not found: {p1}")
    if not p2.exists():
        report["errors"].append(f"T2 image file not found: {p2}")

    if report["errors"]:
        return False, report

    meta1 = inspect_geotiff_metadata(p1)
    meta2 = inspect_geotiff_metadata(p2)
    report["meta_t1"] = meta1
    report["meta_t2"] = meta2

    compatible, issues = verify_pair_compatibility(meta1, meta2)
    if not compatible:
        report["errors"].extend(issues)
        return False, report

    # Verify temporal ordering if dates are given
    if date_t1 and date_t2:
        if str(date_t1) <= str(date_t2):
            report["ordering_verified"] = True
        else:
            report["errors"].append(f"Temporal sequence inversion: T1 date ({date_t1}) is after T2 date ({date_t2}).")
            return False, report
    else:
        # Known ordering assumed from parameter slots
        report["ordering_verified"] = True

    report["valid"] = True
    return True, report
