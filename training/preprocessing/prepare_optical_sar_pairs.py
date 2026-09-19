"""
Optical–SAR Cross-Modal Sensor Alignment & Preprocessing.
Supports generic optical (Cartosat, Sentinel-2, Planet, aerial) and SAR (RISAT, Sentinel-1, TerraSAR-X).
Strictly enforces co-registration checks before fusion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .validate_geotiff import inspect_geotiff_metadata, verify_pair_compatibility


def prepare_optical_sar_pair(
    optical_path: str | Path,
    sar_path: str | Path,
    optical_sensor: str = "generic_optical",
    sar_sensor: str = "generic_sar"
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates and inspects optical + SAR inputs for genuine cross-sensor fusion.
    Never pretends two unrelated images are co-registered.
    """
    opt_p = Path(optical_path)
    sar_p = Path(sar_path)

    report: Dict[str, Any] = {
        "valid": False,
        "optical_path": str(opt_p),
        "sar_path": str(sar_p),
        "optical_sensor": optical_sensor,
        "sar_sensor": sar_sensor,
        "optical_meta": None,
        "sar_meta": None,
        "coregistration_status": "unverified",
        "errors": [],
        "warnings": []
    }

    if not opt_p.exists():
        report["errors"].append(f"Optical raster file not found: {opt_p}")
    if not sar_p.exists():
        report["errors"].append(f"SAR raster file not found: {sar_p}")

    if report["errors"]:
        return False, report

    opt_meta = inspect_geotiff_metadata(opt_p)
    sar_meta = inspect_geotiff_metadata(sar_p)
    report["optical_meta"] = opt_meta
    report["sar_meta"] = sar_meta

    # Check geometric compatibility
    compatible, issues = verify_pair_compatibility(opt_meta, sar_meta)
    if not compatible:
        report["coregistration_status"] = "rejected"
        report["errors"].extend(issues)
        return False, report

    # Check resolution alignment
    if opt_meta.get("resolution") and sar_meta.get("resolution"):
        r_opt = opt_meta["resolution"]
        r_sar = sar_meta["resolution"]
        if abs(r_opt[0] - r_sar[0]) > 0.5:
            report["warnings"].append(f"Pixel resolution discrepancy: Optical ({r_opt}) vs SAR ({r_sar}). Resampling advised.")

    report["coregistration_status"] = "verified_compatible"
    report["valid"] = True
    return True, report
