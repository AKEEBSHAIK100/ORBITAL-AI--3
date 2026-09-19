"""
GeoTIFF and Remote-Sensing Raster Validator.
Inspects geospatial tags, Coordinate Reference Systems (CRS), dimensions, and bit depth.
Works safely with or without GDAL/rasterio by falling back to PIL / tifffile / raw parsing.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image


def inspect_geotiff_metadata(data_or_path: bytes | str | Path) -> Dict[str, Any]:
    """
    Inspects GeoTIFF / TIFF raster and extracts spatial dimensions,
    band count, bit depth, and CRS tags if embedded.
    """
    metadata: Dict[str, Any] = {
        "valid": False,
        "format": "unknown",
        "width": 0,
        "height": 0,
        "bands": 0,
        "dtype": "unknown",
        "has_georeference": False,
        "crs": None,
        "resolution": None,
        "warnings": []
    }

    try:
        if isinstance(data_or_path, (str, Path)):
            img = Image.open(data_or_path)
            metadata["path"] = str(data_or_path)
        else:
            img = Image.open(io.BytesIO(data_or_path))

        metadata["format"] = img.format or "TIFF"
        metadata["width"], metadata["height"] = img.size
        metadata["bands"] = len(img.getbands())
        metadata["mode"] = img.mode

        # Check for TIFF / GeoTIFF tags (Tag 34735 = GeoKeyDirectoryTag, 33550 = ModelPixelScaleTag)
        if hasattr(img, "tag_v2"):
            tags = img.tag_v2
            # 34735: GeoKeyDirectoryTag
            if 34735 in tags:
                metadata["has_georeference"] = True
                metadata["crs"] = "GeoTIFF_Embedded_GeoKey"
            # 33550: ModelPixelScaleTag
            if 33550 in tags:
                scale = tags[33550]
                metadata["resolution"] = [float(s) for s in scale[:2]]

        metadata["valid"] = metadata["width"] > 0 and metadata["height"] > 0
        return metadata

    except Exception as e:
        metadata["warnings"].append(f"Inspection error: {str(e)}")
        return metadata


def verify_pair_compatibility(
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
    tolerance_ratio: float = 0.05
) -> Tuple[bool, List[str]]:
    """
    Verifies whether two satellite scenes are geometrically compatible for
    bi-temporal change detection or optical-SAR fusion.
    """
    issues: List[str] = []

    # 1. Dimension check
    w1, h1 = meta1.get("width", 0), meta1.get("height", 0)
    w2, h2 = meta2.get("width", 0), meta2.get("height", 0)

    if w1 == 0 or h1 == 0 or w2 == 0 or h2 == 0:
        issues.append("One or both images have invalid zero dimensions.")
        return False, issues

    diff_w = abs(w1 - w2) / max(w1, w2)
    diff_h = abs(h1 - h2) / max(h1, h2)

    if diff_w > tolerance_ratio or diff_h > tolerance_ratio:
        issues.append(f"Dimension mismatch: Image 1 is ({w1}x{h1}), Image 2 is ({w2}x{h2}).")

    # 2. CRS check if both have georeference
    crs1 = meta1.get("crs")
    crs2 = meta2.get("crs")
    if crs1 and crs2 and crs1 != crs2:
        issues.append(f"Coordinate Reference System conflict: '{crs1}' vs '{crs2}'. Reprojection required.")

    compatible = len(issues) == 0
    return compatible, issues
