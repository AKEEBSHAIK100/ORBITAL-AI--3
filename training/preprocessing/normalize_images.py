"""
Remote-Sensing Image Normalization Utilities.
Handles optical RGB, multispectral reflectance scaling, and SAR decibel calibration.
"""

from __future__ import annotations

import numpy as np


def normalize_optical(img: np.ndarray, target_size: tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Normalizes optical 8-bit or 16-bit array to standard float32 [0.0, 1.0].
    Preserves channel order (BGR or RGB).
    """
    arr = img.astype(np.float32)
    if arr.max() > 255.0:
        # Likely 12-bit or 16-bit satellite product (e.g. Sentinel-2 L2A / Cartosat)
        p2, p98 = np.percentile(arr, (2, 98))
        if p98 > p2:
            arr = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
        else:
            arr = np.clip(arr / 65535.0, 0.0, 1.0)
    else:
        arr = arr / 255.0
    return arr


def normalize_sar(img: np.ndarray, clip_min_db: float = -30.0, clip_max_db: float = 0.0) -> np.ndarray:
    """
    Calibrates raw SAR backscatter to calibrated decibel scale and normalizes to [0.0, 1.0].
    """
    arr = img.astype(np.float32)
    if len(arr.shape) == 3:
        arr = arr[:, :, 0]

    # Convert linear amplitude to intensity if needed
    if arr.min() >= 0:
        # Avoid log(0)
        intensity = np.maximum(arr, 1e-6)
        db = 10.0 * np.log10(intensity)
    else:
        db = arr

    norm = np.clip((db - clip_min_db) / (clip_max_db - clip_min_db), 0.0, 1.0)
    return norm


def to_uint8_preview(arr: np.ndarray) -> np.ndarray:
    """Safely converts normalized float [0.0, 1.0] array to 8-bit uint8 representation for OpenCV/display."""
    clipped = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return clipped
