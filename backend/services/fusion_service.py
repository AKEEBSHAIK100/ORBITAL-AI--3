import cv2
import numpy as np
from typing import Dict, Any, Tuple

def ensure_same_size(img_a: np.ndarray, img_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check dimensions. Silent resizing is disabled to prevent unverified cross-modal pixel alignment.
    Returns (img_a, img_b) unchanged.
    """
    return img_a, img_b

def extract_optical_features(img_bgr: np.ndarray) -> Dict[str, float]:
    """
    Extract classical remote-sensing spectral proxies from optical RGB imagery:
    - Visible NDVI / Green-Red Vegetation Index (GRVI) proxy
    - Water surface mask (specular low-reflectance, high relative blue)
    - Built-up / high-albedo structural footprint fraction
    - Texture entropy (Laplacian variance)
    """
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    b, g, r = cv2.split(img_bgr.astype(np.float32))
    total_pixels = float(b.size)

    # 1. Vegetation index proxy: Green-Red difference index
    denom = g + r + 1e-6
    grvi = (g - r) / denom
    veg_mask = (grvi > 0.08) & (g > b)
    vegetation_fraction = float(np.count_nonzero(veg_mask) / total_pixels)

    # 2. Water body index proxy: low overall reflectance, blue dominance
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    water_mask = (brightness < 65) & (b > r * 0.95) & (~veg_mask)
    water_fraction = float(np.count_nonzero(water_mask) / total_pixels)

    # 3. Built-up / impervious surface index proxy: high reflectance, low vegetation
    built_mask = (brightness > 135) & (grvi < 0.05) & (~water_mask)
    built_up_fraction = float(np.count_nonzero(built_mask) / total_pixels)

    # 4. Texture entropy (spatial complexity)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Normalize entropy roughly to 0-100 scale
    texture_entropy = min(100.0, laplacian_var / 15.0)

    return {
        "vegetation_fraction": round(vegetation_fraction, 4),
        "water_fraction": round(water_fraction, 4),
        "built_up_fraction": round(built_up_fraction, 4),
        "texture_entropy": round(texture_entropy, 2),
    }

def extract_sar_features(img_sar: np.ndarray) -> Dict[str, Any]:
    """
    Extract synthetic aperture radar (SAR) microwave telemetry:
    - Signal level mean & standard deviation derived from raw amplitude (relative dB against 8-bit dynamic range; NOT radiometrically calibrated to sigma-nought)
    - Speckle index (variance / mean^2, modeling multiplicative radar noise)
    - High-dielectric structural edge density (double-bounce radar returns)
    - Rough surface / volume scattering fraction
    """
    if len(img_sar.shape) == 3:
        sar_gray = cv2.cvtColor(img_sar, cv2.COLOR_BGR2GRAY)
    else:
        sar_gray = img_sar.copy()

    intensity = sar_gray.astype(np.float32)
    mean_val = float(np.mean(intensity))
    std_val = float(np.std(intensity))
    total_pixels = float(sar_gray.size)

    # Convert to relative dB scale against 8-bit dynamic range (raw amplitude proxy, uncalibrated)
    norm_mean = max(1e-4, mean_val / 255.0)
    norm_std = max(1e-4, std_val / 255.0)
    mean_db = 10.0 * np.log10(norm_mean)
    std_db = 10.0 * np.log10(norm_std)

    # Speckle index: coefficient of variation squared (sigma^2 / mu^2)
    speckle_index = (std_val ** 2) / (max(1e-4, mean_val) ** 2)

    # Edge density: high-frequency double-bounce radar returns (buildings, metallic structures)
    if sar_gray.dtype != np.uint8:
        sar_u8 = cv2.normalize(sar_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        sar_u8 = sar_gray
    edges = cv2.Canny(sar_u8, 60, 160)
    edge_density = float(np.count_nonzero(edges) / total_pixels)

    # Rough surface fraction (elevated backscatter from diffuse rough scattering)
    rough_mask = intensity > (mean_val + 0.65 * std_val)
    rough_surface_fraction = float(np.count_nonzero(rough_mask) / total_pixels)

    return {
        "mean_signal_level_db": round(float(mean_db), 2),
        "mean_backscatter_db": round(float(mean_db), 2),  # preserved for schema compatibility
        "std_signal_level_db": round(float(std_db), 2),
        "std_backscatter_db": round(float(std_db), 2),   # preserved for schema compatibility
        "speckle_index": round(float(speckle_index), 3),
        "edge_density": round(edge_density, 4),
        "rough_surface_fraction": round(rough_surface_fraction, 4),
        "calibration_status": "uncalibrated_raw_amplitude",
        "description": "SAR mean signal level derived from raw amplitude (not radiometrically calibrated to sigma-nought backscatter)",
    }

def compute_cross_modal_metrics(
    img_optical: np.ndarray,
    img_sar: np.ndarray,
    is_coregistered: bool = False
) -> Dict[str, Any]:
    """
    Compute joint optical-SAR cross-modal fusion telemetry:
    - Structural Similarity (SSIM proxy between optical luminance and SAR intensity)
    - Pearson Cross-Correlation Coefficient
    - Complementarity Index (identifying features visible in SAR through clouds/shadows)
    - Fusion Confidence rating

    If dimensions differ or registration is unverified, silent pixel alignment is disabled
    and pixel-level metrics (SSIM, cross-correlation) are reported as unavailable.
    """
    if img_optical.shape[:2] != img_sar.shape[:2] or not is_coregistered:
        diff_desc = (
            f"dimension disparity: optical {img_optical.shape[1]}x{img_optical.shape[0]} vs SAR {img_sar.shape[1]}x{img_sar.shape[0]}"
            if img_optical.shape[:2] != img_sar.shape[:2]
            else "unverified geospatial co-registration"
        )
        return {
            "alignment_status": "unavailable",
            "registration_verified": False,
            "structural_similarity": None,
            "cross_correlation": None,
            "complementarity_index": None,
            "fusion_confidence": "unavailable",
            "message": f"Cross-modal pixel alignment is unavailable ({diff_desc}). Silent pixel alignment is disabled.",
        }

    opt_gray = cv2.cvtColor(img_optical, cv2.COLOR_BGR2GRAY) if len(img_optical.shape) == 3 else img_optical
    sar_gray = cv2.cvtColor(img_sar, cv2.COLOR_BGR2GRAY) if len(img_sar.shape) == 3 else img_sar

    opt_f = opt_gray.astype(np.float32)
    sar_f = sar_gray.astype(np.float32)

    # 1. Pearson cross-correlation
    opt_centered = opt_f - np.mean(opt_f)
    sar_centered = sar_f - np.mean(sar_f)
    denom = np.sqrt(np.sum(opt_centered ** 2) * np.sum(sar_centered ** 2)) + 1e-6
    correlation = float(np.sum(opt_centered * sar_centered) / denom)
    correlation = max(-1.0, min(1.0, correlation))

    # 2. Structural Similarity (SSIM luminance + structure proxy)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu1 = np.mean(opt_f)
    mu2 = np.mean(sar_f)
    sigma1_sq = np.var(opt_f)
    sigma2_sq = np.var(sar_f)
    sigma12 = np.mean((opt_f - mu1) * (sar_f - mu2))
    ssim = float(((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)))
    ssim = max(0.0, min(1.0, ssim))

    # 3. Complementarity index:
    # High when SAR provides structural contrast where optical is low-contrast (shadow/uniform) or vice versa
    opt_grad = cv2.Sobel(opt_gray, cv2.CV_32F, 1, 1)
    sar_grad = cv2.Sobel(sar_gray, cv2.CV_32F, 1, 1)
    diff_grad = np.abs(sar_grad - opt_grad)
    complementarity = float(np.mean(diff_grad) / 255.0 * 2.0)
    complementarity = max(0.0, min(1.0, complementarity))

    # Overall fusion confidence
    if correlation > 0.4 or ssim > 0.35:
        fusion_conf = "high"
    elif correlation > 0.15 or ssim > 0.2:
        fusion_conf = "medium"
    else:
        fusion_conf = "low"

    return {
        "alignment_status": "verified",
        "registration_verified": True,
        "structural_similarity": round(ssim, 3),
        "cross_correlation": round(correlation, 3),
        "complementarity_index": round(complementarity, 3),
        "fusion_confidence": fusion_conf,
    }

def analyze_fusion_pair(
    optical_img: np.ndarray,
    sar_img: np.ndarray,
    is_coregistered: bool = False
) -> Dict[str, Any]:
    """Execute multi-modal fusion analysis pipeline without silent pixel alignment."""
    optical_feats = extract_optical_features(optical_img)
    sar_feats = extract_sar_features(sar_img)
    cross_metrics = compute_cross_modal_metrics(optical_img, sar_img, is_coregistered=is_coregistered)

    return {
        "optical": optical_feats,
        "sar": sar_feats,
        "cross_modal": cross_metrics,
    }
