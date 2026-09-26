import time
from typing import Any, Dict, List, Optional
import numpy as np
import cv2

from .base import BaseTool
from ..services.fusion_service import analyze_fusion_pair

class OpticalSARTool(BaseTool):
    id = "optical_sar"
    name = "Optical–SAR Cross-Modal Fusion Specialist"
    description = "Joint reasoning over co-registered optical and SAR pairs. Computes radar backscatter intensity, surface roughness, speckle index, SSIM structural similarity, and optical NDVI proxy."
    supported_tasks = ["sar_optical_fusion"]
    modalities = ["optical", "sar"]
    adapter = "OpenCV / NumPy Classical Computer Vision & Cross-Modal Telemetry Engine"
    domain_adaptation = "Uncalibrated RGB optical proxies plus raw SAR intensity telemetry; no sensor-specific calibration is assumed."
    model_id = "classical-cv-fusion-engine-v2"
    permitted_parameters = {
        "optical_sensor": "unspecified unless metadata is supplied",
        "sar_sensor": "unspecified unless metadata is supplied",
        "decomposition": "SSIM+CrossCorr"
    }

    def run(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        optical_bgr = inputs.get("optical_image") if inputs.get("optical_image") is not None else inputs.get("image")
        sar_bgr = inputs.get("sar_image") if inputs.get("sar_image") is not None else inputs.get("secondary_image")

        if optical_bgr is None or not isinstance(optical_bgr, np.ndarray):
            return {"error": "Missing optical image for optical-SAR fusion", "status": "error"}

        if sar_bgr is None or not isinstance(sar_bgr, np.ndarray):
            # Joint optical-SAR analysis requires both observations. Never silently\n        # downgrade a fusion request to optical-only analysis.\n        if sar_bgr is None or not isinstance(sar_bgr, np.ndarray):\n            return {\n                "status": "specialist_unavailable",\n                "answer": "Optical-SAR fusion requires both an optical image and a SAR image. No optical-only fallback is returned for a joint-analysis request.",\n                "metrics": None,\n                "is_synthetic_sar": False,\n                "is_coregistered": False,\n                "confidence": None,\n                "confidence_status": "unavailable",\n                "warnings": ["SAR input is missing; joint optical-SAR analysis was not executed."],\n                "duration_ms": (time.time() - t0) * 1000,\n            }\n\n        # Co-registration verification
        meta_opt = inputs.get("metadata") or {}
        meta_sar = inputs.get("secondary_metadata") or {}

        is_coregistered = inputs.get("is_coregistered")
        if is_coregistered is None and parameters:
            is_coregistered = parameters.get("is_coregistered")

        # If dimensions differ, pixel alignment is impossible without unverified resampling/warping
        if optical_bgr.shape[:2] != sar_bgr.shape[:2]:
            is_coregistered = False
        elif is_coregistered is None:
            # Check metadata for verified spatial reference and grid alignment
            crs_opt = meta_opt.get("crs")
            crs_sar = meta_sar.get("crs")
            tf_opt = meta_opt.get("geotransform") or meta_opt.get("transform")
            tf_sar = meta_sar.get("geotransform") or meta_sar.get("transform")
            if crs_opt and crs_sar and str(crs_opt) == str(crs_sar) and tf_opt and tf_sar and tf_opt == tf_sar:
                is_coregistered = True
            else:
                is_coregistered = False

        metrics = analyze_fusion_pair(optical_bgr, sar_bgr, is_coregistered=is_coregistered)
        duration_ms = (time.time() - t0) * 1000

        # Construct truthful evidence interpretation
        sar_metrics = metrics.get("sar", {})
        opt_metrics = metrics.get("optical", {})
        cm_metrics = metrics.get("cross_modal", {})

        built_up_frac = opt_metrics.get("built_up_fraction")
        water_frac = opt_metrics.get("water_fraction")
        mean_db = sar_metrics.get("mean_signal_level_db", sar_metrics.get("mean_backscatter_db"))
        speckle = sar_metrics.get("speckle_index")

        interpretation_parts = [
            f"Classical cross-modal telemetry baseline: optical visible-band proxies indicate {built_up_frac*100:.1f}% built-up fabric "
            f"and {water_frac*100:.1f}% water bodies.",
            f"SAR mean signal level derived from raw amplitude is {mean_db:.1f} dB (speckle index {speckle:.2f}; raw amplitude uncalibrated to sigma-nought backscatter)."
        ]

        if is_coregistered and cm_metrics.get("structural_similarity") is not None:
            ssim = cm_metrics["structural_similarity"]
            cross_corr = cm_metrics.get("cross_correlation", 0.0)
            interpretation_parts.append(f"Cross-modal structural alignment: SSIM = {ssim:.2f}, Cross-Correlation = {cross_corr:.2f}.")
        else:
            interpretation_parts.append("Cross-modal pixel alignment is unavailable because spatial co-registration is unverified; silent pixel alignment is disabled.")

        interpretation = " ".join(interpretation_parts)

        return {
            "status": "success",
            "interpretation": interpretation,
            "metrics": metrics,
            "is_synthetic_sar": False,
            "is_coregistered": is_coregistered,
            "confidence": None,
            "confidence_status": "not_calibrated",
            "confidence_source": "classical_cv_uncalibrated",
            "method": "Visible-band optical proxies & raw SAR signal-level telemetry baseline",
            "duration_ms": duration_ms
        }
