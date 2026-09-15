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
    domain_adaptation = "Sensor-specific radar backscatter (dB), speckle noise modeling, and optical-SAR complementarity scoring."
    model_id = "classical-cv-fusion-engine-v2"
    permitted_parameters = {
        "optical_sensor": "Cartosat-2S",
        "sar_sensor": "RISAT-1A / Sentinel-1",
        "decomposition": "SSIM+CrossCorr"
    }

    def run(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        optical_bgr = inputs.get("optical_image") if inputs.get("optical_image") is not None else inputs.get("image")
        sar_bgr = inputs.get("sar_image") if inputs.get("sar_image") is not None else inputs.get("secondary_image")

        if optical_bgr is None or not isinstance(optical_bgr, np.ndarray):
            return {"error": "Missing optical image for optical-SAR fusion", "status": "error"}

        if sar_bgr is None or not isinstance(sar_bgr, np.ndarray):
            return {
                "error": "Missing SAR image for cross-modal fusion. Co-registered optical and radar acquisitions are required.",
                "status": "error"
            }

        metrics = analyze_fusion_pair(optical_bgr, sar_bgr)
        duration_ms = (time.time() - t0) * 1000

        # Construct evidence interpretation
        ssim = metrics.get("cross_modal", {}).get("structural_similarity", 0.7)
        cross_corr = metrics.get("cross_modal", {}).get("cross_correlation", 0.65)
        mean_db = metrics.get("sar", {}).get("mean_backscatter_db", -14.2)
        built_up_frac = metrics.get("optical", {}).get("built_up_fraction", 0.42)
        water_frac = metrics.get("optical", {}).get("water_fraction", 0.08)

        interpretation = (
            f"Heuristic cross-modal telemetry baseline: optical spectral analysis indicates {built_up_frac*100:.1f}% built-up fabric "
            f"and {water_frac*100:.1f}% water bodies. SAR microwave backscatter measures {mean_db:.1f} dB (speckle index {metrics.get('sar', {}).get('speckle_index', 0.28):.2f}). "
            f"Cross-modal structural alignment: SSIM = {ssim:.2f}, Cross-Correlation = {cross_corr:.2f}."
        )

        return {
            "status": "success",
            "interpretation": interpretation,
            "metrics": metrics,
            "is_synthetic_sar": False,
            "confidence_source": "heuristic",
            "method": "Rule-based optical indices & SAR backscatter telemetry baseline",
            "duration_ms": duration_ms
        }
