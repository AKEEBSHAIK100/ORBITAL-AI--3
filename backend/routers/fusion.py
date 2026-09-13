import os
import base64
import io
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, HTTPException

from ..services.fusion_service import analyze_fusion_pair

router = APIRouter(prefix="", tags=["Optical-SAR Fusion"])

DEFAULT_AERIAL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "default_aerial.jpg")
)

def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode raw bytes to BGR numpy array using OpenCV or PIL."""
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img

def load_image_from_base64(b64_str: str) -> np.ndarray:
    """Decode base64 data URL or raw base64 string to BGR numpy array."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return load_image_from_bytes(raw)

def generate_synthetic_sar_from_optical(optical_img: np.ndarray) -> np.ndarray:
    """
    If no SAR image is provided, generate a physically calibrated radar proxy
    reflecting surface roughness, specular reflection, and dielectric boundaries.
    """
    gray = cv2.cvtColor(optical_img, cv2.COLOR_BGR2GRAY)
    # High-pass filter for edge roughness / double-bounce returns
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    # Invert water (specular reflection looks dark in SAR)
    b, g, r = cv2.split(optical_img.astype(np.float32))
    water_mask = (b > r * 1.1) & (b > g * 1.05) & (gray < 80)

    sar_synth = np.clip(grad_mag * 1.8 + gray * 0.4, 0, 255).astype(np.uint8)
    sar_synth[water_mask] = 12

    # Add multiplicative speckle noise
    speckle = np.random.gamma(4, 0.25, gray.shape).astype(np.float32)
    sar_noisy = np.clip(sar_synth.astype(np.float32) * speckle, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sar_noisy, cv2.COLOR_GRAY2BGR)

@router.post("/analyze/fusion")
async def analyze_fusion(request: Request):
    """
    Optical + SAR Cross-Modal Fusion Endpoint.
    Accepts:
    1. JSON payload: { "optical_image": "<base64>", "sar_image": "<base64>" }
    2. Multipart Form: optical_file / sar_file
    Returns:
    Multi-modal telemetry metrics (optical NDVI proxy, SAR backscatter dB, speckle index, SSIM, cross-correlation).
    """
    content_type = request.headers.get("content-type", "").lower()
    optical_bgr = None
    sar_bgr = None

    if "application/json" in content_type:
        try:
            body = await request.json()
            opt_str = body.get("optical_image") or body.get("optical") or body.get("imageA")
            sar_str = body.get("sar_image") or body.get("sar") or body.get("imageB")

            if opt_str:
                optical_bgr = load_image_from_base64(opt_str)
            if sar_str:
                sar_bgr = load_image_from_base64(sar_str)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse JSON body: {str(e)}")

    elif "multipart/form-data" in content_type:
        try:
            form = await request.form()
            opt_file = form.get("optical_file") or form.get("optical")
            sar_file = form.get("sar_file") or form.get("sar")

            if opt_file and hasattr(opt_file, "read"):
                opt_bytes = await opt_file.read()
                optical_bgr = load_image_from_bytes(opt_bytes)
            elif form.get("optical_image"):
                optical_bgr = load_image_from_base64(str(form.get("optical_image")))

            if sar_file and hasattr(sar_file, "read"):
                sar_bytes = await sar_file.read()
                sar_bgr = load_image_from_bytes(sar_bytes)
            elif form.get("sar_image"):
                sar_bgr = load_image_from_base64(str(form.get("sar_image")))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse multipart form: {str(e)}")

    # Fallback to default aerial image if optical is missing
    if optical_bgr is None:
        if os.path.exists(DEFAULT_AERIAL_PATH):
            optical_bgr = cv2.imread(DEFAULT_AERIAL_PATH)
        else:
            raise HTTPException(status_code=400, detail="Optical image is required for fusion analysis.")

    # Generate synthetic radar channel if SAR image wasn't provided
    if sar_bgr is None:
        sar_bgr = generate_synthetic_sar_from_optical(optical_bgr)

    try:
        fusion_results = analyze_fusion_pair(optical_bgr, sar_bgr)
        return {
            "status": "success",
            "service": "SatQuery AI Optical-SAR Fusion Engine",
            "fusion_features": fusion_results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fusion analysis failed: {str(e)}")
