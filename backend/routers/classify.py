"""
SatQuery AI — BigEarthNet v2.0 Land-Cover Classification Router
POST /classify  — accepts image upload or base64 JSON
Returns 19-class multi-label predictions from the BIFOLD pretrained ResNet-50.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.ben_classifier import BENClassifier

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/classify", tags=["classify"])

# ── Pydantic models ───────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image (data URL or raw base64)")
    top_k: int = Field(5, ge=1, le=19, description="Number of top labels to return")
    threshold: float = Field(0.25, ge=0.0, le=1.0, description="Sigmoid activation threshold")


class LabelScore(BaseModel):
    name: str
    short: str
    score: float
    active: bool


class ClassifyResponse(BaseModel):
    labels: List[LabelScore]
    active_labels: List[LabelScore]
    top_label: str
    confidence: float
    model_id: str
    available: bool
    device: str
    note: str
    citation: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_base64_image(data: str) -> bytes:
    """Strip data-URL prefix and decode base64 → raw bytes."""
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        return base64.b64decode(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/", response_model=ClassifyResponse)
@router.post("", response_model=ClassifyResponse)
async def classify_from_json(req: ClassifyRequest):
    """
    Classify a satellite image using BigEarthNet v2.0 (ResNet-50 pretrained).
    Accepts a base64-encoded image in the JSON body.
    """
    img_bytes = _parse_base64_image(req.image)
    classifier = BENClassifier.get_instance()
    result = classifier.classify_image(
        img_bytes,
        top_k=req.top_k,
        threshold=req.threshold,
    )
    return JSONResponse(content=result)


@router.post("/upload", response_model=ClassifyResponse)
async def classify_from_upload(
    file: UploadFile = File(...),
    top_k: int = 5,
    threshold: float = 0.25,
):
    """
    Classify a satellite image using BigEarthNet v2.0 (ResNet-50 pretrained).
    Accepts a multipart file upload (JPEG / PNG / TIFF).
    """
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/tiff", "image/webp"}
    if file.content_type and file.content_type.lower() not in allowed:
        # Be permissive — also allow generic content types from file pickers
        if not any(ext in (file.filename or "").lower() for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"]):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or TIFF.",
            )

    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file received.")

    classifier = BENClassifier.get_instance()
    result = classifier.classify_image(img_bytes, top_k=top_k, threshold=threshold)
    return JSONResponse(content=result)


@router.get("/status")
async def classifier_status():
    """Return the current state of the BigEarthNet v2.0 classifier."""
    classifier = BENClassifier.get_instance()
    return {
        "available": classifier.is_available,
        "device": classifier.device,
        "model_id": classifier.model_id,
        "classes": 19,
        "architecture": "ResNet-50",
        "dataset": "BigEarthNet v2.0 (reBEN)",
        "paper": "Clasen et al., IGARSS 2025 · arXiv:2407.03653",
        "load_error": classifier.load_error or None,
    }
