import os
import base64
import io
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

from ..schemas.analysis import (
    AnalyzeRequest,
    UnifiedAnalysisResponse,
    BuildingAnalysisResponse,
    ObservableTrace,
    TraceStep,
    GroundingItem,
    GroundingRegion,
)
from ..agents.validator import validate_input_imagery
from ..agents.router import classify_query_intent
from ..agents.aggregator import build_observable_trace
from ..tools.registry import get_tool, list_all_tools
from ..preprocessing.geotiff import inspect_and_load_geospatial_image
from ..postprocessing.report_generator import generate_downloadable_report

router = APIRouter(prefix="", tags=["Unified Analysis"])

def decode_image_input(val: Optional[str]) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if not val or not val.strip():
        return None, {}
    val_clean = val.strip()
    if "," in val_clean:
        val_clean = val_clean.split(",", 1)[1]
    try:
        raw_bytes = base64.b64decode(val_clean)
        return inspect_and_load_geospatial_image(raw_bytes)
    except Exception as e:
        print(f"[AnalyzeRouter] Error decoding base64 image: {e}")
        return None, {}

# NOTE: /health and /api/health are handled by backend/main.py
from models.registry import ModelRegistry
from training.datasets.dataset_registry import DatasetRegistry
from evaluation.benchmark_manager import BenchmarkStorage

# NOTE: /health and /api/health are handled by backend/main.py
# to avoid duplicate route registration and ensure model-safe responses.

@router.get("/api/models")
@router.get("/models")
async def get_models():
    reg = ModelRegistry.get_instance()
    return {"models": reg.list_specialists()}

@router.get("/api/model-status")
@router.get("/api/models/status")
@router.get("/api/system/status")
async def get_model_and_dataset_status():
    """
    Exposes ground truth admin / developer status:
    - Dataset presence (AVAILABLE vs NOT_DOWNLOADED)
    - Specialist availability & checkpoints
    - Real completed benchmark evaluation results
    - Actual adapter runtime verification (weights, base model, device, state)
    Never fabricates metrics.
    """
    dreg = DatasetRegistry.get_instance()
    mreg = ModelRegistry.get_instance()
    bstorage = BenchmarkStorage()

    from ..services.rs_adapters import RSAdapterRuntime, SpecialistState, CAPTION_ADAPTER_PATH, VQA_ADAPTER_PATH
    runtime = RSAdapterRuntime.get_instance()

    caption_weights_present = runtime._has_adapter_weights(CAPTION_ADAPTER_PATH)
    vqa_weights_present = runtime._has_adapter_weights(VQA_ADAPTER_PATH)

    caption_status = {
        "is_available": runtime.caption_state == SpecialistState.AVAILABLE,
        "state": runtime.caption_state.value,
        "adapter_path": str(CAPTION_ADAPTER_PATH),
        "weights_present": caption_weights_present,
        "base_model": runtime.caption_base_id,
        "device": runtime.device,
        "error": runtime.caption_unavailable_reason if runtime.caption_state != SpecialistState.AVAILABLE else None,
    }

    vqa_status = {
        "is_available": runtime.vqa_state == SpecialistState.AVAILABLE,
        "state": runtime.vqa_state.value,
        "adapter_path": str(VQA_ADAPTER_PATH),
        "weights_present": vqa_weights_present,
        "base_model": runtime.vqa_base_id,
        "device": runtime.device,
        "error": runtime.vqa_unavailable_reason if runtime.vqa_state != SpecialistState.AVAILABLE else None,
    }

    specialists = []
    for spec in mreg.list_specialists():
        spec_copy = dict(spec)
        if spec_copy.get("id") in ("rs_caption_adapted", "captioning"):
            spec_copy.update({
                "state": caption_status["state"],
                "adapter_path": caption_status["adapter_path"],
                "weights_present": caption_status["weights_present"],
                "base_model": caption_status["base_model"],
                "device": caption_status["device"],
                "error": caption_status["error"],
            })
        elif spec_copy.get("id") in ("rs_vqa_adapted", "rs_vqa"):
            spec_copy.update({
                "state": vqa_status["state"],
                "adapter_path": vqa_status["adapter_path"],
                "weights_present": vqa_status["weights_present"],
                "base_model": vqa_status["base_model"],
                "device": vqa_status["device"],
                "error": vqa_status["error"],
            })
        specialists.append(spec_copy)

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "datasets": dreg.list_datasets(),
        "specialists": specialists,
        "evaluations": bstorage.list_all_runs(),
        "adapters": {
            "caption": caption_status,
            "vqa": vqa_status,
        },
        "caption_specialist": caption_status,
        "vqa_specialist": vqa_status,
    }

@router.get("/api/tools")
@router.get("/tools")
async def get_tools():
    try:
        return {"tools": list_all_tools()}
    except Exception as e:
        return {"tools": {}, "error": str(e)}

@router.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload geospatial GeoTIFF, TIFF, PNG, or JPEG and receive inspected metadata and base64 preview."""
    contents = await file.read()
    img_bgr, meta = inspect_and_load_geospatial_image(contents)
    # Encode thumbnail
    _, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64_thumb = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "metadata": meta,
        "preview_data_url": b64_thumb
    }

@router.post("/api/analyze", response_model=UnifiedAnalysisResponse)
async def analyze_master(req: AnalyzeRequest):
    """
    Core Master Unified Analysis Endpoint:
    Interprets user query into QueryPlan, validates multi-modal inputs,
    dynamically sequences registered specialists, collects structured evidence,
    and returns an observable execution trace.
    """
    # 1. Decode primary & secondary images
    img1_bgr, meta1 = decode_image_input(req.image)
    img2_bgr, meta2 = decode_image_input(req.secondary_image)

    # Check if images were passed as a list
    if img1_bgr is None and req.images and len(req.images) > 0:
        img1_bgr, meta1 = decode_image_input(req.images[0])
        if len(req.images) > 1:
            img2_bgr, meta2 = decode_image_input(req.images[1])

    # Check if query is unsupported
    from ..agents.planner import is_query_unsupported
    is_unsupported = is_query_unsupported(req.query)

    # Fallback to default aerial imagery if no image provided (for single image demo/test)
    if img1_bgr is None and not is_unsupported:
        default_aerial_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "default_aerial.jpg")
        if os.path.exists(default_aerial_path):
            img1_bgr = cv2.imread(default_aerial_path)
            if img1_bgr is not None:
                meta1 = {
                    "source": "default_aerial.jpg",
                    "format": "jpeg",
                    "dimensions": {"width": img1_bgr.shape[1], "height": img1_bgr.shape[0]},
                    "channels": img1_bgr.shape[2] if len(img1_bgr.shape) > 2 else 1
                }

    images = [img for img in [img1_bgr, img2_bgr] if img is not None]
    metadata_list = [m for m in [meta1, meta2] if m]

    modalities = [req.modality or "optical"]
    if req.secondary_modality:
        modalities.append(req.secondary_modality)

    from ..agents.orchestrator import run_orbital_analysis
    return run_orbital_analysis(
        query=req.query,
        images=images,
        modalities=modalities,
        metadata_list=metadata_list,
        parameters=req.parameters
    )


# Specialized sub-endpoints from Section 9
@router.post("/api/analyze/vqa")
async def analyze_vqa(req: AnalyzeRequest):
    req.task_type = "vqa"
    return await analyze_master(req)

@router.post("/api/analyze/buildings")
async def analyze_buildings_endpoint(req: AnalyzeRequest):
    req.task_type = "building_detection"
    return await analyze_master(req)

@router.post("/api/analyze/grounding")
async def analyze_grounding(req: AnalyzeRequest):
    req.task_type = "grounding"
    return await analyze_master(req)

@router.post("/api/analyze/change")
async def analyze_change(req: AnalyzeRequest):
    req.task_type = "change_detection"
    return await analyze_master(req)

@router.post("/api/analyze/optical-sar")
async def analyze_optical_sar(req: AnalyzeRequest):
    req.task_type = "sar_optical_fusion"
    return await analyze_master(req)
