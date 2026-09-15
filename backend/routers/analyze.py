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
# to avoid duplicate route registration and ensure model-safe responses.

@router.get("/api/models")
@router.get("/models")
async def get_models():
    return {
        "models": [
            {
                "id": "yolo-segmentation-building_model.pt",
                "name": "YOLO Structural Footprint Instance Segmentation",
                "task": "building_footprint_extraction",
                "weights": "SpaceNet & Satellite Aerial Imagery",
                "inference": "Tiled 512px with IoU overlap suppression"
            },
            {
                "id": "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
                "name": "BigEarthNet v2.0 Corine Land Cover Classifier (reBEN)",
                "task": "multispectral_land_cover",
                "taxonomy": "19-class Corine Land Cover",
                "reference": "Clasen et al., IGARSS 2025"
            },
            {
                "id": "classical-cv-fusion-engine-v2",
                "name": "Optical-SAR Cross-Modal Telemetry Extractor",
                "task": "joint_optical_sar_reasoning",
                "metrics": ["SSIM", "CrossCorrelation", "Backscatter_dB", "SpeckleIndex"]
            },
            {
                "id": "rs-change-detector-v2",
                "name": "Bi-Temporal Radiometric & Structural Change Model",
                "task": "bi_temporal_change_vqa",
                "benchmark": "CDVQA"
            }
        ]
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
    Interprets user query, validates inputs, dynamically executes specialist tools,
    aggregates evidence, and returns an observable execution trace.
    """
    t_start = time.time()
    steps_log: List[Dict[str, Any]] = []

    # 1. Decode primary & secondary images
    img1_bgr, meta1 = decode_image_input(req.image)
    img2_bgr, meta2 = decode_image_input(req.secondary_image)
    
    # Fallback to default aerial imagery if no image provided
    if img1_bgr is None:
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
    if not images:
        raise HTTPException(status_code=400, detail="No valid image provided or found for remote sensing analysis.")

    modalities = [req.modality or "optical"]
    if req.secondary_modality:
        modalities.append(req.secondary_modality)

    # 2. Determine task intent
    task_type = req.task_type or classify_query_intent(req.query, image_count=len(images), modalities=modalities)

    # 3. Input Validation
    val_t0 = time.time()
    validation = validate_input_imagery(task_type, images, modalities=modalities)
    if validation.get("compatibility") == "error":
        err_msg = "; ".join(validation.get("errors", ["Input validation rejected request."]))
        raise HTTPException(status_code=400, detail=err_msg)

    steps_log.append({
        "step": 1,
        "tool": "input_validator",
        "description": "Inspecting geospatial format, dimensions, sensor modalities, and compatibility",
        "input_summary": f"{len(images)} image(s) provided; Modalities: {', '.join(modalities)}",
        "output_summary": f"Compatibility: {validation['compatibility'].upper()} ({len(validation['notes'])} checks verified)",
        "duration_ms": (time.time() - val_t0) * 1000,
        "status": "success",
        "success": True,
        "confidence_source": "none",
        "parameters": {"task": task_type, "modalities": modalities}
    })

    # 4. Route and Execute Tools
    building_response = None
    grounding_response = None
    change_response = None
    fusion_response = None
    land_cover_response = None
    answer = ""
    confidence = 0.85
    confidence_level = "High"
    tools_used = []

    tool_inputs = {
        "query": req.query,
        "image": img1_bgr,
        "secondary_image": img2_bgr,
        "optical_image": img1_bgr,
        "sar_image": img2_bgr,
        "metadata": meta1
    }

    # Execute specific tool workflows
    if task_type == "building_detection":
        b_tool = get_tool("building_detection")
        tools_used.append(b_tool.id)
        b_t0 = time.time()
        b_res = b_tool.run(tool_inputs, req.parameters)
        b_dur = (time.time() - b_t0) * 1000

        if "error" in b_res:
            raise HTTPException(status_code=503, detail=f"Building detection unavailable: {b_res['error']}")

        building_response = BuildingAnalysisResponse(
            success=True,
            image_dimensions=b_res["image_dimensions"],
            tiles_processed=b_res["tiles_processed"],
            raw_detections_count=b_res["raw_detections_count"],
            merged_detections_count=b_res["merged_detections_count"],
            building_count=b_res["building_count"],
            high_confidence_count=b_res["high_confidence_count"],
            medium_confidence_count=b_res["medium_confidence_count"],
            low_confidence_count=b_res["low_confidence_count"],
            partial_count=b_res["partial_count"],
            confidence=b_res["confidence"],
            confidence_level=b_res["confidence_level"],
            validation_status=b_res["validation_status"],
            validation=b_res["validation"],
            detections=b_res["detections"],
            geojson=b_res.get("geojson")
        )
        answer = (
            f"The deep-learning building footprint pipeline extracted {b_res['building_count']} structures "
            f"across {b_res['tiles_processed']} overlapping tiles. Confidence breakdown: {b_res['high_confidence_count']} high, "
            f"{b_res['medium_confidence_count']} medium, and {b_res['low_confidence_count']} low certainty footprints."
        )
        confidence = b_res["confidence"]
        confidence_level = b_res["confidence_level"]

        steps_log.append({
            "step": 2,
            "tool": b_tool.name,
            "description": "512px sliding-window tiling and YOLO instance segmentation with polygon IoU deduplication",
            "input_summary": f"Full scene ({meta1.get('dimensions', {}).get('width', 1000)}x{meta1.get('dimensions', {}).get('height', 1000)})",
            "output_summary": f"Detected {b_res['building_count']} building footprints ({b_res['high_confidence_count']} high confidence)",
            "duration_ms": b_dur,
            "status": "success",
            "success": True,
            "confidence_source": "real_inference",
            "parameters": b_tool.permitted_parameters
        })

    elif task_type == "sar_optical_fusion":
        f_tool = get_tool("optical_sar")
        tools_used.append(f_tool.id)
        f_t0 = time.time()
        f_res = f_tool.run(tool_inputs, req.parameters)
        f_dur = (time.time() - f_t0) * 1000

        if "error" in f_res:
            raise HTTPException(status_code=400, detail=f"Optical-SAR fusion unavailable: {f_res['error']}")

        fusion_response = f_res.get("metrics")
        answer = f_res.get("interpretation", "Optical-SAR fusion analysis complete.")
        confidence = 0.82
        confidence_level = "High"

        steps_log.append({
            "step": 2,
            "tool": f_tool.name,
            "description": "Heuristic cross-modal telemetry baseline: spectral indices, SAR backscatter dB, SSIM",
            "input_summary": "Co-registered Optical + SAR sensor matrices",
            "output_summary": f"SSIM: {f_res.get('metrics', {}).get('cross_modal', {}).get('structural_similarity', 0.7):.2f}; Mean Backscatter: {f_res.get('metrics', {}).get('sar', {}).get('mean_backscatter_db', -14):.1f} dB",
            "duration_ms": f_dur,
            "status": "success",
            "success": True,
            "confidence_source": "heuristic",
            "parameters": f_tool.permitted_parameters
        })

    elif task_type in ["change_detection", "change_vqa"]:
        c_tool = get_tool("change_detection")
        tools_used.append(c_tool.id)
        c_t0 = time.time()
        c_res = c_tool.run(tool_inputs, req.parameters)
        c_dur = (time.time() - c_t0) * 1000

        if "error" in c_res:
            raise HTTPException(status_code=400, detail=f"Change detection unavailable for this image pair: {c_res['error']}")

        change_response = c_res
        answer = c_res.get("answer", "Bi-temporal change analysis completed.")
        confidence = c_res.get("confidence", 0.85)
        confidence_level = c_res.get("confidence_level", "High")

        steps_log.append({
            "step": 2,
            "tool": c_tool.name,
            "description": "Pixel-differencing heuristic baseline: adaptive thresholding & morphological contour clustering",
            "input_summary": "T1 (initial) and T2 (subsequent) temporal captures",
            "output_summary": f"Alteration detected across {c_res.get('change_percentage', 0):.1f}% surface ({c_res.get('change_clusters', 0)} clusters)",
            "duration_ms": c_dur,
            "status": "success",
            "success": True,
            "confidence_source": "heuristic",
            "parameters": c_tool.permitted_parameters
        })

    elif task_type == "grounding":
        g_tool = get_tool("grounding")
        tools_used.append(g_tool.id)
        g_t0 = time.time()
        g_res = g_tool.run(tool_inputs, req.parameters)
        g_dur = (time.time() - g_t0) * 1000

        if "error" in g_res:
            raise HTTPException(status_code=422, detail=f"Spatial grounding unavailable: {g_res['error']}")

        grounding_response = [
            GroundingItem(
                target=r["target"],
                region=GroundingRegion(**r["region"]),
                confidence=r["confidence"],
                label=r["label"]
            )
            for r in g_res.get("regions", [])
        ]
        top_reg = g_res.get("primary_region")
        if top_reg:
            answer = (
                f"Spectral grounding baseline localized '{g_res.get('target', 'feature')}' at: "
                f"X: {top_reg.get('x_percent', 0)}%, Y: {top_reg.get('y_percent', 0)}%, "
                f"Width: {top_reg.get('w_percent', 0)}%, Height: {top_reg.get('h_percent', 0)}%."
            )
            confidence = 0.80
        else:
            answer = f"No '{g_res.get('target', 'target feature')}' was localized in this scene by spectral thresholding."
            confidence = 0.40
        confidence_level = "High" if confidence >= 0.75 else "Medium"

        steps_log.append({
            "step": 2,
            "tool": g_tool.name,
            "description": "Spectral thresholding & connected component contour extraction for target localization",
            "input_summary": f"Query targeting '{g_res.get('target', 'grounding feature')}'",
            "output_summary": f"{len(grounding_response)} region(s) localized via spectral heuristic",
            "duration_ms": g_dur,
            "status": "success",
            "success": True,
            "confidence_source": "heuristic",
            "parameters": g_tool.permitted_parameters
        })

    elif task_type == "caption":
        cap_tool = get_tool("caption")
        tools_used.append(cap_tool.id)
        cap_t0 = time.time()
        cap_res = cap_tool.run(tool_inputs, req.parameters)
        cap_dur = (time.time() - cap_t0) * 1000

        answer = cap_res.get("caption", "Scene caption generated.")
        confidence = cap_res.get("confidence", 0.88)
        confidence_level = cap_res.get("confidence_level", "High")
        land_cover_response = {
            "top_label": cap_res.get("top_label"),
            "active_labels": cap_res.get("active_labels"),
            "surface_breakdown": cap_res.get("surface_breakdown")
        }

        steps_log.append({
            "step": 2,
            "tool": cap_tool.name,
            "description": "VRSBench multi-attribute scene captioning with BigEarthNet land-cover grounding",
            "input_summary": "Single optical remote-sensing scene",
            "output_summary": f"Primary terrain: {cap_res.get('top_label')}",
            "duration_ms": cap_dur,
            "status": "success",
            "success": True,
            "confidence_source": "real_inference",
            "parameters": cap_tool.permitted_parameters
        })

    else:  # General VQA / Land-cover
        vqa_tool = get_tool("vqa")
        tools_used.append(vqa_tool.id)
        v_t0 = time.time()
        v_res = vqa_tool.run(tool_inputs, req.parameters)
        v_dur = (time.time() - v_t0) * 1000

        answer = v_res.get("answer", "Remote sensing analysis completed.")
        confidence = v_res.get("confidence", 0.85)
        confidence_level = v_res.get("confidence_level", "High")
        land_cover_response = {
            "top_label": v_res.get("top_label"),
            "surface_breakdown": v_res.get("surface_breakdown")
        }
        if v_res.get("building_analysis"):
            ba = v_res["building_analysis"]
            building_response = BuildingAnalysisResponse(
                success=True,
                image_dimensions=ba["image_dimensions"],
                tiles_processed=ba["tiles_processed"],
                raw_detections_count=ba["raw_detections_count"],
                merged_detections_count=ba["merged_detections_count"],
                building_count=ba["building_count"],
                high_confidence_count=ba["high_confidence_count"],
                medium_confidence_count=ba["medium_confidence_count"],
                low_confidence_count=ba["low_confidence_count"],
                partial_count=ba["partial_count"],
                confidence=ba["confidence"],
                confidence_level=ba["confidence_level"],
                validation_status=ba["validation_status"],
                validation=ba["validation"],
                detections=ba["detections"],
                geojson=ba.get("geojson")
            )

        steps_log.append({
            "step": 2,
            "tool": vqa_tool.name,
            "description": "Remote-sensing visual question answering and BigEarthNet taxonomy inference",
            "input_summary": f"User query: '{req.query}'",
            "output_summary": f"Answer formulated with {confidence_level} confidence ({confidence*100:.0f}%)",
            "duration_ms": v_dur,
            "status": "success",
            "success": True,
            "confidence_source": "real_inference",
            "parameters": vqa_tool.permitted_parameters
        })

    total_duration_ms = (time.time() - t_start) * 1000
    primary_tool = get_tool(tools_used[0]).to_spec() if tools_used else {}

    # Build observable execution trace
    trace = build_observable_trace(
        task_type=task_type,
        steps=steps_log,
        total_duration_ms=total_duration_ms,
        validation=validation,
        primary_tool=primary_tool
    )

    evidence_dict: Dict[str, Any] = {}
    if land_cover_response:
        evidence_dict["land_cover"] = land_cover_response
    if building_response:
        evidence_dict["building_count"] = building_response.building_count
        evidence_dict["high_confidence_count"] = building_response.high_confidence_count
    if fusion_response:
        evidence_dict["fusion_metrics"] = fusion_response
    if change_response:
        evidence_dict["change_percentage"] = change_response.get("change_percentage")
    if grounding_response:
        evidence_dict["grounding_regions"] = [g.dict() for g in grounding_response]

    return UnifiedAnalysisResponse(
        success=True,
        task_type=task_type,
        answer=answer,
        confidence=round(confidence, 3),
        confidence_level=confidence_level,
        tools_used=tools_used,
        input_modality=modalities[0],
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
        execution_time_ms=round(total_duration_ms, 2),
        evidence=evidence_dict,
        execution_trace=trace,
        warnings=validation.get("warnings", []),
        building_analysis=building_response,
        grounding=grounding_response,
        change_map=change_response,
        fusion_metrics=fusion_response,
        land_cover=land_cover_response,
        mode="model",
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
