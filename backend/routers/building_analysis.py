import base64
import io
import cv2
import time
import asyncio
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from ..services.building_detector import BuildingDetector
from ..services.tiling import generate_tiles, translate_detection_to_global
from ..services.duplicate_removal import merge_duplicate_detections
from ..services.postprocessing import filter_and_postprocess_detections
from ..services.validation import evaluate_accuracy

router = APIRouter(prefix="", tags=["Building Analysis"])

class AnalyzeBuildingRequest(BaseModel):
    image: Optional[str] = None
    tile_size: Optional[int] = 640
    overlap: Optional[float] = 0.20
    conf_threshold: Optional[float] = 0.15
    iou_threshold: Optional[float] = 0.45
    ground_truth_count: Optional[int] = None

def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes to BGR numpy array using OpenCV or PIL."""
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback to PIL
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img

def load_image_from_base64(b64_str: str) -> np.ndarray:
    """Decode base64 data URL to BGR numpy array."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return load_image_from_bytes(raw)

@router.post("/analyze/buildings")
async def analyze_buildings(
    file: Optional[UploadFile] = File(None),
    payload: Optional[AnalyzeBuildingRequest] = None
):
    """
    Main building detection endpoint:
    Processes full-resolution satellite image via overlapping tiles,
    extracts rooftop segmentation polygons, merges duplicates, and counts unique buildings.
    """
    start_time = time.perf_counter()
    img_bgr = None

    if file is not None:
        contents = await file.read()
        try:
            img_bgr = load_image_from_bytes(contents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode uploaded image: {e}")
    elif payload and payload.image:
        try:
            img_bgr = load_image_from_base64(payload.image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {e}")
    else:
        raise HTTPException(status_code=400, detail="No image provided. Supply 'file' or JSON 'image'.")

    if img_bgr is None or img_bgr.size == 0:
        raise HTTPException(status_code=400, detail="Invalid or empty image.")

    img_height, img_width = img_bgr.shape[:2]

    # Preprocess: downscale very large images to reduce tile count and inference time.
    # Building detection works well at 1500px — structures are still clearly resolved.
    MAX_DIM = 1500
    if max(img_width, img_height) > MAX_DIM:
        scale = MAX_DIM / max(img_width, img_height)
        new_w = int(img_width * scale)
        new_h = int(img_height * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_height, img_width = img_bgr.shape[:2]

    # Parameters
    tile_size = payload.tile_size if payload and payload.tile_size else 640
    overlap = payload.overlap if payload and payload.overlap else 0.20
    conf_threshold = payload.conf_threshold if payload and payload.conf_threshold else 0.15
    iou_threshold = payload.iou_threshold if payload and payload.iou_threshold else 0.45
    gt_count = payload.ground_truth_count if payload else None

    # Step 1: Generate tiles
    tiles = generate_tiles(img_width, img_height, tile_size=tile_size, overlap=overlap)
    detector = BuildingDetector.get_instance()

    # Step 2: Crop all tiles and run batch inference (offloaded to thread pool)
    tile_crops = [img_bgr[y1:y2, x1:x2] for (x1, y1, x2, y2) in tiles]
    loop = asyncio.get_event_loop()
    batch_results = await loop.run_in_executor(None, detector.predict_batch, tile_crops, conf_threshold)

    raw_tile_detections: List[Dict[str, Any]] = []
    for tile_coords, tile_results in zip(tiles, batch_results):
        for det in tile_results:
            global_det = translate_detection_to_global(
                tile_coords=tile_coords,
                bbox=det["bbox"],
                polygon=det["polygon"],
                img_width=img_width,
                img_height=img_height
            )
            raw_tile_detections.append({
                "confidence": det["confidence"],
                **global_det,
            })

    # Step 3: Duplicate removal across tiles
    merged_detections = merge_duplicate_detections(
        raw_tile_detections,
        box_iou_threshold=iou_threshold,
        mask_iou_threshold=0.35,
        centroid_dist_ratio=0.40
    )

    # Step 4: False-positive filtering, edge building handling, and ID assignment
    final_detections, stats = filter_and_postprocess_detections(
        merged_detections,
        img_width=img_width,
        img_height=img_height,
        min_confidence=conf_threshold
    )

    # Step 5: Accuracy evaluation against ground truth if provided
    validation = evaluate_accuracy(
        predicted_count=stats["building_count"],
        ground_truth_count=gt_count,
        predicted_bboxes=[d["bbox"] for d in final_detections]
    )

    return {
        "success": True,
        "image_dimensions": {"width": img_width, "height": img_height},
        "tiles_processed": len(tiles),
        "raw_detections_count": len(raw_tile_detections),
        "merged_detections_count": len(merged_detections),
        "building_count": stats["building_count"],
        "high_confidence_count": stats["high_confidence_count"],
        "medium_confidence_count": stats["medium_confidence_count"],
        "low_confidence_count": stats["low_confidence_count"],
        "partial_count": stats["partial_count"],
        "confidence": stats["confidence"],
        "confidence_level": stats["confidence_level"],
        "validation_status": stats["validation_status"],
        "validation": validation,
        "detections": final_detections,
    }

@router.post("/analyze/buildings/validate")
async def validate_buildings(
    predicted_count: int,
    ground_truth_count: int
):
    """Direct validation metric calculator."""
    return evaluate_accuracy(predicted_count, ground_truth_count)
