import os
import base64
import io
import cv2
import time
import asyncio
import urllib.request
import urllib.error
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from ..services.building_detector import BuildingDetector
from ..services.tiling import generate_tiles, translate_detection_to_global
from ..services.duplicate_removal import merge_duplicate_detections
from ..services.postprocessing import filter_and_postprocess_detections
from ..services.validation import evaluate_accuracy

router = APIRouter(prefix="", tags=["Building Analysis"])

DEFAULT_AERIAL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "default_aerial.jpg")
)

def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes to BGR numpy array using OpenCV or PIL."""
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

def load_image_from_any(val: str) -> np.ndarray:
    """
    Robust loader: accepts HTTP/HTTPS URLs, base64 data URLs, local file paths,
    or falls back to the bundled high-resolution aerial image.
    """
    if not val or not val.strip():
        if os.path.exists(DEFAULT_AERIAL_PATH):
            return cv2.imread(DEFAULT_AERIAL_PATH)
        raise ValueError("Empty image string provided and default image not found.")

    val_clean = val.strip()

    # 1. HTTP / HTTPS URL
    if val_clean.startswith("http://") or val_clean.startswith("https://"):
        # Check if local cache matches default unsplash
        if "photo-1472146936668-d987bf0a6e38" in val_clean and os.path.exists(DEFAULT_AERIAL_PATH):
            img = cv2.imread(DEFAULT_AERIAL_PATH)
            if img is not None and img.size > 0:
                return img
        try:
            req = urllib.request.Request(
                val_clean,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SatQuery/1.0"}
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = resp.read()
            return load_image_from_bytes(data)
        except Exception as err:
            print(f"[BuildingAnalysis] Failed to fetch image URL ({err}), checking local fallback...")
            if os.path.exists(DEFAULT_AERIAL_PATH):
                return cv2.imread(DEFAULT_AERIAL_PATH)
            raise ValueError(f"Could not load image from URL: {err}")

    # 2. Local file path
    if os.path.isfile(val_clean):
        img = cv2.imread(val_clean)
        if img is not None and img.size > 0:
            return img

    # 3. Base64 data URL or raw base64
    try:
        return load_image_from_base64(val_clean)
    except Exception as b64_err:
        if os.path.exists(DEFAULT_AERIAL_PATH):
            return cv2.imread(DEFAULT_AERIAL_PATH)
        raise ValueError(f"Failed to decode base64 image: {b64_err}")

@router.post("/analyze/buildings")
@router.post("/api/buildings")  # alias for Vercel/Express proxy compatibility
async def analyze_buildings(request: Request):
    """
    Main building detection endpoint:
    Accepts BOTH JSON payloads ({ "image": "...", ... })
    AND multipart/form-data with file upload or image field.
    Processes full-resolution satellite image via overlapping tiles,
    extracts rooftop segmentation polygons, merges duplicates, and counts unique buildings.
    """
    content_type = request.headers.get("content-type", "").lower()
    img_bgr = None

    tile_size = 640
    overlap = 0.20
    conf_threshold = 0.15
    iou_threshold = 0.45
    gt_count = None

    if "multipart/form-data" in content_type:
        form = await request.form()
        file_obj = form.get("file")
        if file_obj and hasattr(file_obj, "read"):
            contents = await file_obj.read()
            try:
                img_bgr = load_image_from_bytes(contents)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode uploaded image file: {e}")
        elif "image" in form:
            img_val = form["image"]
            try:
                img_bgr = load_image_from_any(str(img_val))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load image from form: {e}")
        
        if "tile_size" in form:
            tile_size = int(form["tile_size"])
        if "overlap" in form:
            overlap = float(form["overlap"])
        if "conf_threshold" in form:
            conf_threshold = float(form["conf_threshold"])
        if "iou_threshold" in form:
            iou_threshold = float(form["iou_threshold"])
        if "ground_truth_count" in form and str(form["ground_truth_count"]).isdigit():
            gt_count = int(form["ground_truth_count"])
    else:
        # JSON body or empty fallback
        try:
            body = await request.json()
        except Exception:
            body = {}
        
        img_val = body.get("image")
        if img_val:
            try:
                img_bgr = load_image_from_any(str(img_val))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load image from JSON: {e}")
        elif os.path.exists(DEFAULT_AERIAL_PATH):
            img_bgr = cv2.imread(DEFAULT_AERIAL_PATH)

        tile_size = body.get("tile_size", 640)
        overlap = body.get("overlap", 0.20)
        conf_threshold = body.get("conf_threshold", 0.15)
        iou_threshold = body.get("iou_threshold", 0.45)
        gt_count = body.get("ground_truth_count")

    # If still no image, use default aerial fallback
    if img_bgr is None or img_bgr.size == 0:
        if os.path.exists(DEFAULT_AERIAL_PATH):
            img_bgr = cv2.imread(DEFAULT_AERIAL_PATH)
        else:
            raise HTTPException(status_code=400, detail="No image provided. Supply 'file' or JSON 'image'.")

    img_height, img_width = img_bgr.shape[:2]

    # Preprocess: downscale very large images to reduce tile count and inference time.
    MAX_DIM = 1500
    if max(img_width, img_height) > MAX_DIM:
        scale = MAX_DIM / max(img_width, img_height)
        new_w = int(img_width * scale)
        new_h = int(img_height * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_height, img_width = img_bgr.shape[:2]

    # Step 1: Generate tiles
    tiles = generate_tiles(img_width, img_height, tile_size=tile_size, overlap=overlap)
    detector = BuildingDetector.get_instance()

    if not detector.is_available:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "MODEL_UNAVAILABLE",
                "message": f"Building detection model is not loaded: {detector.load_error}",
                "mode": "model_unavailable",
            },
        )

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
        "mode": "model",
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
