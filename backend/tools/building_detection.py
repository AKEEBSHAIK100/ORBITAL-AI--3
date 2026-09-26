import time
from typing import Any, Dict, List, Optional
import numpy as np
import cv2

from .base import BaseTool
from ..services.building_detector import BuildingDetector
from ..services.tiling import generate_tiles, translate_detection_to_global
from ..services.duplicate_removal import merge_duplicate_detections
from ..services.postprocessing import filter_and_postprocess_detections
from ..services.validation import evaluate_accuracy
from ..services.cache_manager import (
    compute_image_hash,
    build_cache_key,
    get_query_result_cache,
)

class BuildingDetectionTool(BaseTool):
    id = "building_detection"
    name = "Tiled Deep-Learning Building Footprint Segmentation Pipeline"
    description = "Dedicated building instance segmentation pipeline with 512px sliding-window tiling, overlap deduplication, polygon extraction, and individual structural rooftop auditing."
    supported_tasks = ["building_detection"]
    modalities = ["optical"]
    adapter = "PyTorch YOLO Segmentation Engine + Tiled Coordinate Translation & IoU Deduplication"
    domain_adaptation = "SpaceNet & high-resolution satellite imagery for dense urban rooftop footprints"
    model_id = "yolo-segmentation-building_model.pt"
    permitted_parameters = {
        "tile_size": 512,
        "tile_overlap": 64,
        "conf_threshold": 0.20,
        "box_iou_threshold": 0.45,
        "mask_iou_threshold": 0.35,
        "min_area_px": 25.0
    }

    def __init__(self, detector: Optional[BuildingDetector] = None):
        self._detector = detector

    @property
    def detector(self) -> BuildingDetector:
        if self._detector is None:
            self._detector = BuildingDetector.get_instance()
        return self._detector

    @detector.setter
    def detector(self, value: Optional[BuildingDetector]) -> None:
        self._detector = value

    def run(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        params = {**self.permitted_parameters, **(parameters or {})}
        img_bgr = inputs.get("image")

        if img_bgr is None or not isinstance(img_bgr, np.ndarray):
            return {"error": "Invalid or missing image array", "status": "error"}

        h, w = img_bgr.shape[:2]
        tile_size = int(params.get("tile_size", 512))
        overlap = int(params.get("tile_overlap", 64))
        conf_thresh = float(params.get("conf_threshold", 0.20))
        box_iou_thresh = float(params.get("box_iou_threshold", 0.45))
        mask_iou_thresh = float(params.get("mask_iou_threshold", 0.35))
        min_area = float(params.get("min_area_px", 25.0))

        # --- Query result cache lookup ---
        result_cache = get_query_result_cache()
        img_hash = compute_image_hash(img_bgr)
        cache_key = build_cache_key(
            image_identity=img_hash,
            query="building_detection",
            task="building_detection",
            model_id=self.model_id,
            model_version="v1",
            adapter_identity="none",
            parameters={
                "tile_size": tile_size,
                "overlap": overlap,
                "conf_threshold": conf_thresh,
                "box_iou_threshold": box_iou_thresh,
                "mask_iou_threshold": mask_iou_thresh,
                "min_area_px": min_area,
            },
        )
        cached = result_cache.get(cache_key)
        if cached is not None:
            cached["duration_ms"] = round((time.time() - t0) * 1000, 2)
            cached["cache_hit"] = True
            return cached

        # 1. Tiling
        overlap_ratio = 0.20 if isinstance(overlap, (int, float)) and overlap > 1 else float(overlap)
        tiles = generate_tiles(w, h, tile_size=tile_size, overlap=overlap_ratio)
        tile_images = [img_bgr[y1:y2, x1:x2] for (x1, y1, x2, y2) in tiles]

        # 2. Batch inference
        batch_results = self.detector.predict_batch(tile_images, conf_threshold=conf_thresh)

        # 3. Global coordinate mapping
        all_raw_detections = []
        for tile_coords, tile_dets in zip(tiles, batch_results):
            for det in tile_dets:
                global_det = translate_detection_to_global(
                    tile_coords=tile_coords,
                    bbox=det["bbox"],
                    polygon=det["polygon"],
                    img_width=w,
                    img_height=h
                )
                all_raw_detections.append({
                    "confidence": det["confidence"],
                    **global_det,
                })

        # 4. Duplicate removal across tile overlaps
        merged = merge_duplicate_detections(
            all_raw_detections,
            box_iou_threshold=box_iou_thresh,
            mask_iou_threshold=mask_iou_thresh
        )

        # 5. Postprocessing & Filtering
        post_result = filter_and_postprocess_detections(
            merged,
            img_width=w,
            img_height=h,
            min_area=min_area
        )
        # Returns (filtered_detections, stats_dict)
        final_detections, pp_stats = post_result if isinstance(post_result, tuple) else (post_result, {})

        # 6. Evaluation against Ground Truth (if provided)
        gt_count = inputs.get("ground_truth_count")
        gt_bboxes = inputs.get("ground_truth_bboxes")
        predicted_bboxes = [d["bbox"] for d in final_detections]
        accuracy_eval = evaluate_accuracy(
            predicted_count=len(final_detections),
            ground_truth_count=gt_count,
            ground_truth_bboxes=gt_bboxes,
            predicted_bboxes=predicted_bboxes
        )

        # 7. Confidence & Counts (prefer stats from postprocessor)
        high_c = pp_stats.get("high_confidence_count", sum(1 for d in final_detections if d.get("confidence_tier") == "high"))
        med_c = pp_stats.get("medium_confidence_count", sum(1 for d in final_detections if d.get("confidence_tier") == "medium"))
        low_c = pp_stats.get("low_confidence_count", sum(1 for d in final_detections if d.get("confidence_tier") == "low"))
        partial_c = pp_stats.get("partial_count", sum(1 for d in final_detections if d.get("is_partial", False)))
        avg_conf = pp_stats.get("confidence")
        if avg_conf is None and final_detections:
            avg_conf = sum(d["confidence"] for d in final_detections) / len(final_detections)
        conf_level = pp_stats.get("confidence_level")

        # 8. Generate GeoJSON features
        features = []
        for d in final_detections:
            poly = d.get("polygon", [])
            if len(poly) >= 3:
                coords = [[float(pt[0]), float(pt[1])] for pt in poly]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                features.append({
                    "type": "Feature",
                    "id": d["id"],
                    "properties": {
                        "id": d["id"],
                        "confidence": round(d["confidence"], 4),
                        "confidence_tier": d["confidence_tier"],
                        "area_px": round(d["area"], 1),
                        "is_partial": d["is_partial"]
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                })
        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }

        duration_ms = (time.time() - t0) * 1000

        result = {
            "status": "success",
            "success": True,
            "image_dimensions": {"width": w, "height": h},
            "tiles_processed": len(tiles),
            "raw_detections_count": len(all_raw_detections),
            "merged_detections_count": len(merged),
            "building_count": len(final_detections),
            "detected_count": len(final_detections),
            "high_confidence_count": high_c,
            "medium_confidence_count": med_c,
            "low_confidence_count": low_c,
            "partial_count": partial_c,
            "partial_detections": partial_c,
            "confidence": round(avg_conf, 3),
            "confidence_level": conf_level,
            "validation_status": accuracy_eval.get("status"),
            "validation": accuracy_eval,
            "detections": final_detections,
            "geojson": geojson_data,
            "duration_ms": round(duration_ms, 2),
            "cache_hit": False,
        }
        result_cache.set(cache_key, result)
        return result
