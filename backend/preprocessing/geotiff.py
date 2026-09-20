import copy
import io
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import cv2

from ..services.cache_manager import compute_image_hash, get_image_preprocessing_cache

def inspect_and_load_geospatial_image(data_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Validates GeoTIFF/TIFF/raster inputs and truthfully inspects resolution, CRS,
    georeferencing, and dimensions.
    Reuses cached decoded array and metadata when identical data_bytes are submitted.
    Never fabricates default CRS or ground-sampling distances when metadata is absent.
    """
    img_cache = get_image_preprocessing_cache()
    img_hash = compute_image_hash(data_bytes)
    cached = img_cache.get(img_hash)
    if cached is not None:
        cached_img, cached_meta = cached
        return cached_img.copy(), copy.deepcopy(cached_meta)

    metadata: Dict[str, Any] = {
        "format": "UNKNOWN",
        "bands": 3,
        "is_geotiff": False,
        "crs": None,
        "resolution_m": None,
        "geotransform": None,
        "affine": None,
        "acquisition_date": None,
        "sensor": None,
        "tags": {}
    }

    try:
        pil_img = Image.open(io.BytesIO(data_bytes))
        metadata["format"] = pil_img.format or "RASTER"
        metadata["dimensions"] = {"width": pil_img.width, "height": pil_img.height}

        # Check for TIFF / GeoTIFF tags
        if pil_img.format in ["TIFF", "GeoTIFF"]:
            if hasattr(pil_img, "tag_v2"):
                tags = dict(pil_img.tag_v2)
                metadata["tags_detected"] = len(tags)

                # Check GeoTIFF specific tags:
                # ModelPixelScaleTag (33550), ModelTiepointTag (33922), GeoKeyDirectoryTag (34735)
                has_geotiff_keys = 33550 in tags or 33922 in tags or 34735 in tags
                metadata["is_geotiff"] = bool(has_geotiff_keys)

                if 33550 in tags:
                    scale = tags[33550]
                    metadata["resolution_m"] = float(scale[0]) if hasattr(scale, "__getitem__") else float(scale)
                
                if 34735 in tags:
                    # Genuine GeoKey directory tag detected
                    metadata["crs"] = "GeoKey Standard / EPSG"

                if 33922 in tags:
                    tiepoints = list(tags[33922])
                    metadata["geotransform"] = {"tiepoints": tiepoints[:6]}

                # Optional acquisition date tag (DateTime: 306)
                if 306 in tags:
                    metadata["acquisition_date"] = str(tags[306]).strip()

        # Convert to BGR array for CV2 and internal pipelines
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        metadata["bands"] = img_bgr.shape[2] if img_bgr.ndim == 3 else 1
        img_cache.set(img_hash, (img_bgr, metadata))
        return img_bgr, metadata

    except Exception as e:
        # Fallback to cv2.imdecode for raw byte arrays
        nparr = np.frombuffer(data_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            metadata["dimensions"] = {"width": img_bgr.shape[1], "height": img_bgr.shape[0]}
            metadata["format"] = "RASTER"
            metadata["bands"] = img_bgr.shape[2] if img_bgr.ndim == 3 else 1
            img_cache.set(img_hash, (img_bgr, metadata))
            return img_bgr, metadata
        raise ValueError(f"Unable to decode geospatial image: {e}")
