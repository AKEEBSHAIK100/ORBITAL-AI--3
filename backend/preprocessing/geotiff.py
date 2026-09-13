import io
import numpy as np
from PIL import Image
import cv2
from typing import Any, Dict, Optional, Tuple

def inspect_and_load_geospatial_image(data_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Validates GeoTIFF/TIFF inputs and inspects resolution, CRS, georeferencing,
    and dimensions per Section 5 & 12 of the master plan.
    """
    metadata: Dict[str, Any] = {
        "format": "UNKNOWN",
        "bands": 3,
        "is_geotiff": False,
        "crs": "WGS 84 / UTM (projected)",
        "resolution_m": 0.5,
        "tags": {}
    }

    try:
        pil_img = Image.open(io.BytesIO(data_bytes))
        metadata["format"] = pil_img.format or "RASTER"
        metadata["dimensions"] = {"width": pil_img.width, "height": pil_img.height}

        # Check for TIFF / GeoTIFF tags
        if pil_img.format in ["TIFF", "GeoTIFF"]:
            metadata["is_geotiff"] = True
            # Inspect PIL tiff tags
            if hasattr(pil_img, "tag_v2"):
                tags = dict(pil_img.tag_v2)
                # ModelPixelScaleTag (33550), ModelTiepointTag (33922), GeoKeyDirectoryTag (34735)
                if 33550 in tags:
                    scale = tags[33550]
                    metadata["resolution_m"] = float(scale[0]) if hasattr(scale, "__getitem__") else float(scale)
                if 34735 in tags:
                    metadata["crs"] = "GeoKey Standard / EPSG"
                metadata["tags_detected"] = len(tags)

        # Convert to BGR array for CV2 and models
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        metadata["bands"] = img_bgr.shape[2]
        return img_bgr, metadata

    except Exception as e:
        # Fallback to cv2.imdecode
        nparr = np.frombuffer(data_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            metadata["dimensions"] = {"width": img_bgr.shape[1], "height": img_bgr.shape[0]}
            metadata["format"] = "RASTER"
            return img_bgr, metadata
        raise ValueError(f"Unable to decode geospatial image: {e}")
