import copy
import io
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image

from ..services.cache_manager import compute_image_hash, get_image_preprocessing_cache


def _metadata_from_raster(src: Any) -> Dict[str, Any]:
    tags = src.tags() or {}
    descriptions = list(src.descriptions or ())
    band_names = [str(x).strip() for x in descriptions if x]

    sensor = (
        tags.get("sensor") or tags.get("SENSOR") or tags.get("platform")
        or tags.get("PLATFORM") or tags.get("SPACECRAFT_NAME")
        or tags.get("spacecraft_name")
    )
    acquisition_date = (
        tags.get("acquisition_date") or tags.get("ACQUISITION_DATE")
        or tags.get("TIFFTAG_DATETIME")
    )

    metadata: Dict[str, Any] = {
        "format": "GeoTIFF" if src.crs is not None or src.transform is not None else "TIFF",
        "bands": int(src.count),
        "is_geotiff": bool(src.crs is not None or src.transform is not None),
        "crs": str(src.crs) if src.crs is not None else None,
        "resolution_m": float(abs(src.transform.a)) if src.transform is not None and src.transform.a else None,
        "geotransform": src.transform.to_gdal() if src.transform is not None else None,
        "affine": tuple(src.transform) if src.transform is not None else None,
        "acquisition_date": str(acquisition_date).strip() if acquisition_date else None,
        "sensor": str(sensor).strip() if sensor else None,
        "tags": tags,
        "dimensions": {"width": int(src.width), "height": int(src.height)},
        "band_names": band_names,
        "channel_order": "source_band_order",
    }

    source_text = " ".join(
        str(v) for v in [
            metadata["sensor"], tags.get("dataset"), tags.get("DATASET"),
            tags.get("source")
        ] if v
    ).lower()
    if src.count == 10 and any(x in source_text for x in ("bigearthnet", "big earth net", "reben")):
        metadata["band_names"] = [
            "B02", "B03", "B04", "B05", "B06",
            "B07", "B08", "B8A", "B11", "B12",
        ]
        metadata["sensor"] = metadata["sensor"] or "Sentinel-2"
        metadata["source_dataset"] = "BigEarthNet-S2/reBEN"

    return metadata


def inspect_and_load_geospatial_image(data_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Decode raster input without destroying real multispectral channels.

    Ordinary 1/3-band imagery retains the legacy BGR representation expected by
    OpenCV tools. Multi-band GeoTIFF/TIFF is returned HxWxBands in source-band
    order and is never silently converted to RGB.
    """
    img_cache = get_image_preprocessing_cache()
    img_hash = compute_image_hash(data_bytes)
    cached = img_cache.get(img_hash)
    if cached is not None:
        cached_img, cached_meta = cached
        return cached_img.copy(), copy.deepcopy(cached_meta)

    try:
        import rasterio

        with rasterio.MemoryFile(data_bytes) as memfile:
            with memfile.open() as src:
                metadata = _metadata_from_raster(src)
                if src.count > 3:
                    img = np.transpose(src.read(), (1, 2, 0))
                    img_cache.set(img_hash, (img, metadata))
                    return img, copy.deepcopy(metadata)

                img_rgb = np.transpose(src.read(), (1, 2, 0))
                if img_rgb.shape[2] == 1:
                    img = img_rgb[:, :, 0]
                    metadata["channel_order"] = "single"
                else:
                    img = cv2.cvtColor(img_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
                    metadata["channel_order"] = "bgr"
                img_cache.set(img_hash, (img, metadata))
                return img, copy.deepcopy(metadata)

    except Exception as raster_error:
        # PIL/OpenCV fallback is intentionally limited to ordinary images.
        # A TIFF that cannot be decoded as a raster must not be collapsed to RGB.
        try:
            pil_img = Image.open(io.BytesIO(data_bytes))
            fmt = pil_img.format or "RASTER"
            if fmt in ("TIFF", "GeoTIFF"):
                raise ValueError(
                    "TIFF/GeoTIFF could not be decoded with rasterio; refusing to "
                    "collapse it to RGB because spectral bands may be lost."
                )
            pil_rgb = pil_img.convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
            metadata = {
                "format": fmt,
                "bands": int(img_bgr.shape[2]) if img_bgr.ndim == 3 else 1,
                "is_geotiff": False,
                "crs": None,
                "resolution_m": None,
                "geotransform": None,
                "affine": None,
                "acquisition_date": None,
                "sensor": None,
                "tags": {},
                "dimensions": {"width": int(img_bgr.shape[1]), "height": int(img_bgr.shape[0])},
                "band_names": [],
                "channel_order": "bgr" if img_bgr.ndim == 3 else "single",
                "decode_warning": f"Ordinary image fallback used after raster decode failure: {raster_error}",
            }
            img_cache.set(img_hash, (img_bgr, metadata))
            return img_bgr, copy.deepcopy(metadata)
        except Exception as fallback_error:
            raise ValueError(
                f"Unable to decode raster input without losing spectral information: "
                f"{raster_error}; fallback error: {fallback_error}"
            )
