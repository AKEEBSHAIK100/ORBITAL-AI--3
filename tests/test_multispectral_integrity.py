import json
from pathlib import Path

import numpy as np
import pytest

from backend.services.rs_adapters import RSAdapterRuntime
from backend.preprocessing.geotiff import inspect_and_load_geospatial_image


def _make_10_band_geotiff():
    rasterio = pytest.importorskip("rasterio")
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    data = np.arange(10 * 8 * 8, dtype=np.float32).reshape(10, 8, 8)
    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 10,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 8, 10, 10),
    }
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(data)
            dst.update_tags(DATASET="BigEarthNet-S2")
        return mem.read()


def test_multiband_geotiff_preserves_all_bands():
    payload = _make_10_band_geotiff()
    image, metadata = inspect_and_load_geospatial_image(payload)

    assert image.shape == (8, 8, 10)
    assert metadata["bands"] == 10
    assert metadata["sensor"] == "Sentinel-2"
    assert metadata["source_dataset"] == "BigEarthNet-S2/reBEN"
    assert metadata["band_names"] == [
        "B02", "B03", "B04", "B05", "B06",
        "B07", "B08", "B8A", "B11", "B12",
    ]


def test_blip_rejects_ambiguous_multispectral_array():
    runtime = RSAdapterRuntime.__new__(RSAdapterRuntime)
    with pytest.raises(ValueError, match="explicit band_names"):
        runtime._convert_image_to_pil(np.zeros((8, 8, 4), dtype=np.float32), {})


def test_blip_uses_sentinel2_rgb_bands_without_truncation():
    runtime = RSAdapterRuntime.__new__(RSAdapterRuntime)
    image = np.zeros((8, 8, 10), dtype=np.float32)
    image[:, :, 0] = 10.0
    image[:, :, 1] = 20.0
    image[:, :, 2] = 30.0
    image[:, :, 3:] = 1000.0

    pil, meta = runtime._convert_image_to_pil(
        image,
        {
            "sensor": "Sentinel-2",
            "band_names": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
        },
    )

    assert pil.mode == "RGB"
    assert meta["bands_used"] == ["B04", "B03", "B02"]
    assert meta["input_type"] == "multispectral"
    assert "percentile" in meta["preprocessing"]


def test_blip_adapter_configs_do_not_target_dense():
    root = Path(__file__).resolve().parents[1]
    for name in ("blip_rs_lora", "blip_vqa_rs_lora"):
        cfg = json.loads(
            (root / "backend" / "models" / "adapters" / name / "adapter_config.json").read_text()
        )
        assert "dense" not in cfg.get("target_modules", [])
        assert "qkv" in cfg.get("target_modules", []) or "projection" in cfg.get("target_modules", [])


def test_capability_inspector_does_not_equate_adapter_files_with_runtime_availability():
    from backend.services.capability_inspector import inspect_capabilities

    result = inspect_capabilities()
    items = {item["id"]: item for item in result["capabilities"]}
    assert items["caption"]["status"] == "unavailable"
    assert items["vqa"]["status"] == "unavailable"
    assert items["caption"]["runtime_verified"] is False
    assert items["vqa"]["runtime_verified"] is False
