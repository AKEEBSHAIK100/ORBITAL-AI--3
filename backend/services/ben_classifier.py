"""
SatQuery AI — BigEarthNet v2.0 Land-Cover Classifier Service
Loads the official BIFOLD-pretrained ResNet-50 (reBEN) from HuggingFace and
runs real 19-class multi-label inference on uploaded satellite imagery.

Model: BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0
Paper: "reBEN: Refined BigEarthNet Dataset for Remote Sensing Image Analysis"
       Clasen et al., IGARSS 2025 · arXiv:2407.03653
Taxonomy: BigEarthNet 19-class Corine Land Cover hierarchy
"""

from __future__ import annotations

import io
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── BigEarthNet v2.0 official 19-class labels (NEW_LABELS from configilm) ────
BIGEARTHNET_19_CLASSES: List[str] = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

# Short display labels (for UI chips)
BIGEARTHNET_19_SHORT: List[str] = [
    "Urban Fabric",
    "Industrial/Commercial",
    "Arable Land",
    "Permanent Crops",
    "Pastures",
    "Complex Cultivation",
    "Agriculture + Natural Veg.",
    "Agro-Forestry",
    "Broad-Leaved Forest",
    "Coniferous Forest",
    "Mixed Forest",
    "Natural Grassland",
    "Moors & Heathland",
    "Transitional Woodland",
    "Beaches & Dunes",
    "Inland Wetlands",
    "Coastal Wetlands",
    "Inland Waters",
    "Marine Waters",
]

# HuggingFace model identifier — pretrained on full BigEarthNet v2.0 (reBEN)
HF_MODEL_ID = "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"

# BigEarthNet v2.0 Sentinel-2 per-band normalization statistics
# (from configilm BENv2DataModule — 10 bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12)
BEN_S2_MEAN = [340.76769064, 429.9430203, 614.21682446, 590.23569706, 951.34811585,
               1792.46290469, 2075.46795189, 2218.94553375, 2266.46036911, 1594.42694882]
BEN_S2_STD = [554.81258967, 572.41639287, 582.87945694, 675.88746967, 729.89827633,
              1096.01480586, 1273.45393088, 1365.45589904, 1356.13789355, 1079.19066363]

# Image size expected by the BEN v2.0 model
BEN_IMG_SIZE = 120


class BENClassifier:
    """
    Singleton wrapper around the BigEarthNet v2.0 pretrained classifier.
    Uses configilm + PyTorchModelHubMixin to load weights from HuggingFace.
    If the real configilm/model stack is unavailable, classification is unavailable.
    """

    _instance: Optional["BENClassifier"] = None
    _model = None
    _device = "cpu"
    _available = False
    _model_id: str = HF_MODEL_ID
    _load_error: str = ""

    def __init__(self):
        self._load_model()

    @classmethod
    def get_instance(cls) -> "BENClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> None:
        """Attempt to load the pretrained BEN v2.0 model from HuggingFace."""
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            import sys, os, types
            # Ensure local reben_publication package is resolvable
            services_dir = Path(__file__).resolve().parent
            if str(services_dir) not in sys.path:
                sys.path.insert(0, str(services_dir))
            reben_dir = Path(__file__).resolve().parent.parent.parent / "data" / "reben-training-scripts-main"
            if reben_dir.exists() and str(reben_dir) not in sys.path:
                sys.path.insert(0, str(reben_dir))

            # Provide compatibility shims for configilm in Python 3.14+
            try:
                import configilm
                if not hasattr(configilm, 'extra'):
                    configilm.extra = types.ModuleType('configilm.extra')
                    sys.modules['configilm.extra'] = configilm.extra
                if not hasattr(configilm.extra, 'BENv2_utils'):
                    ben_utils = types.ModuleType('configilm.extra.BENv2_utils')
                    ben_utils.NEW_LABELS = BIGEARTHNET_19_CLASSES
                    sys.modules['configilm.extra.BENv2_utils'] = ben_utils
                    configilm.extra.BENv2_utils = ben_utils
                if not hasattr(configilm.extra, 'CustomTorchClasses'):
                    custom_torch = types.ModuleType('configilm.extra.CustomTorchClasses')
                    class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
                        pass
                    custom_torch.LinearWarmupCosineAnnealingLR = LinearWarmupCosineAnnealingLR
                    sys.modules['configilm.extra.CustomTorchClasses'] = custom_torch
                    configilm.extra.CustomTorchClasses = custom_torch
                if not hasattr(configilm, 'metrics'):
                    metrics = types.ModuleType('configilm.metrics')
                    metrics.get_classification_metric_collection = lambda *a, **k: None
                    sys.modules['configilm.metrics'] = metrics
                    configilm.metrics = metrics

                import inspect
                from configilm.ConfigILM import ILMConfiguration, ILMType
                orig_ilm_init = ILMConfiguration.__init__
                def patched_ilm_init(self, *args, **kwargs):
                    if '_fusion_activation' in kwargs and 'fusion_activation' not in kwargs:
                        kwargs.pop('_fusion_activation')
                    if '_fusion_method' in kwargs and 'fusion_method' not in kwargs:
                        kwargs.pop('_fusion_method')
                    if not args and 'timm_model_name' not in kwargs:
                        kwargs['timm_model_name'] = kwargs.get('model_name', 'resnet50')
                    kwargs.setdefault('classes', 19)
                    kwargs.setdefault('channels', 10)
                    kwargs.setdefault('network_type', ILMType.IMAGE_CLASSIFICATION)
                    valid_params = set(inspect.signature(orig_ilm_init).parameters.keys()) - {'self'}
                    filtered = {k: v for k, v in kwargs.items() if k in valid_params}
                    return orig_ilm_init(self, *args, **filtered)
                ILMConfiguration.__init__ = patched_ilm_init
            except Exception as shim_err:
                logger.debug(f"[BEN] configilm shim note: {shim_err}")

            from reben_publication.BigEarthNetv2_0_ImageClassifier import (
                BigEarthNetv2_0_ImageClassifier,
            )

            logger.info(f"[BEN] Loading {HF_MODEL_ID} from HuggingFace…")
            t0 = time.time()
            self._model = BigEarthNetv2_0_ImageClassifier.from_pretrained(HF_MODEL_ID)
            self._model.eval()
            self._model.to(self._device)
            elapsed = time.time() - t0
            logger.info(f"[BEN] Model loaded on {self._device.upper()} in {elapsed:.1f}s")
            self._available = True

        except ImportError:
            # configilm is required for the real BigEarthNet classifier
            self._load_error = (
                "configilm not installed. Run: pip install configilm[full]~=0.7.0. "
                "BigEarthNet classification is unavailable until the real classifier stack is installed."
            )
            logger.warning(f"[BEN] {self._load_error}")
            self._available = False

        except Exception as exc:
            self._load_error = str(exc)
            logger.warning(f"[BEN] Could not load pretrained model: {exc}")
            self._available = False

    def ensure_model_loaded(self) -> bool:
        """Ensures the model is loaded in memory. If unloaded, attempts to re-load."""
        if self._model is None and not self._load_error:
            self._load_model()
        return self._available

    def unload_model(self) -> None:
        """Release loaded ResNet-50 model weights from memory and clear GPU cache."""
        import gc
        self._model = None
        self._available = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[BEN] Model unloaded successfully.")

    def classify_image(
        self,
        image_bytes: bytes,
        top_k: int = 5,
        threshold: float = 0.25,
    ) -> Dict:
        """Run BigEarthNet v2.0 inference only when the real model is available."""
        if not self._available and self._model is None and not self._load_error:
            self.ensure_model_loaded()

        if self._available and self._model is not None:
            return self._run_model_inference(image_bytes, top_k, threshold)

        return {
            "labels": [],
            "active_labels": [],
            "top_label": None,
            "confidence": None,
            "model_id": self._model_id,
            "available": False,
            "device": self._device,
            "note": (
                "BigEarthNet v2.0 classifier unavailable; no heuristic or synthetic "
                "land-cover result is returned."
            ),
            "citation": None,
            "status": "unavailable",
            "error": self._load_error or "Model weights are not loaded.",
        }

    def _run_model_inference(
        self, image_bytes: bytes, top_k: int, threshold: float
    ) -> Dict:
        """Real BigEarthNet v2.0 model inference path."""
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image

        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.warning(f"[BEN] Image decode failed: {e}")
            return {
                "labels": [],
                "active_labels": [],
                "top_label": None,
                "confidence": None,
                "model_id": self._model_id,
                "available": False,
                "device": self._device,
                "note": "Image decoding failed; no classification result returned.",
                "citation": None,
                "status": "error",
                "error": str(e),
            }

        pil_img = pil_img.resize((BEN_IMG_SIZE, BEN_IMG_SIZE))
        rgb = np.array(pil_img, dtype=np.float32)
        rgb_scaled = rgb / 255.0 * 3000.0

        channels = np.stack([
            rgb_scaled[:, :, 2],
            rgb_scaled[:, :, 1],
            rgb_scaled[:, :, 0],
            (rgb_scaled[:, :, 0] + rgb_scaled[:, :, 1]) / 2,
            rgb_scaled[:, :, 1],
            rgb_scaled[:, :, 1] * 1.1,
            (rgb_scaled[:, :, 0] + rgb_scaled[:, :, 1]) / 1.8,
            rgb_scaled[:, :, 1] * 0.9,
            rgb_scaled[:, :, 0] * 0.7,
            rgb_scaled[:, :, 0] * 0.5,
        ], axis=0)

        for i in range(10):
            channels[i] = (channels[i] - BEN_S2_MEAN[i]) / (BEN_S2_STD[i] + 1e-8)

        tensor = torch.from_numpy(channels).float().unsqueeze(0).to(self._device)

        try:
            with torch.no_grad():
                logits = self._model(tensor)
                scores = torch.sigmoid(logits).squeeze().cpu().numpy()
        except Exception as e:
            logger.warning(f"[BEN] Inference failed: {e}")
            return {
                "labels": [],
                "active_labels": [],
                "top_label": None,
                "confidence": None,
                "model_id": self._model_id,
                "available": False,
                "device": self._device,
                "note": "BigEarthNet inference failed; no classification result returned.",
                "citation": None,
                "status": "error",
                "error": str(e),
            }

        label_scores = [
            {
                "name": BIGEARTHNET_19_CLASSES[i],
                "short": BIGEARTHNET_19_SHORT[i],
                "score": float(scores[i]),
                "active": bool(scores[i] >= threshold),
            }
            for i in range(19)
        ]
        label_scores.sort(key=lambda x: x["score"], reverse=True)
        top = label_scores[0]
        active_labels = [l for l in label_scores if l["active"]]

        return {
            "labels": label_scores[:top_k],
            "active_labels": active_labels[:top_k],
            "top_label": top["short"],
            "confidence": round(float(top["score"]) * 100, 1),
            "model_id": self._model_id,
            "available": True,
            "device": self._device,
            "note": f"BigEarthNet v2.0 · ResNet-50 · {len(active_labels)} active classes (≥{threshold:.0%})",
            "citation": "Clasen et al., IGARSS 2025 · arXiv:2407.03653",
            "status": "success",
        }

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def load_error(self) -> str:
        return self._load_error
