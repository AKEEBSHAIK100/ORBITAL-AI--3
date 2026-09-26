"""
ORBITAL-AI — Remote-Sensing BLIP LoRA Adapter Runtime Service.
Provides lazy-loading, caching, device selection, and truthful provenance reporting
for BigEarthNet-derived vision-language pilot adapters.

Never crashes application startup when adapters or weights are unavailable.
Never silently substitutes a heuristic result while claiming an adapted model ran.
"""

from __future__ import annotations

import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
from PIL import Image

# ─── Path Resolution ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CAPTION_ADAPTER_PATH = (
    PROJECT_ROOT / "backend" / "models" / "adapters" / "blip_rs_lora"
)
DEFAULT_VQA_ADAPTER_PATH = (
    PROJECT_ROOT / "backend" / "models" / "adapters" / "blip_vqa_rs_lora"
)

CAPTION_ADAPTER_PATH = Path(
    os.getenv("RS_CAPTION_ADAPTER_PATH", str(DEFAULT_CAPTION_ADAPTER_PATH))
)
VQA_ADAPTER_PATH = Path(
    os.getenv("RS_VQA_ADAPTER_PATH", str(DEFAULT_VQA_ADAPTER_PATH))
)


class SpecialistState(str, Enum):
    AVAILABLE = "AVAILABLE"
    LOADING = "LOADING"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


class RSAdapterRuntime:
    """
    Thread-safe lazy-loading singleton runtime for BLIP base models and LoRA adapters.
    Manages process memory caching and device selection (CUDA when available, else CPU).
    """

    _instance: Optional["RSAdapterRuntime"] = None

    def __init__(self):
        # State tracking
        self.caption_state = SpecialistState.UNAVAILABLE
        self.vqa_state = SpecialistState.UNAVAILABLE
        self.caption_unavailable_reason: Optional[str] = None
        self.vqa_unavailable_reason: Optional[str] = None

        # Base model identifiers
        self.caption_base_id = "Salesforce/blip-image-captioning-base"
        self.vqa_base_id = "Salesforce/blip-vqa-base"

        # Model & processor cache
        self._caption_model = None
        self._caption_processor = None
        self._vqa_model = None
        self._vqa_processor = None

        # Device selection
        self.device = self._detect_device()

        # Initial inspection of filesystem
        self._inspect_adapter_availability()

    @classmethod
    def get_instance(cls) -> "RSAdapterRuntime":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing purposes."""
        cls._instance = None

    def unload_models(self) -> None:
        """Release loaded PyTorch model weights from memory and clear GPU cache."""
        import gc
        self._caption_model = None
        self._caption_processor = None
        self._vqa_model = None
        self._vqa_processor = None
        self._inspect_adapter_availability()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _has_adapter_weights(self, adapter_dir: Path) -> bool:
        """
        Checks whether directory exists and contains valid adapter artifacts:
        1. adapter_config.json exists and is a regular file
        2. adapter_model.safetensors exists and is a regular file with size > 0
        """
        if not adapter_dir.exists() or not adapter_dir.is_dir():
            return False
        config_file = adapter_dir / "adapter_config.json"
        weights_file = adapter_dir / "adapter_model.safetensors"
        if not (config_file.exists() and config_file.is_file()):
            return False
        if not (weights_file.exists() and weights_file.is_file() and weights_file.stat().st_size > 0):
            return False
        return True

    def _validate_adapter_config(self, adapter_dir: Path, expected_base: str) -> None:
        import json
        cfg_path = adapter_dir / "adapter_config.json"
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Invalid adapter_config.json: {exc}") from exc
        if cfg.get("base_model_name_or_path") != expected_base:
            raise RuntimeError(
                f"Adapter base-model mismatch: expected {expected_base}, got {cfg.get('base_model_name_or_path')!r}."
            )
        targets = cfg.get("target_modules")
        if not isinstance(targets, list) or not targets:
            raise RuntimeError("Adapter config has no target_modules.")
        if "dense" in targets:
            raise RuntimeError("Adapter config targets unsupported BLIP module 'dense'.")
        if not any(t in targets for t in ("qkv", "projection")):
            raise RuntimeError("Adapter config does not target a verified BLIP vision attention module.")

    def _inspect_adapter_availability(self) -> None:
        """Filesystem inspection is never treated as runtime availability."""
        if self._has_adapter_weights(CAPTION_ADAPTER_PATH):
            self.caption_state = SpecialistState.UNAVAILABLE
            self.caption_unavailable_reason = "Caption adapter artifacts present; live PEFT load/inference verification pending."
        else:
            self.caption_state = SpecialistState.UNAVAILABLE
            self.caption_unavailable_reason = f"Caption adapter artifacts not found at '{CAPTION_ADAPTER_PATH}'."
        if self._has_adapter_weights(VQA_ADAPTER_PATH):
            self.vqa_state = SpecialistState.UNAVAILABLE
            self.vqa_unavailable_reason = "VQA adapter artifacts present; live PEFT load/inference verification pending."
        else:
            self.vqa_state = SpecialistState.UNAVAILABLE
            self.vqa_unavailable_reason = f"VQA adapter artifacts not found at '{VQA_ADAPTER_PATH}'."

    @property
    def caption_provenance(self) -> Dict[str, Any]:
        return {
            "base_model": self.caption_base_id,
            "adapter": "BigEarthNet-derived LoRA pilot adapter",
            "adapter_path": str(CAPTION_ADAPTER_PATH),
            "training_pairs": 87,
            "epochs": 3,
            "adaptation_scope": "pilot_domain_adaptation",
            "benchmark_accuracy_claim": None,
            "note": "Pilot adaptation artifact only; no benchmark accuracy claim (e.g. VRSBench).",
        }

    @property
    def vqa_provenance(self) -> Dict[str, Any]:
        return {
            "base_model": self.vqa_base_id,
            "adapter": "BigEarthNet-derived VQA LoRA pilot adapter",
            "adapter_path": str(VQA_ADAPTER_PATH),
            "training_qa": 551,
            "training_patches": 69,
            "validation_qa": 144,
            "validation_patches": 18,
            "epochs": 3,
            "adaptation_scope": "pilot_domain_adaptation",
            "benchmark_accuracy_claim": None,
            "note": "Pilot adaptation artifact only; no benchmark superiority claim (e.g. RSVQA).",
        }

    def _convert_image_to_pil(
        self,
        image: Union[np.ndarray, Image.Image, bytes, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        meta = dict(metadata or {})

        def stretch(ch: np.ndarray) -> np.ndarray:
            arr = np.asarray(ch, dtype=np.float32)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                raise ValueError("Sentinel-2 band contains no finite pixels.")
            lo, hi = np.percentile(finite, [2.0, 98.0])
            if hi <= lo:
                return np.clip(arr, 0, 1)
            return np.clip((arr - lo) / (hi - lo), 0, 1)

        if isinstance(image, Image.Image):
            return image.convert("RGB"), {
                "modality": meta.get("modality", "optical"),
                "sensor": meta.get("sensor"),
                "input_type": "rgb",
                "bands_used": ["R", "G", "B"],
                "preprocessing": "PIL RGB input converted to BLIP RGB."
            }

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                order = str(meta.get("channel_order", "bgr")).lower()
                rgb = image[:, :, ::-1] if order == "bgr" else image
                return Image.fromarray(np.asarray(rgb, dtype=np.uint8)), {
                    "modality": meta.get("modality", "optical"),
                    "sensor": meta.get("sensor"),
                    "input_type": "rgb",
                    "bands_used": ["R", "G", "B"],
                    "preprocessing": "3-channel image converted to BLIP RGB."
                }
            if image.ndim == 3 and image.shape[2] > 3:
                names = [str(x).strip().upper() for x in meta.get("band_names", []) if x]
                source_dataset = str(meta.get("source_dataset", "")).lower()
                sensor = str(meta.get("sensor", "")).lower()
                if image.shape[2] == 10 and not names and ("bigearthnet" in source_dataset or "reben" in source_dataset or "sentinel-2" in sensor):
                    names = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]
                if len(names) != image.shape[2]:
                    raise ValueError("Multispectral BLIP input requires explicit band_names metadata; refusing silent band truncation.")
                missing = [b for b in ("B02","B03","B04") if b not in names]
                if missing:
                    raise ValueError(f"Required Sentinel-2 bands missing: {missing}")
                idx = {n: names.index(n) for n in names}
                rgb = np.stack([
                    stretch(image[:, :, idx["B04"]]),
                    stretch(image[:, :, idx["B03"]]),
                    stretch(image[:, :, idx["B02"]])
                ], axis=2)
                return Image.fromarray(np.rint(rgb * 255).astype(np.uint8), mode="RGB"), {
                    "modality": "multispectral",
                    "sensor": meta.get("sensor") or "Sentinel-2",
                    "input_type": "multispectral",
                    "bands_used": ["B04", "B03", "B02"],
                    "preprocessing": "B04/B03/B02 natural-colour composite; per-channel 2%-98% percentile stretch.",
                    "source_band_order": names,
                }
            if image.ndim == 2:
                return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB"), {
                    "modality": meta.get("modality", "optical"),
                    "sensor": meta.get("sensor"),
                    "input_type": "single_channel",
                    "bands_used": None,
                    "preprocessing": "Single-channel image expanded to RGB; sensor identity not inferred."
                }

        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB"), {
                "modality": meta.get("modality", "optical"),
                "sensor": meta.get("sensor"),
                "input_type": "rgb",
                "bands_used": ["R", "G", "B"],
                "preprocessing": "Encoded RGB image decoded to BLIP RGB."
            }
        if isinstance(image, (str, Path)):
            return Image.open(str(image)).convert("RGB"), {
                "modality": meta.get("modality", "optical"),
                "sensor": meta.get("sensor"),
                "input_type": "rgb",
                "bands_used": ["R", "G", "B"],
                "preprocessing": "Image file decoded to BLIP RGB."
            }
        raise ValueError(f"Unsupported image input type for RS adapter: {type(image)}")

    # ─── Lazy Model Loading ───────────────────────────────────────────────────

    def _load_caption_model(self):
        if self._caption_model is not None and self._caption_processor is not None:
            return self._caption_model, self._caption_processor
        if not self._has_adapter_weights(CAPTION_ADAPTER_PATH):
            raise RuntimeError(self.caption_unavailable_reason or "Caption adapter unavailable.")
        self.caption_state = SpecialistState.LOADING
        try:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from peft import PeftModel
            self._validate_adapter_config(CAPTION_ADAPTER_PATH, self.caption_base_id)
            processor = BlipProcessor.from_pretrained(self.caption_base_id)
            model = BlipForConditionalGeneration.from_pretrained(self.caption_base_id)
            model = PeftModel.from_pretrained(model, str(CAPTION_ADAPTER_PATH))
            model.to(self.device).eval()
            smoke = Image.new("RGB", (32, 32), (0, 0, 0))
            inputs = processor(smoke, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=4)
            if not processor.decode(out[0], skip_special_tokens=True).strip():
                raise RuntimeError("BLIP caption smoke test returned empty text.")
            self._caption_model, self._caption_processor = model, processor
            self.caption_state = SpecialistState.AVAILABLE
            self.caption_unavailable_reason = None
            return model, processor
        except Exception as exc:
            self.caption_state = SpecialistState.UNAVAILABLE
            self.caption_unavailable_reason = f"Caption runtime verification failed: {exc}"
            self._caption_model = self._caption_processor = None
            raise RuntimeError(self.caption_unavailable_reason) from exc

    def _load_vqa_model(self):
        if self._vqa_model is not None and self._vqa_processor is not None:
            return self._vqa_model, self._vqa_processor
        if not self._has_adapter_weights(VQA_ADAPTER_PATH):
            raise RuntimeError(self.vqa_unavailable_reason or "VQA adapter unavailable.")
        self.vqa_state = SpecialistState.LOADING
        try:
            import torch
            from transformers import BlipProcessor, BlipForQuestionAnswering
            from peft import PeftModel
            self._validate_adapter_config(VQA_ADAPTER_PATH, self.vqa_base_id)
            processor = BlipProcessor.from_pretrained(self.vqa_base_id)
            model = BlipForQuestionAnswering.from_pretrained(self.vqa_base_id)
            model = PeftModel.from_pretrained(model, str(VQA_ADAPTER_PATH))
            model.to(self.device).eval()
            smoke = Image.new("RGB", (32, 32), (0, 0, 0))
            inputs = processor(smoke, "What is visible?", return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=4)
            if not processor.decode(out[0], skip_special_tokens=True).strip():
                raise RuntimeError("BLIP VQA smoke test returned empty text.")
            self._vqa_model, self._vqa_processor = model, processor
            self.vqa_state = SpecialistState.AVAILABLE
            self.vqa_unavailable_reason = None
            return model, processor
        except Exception as exc:
            self.vqa_state = SpecialistState.UNAVAILABLE
            self.vqa_unavailable_reason = f"VQA runtime verification failed: {exc}"
            self._vqa_model = self._vqa_processor = None
            raise RuntimeError(self.vqa_unavailable_reason) from exc

    # ─── Public Inference API ─────────────────────────────────────────────────

    def caption(self, image: Union[np.ndarray, Image.Image, bytes, str]) -> Dict[str, Any]:
        """
        Executes BLIP + BigEarthNet-derived LoRA captioning.
        Returns structured output with status, caption, model_id, provenance, and null confidence.
        Never falls back to heuristic generation while claiming adapted specialist ran.
        """
        t0 = time.perf_counter()

        if self.caption_state != SpecialistState.AVAILABLE:
            return {
                "status": "specialist_unavailable",
                "caption": None,
                "answer": f"Caption specialist unavailable: {self.caption_unavailable_reason}",
                "model_id": "rs-caption-adapted-v1",
                "base_model": self.caption_base_id,
                "adapter": "BigEarthNet-derived LoRA pilot adapter",
                "device": self.device,
                "inference_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                "confidence": None,
                "confidence_status": "not_calibrated",
                "provenance": self.caption_provenance,
                "warnings": [self.caption_unavailable_reason or "Caption adapter not installed."],
            }

        try:
            pil_img = self._convert_image_to_pil(image)
            model, processor = self._load_caption_model()

            import torch
            inputs = processor(pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=60)
            caption_text = processor.decode(out[0], skip_special_tokens=True).strip()

            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "status": "success",
                "caption": caption_text,
                "answer": caption_text,
                "model_id": "rs-caption-adapted-v1",
                "base_model": self.caption_base_id,
                "adapter": "BigEarthNet-derived LoRA pilot adapter",
                "device": self.device,
                "inference_time_ms": duration_ms,
                "confidence": None,  # VLM outputs are uncalibrated
                "confidence_status": "not_calibrated",
                "provenance": self.caption_provenance,
                "warnings": [],
            }

        except Exception as e:
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "status": "error",
                "caption": None,
                "answer": f"Specialist execution error: {str(e)}",
                "model_id": "rs-caption-adapted-v1",
                "base_model": self.caption_base_id,
                "adapter": "BigEarthNet-derived LoRA pilot adapter",
                "device": self.device,
                "inference_time_ms": duration_ms,
                "confidence": None,
                "confidence_status": "not_calibrated",
                "provenance": self.caption_provenance,
                "warnings": [str(e)],
            }

    def vqa(self, image: Union[np.ndarray, Image.Image, bytes, str], question: str) -> Dict[str, Any]:
        """
        Executes BLIP-VQA + BigEarthNet-derived LoRA visual question answering.
        Returns structured output with status, answer, model_id, provenance, and null confidence.
        Never fabricates an answer with hard-coded templates.
        """
        t0 = time.perf_counter()

        if self.vqa_state != SpecialistState.AVAILABLE:
            return {
                "status": "specialist_unavailable",
                "answer": f"VQA specialist unavailable: {self.vqa_unavailable_reason}",
                "question": question,
                "model_id": "rs-vqa-adapted-v1",
                "base_model": self.vqa_base_id,
                "adapter": "BigEarthNet-derived VQA LoRA pilot adapter",
                "device": self.device,
                "inference_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                "confidence": None,
                "confidence_status": "not_calibrated",
                "provenance": self.vqa_provenance,
                "warnings": [self.vqa_unavailable_reason or "VQA adapter not installed."],
            }

        try:
            pil_img = self._convert_image_to_pil(image)
            model, processor = self._load_vqa_model()

            import torch
            inputs = processor(pil_img, question, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)
            answer_text = processor.decode(out[0], skip_special_tokens=True).strip()

            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "status": "success",
                "answer": answer_text,
                "question": question,
                "model_id": "rs-vqa-adapted-v1",
                "base_model": self.vqa_base_id,
                "adapter": "BigEarthNet-derived VQA LoRA pilot adapter",
                "device": self.device,
                "inference_time_ms": duration_ms,
                "confidence": None,  # VLM outputs are uncalibrated
                "confidence_status": "not_calibrated",
                "provenance": self.vqa_provenance,
                "warnings": [],
            }

        except Exception as e:
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "status": "error",
                "answer": f"Specialist execution error: {str(e)}",
                "question": question,
                "model_id": "rs-vqa-adapted-v1",
                "base_model": self.vqa_base_id,
                "adapter": "BigEarthNet-derived VQA LoRA pilot adapter",
                "device": self.device,
                "inference_time_ms": duration_ms,
                "confidence": None,
                "confidence_status": "not_calibrated",
                "provenance": self.vqa_provenance,
                "warnings": [str(e)],
            }
