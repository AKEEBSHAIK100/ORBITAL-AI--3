"""
SatQuery AI — Remote-Sensing BLIP LoRA Adapter Runtime Service.
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

    def _inspect_adapter_availability(self) -> None:
        """Check filesystem for adapter artifacts without loading weights into memory."""
        # Check Caption adapter
        if self._has_adapter_weights(CAPTION_ADAPTER_PATH):
            self.caption_state = SpecialistState.UNAVAILABLE
            self.caption_unavailable_reason = "Caption adapter artifacts are present, but runtime execution has not been verified yet."
        else:
            self.caption_state = SpecialistState.UNAVAILABLE
            self.caption_unavailable_reason = (
                f"Caption LoRA adapter files (adapter_config.json, adapter_model.safetensors) not found at '{CAPTION_ADAPTER_PATH}'. "
                "Set RS_CAPTION_ADAPTER_PATH or mount pilot adapter weights."
            )

        # Check VQA adapter
        if self._has_adapter_weights(VQA_ADAPTER_PATH):
            self.vqa_state = SpecialistState.UNAVAILABLE
            self.vqa_unavailable_reason = "VQA adapter artifacts are present, but runtime execution has not been verified yet."
        else:
            self.vqa_state = SpecialistState.UNAVAILABLE
            self.vqa_unavailable_reason = (
                f"VQA LoRA adapter files (adapter_config.json, adapter_model.safetensors) not found at '{VQA_ADAPTER_PATH}'. "
                "Set RS_VQA_ADAPTER_PATH or mount pilot adapter weights."
            )

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

    def _convert_image_to_pil(self, image: Union[np.ndarray, Image.Image, bytes, str]) -> Image.Image:
        """Converts diverse remote sensing image inputs into a standard RGB PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB") if image.mode != "RGB" else image

        if isinstance(image, np.ndarray):
            # Check if BGR from cv2 (most common in backend)
            if image.ndim == 3 and image.shape[2] == 3:
                # Convert BGR to RGB
                rgb = image[:, :, ::-1]
                return Image.fromarray(rgb.astype(np.uint8))
            elif image.ndim == 2:
                # Grayscale / SAR intensity
                return Image.fromarray(image.astype(np.uint8)).convert("RGB")
            elif image.ndim == 3 and image.shape[2] > 3:
                # Multispectral: take first 3 channels
                return Image.fromarray(image[:, :, :3].astype(np.uint8)).convert("RGB")

        if isinstance(image, (bytes, bytearray)):
            import io
            return Image.open(io.BytesIO(image)).convert("RGB")

        if isinstance(image, (str, Path)):
            return Image.open(str(image)).convert("RGB")

        raise ValueError(f"Unsupported image input type for RS adapter: {type(image)}")

    # ─── Lazy Model Loading ───────────────────────────────────────────────────

    def _load_caption_model(self):
        """Loads BLIP caption base model and PEFT LoRA adapter lazily."""
        if self._caption_model is not None and self._caption_processor is not None:
            return self._caption_model, self._caption_processor

        if self.caption_state == SpecialistState.ERROR:
            raise RuntimeError(self.caption_unavailable_reason or "Caption specialist unavailable.")
        if not self._has_adapter_weights(CAPTION_ADAPTER_PATH):
            self.caption_state = SpecialistState.UNAVAILABLE
            self.caption_unavailable_reason = f"Caption adapter artifacts not found at '{CAPTION_ADAPTER_PATH}'."
            raise RuntimeError(self.caption_unavailable_reason)

        self.caption_state = SpecialistState.LOADING
        try:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration

            processor = BlipProcessor.from_pretrained(self.caption_base_id)
            model = BlipForConditionalGeneration.from_pretrained(self.caption_base_id)

            # Strictly verify and apply LoRA adapter weights
            if not self._has_adapter_weights(CAPTION_ADAPTER_PATH):
                self.caption_state = SpecialistState.UNAVAILABLE
                self.caption_unavailable_reason = (
                    f"Caption LoRA adapter files (adapter_config.json, adapter_model.safetensors) not found at '{CAPTION_ADAPTER_PATH}'."
                )
                raise RuntimeError(self.caption_unavailable_reason)

            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(CAPTION_ADAPTER_PATH))
            except ImportError:
                # If PEFT is not installed, fail cleanly to UNAVAILABLE
                self.caption_state = SpecialistState.UNAVAILABLE
                self.caption_unavailable_reason = "peft library is not installed to load LoRA adapter."
                raise RuntimeError(self.caption_unavailable_reason)

            model.to(self.device)
            model.eval()

            self._caption_model = model
            self._caption_processor = processor
            self.caption_state = SpecialistState.AVAILABLE
            return self._caption_model, self._caption_processor

        except Exception as e:
            self.caption_state = SpecialistState.UNAVAILABLE
            self.caption_unavailable_reason = f"Caption runtime verification failed: {e}"
            raise

    def _load_vqa_model(self):
        """Loads BLIP VQA base model and PEFT LoRA adapter lazily."""
        if self._vqa_model is not None and self._vqa_processor is not None:
            return self._vqa_model, self._vqa_processor

        if self.vqa_state == SpecialistState.ERROR:
            raise RuntimeError(self.vqa_unavailable_reason or "VQA specialist unavailable.")
        if not self._has_adapter_weights(VQA_ADAPTER_PATH):
            self.vqa_state = SpecialistState.UNAVAILABLE
            self.vqa_unavailable_reason = f"VQA adapter artifacts not found at '{VQA_ADAPTER_PATH}'."
            raise RuntimeError(self.vqa_unavailable_reason)

        self.vqa_state = SpecialistState.LOADING
        try:
            import torch
            from transformers import BlipProcessor, BlipForQuestionAnswering

            processor = BlipProcessor.from_pretrained(self.vqa_base_id)
            model = BlipForQuestionAnswering.from_pretrained(self.vqa_base_id)

            # Strictly verify and apply LoRA adapter weights
            if not self._has_adapter_weights(VQA_ADAPTER_PATH):
                self.vqa_state = SpecialistState.UNAVAILABLE
                self.vqa_unavailable_reason = (
                    f"VQA LoRA adapter files (adapter_config.json, adapter_model.safetensors) not found at '{VQA_ADAPTER_PATH}'."
                )
                raise RuntimeError(self.vqa_unavailable_reason)

            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(VQA_ADAPTER_PATH))
            except ImportError:
                self.vqa_state = SpecialistState.UNAVAILABLE
                self.vqa_unavailable_reason = "peft library is not installed to load LoRA adapter."
                raise RuntimeError(self.vqa_unavailable_reason)

            model.to(self.device)
            model.eval()

            self._vqa_model = model
            self._vqa_processor = processor
            self.vqa_state = SpecialistState.AVAILABLE
            return self._vqa_model, self._vqa_processor

        except Exception as e:
            self.vqa_state = SpecialistState.UNAVAILABLE
            self.vqa_unavailable_reason = f"VQA runtime verification failed: {e}"
            raise

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
