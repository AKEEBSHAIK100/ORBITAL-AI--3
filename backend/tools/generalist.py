"""
SatQuery AI / Orbital-AI — Remote-Sensing Generalist Multimodal Fallback Specialist.

Provides a clean specialist interface for open-ended multimodal queries when no
verified domain-specific specialist applies.

Base model: Qwen/Qwen2-VL-2B-Instruct (general multimodal VLM).
Model type: general_multimodal_vlm.
DOES NOT claim remote-sensing adaptation or benchmark superiority until a real
adapter is trained and validated.
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from .base import BaseTool
from ..services.cache_manager import (
    compute_image_hash,
    build_cache_key,
    get_query_result_cache,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERALIST_WEIGHTS_DIR = PROJECT_ROOT / "backend" / "models" / "generalist" / "qwen2_vl"


class GeneralistTool(BaseTool):
    id = "rs_generalist"
    name = "Remote-Sensing Generalist Multimodal Fallback Specialist"
    description = (
        "General multimodal vision-language fallback specialist for open-ended "
        "remote-sensing queries not covered by task-specific specialists."
    )
    supported_tasks = [
        "general_vqa",
        "open_scene_description",
        "open_remote_sensing_question",
    ]
    modalities = ["optical", "multispectral", "sar"]
    supported_modalities = ["optical", "multispectral", "sar"]
    base_model = "Qwen/Qwen2-VL-2B-Instruct"
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    model_type = "general_multimodal_vlm"
    adapter: Optional[str] = None
    device = "cpu"
    domain_adaptation = "None (unadapted general multimodal VLM - not remote-sensing adapted)"

    provenance: Dict[str, Any] = {
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "base_model": "Qwen/Qwen2-VL-2B-Instruct",
        "adapter": None,
        "model_type": "general_multimodal_vlm",
        "is_remote_sensing_adapted": False,
        "is_remote_sensing_expert": False,
        "adaptation_scope": "unadapted_foundation_model",
        "benchmark_accuracy_claim": None,
        "note": (
            "General multimodal vision-language foundation model (Qwen2-VL-2B-Instruct). "
            "Not adapted or fine-tuned on remote sensing imagery. "
            "Outputs are uncalibrated and qualitative."
        ),
    }

    permitted_parameters = {
        "max_new_tokens": 128,
        "temperature": 0.2,
    }

    def __init__(self, weights_dir: Optional[Path] = None):
        self._weights_dir = weights_dir
        self._model = None
        self._processor = None

    def _resolve_weights_path(self) -> Path:
        if self._weights_dir is not None:
            return Path(self._weights_dir).resolve()
        env_path = os.getenv("RS_GENERALIST_WEIGHTS_PATH")
        if env_path:
            return Path(env_path).resolve()
        return GENERALIST_WEIGHTS_DIR.resolve()

    @property
    def is_available(self) -> bool:
        """
        Returns True ONLY if local weights or adapter are verified to exist on disk.
        NEVER triggers network downloads on initialization or inspection.
        """
        weights_path = self._resolve_weights_path()
        if not weights_path.exists():
            return False
        if any(weights_path.glob("*.safetensors")) or any(weights_path.glob("*.bin")):
            return True
        return False

    @property
    def unavailable_reason(self) -> Optional[str]:
        if not self.is_available:
            weights_path = self._resolve_weights_path()
            return (
                f"Base model '{self.base_model}' is not installed locally at '{weights_path}'. "
                "Automatic downloads are disabled to prevent unexpected multi-GB transfers."
            )
        return None

    def _load_model(self):
        """Lazy-loads local Qwen2-VL model and processor with strict offline constraint."""
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        if not self.is_available:
            raise RuntimeError(self.unavailable_reason or "Generalist model weights not installed.")

        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        weights_path = str(self._resolve_weights_path())
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info(f"[GeneralistTool] Loading Qwen2-VL from {weights_path} on {device.upper()}…")
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            weights_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            local_files_only=True,
        )
        if device != "cuda":
            self._model.to(device)
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            weights_path,
            local_files_only=True,
        )
        return self._model, self._processor

    def unload_model(self) -> None:
        """Release loaded model weights and processor from memory."""
        import gc
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[GeneralistTool] Model unloaded successfully.")

    def _convert_to_pil(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            import cv2
            if len(image.shape) == 2:
                return Image.fromarray(image).convert("RGB")
            elif image.shape[2] == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
            elif image.shape[2] == 4:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                return Image.fromarray(rgb)
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB")
        raise ValueError(f"Unsupported image type for generalist: {type(image)}")

    def run(
        self,
        inputs: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes generalist multimodal inference or returns clean specialist-unavailable contract.
        Adheres strictly to the generalist contract:
        - status
        - answer
        - model
        - base_model
        - adapter
        - device
        - inference_time_ms
        - confidence (null / None)
        - confidence_status ("not_calibrated")
        - provenance
        - warnings
        """
        query = inputs.get("query") or inputs.get("question") or ""
        params = {**self.permitted_parameters, **(parameters or {})}

        # If weights are not installed, return clean specialist-unavailable contract
        if not self.is_available:
            return {
                "status": "specialist_unavailable",
                "answer": f"Generalist fallback unavailable: {self.unavailable_reason}",
                "model": self.model_id,
                "base_model": self.base_model,
                "adapter": self.adapter,
                "model_type": self.model_type,
                "device": self.device,
                "inference_time_ms": 0.0,
                "confidence": None,
                "confidence_status": "not_calibrated",
                "evidence_source": "generalist_vlm",
                "evidence": {
                    "question": query,
                    "generalist_status": "unavailable",
                    "model_type": self.model_type,
                },
                "provenance": self.provenance,
                "warnings": [
                    self.unavailable_reason or "Generalist model weights not installed."
                ],
            }

        # --- Cache check ---
        image_input = inputs.get("image")
        raw_bytes = b""
        if image_input is not None:
            try:
                pil_img = self._convert_to_pil(image_input)
                b_io = io.BytesIO()
                pil_img.save(b_io, format="JPEG")
                raw_bytes = b_io.getvalue()
            except Exception:
                pass

        result_cache = get_query_result_cache()
        img_hash = compute_image_hash(raw_bytes)
        cache_key = build_cache_key(
            image_identity=img_hash,
            query=query,
            task="general_vqa",
            model_id=self.model_id,
            model_version="v1",
            adapter_identity="none",
            parameters=params,
        )

        cached_entry = result_cache.get(cache_key)
        if cached_entry is not None:
            cached_res = dict(cached_entry)
            evidence = dict(cached_res.get("evidence", {}))
            evidence["cached"] = True
            cached_res["evidence"] = evidence
            return cached_res

        t_start = time.time()
        warnings = [
            "Qualitative generalist output from unadapted base model (Qwen/Qwen2-VL-2B-Instruct); "
            "confidence is uncalibrated. Not remote-sensing adapted."
        ]

        try:
            model, processor = self._load_model()
            pil_img = self._convert_to_pil(image_input)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": query},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            import torch
            inputs_proc = processor(
                text=[text],
                images=[pil_img],
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            max_tokens = int(params.get("max_new_tokens", 128))
            temp = float(params.get("temperature", 0.2))

            with torch.no_grad():
                gen_kwargs = {"max_new_tokens": max_tokens}
                if temp > 0:
                    gen_kwargs["temperature"] = temp
                    gen_kwargs["do_sample"] = True
                else:
                    gen_kwargs["do_sample"] = False

                generated_ids = model.generate(**inputs_proc, **gen_kwargs)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_proc.input_ids, generated_ids)
            ]
            answer = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            exec_time_ms = round((time.time() - t_start) * 1000, 2)

            res = {
                "status": "success",
                "answer": answer,
                "model": self.model_id,
                "base_model": self.base_model,
                "adapter": self.adapter,
                "model_type": self.model_type,
                "device": self.device,
                "inference_time_ms": exec_time_ms,
                "confidence": None,
                "confidence_status": "not_calibrated",
                "evidence_source": "generalist_vlm",
                "evidence": {
                    "question": query,
                    "qualitative_answer": answer,
                    "model_type": self.model_type,
                    "cached": False,
                },
                "provenance": self.provenance,
                "warnings": warnings,
            }
            result_cache.set(cache_key, res)
            return res

        except Exception as e:
            return {
                "status": "error",
                "answer": f"Generalist execution error: {str(e)}",
                "model": self.model_id,
                "base_model": self.base_model,
                "adapter": self.adapter,
                "model_type": self.model_type,
                "device": self.device,
                "inference_time_ms": round((time.time() - t_start) * 1000, 2),
                "confidence": None,
                "confidence_status": "not_calibrated",
                "evidence_source": "generalist_vlm",
                "evidence": {"error": str(e)},
                "provenance": self.provenance,
                "warnings": [str(e)],
            }

    def to_spec(self) -> Dict[str, Any]:
        base_spec = super().to_spec()
        base_spec.update({
            "base_model": self.base_model,
            "model_type": self.model_type,
            "provenance": self.provenance,
            "is_available": self.is_available,
        })
        return base_spec
