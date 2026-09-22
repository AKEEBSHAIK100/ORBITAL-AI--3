"""
SatQuery AI / Orbital-AI — AdaptLLM Remote-Sensing VLM Specialist.

Integrates AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct as an OPTIONAL
remote-sensing domain-adapted VLM specialist for open-vocabulary remote-sensing
visual questions.

Base model: Qwen/Qwen2-VL-2B-Instruct
Domain adaptation: Remote-sensing domain post-training
Model type: remote_sensing_vlm

Strict offline safety:
- Zero automatic downloads during initialization or runtime.
- Requires explicit local model path or enablement with on-disk weights.
- Returns clean specialist_unavailable status when weights are absent.
- Enforces explicit visual-token and pixel budgets to prevent out-of-memory errors.
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
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
DEFAULT_ADAPTLLM_DIR = PROJECT_ROOT / "backend" / "models" / "generalist" / "adaptllm"

# Default conservative visual token & pixel budget for 2B VLM memory safety
DEFAULT_MIN_PIXELS = 256 * 28 * 28  # 200,704
DEFAULT_MAX_PIXELS = 1024 * 28 * 28  # 802,816
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_MAX_IMAGE_SIZE = 1280  # Max dimension before downscaling


def is_adaptllm_available() -> bool:
    """
    Returns True ONLY if AdaptLLM is enabled and valid weights exist on disk.
    Strictly offline; zero network calls.
    """
    env_path = os.getenv("ADAPTLLM_MODEL_PATH")
    enabled = (
        os.getenv("ENABLE_ADAPTLLM", "false").lower() in ("true", "1", "yes")
        or bool(env_path)
    )
    if not enabled:
        return False
    weights_path = Path(env_path).resolve() if env_path else DEFAULT_ADAPTLLM_DIR
    if not weights_path.exists():
        return False
    return any(weights_path.glob("*.safetensors")) or any(weights_path.glob("*.bin"))


class AdaptLLMTool(BaseTool):
    id = "rs_adaptllm"
    name = "AdaptLLM Remote-Sensing VLM Specialist"
    description = (
        "Optional remote-sensing domain-adapted vision-language specialist for "
        "open-vocabulary remote-sensing visual questions where no dedicated specialist exists."
    )
    supported_tasks = [
        "general_vqa",
        "open_scene_description",
        "open_remote_sensing_question",
    ]
    modalities = ["optical", "multispectral", "sar"]
    supported_modalities = ["optical", "multispectral", "sar"]

    model = "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct"
    model_id = "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct"
    base_model = "Qwen/Qwen2-VL-2B-Instruct"
    model_type = "remote_sensing_vlm"
    adapter: Optional[str] = None
    device = "cpu"
    domain_adaptation = "Remote-sensing domain post-training on Qwen2-VL-2B-Instruct"
    is_remote_sensing_adapted = True
    is_remote_sensing_expert = True
    adaptation_scope = "remote_sensing_domain_post_training"
    benchmark_accuracy_claim = None

    permitted_parameters = {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "temperature": 0.2,
    }

    def __init__(self, weights_dir: Optional[Union[str, Path]] = None):
        self._weights_dir = Path(weights_dir).resolve() if weights_dir else None
        self._model = None
        self._processor = None

    def _resolve_weights_path(self) -> Path:
        if self._weights_dir is not None:
            return self._weights_dir
        env_path = os.getenv("ADAPTLLM_MODEL_PATH")
        if env_path:
            return Path(env_path).resolve()
        return DEFAULT_ADAPTLLM_DIR.resolve()

    @property
    def is_available(self) -> bool:
        """
        True only if local weights exist on disk and model is enabled.
        Never triggers network downloads.
        """
        weights_path = self._resolve_weights_path()
        env_path = os.getenv("ADAPTLLM_MODEL_PATH")
        enabled = (
            os.getenv("ENABLE_ADAPTLLM", "false").lower() in ("true", "1", "yes")
            or bool(env_path)
            or (self._weights_dir is not None and self._weights_dir.exists())
        )
        if not enabled:
            return False
        if not weights_path.exists():
            return False
        return any(weights_path.glob("*.safetensors")) or any(weights_path.glob("*.bin"))

    @property
    def unavailable_reason(self) -> str:
        return "AdaptLLM remote-sensing VLM weights are not installed or enabled."

    def get_token_budget_config(self) -> Dict[str, int]:
        """Resolves visual token / pixel budget from environment with memory-safe defaults."""
        min_pixels = int(os.getenv("SATQUERY_ADAPTLLM_MIN_PIXELS", str(DEFAULT_MIN_PIXELS)))
        max_pixels = int(os.getenv("SATQUERY_ADAPTLLM_MAX_PIXELS", str(DEFAULT_MAX_PIXELS)))
        max_tokens = int(os.getenv("SATQUERY_ADAPTLLM_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS)))
        max_dim = int(os.getenv("SATQUERY_ADAPTLLM_MAX_IMAGE_SIZE", str(DEFAULT_MAX_IMAGE_SIZE)))
        return {
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "max_new_tokens": max_tokens,
            "max_image_size": max_dim,
        }

    def _preprocess_image(
        self,
        raw_image: Any,
        max_dim: int
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Converts image to RGB and applies physical size cap to prevent OOM.
        Never silently upscales. Downscaling is recorded in provenance.
        """
        pil_img = self._convert_to_pil(raw_image)
        orig_w, orig_h = pil_img.size
        downscaled = False

        if max(orig_w, orig_h) > max_dim:
            # Preserve aspect ratio
            pil_img = pil_img.copy()
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            downscaled = True

        proc_w, proc_h = pil_img.size
        meta = {
            "downscaled": downscaled,
            "original_size": [orig_w, orig_h],
            "processed_size": [proc_w, proc_h],
        }
        return pil_img, meta

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
        raise ValueError(f"Unsupported image type for AdaptLLM: {type(image)}")

    def _load_model(self):
        """Lazy-loads local AdaptLLM model and processor with strict offline constraint."""
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        if not self.is_available:
            raise RuntimeError(self.unavailable_reason)

        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        weights_path = str(self._resolve_weights_path())
        torch_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if device == "cuda" else torch.float32
        )

        logger.info(f"[AdaptLLMTool] Loading AdaptLLM from {weights_path} on {device.upper()}…")
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            weights_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            local_files_only=True,
        )
        if device != "cuda":
            self._model.to(device)
        self._model.eval()

        budget = self.get_token_budget_config()
        try:
            self._processor = AutoProcessor.from_pretrained(
                weights_path,
                min_pixels=budget["min_pixels"],
                max_pixels=budget["max_pixels"],
                local_files_only=True,
            )
        except TypeError:
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
        logger.info("[AdaptLLMTool] Model unloaded successfully.")

    def get_provenance(self, budget_meta: Optional[Dict[str, Any]] = None, exec_time_ms: float = 0.0) -> Dict[str, Any]:
        """Constructs standardized, truthful provenance dictionary."""
        budget_cfg = self.get_token_budget_config()
        budget_data = {
            "min_pixels": budget_cfg["min_pixels"],
            "max_pixels": budget_cfg["max_pixels"],
            "max_new_tokens": budget_cfg["max_new_tokens"],
        }
        if budget_meta:
            budget_data.update(budget_meta)

        return {
            "model": self.model_id,
            "base_model": self.base_model,
            "adapter": None,
            "model_type": self.model_type,
            "is_remote_sensing_adapted": True,
            "is_remote_sensing_expert": True,
            "adaptation_scope": self.adaptation_scope,
            "benchmark_accuracy_claim": None,
            "confidence": None,
            "confidence_status": "not_calibrated",
            "visual_token_budget": budget_data,
            "inference_time_ms": exec_time_ms,
            "device": self.device,
            "note": (
                "Remote-sensing domain-adapted VLM candidate "
                "(AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct). "
                "Qualitative observation only; confidence is uncalibrated."
            ),
        }

    def run(
        self,
        inputs: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes AdaptLLM VLM inference or returns clean specialist_unavailable contract.
        Never automatically substitutes another model.
        """
        query = inputs.get("query") or inputs.get("question") or ""
        params = {**self.permitted_parameters, **(parameters or {})}
        budget = self.get_token_budget_config()

        # If weights are not installed or enabled, return clean specialist_unavailable contract
        if not self.is_available:
            return {
                "status": "specialist_unavailable",
                "answer": f"Analysis unavailable: {self.unavailable_reason}",
                "model": self.model_id,
                "base_model": self.base_model,
                "adapter": None,
                "model_type": self.model_type,
                "device": self.device,
                "inference_time_ms": 0.0,
                "confidence": None,
                "confidence_status": "not_calibrated",
                "evidence_source": "adaptllm_vlm",
                "evidence": {
                    "question": query,
                    "adaptllm_status": "unavailable",
                    "model_type": self.model_type,
                },
                "provenance": self.get_provenance(),
                "warnings": [self.unavailable_reason],
            }

        image_input = inputs.get("image")
        if image_input is None:
            return {
                "status": "error",
                "answer": "AdaptLLM execution failed: No image provided.",
                "model": self.model_id,
                "base_model": self.base_model,
                "adapter": None,
                "model_type": self.model_type,
                "device": self.device,
                "inference_time_ms": 0.0,
                "confidence": None,
                "confidence_status": "not_calibrated",
                "evidence_source": "adaptllm_vlm",
                "evidence": {"error": "Missing input image"},
                "provenance": self.get_provenance(),
                "warnings": ["No image provided for visual reasoning."],
            }

        # Safe image preprocessing with token/pixel cap
        pil_img, img_meta = self._preprocess_image(image_input, max_dim=budget["max_image_size"])

        # Cache check
        raw_bytes = b""
        try:
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
            "Qualitative observation from remote-sensing domain-adapted candidate "
            "(AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct); confidence is uncalibrated."
        ]

        try:
            model, processor = self._load_model()

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

            proc_kwargs = {
                "text": [text],
                "images": [pil_img],
                "padding": True,
                "return_tensors": "pt",
            }
            # Provide min/max pixels if supported by processor call
            try:
                inputs_proc = processor(
                    **proc_kwargs,
                    min_pixels=budget["min_pixels"],
                    max_pixels=budget["max_pixels"],
                ).to(self.device)
            except TypeError:
                inputs_proc = processor(**proc_kwargs).to(self.device)

            max_tokens = int(params.get("max_new_tokens", budget["max_new_tokens"]))
            temp = float(params.get("temperature", 0.2))

            inference_ctx = (
                torch.inference_mode()
                if hasattr(torch, "inference_mode")
                else torch.no_grad()
            )

            with inference_ctx:
                gen_kwargs = {"max_new_tokens": max_tokens}
                if temp > 0:
                    gen_kwargs["temperature"] = temp
                    gen_kwargs["do_sample"] = True
                else:
                    gen_kwargs["do_sample"] = False

                generated_ids = model.generate(**inputs_proc, **gen_kwargs)

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs_proc.input_ids, generated_ids)
            ]
            answer = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            exec_time_ms = round((time.time() - t_start) * 1000, 2)
            prov = self.get_provenance(budget_meta=img_meta, exec_time_ms=exec_time_ms)

            res = {
                "status": "success",
                "answer": answer,
                "model": self.model_id,
                "base_model": self.base_model,
                "adapter": None,
                "model_type": self.model_type,
                "device": self.device,
                "inference_time_ms": exec_time_ms,
                "confidence": None,
                "confidence_status": "not_calibrated",
                "evidence_source": "adaptllm_vlm",
                "evidence": {
                    "question": query,
                    "qualitative_answer": answer,
                    "model_type": self.model_type,
                    "cached": False,
                },
                "provenance": prov,
                "warnings": warnings,
            }
            result_cache.set(cache_key, res)
            return res

        except Exception as e:
            exec_time_ms = round((time.time() - t_start) * 1000, 2)
            prov = self.get_provenance(budget_meta=img_meta, exec_time_ms=exec_time_ms)
            return {
                "status": "error",
                "answer": f"AdaptLLM execution error: {str(e)}",
                "model": self.model_id,
                "base_model": self.base_model,
                "adapter": None,
                "model_type": self.model_type,
                "device": self.device,
                "inference_time_ms": exec_time_ms,
                "confidence": None,
                "confidence_status": "not_calibrated",
                "evidence_source": "adaptllm_vlm",
                "evidence": {"error": str(e)},
                "provenance": prov,
                "warnings": [str(e)],
            }

    def to_spec(self) -> Dict[str, Any]:
        base_spec = super().to_spec()
        base_spec.update({
            "model": self.model_id,
            "base_model": self.base_model,
            "model_type": self.model_type,
            "is_remote_sensing_adapted": self.is_remote_sensing_adapted,
            "is_remote_sensing_expert": self.is_remote_sensing_expert,
            "adaptation_scope": self.adaptation_scope,
            "benchmark_accuracy_claim": self.benchmark_accuracy_claim,
            "confidence": None,
            "confidence_status": "not_calibrated",
            "provenance": self.get_provenance(),
            "is_available": self.is_available,
            "unavailable_reason": None if self.is_available else self.unavailable_reason,
        })
        return base_spec
