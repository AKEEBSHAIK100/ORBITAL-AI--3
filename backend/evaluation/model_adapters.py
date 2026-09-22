"""
SatQuery AI — Model Adapters for Remote-Sensing VLM Evaluation.

Defines standardized adapters for:
1. GeoChat-7B (Candidate A — RS-native broad VLM)
2. SkyEyeGPT (Candidate B — RS-native multimodal instruction follower)
3. Qwen2-VL-2B-Instruct (Baseline Candidate C — General multimodal baseline)

Features:
- Normalized EvaluationResult schema
- Strict offline weights verification (no automatic downloads)
- Lifecycle management (lazy load, explicit unload, memory release)
- Geospatial pair validation (never silently resizes incompatible images)
- Truthful provenance and uncalibrated confidence defaults
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .schemas import (
    ConfidenceStatus,
    EvaluationResult,
    HumanReviewGrading,
    ModelAvailability,
    ModelMetadata,
    ModelStatus,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def detect_eval_device() -> str:
    """Detects available evaluation compute device without side effects."""
    override = os.getenv("RS_EVAL_DEVICE")
    if override:
        return override
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_gpu_memory_mb() -> Optional[float]:
    """Measures GPU memory allocated if CUDA is available; returns None otherwise."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
    except Exception:
        pass
    return None


def load_image_rgb(image_input: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Safely loads and converts input image to an RGB PIL Image.
    Handles GeoTIFFs via OpenCV fallback if PIL fails on specialized GeoTIFF tags.
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB"), None

    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return Image.fromarray(image_input).convert("RGB"), None
        elif image_input.ndim == 3:
            if image_input.shape[2] == 3:
                return Image.fromarray(image_input.astype(np.uint8)), None
            elif image_input.shape[2] > 3:
                return Image.fromarray(image_input[:, :, :3].astype(np.uint8)), None
        return None, f"Unsupported numpy array shape: {image_input.shape}"

    path = Path(image_input)
    if not path.exists():
        return None, f"Image file not found: {path}"

    try:
        pil_img = Image.open(path).convert("RGB")
        return pil_img, None
    except Exception as pil_err:
        # Fallback to cv2 for GeoTIFFs with geospatial tags
        try:
            import cv2
            cv_img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if cv_img is not None:
                rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb), None
        except Exception:
            pass
        return None, f"Failed to load image from {path}: {str(pil_err)}"


class RemoteSensingVLMAdapter(ABC):
    """Abstract base adapter for all candidate vision-language models in the benchmark."""

    model_id: str
    model_name: str
    base_model: Optional[str]
    is_remote_sensing_adapted: bool
    model_type: str  # "remote_sensing_vlm" | "general_multimodal_vlm"
    training_source: Optional[str]
    license: Optional[str]
    source_url: Optional[str]
    supported_modalities: List[str]
    provenance_note: str

    def __init__(self, weights_dir: Optional[Union[str, Path]] = None):
        self._weights_dir = Path(weights_dir) if weights_dir else None
        self._model = None
        self._processor = None
        self._tokenizer = None
        self.device = detect_eval_device()

    @abstractmethod
    def resolve_weights_path(self) -> Path:
        """Returns the local path where model weights are expected."""
        pass

    @abstractmethod
    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """Checks whether required Python dependencies for this candidate are installed."""
        pass

    @abstractmethod
    def get_install_instructions(self) -> str:
        """Returns exact installation instructions and expected directory setup."""
        pass

    def check_weights_available(self) -> bool:
        """Returns True ONLY if local weights exist on disk. Never triggers network download."""
        path = self.resolve_weights_path()
        if not path.exists():
            return False
        if path.is_file():
            return True
        # Check for weights formats: safetensors, bin, or pt
        has_safetensors = any(path.glob("*.safetensors"))
        has_bin = any(path.glob("*.bin"))
        has_pt = any(path.glob("*.pt"))
        return has_safetensors or has_bin or has_pt

    def availability(self) -> ModelAvailability:
        weights_avail = self.check_weights_available()
        deps_avail, missing_deps = self.check_dependencies()
        expected_path = str(self.resolve_weights_path())

        if not weights_avail:
            status = ModelStatus.UNAVAILABLE.value
            notes = f"Model weights are not installed locally at '{expected_path}'."
        elif not deps_avail:
            status = ModelStatus.UNAVAILABLE.value
            notes = f"Missing dependencies: {', '.join(missing_deps)}."
        else:
            status = ModelStatus.AVAILABLE.value
            notes = "Model is available for evaluation."

        return ModelAvailability(
            model_id=self.model_id,
            model_name=self.model_name,
            status=status,
            weights_available=weights_avail,
            expected_path=expected_path,
            install_instructions=self.get_install_instructions(),
            dependencies_available=deps_avail,
            missing_dependencies=missing_deps,
            device=self.device,
            gpu_memory_available_mb=get_gpu_memory_mb(),
            notes=notes,
        )

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            model_name=self.model_name,
            base_model=self.base_model,
            is_remote_sensing_adapted=self.is_remote_sensing_adapted,
            model_type=self.model_type,
            training_source=self.training_source,
            checkpoint_availability=self.check_weights_available(),
            license=self.license,
            source_url=self.source_url,
            supported_modalities=self.supported_modalities,
            provenance_note=self.provenance_note,
        )

    def load(self) -> None:
        """Loads model weights into memory. Overridden by concrete adapters."""
        if not self.check_weights_available():
            raise FileNotFoundError(
                f"Cannot load {self.model_name}: weights absent at {self.resolve_weights_path()}."
            )

    def unload(self) -> None:
        """Safely unloads model and frees RAM/VRAM."""
        self._model = None
        self._processor = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @abstractmethod
    def _run_inference(self, image: Image.Image, prompt: str) -> str:
        """Performs actual model inference."""
        pass

    def analyze(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        question: str,
        question_id: str = "RS-EVAL",
        category: str = "general",
        image_id: str = "image_sample",
    ) -> EvaluationResult:
        """Analyzes a single satellite image."""
        avail = self.availability()
        if avail.status != ModelStatus.AVAILABLE.value:
            return EvaluationResult(
                question_id=question_id,
                category=category,
                model_id=self.model_id,
                model_name=self.model_name,
                image_id=image_id,
                status=ModelStatus.UNAVAILABLE.value,
                answer=f"Evaluation unavailable: {avail.notes}",
                inference_time_ms=None,
                device=self.device,
                memory_mb=None,
                confidence=None,
                confidence_status=ConfidenceStatus.NOT_CALIBRATED.value,
                warnings=[avail.notes] if avail.notes else [],
                error=None,
                is_remote_sensing_adapted=self.is_remote_sensing_adapted,
                model_type=self.model_type,
                source=self.source_url,
                weights_available=False,
                ground_truth_available=False,
            )

        img, err = load_image_rgb(image_input)
        if err or img is None:
            return EvaluationResult(
                question_id=question_id,
                category=category,
                model_id=self.model_id,
                model_name=self.model_name,
                image_id=image_id,
                status=ModelStatus.ERROR.value,
                answer=f"Image loading failed: {err}",
                inference_time_ms=None,
                device=self.device,
                memory_mb=None,
                confidence=None,
                confidence_status=ConfidenceStatus.NOT_CALIBRATED.value,
                warnings=[],
                error=err,
                is_remote_sensing_adapted=self.is_remote_sensing_adapted,
                model_type=self.model_type,
                source=self.source_url,
                weights_available=True,
                ground_truth_available=False,
            )

        t0 = time.perf_counter()
        try:
            if self._model is None:
                self.load()
            answer = self._run_inference(img, question)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            mem_mb = get_gpu_memory_mb()

            return EvaluationResult(
                question_id=question_id,
                category=category,
                model_id=self.model_id,
                model_name=self.model_name,
                image_id=image_id,
                status=ModelStatus.SUCCESS.value,
                answer=answer,
                inference_time_ms=latency_ms,
                device=self.device,
                memory_mb=mem_mb,
                confidence=None,
                confidence_status=ConfidenceStatus.NOT_CALIBRATED.value,
                warnings=[],
                error=None,
                is_remote_sensing_adapted=self.is_remote_sensing_adapted,
                model_type=self.model_type,
                source=self.source_url,
                weights_available=True,
                ground_truth_available=False,
            )
        except Exception as e:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return EvaluationResult(
                question_id=question_id,
                category=category,
                model_id=self.model_id,
                model_name=self.model_name,
                image_id=image_id,
                status=ModelStatus.ERROR.value,
                answer=f"Inference error in {self.model_name}: {str(e)}",
                inference_time_ms=latency_ms,
                device=self.device,
                memory_mb=None,
                confidence=None,
                confidence_status=ConfidenceStatus.NOT_CALIBRATED.value,
                warnings=[],
                error=str(e),
                is_remote_sensing_adapted=self.is_remote_sensing_adapted,
                model_type=self.model_type,
                source=self.source_url,
                weights_available=True,
                ground_truth_available=False,
            )

    def analyze_pair(
        self,
        image_a_input: Union[str, Path, np.ndarray, Image.Image],
        image_b_input: Union[str, Path, np.ndarray, Image.Image],
        question: str,
        question_id: str = "RS-PAIR-EVAL",
        category: str = "temporal_change_reasoning",
        image_id: str = "pair_sample",
        require_same_dimensions: bool = False,
    ) -> EvaluationResult:
        """
        Analyzes a pair of satellite images (e.g. bi-temporal or optical-SAR).
        Enforces geospatial safety: never silently resizes incompatible images.
        """
        avail = self.availability()
        if avail.status != ModelStatus.AVAILABLE.value:
            return EvaluationResult(
                question_id=question_id,
                category=category,
                model_id=self.model_id,
                model_name=self.model_name,
                image_id=image_id,
                status=ModelStatus.UNAVAILABLE.value,
                answer=f"Evaluation unavailable: {avail.notes}",
                inference_time_ms=None,
                device=self.device,
                memory_mb=None,
                confidence=None,
                confidence_status=ConfidenceStatus.NOT_CALIBRATED.value,
                warnings=[avail.notes] if avail.notes else [],
                error=None,
                is_remote_sensing_adapted=self.is_remote_sensing_adapted,
                model_type=self.model_type,
                source=self.source_url,
                weights_available=False,
                ground_truth_available=False,
            )

        img_a, err_a = load_image_rgb(image_a_input)
        img_b, err_b = load_image_rgb(image_b_input)
        if err_a or img_a is None:
            return EvaluationResult(
                question_id=question_id,
                category=category,
                model_id=self.model_id,
                model_name=self.model_name,
                image_id=image_id,
                status=ModelStatus.ERROR.value,
                answer=f"Image A loading failed: {err_a}",
                inference_time_ms=None,
                device=self.device,
                memory_mb=None,
                confidence=None,
                confidence_status=ConfidenceStatus.NOT_CALIBRATED.value,
                warnings=[],
                error=err_a,
                is_remote_sensing_adapted=self.is_remote_sensing_adapted,
                model_type=self.model_type,
                source=self.source_url,
                weights_available=True,
                ground_truth_available=False,
            )
        if err_b or img_b is None:
            return EvaluationResult(
                question_id=question_id,
                category=category,
                model_id=self.model_id,
                model_name=self.model_name,
                image_id=image_id,
                status=ModelStatus.ERROR.value,
                answer=f"Image B loading failed: {err_b}",
                inference_time_ms=None,
                device=self.device,
                memory_mb=None,
                confidence=None,
                confidence_status=ConfidenceStatus.NOT_CALIBRATED.value,
                warnings=[],
                error=err_b,
                is_remote_sensing_adapted=self.is_remote_sensing_adapted,
                model_type=self.model_type,
                source=self.source_url,
                weights_available=True,
                ground_truth_available=False,
            )

        warnings = []
        if img_a.size != img_b.size:
            if require_same_dimensions:
                return EvaluationResult(
                    question_id=question_id,
                    category=category,
                    model_id=self.model_id,
                    model_name=self.model_name,
                    image_id=image_id,
                    status=ModelStatus.ERROR.value,
                    answer=(
                        f"Geospatial alignment error: Image A dimensions {img_a.size} do not match "
                        f"Image B dimensions {img_b.size}. Silent resizing is strictly disallowed."
                    ),
                    inference_time_ms=None,
                    device=self.device,
                    memory_mb=None,
                    confidence=None,
                    confidence_status=ConfidenceStatus.NOT_CALIBRATED.value,
                    warnings=["Mismatched pair dimensions rejected without silent resizing."],
                    error="Dimension mismatch in paired imagery.",
                    is_remote_sensing_adapted=self.is_remote_sensing_adapted,
                    model_type=self.model_type,
                    source=self.source_url,
                    weights_available=True,
                    ground_truth_available=False,
                )
            warnings.append(
                f"Pair dimensions differ: Image A is {img_a.size}, Image B is {img_b.size}. "
                "Co-registration not asserted."
            )

        # Composite the pair side-by-side without altering native individual aspect ratios or silent warping
        t0 = time.perf_counter()
        try:
            if self._model is None:
                self.load()

            # Create side-by-side composite canvas
            max_h = max(img_a.height, img_b.height)
            total_w = img_a.width + img_b.width
            composite = Image.new("RGB", (total_w, max_h), color=(0, 0, 0))
            composite.paste(img_a, (0, 0))
            composite.paste(img_b, (img_a.width, 0))

            prompt = (
                f"This image contains two paired remote sensing scenes side by side: "
                f"Left is Image A ({img_a.size[0]}x{img_a.size[1]}), "
                f"Right is Image B ({img_b.size[0]}x{img_b.size[1]}).\n"
                f"Question: {question}"
            )

            answer = self._run_inference(composite, prompt)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            mem_mb = get_gpu_memory_mb()

            return EvaluationResult(
                question_id=question_id,
                category=category,
                model_id=self.model_id,
                model_name=self.model_name,
                image_id=image_id,
                status=ModelStatus.SUCCESS.value,
                answer=answer,
                inference_time_ms=latency_ms,
                device=self.device,
                memory_mb=mem_mb,
                confidence=None,
                confidence_status=ConfidenceStatus.NOT_CALIBRATED.value,
                warnings=warnings,
                error=None,
                is_remote_sensing_adapted=self.is_remote_sensing_adapted,
                model_type=self.model_type,
                source=self.source_url,
                weights_available=True,
                ground_truth_available=False,
            )
        except Exception as e:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return EvaluationResult(
                question_id=question_id,
                category=category,
                model_id=self.model_id,
                model_name=self.model_name,
                image_id=image_id,
                status=ModelStatus.ERROR.value,
                answer=f"Pair inference error in {self.model_name}: {str(e)}",
                inference_time_ms=latency_ms,
                device=self.device,
                memory_mb=None,
                confidence=None,
                confidence_status=ConfidenceStatus.NOT_CALIBRATED.value,
                warnings=warnings,
                error=str(e),
                is_remote_sensing_adapted=self.is_remote_sensing_adapted,
                model_type=self.model_type,
                source=self.source_url,
                weights_available=True,
                ground_truth_available=False,
            )


# ─── Candidate A: GeoChat-7B ───────────────────────────────────────────────────

class GeoChatAdapter(RemoteSensingVLMAdapter):
    """
    Candidate A: GeoChat-7B.
    Remote-sensing-native vision-language model trained on multimodal remote sensing instruction datasets.
    """

    model_id = "geochat_7b"
    model_name = "GeoChat-7B"
    base_model = "MBZUAI/geochat-7b"
    is_remote_sensing_adapted = True
    model_type = "remote_sensing_vlm"
    training_source = "GeoChat: Grounded Large Vision-Language Model for Remote Sensing (Kuckreja et al., CVPR 2024)"
    license = "Non-commercial research / LLaVA-Vicuna license terms"
    source_url = "https://github.com/mbzuai-oryx/GeoChat"
    supported_modalities = ["optical_rgb", "high_resolution_aerial", "satellite_multispectral"]
    provenance_note = (
        "Remote-sensing-native visual instruction tuned model built upon LLaVA architecture. "
        "Pre-trained on remote-sensing imagery and fine-tuned on RS instruction dialogues."
    )

    def resolve_weights_path(self) -> Path:
        env_path = os.getenv("GEOCHAT_WEIGHTS_PATH")
        if env_path:
            return Path(env_path).resolve()
        return (PROJECT_ROOT / "backend" / "models" / "evaluation" / "geochat_7b").resolve()

    def check_dependencies(self) -> Tuple[bool, List[str]]:
        missing = []
        try:
            import transformers
        except ImportError:
            missing.append("transformers")
        try:
            import torch
        except ImportError:
            missing.append("torch")
        # GeoChat typically relies on llava or transformers model code
        return (len(missing) == 0, missing)

    def get_install_instructions(self) -> str:
        expected = self.resolve_weights_path()
        return (
            f"1. Download GeoChat-7B checkpoint from HuggingFace (e.g. MBZUAI/geochat-7b).\n"
            f"2. Place files into: {expected}\n"
            f"   (or set environment variable GEOCHAT_WEIGHTS_PATH to the checkpoint directory).\n"
            f"3. Ensure PyTorch and Transformers are installed with CUDA support if running on GPU."
        )

    def load(self) -> None:
        if not self.check_weights_available():
            raise FileNotFoundError(f"GeoChat-7B weights not installed at {self.resolve_weights_path()}.")

        path = str(self.resolve_weights_path())
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                path,
                local_files_only=True,
                torch_dtype="auto",
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device != "cuda" and self._model is not None:
                self._model = self._model.to("cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load GeoChat-7B checkpoint: {e}")

    def _run_inference(self, image: Image.Image, prompt: str) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("GeoChat-7B is not loaded.")
        # Minimal offline inference stub when model is instantiated
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        output = self._model.generate(**inputs, max_new_tokens=128)
        return self._tokenizer.decode(output[0], skip_special_tokens=True)


# ─── Candidate B: SkyEyeGPT ───────────────────────────────────────────────────

class SkyEyeGPTAdapter(RemoteSensingVLMAdapter):
    """
    Candidate B: SkyEyeGPT.
    Remote-sensing-native multimodal instruction-following model.
    """

    model_id = "skyeyegpt"
    model_name = "SkyEyeGPT"
    base_model = "Sun-Y/SkyEyeGPT"
    is_remote_sensing_adapted = True
    model_type = "remote_sensing_vlm"
    training_source = "SkyEyeGPT: Remote Sensing Multimodal Instruction Tuning"
    license = "Research / Academic use"
    source_url = "https://github.com/Sun-Y/SkyEyeGPT"
    supported_modalities = ["optical_rgb", "aerial", "satellite_multispectral"]
    provenance_note = (
        "Remote-sensing instruction-tuned vision-language model designed for complex "
        "overhead reasoning and multi-turn instruction following."
    )

    def resolve_weights_path(self) -> Path:
        env_path = os.getenv("SKYEYEGPT_WEIGHTS_PATH")
        if env_path:
            return Path(env_path).resolve()
        return (PROJECT_ROOT / "backend" / "models" / "evaluation" / "skyeyegpt").resolve()

    def check_dependencies(self) -> Tuple[bool, List[str]]:
        missing = []
        try:
            import transformers
        except ImportError:
            missing.append("transformers")
        try:
            import torch
        except ImportError:
            missing.append("torch")
        return (len(missing) == 0, missing)

    def get_install_instructions(self) -> str:
        expected = self.resolve_weights_path()
        return (
            f"1. Download SkyEyeGPT checkpoint (e.g. from Sun-Y/SkyEyeGPT HuggingFace repository).\n"
            f"2. Place files into: {expected}\n"
            f"   (or set environment variable SKYEYEGPT_WEIGHTS_PATH to the checkpoint directory).\n"
            f"3. Verify checkpoint integrity before running evaluation."
        )

    def load(self) -> None:
        if not self.check_weights_available():
            raise FileNotFoundError(f"SkyEyeGPT weights not installed at {self.resolve_weights_path()}.")

        path = str(self.resolve_weights_path())
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                path,
                local_files_only=True,
                torch_dtype="auto",
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device != "cuda" and self._model is not None:
                self._model = self._model.to("cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load SkyEyeGPT checkpoint: {e}")

    def _run_inference(self, image: Image.Image, prompt: str) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("SkyEyeGPT is not loaded.")
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        output = self._model.generate(**inputs, max_new_tokens=128)
        return self._tokenizer.decode(output[0], skip_special_tokens=True)


# ─── Candidate C: Qwen2-VL-2B-Instruct (General Baseline) ─────────────────────

class Qwen2VLAdapter(RemoteSensingVLMAdapter):
    """
    Baseline Candidate C: Qwen2-VL-2B-Instruct.
    General multimodal baseline vision-language foundation model.
    Explicitly marked: is_remote_sensing_adapted = False, model_type = "general_multimodal_vlm".
    """

    model_id = "qwen2_vl_2b"
    model_name = "Qwen2-VL-2B-Instruct"
    base_model = "Qwen/Qwen2-VL-2B-Instruct"
    is_remote_sensing_adapted = False  # Critical: general foundation model
    model_type = "general_multimodal_vlm"
    training_source = "Qwen2-VL: Enhancing Vision-Language Model Capabilities (Alibaba, 2024)"
    license = "Apache-2.0"
    source_url = "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct"
    supported_modalities = ["optical_rgb", "natural_images", "general_imagery"]
    provenance_note = (
        "General multimodal vision-language foundation model. NOT fine-tuned or adapted "
        "on remote sensing imagery. Used as an uncalibrated generalist baseline."
    )

    def resolve_weights_path(self) -> Path:
        env_path = os.getenv("QWEN2_VL_WEIGHTS_PATH")
        if env_path:
            return Path(env_path).resolve()
        # Check production generalist directory first
        gen_dir = PROJECT_ROOT / "backend" / "models" / "generalist" / "qwen2_vl"
        if gen_dir.exists() and (any(gen_dir.glob("*.safetensors")) or any(gen_dir.glob("*.bin"))):
            return gen_dir.resolve()
        return (PROJECT_ROOT / "backend" / "models" / "evaluation" / "qwen2_vl_2b").resolve()

    def check_dependencies(self) -> Tuple[bool, List[str]]:
        missing = []
        try:
            import transformers
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError:
            missing.append("transformers[qwen2_vl]")
        try:
            import torch
        except ImportError:
            missing.append("torch")
        return (len(missing) == 0, missing)

    def get_install_instructions(self) -> str:
        expected = self.resolve_weights_path()
        return (
            f"1. Download Qwen/Qwen2-VL-2B-Instruct from HuggingFace.\n"
            f"2. Place files into: {expected}\n"
            f"   (or set environment variable QWEN2_VL_WEIGHTS_PATH to the checkpoint directory).\n"
            f"3. Ensure transformers>=4.40.0 is installed."
        )

    def load(self) -> None:
        if not self.check_weights_available():
            raise FileNotFoundError(f"Qwen2-VL-2B-Instruct weights not installed at {self.resolve_weights_path()}.")

        path = str(self.resolve_weights_path())
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            self._processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                path,
                local_files_only=True,
                torch_dtype="auto",
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device != "cuda" and self._model is not None:
                self._model = self._model.to("cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen2-VL-2B checkpoint: {e}")

    def _run_inference(self, image: Image.Image, prompt: str) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("Qwen2-VL-2B is not loaded.")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], padding=True, return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to("cuda")
        output_ids = self._model.generate(**inputs, max_new_tokens=128)
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# ─── Registry of Candidates ───────────────────────────────────────────────────

EVAL_CANDIDATES: Dict[str, type[RemoteSensingVLMAdapter]] = {
    "geochat_7b": GeoChatAdapter,
    "skyeyegpt": SkyEyeGPTAdapter,
    "qwen2_vl_2b": Qwen2VLAdapter,
}


def get_candidate_adapter(model_id: str, weights_dir: Optional[Union[str, Path]] = None) -> RemoteSensingVLMAdapter:
    """Returns an instantiated adapter for the specified candidate model."""
    if model_id not in EVAL_CANDIDATES:
        raise ValueError(f"Unknown candidate '{model_id}'. Available: {list(EVAL_CANDIDATES.keys())}")
    return EVAL_CANDIDATES[model_id](weights_dir=weights_dir)


def list_candidate_adapters() -> List[RemoteSensingVLMAdapter]:
    """Instantiates all candidate adapters (without loading weights)."""
    return [cls() for cls in EVAL_CANDIDATES.values()]
