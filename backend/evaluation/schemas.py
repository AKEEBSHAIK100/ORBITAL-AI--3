"""
SatQuery AI — Remote-Sensing VLM Evaluation Schemas.

Defines normalized data structures for:
- Model availability and hardware environment
- Model metadata and provenance
- Satellite image samples (single and paired)
- 40-question deterministic test suite
- Normalized evaluation results
- Objective performance measurements
- Human/review grading and hallucination tracking
- Summary reports
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ModelStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    SUCCESS = "success"


class ConfidenceStatus(str, Enum):
    NOT_CALIBRATED = "not_calibrated"
    CALIBRATED = "calibrated"
    UNAVAILABLE = "unavailable"


class InputType(str, Enum):
    SINGLE = "single"
    PAIR = "pair"


class ImageRole(str, Enum):
    OPTICAL = "optical"
    TEMPORAL_PAIR = "temporal_pair"
    OPTICAL_SAR = "optical_sar"


@dataclass
class ModelAvailability:
    model_id: str
    model_name: str
    status: str  # "available" | "unavailable" | "error"
    weights_available: bool
    expected_path: str
    install_instructions: str
    dependencies_available: bool
    missing_dependencies: List[str] = field(default_factory=list)
    device: str = "cpu"
    gpu_memory_available_mb: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "status": self.status,
            "weights_available": self.weights_available,
            "expected_path": self.expected_path,
            "install_instructions": self.install_instructions,
            "dependencies_available": self.dependencies_available,
            "missing_dependencies": self.missing_dependencies,
            "device": self.device,
            "gpu_memory_available_mb": self.gpu_memory_available_mb,
            "notes": self.notes,
        }


@dataclass
class ModelMetadata:
    model_id: str
    model_name: str
    base_model: Optional[str]
    is_remote_sensing_adapted: bool
    model_type: str  # "remote_sensing_vlm" | "general_multimodal_vlm"
    training_source: Optional[str]
    checkpoint_availability: bool
    license: Optional[str]
    source_url: Optional[str]
    supported_modalities: List[str]
    provenance_note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "base_model": self.base_model,
            "is_remote_sensing_adapted": self.is_remote_sensing_adapted,
            "model_type": self.model_type,
            "training_source": self.training_source,
            "checkpoint_availability": self.checkpoint_availability,
            "license": self.license,
            "source_url": self.source_url,
            "supported_modalities": self.supported_modalities,
            "provenance_note": self.provenance_note,
        }


@dataclass
class ImageSample:
    image_id: str
    image_path: str
    secondary_image_path: Optional[str] = None
    dimensions: Optional[List[int]] = None  # [width, height]
    secondary_dimensions: Optional[List[int]] = None
    modality: str = "optical"
    secondary_modality: Optional[str] = None
    crs: Optional[str] = None
    temporal_info: Optional[str] = None
    is_paired: bool = False
    is_coregistered: Optional[bool] = None
    metadata_verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "secondary_image_path": self.secondary_image_path,
            "dimensions": self.dimensions,
            "secondary_dimensions": self.secondary_dimensions,
            "modality": self.modality,
            "secondary_modality": self.secondary_modality,
            "crs": self.crs,
            "temporal_info": self.temporal_info,
            "is_paired": self.is_paired,
            "is_coregistered": self.is_coregistered,
            "metadata_verified": self.metadata_verified,
        }


@dataclass
class EvaluationQuestion:
    question_id: str
    category: str
    category_name: str
    question_text: str
    input_type: str  # "single" | "pair"
    image_role: str  # "optical" | "temporal_pair" | "optical_sar"
    sample_id: str
    ground_truth_available: bool = False
    ground_truth_answer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "category_name": self.category_name,
            "question_text": self.question_text,
            "input_type": self.input_type,
            "image_role": self.image_role,
            "sample_id": self.sample_id,
            "ground_truth_available": self.ground_truth_available,
            "ground_truth_answer": self.ground_truth_answer,
        }


@dataclass
class HumanReviewGrading:
    correctness: Optional[int] = None  # 0 = poor/incorrect, 1 = partially useful, 2 = good, 3 = strong
    relevance: Optional[int] = None
    completeness: Optional[int] = None
    rs_terminology: Optional[int] = None
    spatial_correctness: Optional[int] = None
    hallucination_observed: bool = False
    unsupported_claim: bool = False
    uncertainty_appropriate: Optional[bool] = None
    review_status: str = "manual_review_required"
    reviewer_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correctness": self.correctness,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "rs_terminology": self.rs_terminology,
            "spatial_correctness": self.spatial_correctness,
            "hallucination_observed": self.hallucination_observed,
            "unsupported_claim": self.unsupported_claim,
            "uncertainty_appropriate": self.uncertainty_appropriate,
            "review_status": self.review_status,
            "reviewer_notes": self.reviewer_notes,
        }


@dataclass
class EvaluationResult:
    question_id: str
    category: str
    model_id: str
    model_name: str
    image_id: str
    status: str  # "success" | "unavailable" | "error"
    answer: str
    inference_time_ms: Optional[float] = None
    device: str = "cpu"
    memory_mb: Optional[float] = None
    confidence: Optional[float] = None
    confidence_status: str = ConfidenceStatus.NOT_CALIBRATED.value
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    is_remote_sensing_adapted: bool = False
    model_type: str = "general_multimodal_vlm"
    source: Optional[str] = None
    weights_available: bool = False
    ground_truth_available: bool = False
    review_grading: Optional[HumanReviewGrading] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "image_id": self.image_id,
            "status": self.status,
            "answer": self.answer,
            "inference_time_ms": self.inference_time_ms,
            "device": self.device,
            "memory_mb": self.memory_mb,
            "confidence": self.confidence,
            "confidence_status": self.confidence_status,
            "error": self.error,
            "warnings": self.warnings,
            "is_remote_sensing_adapted": self.is_remote_sensing_adapted,
            "model_type": self.model_type,
            "source": self.source,
            "weights_available": self.weights_available,
            "ground_truth_available": self.ground_truth_available,
            "review_grading": self.review_grading.to_dict() if self.review_grading else HumanReviewGrading().to_dict(),
        }


@dataclass
class ModelBenchmarkMetrics:
    model_id: str
    model_name: str
    status: str
    weights_available: bool
    total_questions: int
    successful_inferences: int
    failed_inferences: int
    unavailable_inferences: int
    avg_inference_time_ms: Optional[float] = None
    median_inference_time_ms: Optional[float] = None
    max_inference_time_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    manual_review_status: str = "manual_review_required"
    known_limitations: List[str] = field(default_factory=list)
    category_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "status": self.status,
            "weights_available": self.weights_available,
            "total_questions": self.total_questions,
            "successful_inferences": self.successful_inferences,
            "failed_inferences": self.failed_inferences,
            "unavailable_inferences": self.unavailable_inferences,
            "avg_inference_time_ms": self.avg_inference_time_ms,
            "median_inference_time_ms": self.median_inference_time_ms,
            "max_inference_time_ms": self.max_inference_time_ms,
            "memory_mb": self.memory_mb,
            "manual_review_status": self.manual_review_status,
            "known_limitations": self.known_limitations,
            "category_performance": self.category_performance,
        }
