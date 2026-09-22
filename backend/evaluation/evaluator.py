"""
SatQuery AI — Remote-Sensing VLM Benchmark Evaluator.

Executes the 40-question benchmark suite against candidate models.
Key guarantees:
1. Strict one-model-at-a-time lifecycle (load -> evaluate -> unload).
2. Never loads all candidates simultaneously.
3. Does not fabricate ground truth or accuracy percentages.
4. Objective telemetry (latency, memory, success/failure counts).
5. Human review grading templates initialized with "manual_review_required".
6. Clean error isolation (OOM or missing weights never crash the runner).
"""

from __future__ import annotations

import logging
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

from .model_adapters import RemoteSensingVLMAdapter, get_candidate_adapter
from .question_set import get_40_question_suite, get_evaluation_samples
from .schemas import (
    EvaluationQuestion,
    EvaluationResult,
    HumanReviewGrading,
    ImageSample,
    ModelBenchmarkMetrics,
    ModelStatus,
)

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Orchestrates benchmark evaluation over candidate vision-language models."""

    def __init__(self, samples: Optional[Dict[str, ImageSample]] = None, questions: Optional[List[EvaluationQuestion]] = None):
        self.samples = samples or get_evaluation_samples()
        self.questions = questions or get_40_question_suite()

    def evaluate_model(
        self,
        adapter: RemoteSensingVLMAdapter,
        dry_run: bool = False,
    ) -> Tuple[List[EvaluationResult], ModelBenchmarkMetrics]:
        """
        Evaluates a single candidate adapter across all 40 benchmark questions.
        Loads the model, executes questions sequentially, and unloads the model immediately.
        """
        results: List[EvaluationResult] = []
        latencies: List[float] = []
        success_count = 0
        failed_count = 0
        unavailable_count = 0

        avail = adapter.availability()
        weights_avail = avail.weights_available

        # If weights absent or dry_run, we do not attempt to load weights
        if not dry_run and weights_avail:
            try:
                adapter.load()
            except Exception as e:
                logger.error(f"Failed to load {adapter.model_name}: {e}")
                # Treat as error / unavailable for all questions
                for q in self.questions:
                    res = EvaluationResult(
                        question_id=q.question_id,
                        category=q.category,
                        model_id=adapter.model_id,
                        model_name=adapter.model_name,
                        image_id=q.sample_id,
                        status=ModelStatus.ERROR.value,
                        answer=f"Model failed to load: {str(e)}",
                        inference_time_ms=None,
                        device=adapter.device,
                        memory_mb=None,
                        confidence=None,
                        confidence_status="not_calibrated",
                        warnings=[f"Model loading exception: {str(e)}"],
                        error=str(e),
                        is_remote_sensing_adapted=adapter.is_remote_sensing_adapted,
                        model_type=adapter.model_type,
                        source=adapter.source_url,
                        weights_available=True,
                        ground_truth_available=q.ground_truth_available,
                        review_grading=HumanReviewGrading(review_status="manual_review_required"),
                    )
                    results.append(res)
                    failed_count += 1

                adapter.unload()
                metrics = ModelBenchmarkMetrics(
                    model_id=adapter.model_id,
                    model_name=adapter.model_name,
                    status=ModelStatus.ERROR.value,
                    weights_available=True,
                    total_questions=len(self.questions),
                    successful_inferences=0,
                    failed_inferences=failed_count,
                    unavailable_inferences=0,
                    avg_inference_time_ms=None,
                    median_inference_time_ms=None,
                    max_inference_time_ms=None,
                    memory_mb=None,
                    manual_review_status="manual_review_required",
                    known_limitations=[f"Load failure: {str(e)}"],
                )
                return results, metrics

        try:
            for q in self.questions:
                sample = self.samples.get(q.sample_id)
                if sample is None:
                    res = EvaluationResult(
                        question_id=q.question_id,
                        category=q.category,
                        model_id=adapter.model_id,
                        model_name=adapter.model_name,
                        image_id=q.sample_id,
                        status=ModelStatus.ERROR.value,
                        answer=f"Evaluation sample '{q.sample_id}' not found in registry.",
                        error="Missing image sample.",
                        is_remote_sensing_adapted=adapter.is_remote_sensing_adapted,
                        model_type=adapter.model_type,
                        weights_available=weights_avail,
                        ground_truth_available=q.ground_truth_available,
                        review_grading=HumanReviewGrading(review_status="manual_review_required"),
                    )
                    results.append(res)
                    failed_count += 1
                    continue

                if dry_run:
                    # Dry-run execution: simulate evaluation without real model inference
                    res = EvaluationResult(
                        question_id=q.question_id,
                        category=q.category,
                        model_id=adapter.model_id,
                        model_name=adapter.model_name,
                        image_id=sample.image_id,
                        status=ModelStatus.SUCCESS.value if weights_avail else ModelStatus.UNAVAILABLE.value,
                        answer=f"[DRY-RUN] Simulated evaluation response for question: {q.question_text}",
                        inference_time_ms=15.0,
                        device=adapter.device,
                        memory_mb=None,
                        confidence=None,
                        confidence_status="not_calibrated",
                        warnings=["Dry-run execution without model weights."],
                        is_remote_sensing_adapted=adapter.is_remote_sensing_adapted,
                        model_type=adapter.model_type,
                        source=adapter.source_url,
                        weights_available=weights_avail,
                        ground_truth_available=q.ground_truth_available,
                        review_grading=HumanReviewGrading(review_status="manual_review_required"),
                    )
                    results.append(res)
                    if weights_avail:
                        success_count += 1
                        latencies.append(15.0)
                    else:
                        unavailable_count += 1
                    continue

                # Real evaluation execution
                if q.input_type == "pair":
                    res = adapter.analyze_pair(
                        image_a_input=sample.image_path,
                        image_b_input=sample.secondary_image_path or sample.image_path,
                        question=q.question_text,
                        question_id=q.question_id,
                        category=q.category,
                        image_id=sample.image_id,
                    )
                else:
                    res = adapter.analyze(
                        image_input=sample.image_path,
                        question=q.question_text,
                        question_id=q.question_id,
                        category=q.category,
                        image_id=sample.image_id,
                    )

                res.ground_truth_available = q.ground_truth_available
                res.review_grading = HumanReviewGrading(review_status="manual_review_required")
                results.append(res)

                if res.status == ModelStatus.SUCCESS.value:
                    success_count += 1
                    if res.inference_time_ms is not None:
                        latencies.append(res.inference_time_ms)
                elif res.status == ModelStatus.UNAVAILABLE.value:
                    unavailable_count += 1
                else:
                    failed_count += 1

        finally:
            # Performance Safety: unload candidate model before returning
            adapter.unload()

        # Compute objective latency statistics
        avg_lat = round(statistics.mean(latencies), 2) if latencies else None
        med_lat = round(statistics.median(latencies), 2) if latencies else None
        max_lat = round(max(latencies), 2) if latencies else None

        # Build category performance summary
        cat_perf: Dict[str, Dict[str, Any]] = {}
        for r in results:
            if r.category not in cat_perf:
                cat_perf[r.category] = {"total": 0, "success": 0, "unavailable": 0, "failed": 0}
            cat_perf[r.category]["total"] += 1
            if r.status == ModelStatus.SUCCESS.value:
                cat_perf[r.category]["success"] += 1
            elif r.status == ModelStatus.UNAVAILABLE.value:
                cat_perf[r.category]["unavailable"] += 1
            else:
                cat_perf[r.category]["failed"] += 1

        overall_status = ModelStatus.SUCCESS.value if success_count > 0 else (
            ModelStatus.UNAVAILABLE.value if unavailable_count > 0 else ModelStatus.ERROR.value
        )

        known_limits = []
        if not weights_avail:
            known_limits.append("Model weights not installed locally; evaluation reported unavailable.")
        if not adapter.is_remote_sensing_adapted:
            known_limits.append("General vision-language model not fine-tuned on satellite imagery.")

        metrics = ModelBenchmarkMetrics(
            model_id=adapter.model_id,
            model_name=adapter.model_name,
            status=overall_status,
            weights_available=weights_avail,
            total_questions=len(self.questions),
            successful_inferences=success_count,
            failed_inferences=failed_count,
            unavailable_inferences=unavailable_count,
            avg_inference_time_ms=avg_lat,
            median_inference_time_ms=med_lat,
            max_inference_time_ms=max_lat,
            memory_mb=None,
            manual_review_status="manual_review_required",
            known_limitations=known_limits,
            category_performance=cat_perf,
        )

        return results, metrics
