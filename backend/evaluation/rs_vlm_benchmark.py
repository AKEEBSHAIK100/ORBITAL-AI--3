"""
SatQuery AI — Remote-Sensing VLM Benchmark CLI.

Commands:
  python -m backend.evaluation.rs_vlm_benchmark --list-models
  python -m backend.evaluation.rs_vlm_benchmark --check
  python -m backend.evaluation.rs_vlm_benchmark --model geochat_7b
  python -m backend.evaluation.rs_vlm_benchmark --model skyeyegpt
  python -m backend.evaluation.rs_vlm_benchmark --model qwen2_vl_2b
  python -m backend.evaluation.rs_vlm_benchmark --all
  python -m backend.evaluation.rs_vlm_benchmark --report
  python -m backend.evaluation.rs_vlm_benchmark --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .evaluator import BenchmarkEvaluator
from .model_adapters import (
    EVAL_CANDIDATES,
    RemoteSensingVLMAdapter,
    get_candidate_adapter,
    list_candidate_adapters,
)
from .question_set import get_40_question_suite, get_evaluation_samples
from .report import generate_json_report, generate_markdown_report
from .schemas import EvaluationResult, ModelBenchmarkMetrics, ModelStatus


def handle_list_models() -> None:
    """Lists registered candidate models and their provenance."""
    print("\n" + "=" * 75)
    print("SatQuery AI — Registered Remote-Sensing VLM Candidates")
    print("=" * 75)
    adapters = list_candidate_adapters()
    for adapter in adapters:
        meta = adapter.metadata()
        adapt_str = "Remote-Sensing Native" if meta.is_remote_sensing_adapted else "General Multimodal (Unadapted)"
        print(f"\nModel ID:      {meta.model_id}")
        print(f"Model Name:    {meta.model_name}")
        print(f"Base Model:    {meta.base_model}")
        print(f"Adaptation:    {adapt_str}")
        print(f"Model Type:    {meta.model_type}")
        print(f"License:       {meta.license}")
        print(f"Source URL:    {meta.source_url}")
        print(f"Modalities:    {', '.join(meta.supported_modalities)}")
        print(f"Notes:         {meta.provenance_note}")
    print("\n" + "=" * 75)


def handle_check() -> None:
    """Checks weight availability and hardware device status for all candidates."""
    print("\n" + "=" * 75)
    print("SatQuery AI — Remote-Sensing VLM Candidate Availability Check")
    print("=" * 75)
    adapters = list_candidate_adapters()
    for adapter in adapters:
        avail = adapter.availability()
        weights_str = "PRESENT" if avail.weights_available else "ABSENT (Not Installed)"
        status_str = avail.status.upper()

        print(f"\n[{adapter.model_name}] (ID: {avail.model_id})")
        print(f"  Status:            {status_str}")
        print(f"  Local Weights:     {weights_str}")
        print(f"  Expected Path:     {avail.expected_path}")
        print(f"  Compute Device:    {avail.device}")
        if avail.gpu_memory_available_mb is not None:
            print(f"  GPU VRAM:          {avail.gpu_memory_available_mb} MB")
        if not avail.dependencies_available:
            print(f"  Missing Packages:  {', '.join(avail.missing_dependencies)}")
        if not avail.weights_available:
            print(f"  Install Guide:\n    " + avail.install_instructions.replace("\n", "\n    "))
    print("\n" + "=" * 75)


def run_benchmark_for_models(
    model_ids: List[str],
    dry_run: bool = False,
    output_path: Optional[str] = None,
    output_format: str = "md",
) -> None:
    """Runs the 40-question benchmark suite sequentially across specified candidates."""
    samples = get_evaluation_samples()
    questions = get_40_question_suite()
    evaluator = BenchmarkEvaluator(samples=samples, questions=questions)

    all_metrics: List[ModelBenchmarkMetrics] = []
    all_results: dict[str, List[EvaluationResult]] = {}

    print(f"\nStarting benchmark: {len(questions)} questions across {len(samples)} image samples.")
    print(f"Evaluation candidates: {', '.join(model_ids)}")
    if dry_run:
        print("Mode: DRY-RUN (simulated responses, no heavy model weights loaded)")

    for model_id in model_ids:
        print(f"\n--- Evaluating Candidate: {model_id} ---")
        adapter = get_candidate_adapter(model_id)
        avail = adapter.availability()

        if not dry_run and not avail.weights_available:
            print(f"Notice: Local weights absent for {adapter.model_name}. Executing availability evaluation.")

        # Evaluate one candidate at a time, automatic unload in evaluator
        results, metrics = evaluator.evaluate_model(adapter, dry_run=dry_run)
        all_metrics.append(metrics)
        all_results[model_id] = results

        print(f"Completed {adapter.model_name}: "
              f"{metrics.successful_inferences} successful, "
              f"{metrics.unavailable_inferences} unavailable, "
              f"{metrics.failed_inferences} failed.")

    # Generate Report
    if output_format == "json":
        report_data = generate_json_report(
            model_metrics=all_metrics,
            detailed_results=all_results,
            image_count=len(samples),
            total_questions=len(questions),
        )
        report_str = json.dumps(report_data, indent=2)
    else:
        report_str = generate_markdown_report(
            model_metrics=all_metrics,
            detailed_results=all_results,
            image_count=len(samples),
            total_questions=len(questions),
        )

    print("\n" + report_str)

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(report_str, encoding="utf-8")
        print(f"\nReport written to: {out_file.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SatQuery AI Remote-Sensing VLM Benchmark Runner (Evaluation Phase)"
    )
    parser.add_argument("--list-models", action="store_true", help="List registered candidate models and provenance")
    parser.add_argument("--check", action="store_true", help="Check local weight and hardware availability")
    parser.add_argument("--model", type=str, choices=list(EVAL_CANDIDATES.keys()), help="Evaluate a specific model")
    parser.add_argument("--all", action="store_true", help="Evaluate all registered candidate models")
    parser.add_argument("--dry-run", action="store_true", help="Simulate benchmark execution without model weights")
    parser.add_argument("--report", action="store_true", help="Generate report from current model availability")
    parser.add_argument("--format", choices=["md", "json"], default="md", help="Report output format")
    parser.add_argument("--output", type=str, default=None, help="Save report to specified path")

    args = parser.parse_args()

    if args.list_models:
        handle_list_models()
        return

    if args.check:
        handle_check()
        return

    if args.report:
        # Run availability report across all models
        run_benchmark_for_models(list(EVAL_CANDIDATES.keys()), dry_run=True, output_path=args.output, output_format=args.format)
        return

    if args.model:
        run_benchmark_for_models([args.model], dry_run=args.dry_run, output_path=args.output, output_format=args.format)
        return

    if args.all:
        run_benchmark_for_models(list(EVAL_CANDIDATES.keys()), dry_run=args.dry_run, output_path=args.output, output_format=args.format)
        return

    # Default to check
    handle_check()


if __name__ == "__main__":
    main()
