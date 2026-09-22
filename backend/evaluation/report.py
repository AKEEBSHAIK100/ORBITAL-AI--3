"""
SatQuery AI — Remote-Sensing VLM Benchmark Report Generator.

Generates neutral, objective comparison reports across evaluated models.
Strict constraints:
1. NEVER declares a 'winner' or 'best model'.
2. NEVER claims accuracy without empirical ground truth.
3. Separates objective measurements from manual review grading.
4. Generates both human-readable Markdown and structured JSON.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import EvaluationResult, ModelBenchmarkMetrics


def generate_markdown_report(
    model_metrics: List[ModelBenchmarkMetrics],
    detailed_results: Optional[Dict[str, List[EvaluationResult]]] = None,
    image_count: int = 5,
    total_questions: int = 40,
) -> str:
    """Renders a neutral Markdown comparison report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines = [
        "# RS-VLM BAKE-OFF BENCHMARK REPORT",
        "",
        f"**Date (UTC)**: {timestamp}  ",
        f"**Benchmark Scope**: Remote-Sensing-Native VLM Evaluation  ",
        f"**Candidate Models**: {', '.join(m.model_name for m in model_metrics)}  ",
        f"**Total Questions**: {total_questions} (20 categories × 2 questions)  ",
        f"**Satellite Images Used**: {image_count} (Optical aerial, bi-temporal LEVIR-CD pair, Optical-SAR pair)  ",
        "",
        "> [!IMPORTANT]",
        "> **Objective Telemetry & Manual Review Notice**:",
        "> This benchmark separates objective execution metrics (availability, execution latency, error counts) ",
        "> from qualitative visual-language grading. No automatic model rankings, winners, or synthetic accuracy ",
        "> percentages are produced without verified ground truth. All semantic evaluations are marked `manual_review_required`.",
        "",
        "---",
        "",
        "## 1. Candidate Model Summary & Objective Telemetry",
        "",
        "| Candidate | Adaptation | Status | Weights Local | Success | Unavailable | Failed | Avg Latency (ms) | Median Latency (ms) | Manual Review |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for m in model_metrics:
        adaptation = "RS-Native" if "qwen" not in m.model_id else "General Multimodal"
        weights_str = "Yes" if m.weights_available else "No (Not Installed)"
        avg_lat = f"{m.avg_inference_time_ms:.1f}" if m.avg_inference_time_ms is not None else "N/A"
        med_lat = f"{m.median_inference_time_ms:.1f}" if m.median_inference_time_ms is not None else "N/A"

        lines.append(
            f"| **{m.model_name}** | {adaptation} | `{m.status}` | {weights_str} | "
            f"{m.successful_inferences}/{m.total_questions} | {m.unavailable_inferences} | {m.failed_inferences} | "
            f"{avg_lat} | {med_lat} | `{m.manual_review_status}` |"
        )

    lines.extend([
        "",
        "## 2. Category-Level Evaluation Breakdown",
        "",
        "| Category | GeoChat-7B | SkyEyeGPT | Qwen2-VL-2B (Baseline) |",
        "| :--- | :--- | :--- | :--- |",
    ])

    # Extract all distinct categories from first model's category_performance
    sample_perf = model_metrics[0].category_performance if model_metrics else {}
    categories = list(sample_perf.keys())

    metrics_map = {m.model_id: m for m in model_metrics}

    for cat in categories:
        cat_title = cat.replace("_", " ").title()
        statuses = []
        for m_id in ["geochat_7b", "skyeyegpt", "qwen2_vl_2b"]:
            m = metrics_map.get(m_id)
            if m and cat in m.category_performance:
                perf = m.category_performance[cat]
                if perf["success"] > 0:
                    statuses.append(f"{perf['success']}/{perf['total']} success")
                elif perf["unavailable"] > 0:
                    statuses.append("unavailable")
                else:
                    statuses.append("failed")
            else:
                statuses.append("N/A")
        lines.append(f"| **{cat_title}** | {statuses[0]} | {statuses[1]} | {statuses[2]} |")

    lines.extend([
        "",
        "## 3. Provenance & Known Limitations",
        "",
    ])

    for m in model_metrics:
        lines.append(f"### {m.model_name}")
        lines.append(f"- **Model Identifier**: `{m.model_id}`")
        lines.append(f"- **Local Weights Available**: `{m.weights_available}`")
        lines.append(f"- **Known Limitations / Notes**:")
        if m.known_limitations:
            for lim in m.known_limitations:
                lines.append(f"  - {lim}")
        else:
            lines.append("  - None recorded during evaluation.")
        lines.append("")

    lines.extend([
        "## 4. Human Review & Hallucination Protocol",
        "",
        "For each qualitative response produced by candidate models, human reviewers must grade:",
        "- **Correctness** (0 = poor/incorrect, 1 = partially useful, 2 = good, 3 = strong)",
        "- **Relevance & Completeness** (0-3 scale)",
        "- **Remote-Sensing Terminology** (0-3 scale)",
        "- **Spatial Correctness & Grounding** (0-3 scale)",
        "- **Hallucination Observed** (true/false: claims objects not present in image)",
        "- **Unsupported Claims** (true/false: confident assertions beyond visible sensor resolution)",
        "- **Uncertainty Handling** (true/false: appropriately states limitations on low resolution)",
        "",
        "> [!NOTE]",
        "> A fluent, syntactically confident answer that misidentifies agricultural fields as runways or ",
        "> reports non-existent water bodies is classified as a hallucination failure.",
        "",
    ])

    return "\n".join(lines)


def generate_json_report(
    model_metrics: List[ModelBenchmarkMetrics],
    detailed_results: Optional[Dict[str, List[EvaluationResult]]] = None,
    image_count: int = 5,
    total_questions: int = 40,
) -> Dict[str, Any]:
    """Generates structured JSON benchmark export."""
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "benchmark_title": "SatQuery Remote-Sensing VLM Bake-Off Evaluation",
        "timestamp_utc": timestamp,
        "total_questions": total_questions,
        "satellite_images_count": image_count,
        "models_evaluated": [m.model_id for m in model_metrics],
        "metrics": [m.to_dict() for m in model_metrics],
        "detailed_results": {
            m_id: [r.to_dict() for r in res_list]
            for m_id, res_list in (detailed_results or {}).items()
        },
    }
