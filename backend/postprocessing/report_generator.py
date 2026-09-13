import json
from datetime import datetime
from typing import Any, Dict, List, Optional

def generate_downloadable_report(
    query: str,
    task_type: str,
    tools_used: List[str],
    parameters: Dict[str, Any],
    answer: str,
    confidence: float,
    confidence_level: str,
    evidence_summary: Dict[str, Any],
    detection_statistics: Optional[Dict[str, Any]] = None,
    input_metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Implements Section 15: Downloadable Analysis Report Generator.
    Produces comprehensive, auditable Markdown report containing all parameters and evidence.
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    metadata = input_metadata or {}
    dims = metadata.get("dimensions", {})
    dim_str = f"{dims.get('width', '—')} x {dims.get('height', '—')} px"

    md = []
    md.append("# SatQuery AI — Remote Sensing Mission Analysis Report")
    md.append(f"**Generated:** {timestamp} · **Agent Version:** SatQuery-Agent-v3.0")
    md.append("---")
    md.append("## 1. Mission Query & Objective")
    md.append(f"- **User Query:** \"{query}\"")
    md.append(f"- **Determined Task:** `{task_type}`")
    md.append(f"- **Input Modality:** `{metadata.get('modality', 'optical')}`")
    md.append(f"- **Input Dimensions:** {dim_str} ({metadata.get('format', 'Raster')})")
    md.append("")
    md.append("## 2. Specialist Model & Tool Execution")
    md.append(f"- **Tools Invoked:** {', '.join(f'`{t}`' for t in tools_used)}")
    md.append("- **Permitted Parameters Applied:**")
    for k, v in parameters.items():
        md.append(f"  - `{k}`: `{v}`")
    md.append("")
    md.append("## 3. Executive Interpretation & Answer")
    md.append(f"> {answer}")
    md.append(f"- **Aggregate Confidence:** **{confidence_level}** ({confidence*100:.1f}%)")
    md.append("")

    if detection_statistics:
        md.append("## 4. Building Footprint Audit Statistics")
        md.append(f"- **Total Structures Detected:** {detection_statistics.get('building_count', 0)}")
        md.append(f"- **High-Confidence Roofs:** {detection_statistics.get('high_confidence_count', 0)}")
        md.append(f"- **Medium-Confidence Roofs:** {detection_statistics.get('medium_confidence_count', 0)}")
        md.append(f"- **Low-Confidence Roofs:** {detection_statistics.get('low_confidence_count', 0)}")
        md.append(f"- **Tiles Processed:** {detection_statistics.get('tiles_processed', 0)} (512px overlapping sliding windows)")
        md.append(f"- **Validation Status:** {detection_statistics.get('validation_status', 'Deep-learning verified')}")
        md.append("")

    md.append("## 5. Physical & Spectral Evidence")
    for ek, ev in evidence_summary.items():
        if isinstance(ev, dict):
            md.append(f"### {ek.replace('_', ' ').title()}")
            for subk, subv in ev.items():
                md.append(f"- **{subk.replace('_', ' ').title()}:** {subv}")
        else:
            md.append(f"- **{ek.replace('_', ' ').title()}:** {ev}")
    md.append("")
    md.append("---")
    md.append("*Report certified by SatQuery AI Agentic Remote-Sensing Platform.*")

    return "\n".join(md)
