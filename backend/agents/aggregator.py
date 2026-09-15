import time
from typing import Any, Dict, List, Optional
from ..schemas.analysis import ObservableTrace, TraceStep

def build_observable_trace(
    task_type: str,
    steps: List[Dict[str, Any]],
    total_duration_ms: float,
    validation: Dict[str, Any],
    primary_tool: Dict[str, Any]
) -> ObservableTrace:
    """
    Constructs an auditable execution trace adhering to Section 11:
    Only observable execution details are exposed (task, tools, parameters, outputs, confidence).
    Never exposes hidden chain-of-thought or internal reasoning text.
    """
    trace_steps = [
        TraceStep(
            step=s.get("step", idx + 1),
            tool=s.get("tool", "specialist"),
            description=s.get("description", ""),
            input_summary=s.get("input_summary", ""),
            output_summary=s.get("output_summary", ""),
            duration_ms=s.get("duration_ms", 0.0),
            status=s.get("status", "success"),
            success=s.get("success", s.get("status") == "success"),
            confidence_source=s.get("confidence_source", "heuristic"),
            parameters=s.get("parameters")
        )
        for idx, s in enumerate(steps)
    ]

    tools_invoked = list(dict.fromkeys(s.tool for s in trace_steps))

    return ObservableTrace(
        agent_version="SatQuery-Agent-v3.0",
        task_type=task_type,
        tools_invoked=tools_invoked,
        steps=trace_steps,
        total_duration_ms=round(total_duration_ms, 2),
        input_validation=validation,
        model_registry_entry={
            "model_id": primary_tool.get("model_id", "satquery-v3"),
            "adapter": primary_tool.get("adapter", "RS Specialist Engine"),
            "domain_adaptation": primary_tool.get("domain_adaptation", "Calibrated on remote sensing benchmarks"),
            "permitted_parameters": primary_tool.get("permitted_parameters", {})
        }
    )
