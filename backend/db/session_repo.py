"""
backend/db/session_repo.py — Write analysis results from Python backend to shared DB (Phase 4)

All functions are non-fatal no-ops when DB is unavailable.
Used by FastAPI routers to persist Python-side analysis results.
"""
from __future__ import annotations
import json
import logging
import uuid
from typing import Any

from .connection import db_execute

logger = logging.getLogger(__name__)


async def write_analysis_result(
    run_id: str,
    task_type: str,
    result: dict[str, Any],
    confidence: str | None = None,
    confidence_pct: float | None = None,
) -> bool:
    """
    Writes an analysis result to the shared analysis_results table.
    Returns True on success, False if DB unavailable.
    """
    result_id = str(uuid.uuid4())
    return await db_execute(
        """
        INSERT INTO analysis_results (id, run_id, task_type, result_json, confidence, confidence_pct)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT DO NOTHING
        """,
        result_id,
        run_id,
        task_type,
        json.dumps(result),
        confidence,
        confidence_pct,
    )


async def write_trace_step(
    run_id: str,
    step_index: int,
    tool: str,
    description: str | None = None,
    input_summary: str | None = None,
    output_summary: str | None = None,
    duration_ms: float | None = None,
    status: str = "success",
    parameters: dict[str, Any] | None = None,
) -> bool:
    """
    Writes an agent trace step from the Python backend.
    """
    return await db_execute(
        """
        INSERT INTO agent_trace_steps
          (run_id, step_index, tool, description, input_summary, output_summary, duration_ms, status, parameters)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        run_id,
        step_index,
        tool,
        description,
        input_summary,
        output_summary,
        duration_ms,
        status,
        json.dumps(parameters) if parameters else None,
    )


async def mark_run_complete(
    run_id: str,
    duration_ms: int,
    status: str = "success",
    error_tag: str | None = None,
) -> bool:
    """
    Marks an analysis_run as completed.
    """
    return await db_execute(
        """
        UPDATE analysis_runs
        SET status = $2, completed_at = NOW(), duration_ms = $3, error_tag = $4
        WHERE id = $1
        """,
        run_id,
        status,
        duration_ms,
        error_tag,
    )
