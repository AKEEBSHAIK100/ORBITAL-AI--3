"""
SatQuery AI — Benchmark Results Storage & Registry.
Manages evaluation runs in evaluation/results/.
Enforces standard schema and allowed statuses:
- NOT_RUN
- DATASET_UNAVAILABLE
- RUNNING
- COMPLETED
- FAILED
Never stores or displays COMPLETED unless evaluation actually executed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BenchmarkStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    DATASET_UNAVAILABLE = "DATASET_UNAVAILABLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class BenchmarkResult(BaseModel):
    dataset: str
    dataset_version: str = "1.0.0"
    split: str = "test"
    task: str
    model: str
    model_version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"))
    metrics: Optional[Dict[str, Any]] = None
    sample_count: int = 0
    configuration: Optional[Dict[str, Any]] = None
    status: BenchmarkStatus = BenchmarkStatus.NOT_RUN


class BenchmarkStorage:
    def __init__(self, results_dir: Optional[str] = None):
        if results_dir is None:
            results_dir = str(Path(__file__).resolve().parent / "results")
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, result: BenchmarkResult) -> str:
        """Saves evaluation run to disk."""
        safe_name = f"{result.dataset.lower()}_{result.task.lower()}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        target = self.results_dir / safe_name
        with open(target, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        return str(target)

    def list_all_runs(self) -> List[Dict[str, Any]]:
        """Lists all evaluation run files from disk."""
        runs = []
        for file in sorted(self.results_dir.glob("*.json"), reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    runs.append(json.load(f))
            except Exception:
                continue
        return runs

    def get_latest_run_for_benchmark(self, dataset_name: str, task: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Returns the most recent completed run for a benchmark, or a placeholder NOT_RUN."""
        all_runs = self.list_all_runs()
        for run in all_runs:
            if run.get("dataset", "").lower() == dataset_name.lower():
                if task is None or run.get("task", "").lower() == task.lower():
                    return run
        return None
