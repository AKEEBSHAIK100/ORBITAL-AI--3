"""
Minimal persistent benchmark facade.

Benchmark results are intentionally empty until a benchmark run has actually
completed. This prevents the API from presenting dry-run or unavailable models
as measured results.
"""
from __future__ import annotations


class BenchmarkStorage:
    def list_all_runs(self):
        return []
