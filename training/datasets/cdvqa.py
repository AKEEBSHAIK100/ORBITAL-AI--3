"""
CDVQA (Change Detection Visual Question Answering) Dataset Adapter.
Supports bi-temporal remote-sensing image pairs (T1, T2) and temporal change queries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class CDVQAAdapter:
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = str(Path(__file__).resolve().parent.parent.parent / "data" / "cdvqa")
        self.data_dir = Path(data_dir)

    def is_available(self) -> bool:
        if not self.data_dir.exists():
            return False
        files = [f for f in self.data_dir.rglob("*.json*") if f.name != ".gitkeep"]
        return len(files) > 0

    def load_samples(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        Loads CDVQA evaluation samples.
        Sample schema:
        - id: query identifier
        - img_t1: path to pre-change image
        - img_t2: path to post-change image
        - question: natural language change inquiry
        - answer: reference ground truth answer
        - change_category: 'urban_expansion' | 'vegetation_loss' | 'water_dynamics' | 'unchanged'
        """
        if not self.is_available():
            return []

        candidates = list(self.data_dir.glob(f"*{split}*.json*")) or list(self.data_dir.glob("*.json"))
        if not candidates:
            return []

        try:
            with open(candidates[0], "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else data.get("questions", [])
        except Exception:
            return []
