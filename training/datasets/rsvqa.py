"""
RSVQA Dataset Adapter for SatQuery AI.
Supports Sylvain Lobry et al. RSVQA protocols (LR: Low Resolution, HR: High Resolution).
Question categories:
1. Presence questions ("Is there a river?")
2. Comparison questions ("Are there more fields than urban areas?")
3. Counting questions ("How many residential buildings are there?")
4. Area / land-cover questions ("Which area has more vegetation?")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class RSVQAAdapter:
    def __init__(self, data_dir: Optional[str] = None, variant: str = "lr"):
        self.variant = variant.lower()
        if data_dir is None:
            folder = "rsvqa_hr" if "hr" in self.variant else "rsvqa_lr"
            data_dir = str(Path(__file__).resolve().parent.parent.parent / "data" / folder)
        self.data_dir = Path(data_dir)

    def is_available(self) -> bool:
        if not self.data_dir.exists():
            return False
        files = [f for f in self.data_dir.rglob("*.json*") if f.name != ".gitkeep"]
        return len(files) > 0

    def load_samples(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        Loads RSVQA questions and answers.
        Standard format includes:
        - id: question ID
        - img_id: corresponding remote-sensing image
        - question: natural language query
        - answer: ground truth label or numeric count
        - type: 'presence' | 'comparison' | 'count' | 'area'
        """
        if not self.is_available():
            return []

        # Look for standard questions.json and answers.json or unified test.json
        q_file = self.data_dir / f"questions_{split}.json"
        if not q_file.exists():
            candidates = list(self.data_dir.glob("*.json"))
            if not candidates:
                return []
            q_file = candidates[0]

        try:
            with open(q_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return data.get("questions", data.get("annotations", []))
                return []
        except Exception:
            return []
