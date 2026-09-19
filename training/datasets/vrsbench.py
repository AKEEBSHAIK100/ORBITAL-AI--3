"""
VRSBench Dataset Parser for SatQuery AI.
Supports VRSBench 3 core tasks:
1. Scene Captioning (multi-attribute remote-sensing captions)
2. Visual Grounding (text-guided [xmin, ymin, xmax, ymax] or normalized bounding boxes)
3. Visual Question Answering (RS questions and reference answers)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class VRSBenchAdapter:
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = str(Path(__file__).resolve().parent.parent.parent / "data" / "vrsbench")
        self.data_dir = Path(data_dir)

    def is_available(self) -> bool:
        if not self.data_dir.exists():
            return False
        files = [f for f in self.data_dir.rglob("*.json*") if f.name != ".gitkeep"]
        return len(files) > 0

    def load_caption_samples(self, split: str = "test") -> List[Dict[str, Any]]:
        """Loads caption references. Returns empty list if dataset is not downloaded."""
        if not self.is_available():
            return []
        cap_file = self.data_dir / f"caption_{split}.json"
        if not cap_file.exists():
            candidates = list(self.data_dir.glob("*caption*.json*"))
            if not candidates:
                return []
            cap_file = candidates[0]

        try:
            with open(cap_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else data.get("annotations", [])
        except Exception:
            return []

    def load_grounding_samples(self, split: str = "test") -> List[Dict[str, Any]]:
        """Loads visual grounding target queries and ground truth bounding boxes."""
        if not self.is_available():
            return []
        grd_file = self.data_dir / f"grounding_{split}.json"
        if not grd_file.exists():
            candidates = list(self.data_dir.glob("*grounding*.json*"))
            if not candidates:
                return []
            grd_file = candidates[0]

        try:
            with open(grd_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else data.get("annotations", [])
        except Exception:
            return []

    def load_vqa_samples(self, split: str = "test") -> List[Dict[str, Any]]:
        """Loads VQA queries, images, and reference answers."""
        if not self.is_available():
            return []
        vqa_file = self.data_dir / f"vqa_{split}.json"
        if not vqa_file.exists():
            candidates = list(self.data_dir.glob("*vqa*.json*"))
            if not candidates:
                return []
            vqa_file = candidates[0]

        try:
            with open(vqa_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else data.get("annotations", [])
        except Exception:
            return []
