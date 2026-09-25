"""
Lightweight dataset registry.

The registry reports declared datasets and whether their local data directory
exists. It does not download data or claim benchmark availability.
"""
from __future__ import annotations

from pathlib import Path


class DatasetRegistry:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.root = Path(__file__).resolve().parents[2] / "data"

    def list_datasets(self):
        definitions = {
            "bigearthnet_txt": "data/bigearthnet_txt",
            "bigearthnet_v2": "data/bigearthnet_v2",
            "rsvqa_hr": "data/rsvqa_hr",
            "rsvqa_lr": "data/rsvqa_lr",
            "cdvqa": "data/cdvqa",
            "levir_cc": "data/levir_cc",
            "levir_mci": "data/levir_mci",
        }
        return [
            {
                "id": dataset_id,
                "path": rel_path,
                "status": "available" if (self.root / Path(rel_path).relative_to("data")).exists() else "not_downloaded",
                "download_required": not (self.root / Path(rel_path).relative_to("data")).exists(),
            }
            for dataset_id, rel_path in definitions.items()
        ]
