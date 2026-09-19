"""
SatQuery AI — Dataset Registry System
Tracks remote-sensing datasets, their availability, local paths, modalities, and task splits.
Ensures zero runtime blockage when datasets are NOT_DOWNLOADED.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DatasetStatus(str, Enum):
    NOT_DOWNLOADED = "NOT_DOWNLOADED"
    AVAILABLE = "AVAILABLE"
    INVALID = "INVALID"
    PROCESSING = "PROCESSING"
    READY = "READY"


class DatasetEntry(BaseModel):
    id: str
    name: str
    version: str
    local_path: str
    modality: str  # optical, sar, multispectral, bi-temporal, optical-sar
    task: str      # land_cover, vqa, captioning, grounding, change_detection
    split: str     # train, val, test
    annotation_format: str  # json, jsonl, txt, geojson, csv
    download_url: str
    license_attribution: str
    status: DatasetStatus = DatasetStatus.NOT_DOWNLOADED
    sample_count: Optional[int] = None
    notes: Optional[str] = None

    def refresh_status(self) -> DatasetStatus:
        """Inspects disk to determine actual local status without downloading."""
        target = Path(self.local_path)
        if not target.exists():
            self.status = DatasetStatus.NOT_DOWNLOADED
            return self.status

        # If it's a directory, check if it contains actual data files (excluding .gitkeep)
        if target.is_dir():
            entries = [p for p in target.iterdir() if p.name != ".gitkeep"]
            if not entries:
                self.status = DatasetStatus.NOT_DOWNLOADED
            else:
                self.status = DatasetStatus.AVAILABLE
        elif target.is_file():
            if target.stat().st_size > 0:
                self.status = DatasetStatus.AVAILABLE
            else:
                self.status = DatasetStatus.INVALID
        return self.status


class DatasetRegistry:
    _instance: Optional["DatasetRegistry"] = None

    def __init__(self, workspace_root: Optional[str] = None):
        if workspace_root is None:
            # Anchor to repository root
            workspace_root = str(Path(__file__).resolve().parent.parent.parent)
        self.workspace_root = Path(workspace_root)
        self._datasets: Dict[str, DatasetEntry] = {}
        self._register_known_datasets()

    @classmethod
    def get_instance(cls) -> "DatasetRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _register_known_datasets(self) -> None:
        data_dir = self.workspace_root / "data"

        known = [
            DatasetEntry(
                id="bigearthnet_v2",
                name="BigEarthNet-S2 v2.0 (reBEN)",
                version="2.0.0",
                local_path=str(data_dir / "bigearthnet_v2"),
                modality="multispectral",
                task="land_cover",
                split="train",
                annotation_format="jsonl",
                download_url="https://bigearth.net/",
                license_attribution="Community Data License Agreement – Permissive (CDLA-Permissive-1.0)",
                notes="Refined BigEarthNet dataset for remote sensing domain adaptation (Clasen et al., IGARSS 2025)."
            ),
            DatasetEntry(
                id="vrsbench",
                name="VRSBench Remote-Sensing Benchmark",
                version="1.0.0",
                local_path=str(data_dir / "vrsbench"),
                modality="optical",
                task="captioning_grounding_vqa",
                split="test",
                annotation_format="json",
                download_url="https://github.com/IRVL-group/VRSBench",
                license_attribution="Creative Commons Attribution 4.0 International (CC BY 4.0)",
                notes="Multi-task RS benchmark covering captioning, visual grounding, and VQA."
            ),
            DatasetEntry(
                id="vrsbench_sar",
                name="VRSBench SAR Benchmark Subset",
                version="1.0.0",
                local_path=str(data_dir / "vrsbench_sar"),
                modality="sar",
                task="sar_reasoning",
                split="test",
                annotation_format="json",
                download_url="https://github.com/IRVL-group/VRSBench",
                license_attribution="CC BY 4.0",
                notes="SAR subset of VRSBench for radar interpretation benchmarks."
            ),
            DatasetEntry(
                id="rsvqa_lr",
                name="RSVQA Low-Resolution (LR)",
                version="1.0.0",
                local_path=str(data_dir / "rsvqa_lr"),
                modality="optical",
                task="vqa",
                split="test",
                annotation_format="json",
                download_url="https://rsvqa.sylvainlobry.com/",
                license_attribution="Research Only / Open Academic License (Lobry et al., TGRS 2020)",
                notes="Sentinel-2 low resolution VQA subset (presence, comparison, counting)."
            ),
            DatasetEntry(
                id="rsvqa_hr",
                name="RSVQA High-Resolution (HR)",
                version="1.0.0",
                local_path=str(data_dir / "rsvqa_hr"),
                modality="optical",
                task="vqa",
                split="test",
                annotation_format="json",
                download_url="https://rsvqa.sylvainlobry.com/",
                license_attribution="Research Only / Open Academic License (Lobry et al., TGRS 2020)",
                notes="Aerial high-resolution VQA benchmark (urban structure counting & grounding)."
            ),
            DatasetEntry(
                id="cdvqa",
                name="CDVQA Bi-Temporal Change VQA",
                version="1.0.0",
                local_path=str(data_dir / "cdvqa"),
                modality="bi-temporal",
                task="change_detection",
                split="test",
                annotation_format="json",
                download_url="https://github.com/ywh9281/CDVQA",
                license_attribution="Academic Research License (Yuan et al.)",
                notes="Co-registered bi-temporal pairs with natural language change questions."
            ),
        ]

        for d in known:
            d.refresh_status()
            self._datasets[d.id] = d

    def register(self, entry: DatasetEntry) -> None:
        entry.refresh_status()
        self._datasets[entry.id] = entry

    def get(self, dataset_id: str) -> Optional[DatasetEntry]:
        entry = self._datasets.get(dataset_id)
        if entry:
            entry.refresh_status()
        return entry

    def list_datasets(self) -> List[Dict[str, Any]]:
        results = []
        for entry in self._datasets.values():
            entry.refresh_status()
            results.append(entry.model_dump())
        return results

    def is_available(self, dataset_id: str) -> bool:
        entry = self.get(dataset_id)
        return bool(entry and entry.status in (DatasetStatus.AVAILABLE, DatasetStatus.READY))
