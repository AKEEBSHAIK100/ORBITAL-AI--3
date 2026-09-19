"""
BigEarthNet Dataset Parser & Remote-Sensing Adaptation Adapter.
Supports BigEarthNet v2.0 (reBEN) multi-label Corine Land Cover annotations.
Extracts genuine ground truth labels and builds grounded instruction-tuning examples.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


# Standard 19-class Corine Land Cover taxonomy (BigEarthNet-19)
BIGEARTHNET_19_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]


class BigEarthNetAdapter:
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = str(Path(__file__).resolve().parent.parent.parent / "data" / "bigearthnet_v2")
        self.data_dir = Path(data_dir)

    def is_available(self) -> bool:
        if not self.data_dir.exists():
            return False
        # Check if there are json, jsonl, or txt annotations
        valid_extensions = {".json", ".jsonl", ".txt", ".parquet"}
        files = [f for f in self.data_dir.rglob("*") if f.suffix in valid_extensions and f.name != ".gitkeep"]
        return len(files) > 0

    def parse_annotations(self, annotation_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parses actual BigEarthNet annotations from json, jsonl, or txt without fabricating labels.
        """
        if not self.is_available() and annotation_file is None:
            return []

        file_path = Path(annotation_file) if annotation_file else None
        if file_path is None or not file_path.exists():
            candidates = list(self.data_dir.glob("*.jsonl")) + list(self.data_dir.glob("*.json")) + list(self.data_dir.glob("*.txt"))
            if not candidates:
                return []
            file_path = candidates[0]

        records: List[Dict[str, Any]] = []

        if file_path.suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            records.append(self._normalize_record(item))
                        except Exception:
                            continue

        elif file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        records.append(self._normalize_record(item))
                elif isinstance(data, dict):
                    # Could be keyed by patch_name
                    for patch_id, item in data.items():
                        if isinstance(item, dict):
                            item["patch_id"] = patch_id
                            records.append(self._normalize_record(item))

        elif file_path.suffix == ".txt":
            # Tab or comma separated: patch_name \t label1,label2...
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t") if "\t" in line else line.split(",")
                    patch_id = parts[0].strip()
                    labels = [p.strip() for p in parts[1:] if p.strip() in BIGEARTHNET_19_CLASSES]
                    records.append({
                        "patch_id": patch_id,
                        "image_path": str(self.data_dir / f"{patch_id}.tif"),
                        "labels": labels
                    })

        return records

    def _normalize_record(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes heterogeneous BigEarthNet record schemas into a unified format."""
        patch_id = item.get("patch_id") or item.get("name") or item.get("id") or "unknown_patch"
        image_path = item.get("image_path") or item.get("path") or str(self.data_dir / f"{patch_id}.tif")

        labels = item.get("labels") or item.get("classes") or item.get("labels_19") or []
        if isinstance(labels, str):
            labels = [l.strip() for l in labels.split(",") if l.strip()]

        # Filter strictly to valid Corine labels
        valid_labels = [l for l in labels if l in BIGEARTHNET_19_CLASSES]

        return {
            "patch_id": patch_id,
            "image_path": image_path,
            "labels": valid_labels,
            "metadata": {k: v for k, v in item.items() if k not in ("labels", "classes", "labels_19", "image_path")}
        }

    def generate_instruction_examples(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates remote-sensing adaptation instruction-tuning examples.
        Instruction types:
        1. land-cover identification
        2. land-cover presence
        3. multi-label land-cover description
        4. scene description
        """
        examples: List[Dict[str, Any]] = []

        for r in records:
            labels = r.get("labels", [])
            if not labels:
                continue

            img_ref = r.get("image_path", f"{r['patch_id']}.tif")
            labels_str = ", ".join(labels)

            # 1. Multi-label scene description
            examples.append({
                "image": img_ref,
                "task": "land_cover_description",
                "instruction": "Describe the dominant land-cover characteristics of this remote-sensing image.",
                "response": f"The dominant land-cover classes identified in this observation scene are: {labels_str}."
            })

            # 2. Structured land-cover identification
            examples.append({
                "image": img_ref,
                "task": "land_cover_identification",
                "instruction": "Identify all Corine Land Cover categories present in this satellite observation.",
                "response": f"Identified categories: {labels_str}."
            })

            # 3. Presence verification for the first label
            first_label = labels[0]
            examples.append({
                "image": img_ref,
                "task": "land_cover_presence",
                "instruction": f"Is '{first_label}' present in this satellite patch?",
                "response": f"Yes, '{first_label}' is present in this remote-sensing scene based on spectral and spatial signatures."
            })

        return examples
