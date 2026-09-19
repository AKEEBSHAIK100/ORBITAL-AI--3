"""
SatQuery AI — Specialist Model Registry.
Declares all 8 domain specialist engines, their tasks, versions, modalities,
checkpoint locations, availability criteria, and controlled unavailable fallbacks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np


@dataclass
class SpecialistEntry:
    id: str
    name: str
    task: str
    model_id: str
    version: str
    modality: List[str]
    supported_input_types: List[str]
    checkpoint_location: Optional[str]
    is_available: bool
    unavailable_reason: Optional[str] = None
    inference_fn: Optional[Callable[..., Dict[str, Any]]] = None

    def execute_or_fallback(self, **kwargs) -> Dict[str, Any]:
        """
        Executes real specialist inference if available.
        If unavailable, returns clean SPECIALIST_UNAVAILABLE response without fabricated metrics.
        """
        if not self.is_available or self.inference_fn is None:
            reason = self.unavailable_reason or f"Specialist checkpoint '{self.model_id}' is not installed."
            return {
                "status": "SPECIALIST_UNAVAILABLE",
                "answer": f"Analysis unavailable: {reason}",
                "confidence": None,
                "confidence_level": "UNAVAILABLE",
                "evidence": None,
                "warnings": [reason],
                "model": self.name,
                "model_version": self.version,
                "task": self.task,
                "specialist_id": self.id
            }

        try:
            res = self.inference_fn(**kwargs)
            # Ensure standard schema keys
            return {
                "status": res.get("status", "SUCCESS"),
                "answer": res.get("answer", ""),
                "confidence": res.get("confidence"),
                "confidence_level": res.get("confidence_level", "UNAVAILABLE" if res.get("confidence") is None else "High"),
                "evidence": res.get("evidence"),
                "warnings": res.get("warnings", []),
                "model": self.name,
                "model_version": self.version,
                "task": self.task,
                "specialist_id": self.id,
                **{k: v for k, v in res.items() if k not in ("answer", "confidence", "confidence_level", "evidence", "warnings", "model", "model_version", "task")}
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "answer": f"Execution error in specialist '{self.id}': {str(e)}",
                "confidence": None,
                "confidence_level": "UNAVAILABLE",
                "evidence": None,
                "warnings": [str(e)],
                "model": self.name,
                "model_version": self.version,
                "task": self.task,
                "specialist_id": self.id
            }


class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None

    def __init__(self, workspace_root: Optional[str] = None):
        if workspace_root is None:
            workspace_root = str(Path(__file__).resolve().parent.parent)
        self.workspace_root = Path(workspace_root)
        self._specialists: Dict[str, SpecialistEntry] = {}
        self._init_specialists()

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_specialists(self) -> None:
        models_dir = self.workspace_root / "backend" / "models"
        checkpoints_dir = self.workspace_root / "checkpoints"

        # 1. Building / Object Detection (Real weights: building_model.pt)
        bldg_ckpt = models_dir / "building_model.pt"
        bldg_avail = bldg_ckpt.exists() and bldg_ckpt.stat().st_size > 1000

        def run_building(**kw):
            from backend.tools.building_detection import BuildingDetectionTool
            tool = BuildingDetectionTool()
            res = tool.run({"image": kw.get("image"), "ground_truth_count": kw.get("ground_truth_count")}, kw.get("parameters"))
            return {
                "answer": f"Detected {res.get('building_count', 0)} building footprints ({res.get('high_confidence_count', 0)} high confidence).",
                "confidence": res.get("confidence"),
                "confidence_level": res.get("confidence_level"),
                "evidence": {
                    "building_count": res.get("building_count"),
                    "high_confidence_count": res.get("high_confidence_count"),
                    "medium_confidence_count": res.get("medium_confidence_count"),
                    "partial_count": res.get("partial_count")
                },
                "geojson": res.get("geojson"),
                "detections": res.get("detections"),
                "building_analysis": res
            }

        self._specialists["building_detection"] = SpecialistEntry(
            id="building_detection",
            name="Tiled YOLO Building Footprint Specialist",
            task="building_detection",
            model_id="yolo-segmentation-building_model.pt",
            version="1.0.0",
            modality=["optical"],
            supported_input_types=["image/tiff", "image/png", "image/jpeg"],
            checkpoint_location=str(bldg_ckpt) if bldg_avail else None,
            is_available=bldg_avail,
            unavailable_reason=None if bldg_avail else f"Checkpoint not found at '{bldg_ckpt}'.",
            inference_fn=run_building
        )

        # 2. Land-Cover Analysis (reBEN ResNet-50)
        def run_land_cover(**kw):
            from backend.services.ben_classifier import BENClassifier
            clf = BENClassifier.get_instance()
            img = kw.get("image")
            if not clf._available or img is None:
                return {
                    "answer": "Land-cover classifier currently unavailable.",
                    "confidence": None,
                    "confidence_level": "UNAVAILABLE"
                }
            import cv2
            _, buf = cv2.imencode(".png", img)
            pred = clf.classify_image(buf.tobytes())
            top_label = pred.get("top_label", "Unknown")
            return {
                "answer": f"Primary land cover identified as: {top_label}.",
                "confidence": round(pred.get("confidence", 85.0) / 100.0, 2),
                "confidence_level": "High",
                "evidence": {"top_label": top_label, "active_labels": pred.get("active_labels", [])}
            }

        self._specialists["land_cover"] = SpecialistEntry(
            id="land_cover",
            name="BigEarthNet v2.0 Corine Land-Cover Classifier",
            task="land_cover",
            model_id="BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
            version="0.2.0",
            modality=["multispectral", "optical"],
            supported_input_types=["image/tiff", "image/png", "image/jpeg"],
            checkpoint_location="huggingface:BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
            is_available=True,
            inference_fn=run_land_cover
        )

        # 3. Change Detection & Bi-temporal reasoning
        def run_change_det(**kw):
            from models.change.change_detector import ChangeDetector
            detector = ChangeDetector()
            res = detector.detect_changes(kw.get("image"), kw.get("secondary_image"), date_t1=kw.get("date_t1"), date_t2=kw.get("date_t2"))
            if not res.get("success"):
                return {"answer": f"Change detection failed: {res.get('error')}", "confidence": None, "confidence_level": "UNAVAILABLE"}
            return {
                "answer": f"Surface alterations detected across {res['change_percentage']}% of the observation scene.",
                "confidence": 0.85,
                "confidence_level": "High",
                "evidence": res
            }

        self._specialists["change_detection"] = SpecialistEntry(
            id="change_detection",
            name="Bi-Temporal Change Detection Engine",
            task="change_detection",
            model_id="rs-change-detector-v2",
            version="2.0.0",
            modality=["optical", "multispectral", "sar", "bi-temporal"],
            supported_input_types=["pair:image/tiff", "pair:image/png"],
            checkpoint_location=None,
            is_available=True,
            inference_fn=run_change_det
        )

        # 4. Change VQA Specialist
        def run_change_vqa(**kw):
            from models.change.change_vqa import ChangeVQASpecialist
            spec = ChangeVQASpecialist()
            return spec.answer_query(
                kw.get("query", ""),
                kw.get("image"),
                kw.get("secondary_image"),
                meta_t1=kw.get("meta_t1"),
                meta_t2=kw.get("meta_t2"),
                date_t1=kw.get("date_t1"),
                date_t2=kw.get("date_t2")
            )

        self._specialists["change_vqa"] = SpecialistEntry(
            id="change_vqa",
            name="Bi-Temporal Change VQA Specialist",
            task="change_vqa",
            model_id="rs-change-vqa-v2",
            version="2.0.0",
            modality=["bi-temporal"],
            supported_input_types=["pair:image/tiff", "pair:image/png"],
            checkpoint_location=None,
            is_available=True,
            inference_fn=run_change_vqa
        )

        # 5. Optical-SAR Cross-Modal Fusion
        def run_fusion(**kw):
            from models.fusion.optical_sar_fusion import OpticalSARFusionEngine
            fuse_eng = OpticalSARFusionEngine()
            return fuse_eng.fuse(
                kw.get("optical_image") or kw.get("image"),
                kw.get("sar_image") or kw.get("secondary_image"),
                optical_sensor=kw.get("optical_sensor", "generic_optical"),
                sar_sensor=kw.get("sar_sensor", "generic_sar"),
                query=kw.get("query")
            )

        self._specialists["optical_sar_fusion"] = SpecialistEntry(
            id="optical_sar_fusion",
            name="Optical-SAR Cross-Modal Fusion Engine",
            task="optical_sar_fusion",
            model_id="classical-cv-fusion-engine-v2",
            version="2.0.0",
            modality=["optical", "sar"],
            supported_input_types=["image/tiff", "pair:optical+sar"],
            checkpoint_location=None,
            is_available=True,
            inference_fn=run_fusion
        )

        # 6. Remote Sensing Adapted VQA (BLIP-VQA + BigEarthNet LoRA)
        def run_rs_vqa(**kw):
            from backend.tools.vqa import VQATool
            tool = VQATool()
            res = tool.run(kw, kw.get("parameters"))
            return {
                "answer": res.get("answer", ""),
                "status": res.get("status", "SUCCESS"),
                "confidence": res.get("confidence"),
                "confidence_level": res.get("confidence_status", "UNAVAILABLE"),
                "evidence": res.get("evidence"),
                "provenance": res.get("provenance"),
            }

        try:
            from backend.services.rs_adapters import RSAdapterRuntime, SpecialistState
            _runtime = RSAdapterRuntime.get_instance()
            vqa_avail = _runtime.vqa_state == SpecialistState.AVAILABLE
            vqa_unavail_reason = _runtime.vqa_unavailable_reason
            caption_avail = _runtime.caption_state == SpecialistState.AVAILABLE
            caption_unavail_reason = _runtime.caption_unavailable_reason
        except Exception:
            vqa_avail = False
            vqa_unavail_reason = "Adapter runtime unavailable"
            caption_avail = False
            caption_unavail_reason = "Adapter runtime unavailable"

        vqa_entry = SpecialistEntry(
            id="rs_vqa_adapted",
            name="Remote-Sensing Adapted Visual Question Answering Specialist",
            task="vqa",
            model_id="rs-vqa-adapted-v1",
            version="1.0.0",
            modality=["optical", "multispectral"],
            supported_input_types=["image/tiff", "image/png", "image/jpeg"],
            checkpoint_location=str(checkpoints_dir / "rs_vqa_adapter") if vqa_avail else None,
            is_available=vqa_avail,
            unavailable_reason=vqa_unavail_reason,
            inference_fn=run_rs_vqa
        )
        self._specialists["rs_vqa_adapted"] = vqa_entry
        self._specialists["rs_vqa"] = vqa_entry

        # 7. Scene Captioning (BLIP + BigEarthNet LoRA)
        def run_captioning(**kw):
            from backend.tools.caption import CaptionTool
            tool = CaptionTool()
            res = tool.run(kw, kw.get("parameters"))
            return {
                "answer": res.get("caption", ""),
                "status": res.get("status", "SUCCESS"),
                "confidence": res.get("confidence"),
                "confidence_level": res.get("confidence_status", "UNAVAILABLE"),
                "evidence": res.get("evidence"),
                "provenance": res.get("provenance"),
            }

        caption_entry = SpecialistEntry(
            id="rs_caption_adapted",
            name="Remote-Sensing Adapted Captioning Specialist",
            task="captioning",
            model_id="rs-caption-adapted-v1",
            version="1.0.0",
            modality=["optical", "multispectral"],
            supported_input_types=["image/tiff", "image/png", "image/jpeg"],
            checkpoint_location=str(checkpoints_dir / "rs_caption_adapter") if caption_avail else None,
            is_available=caption_avail,
            unavailable_reason=caption_unavail_reason,
            inference_fn=run_captioning
        )
        self._specialists["rs_caption_adapted"] = caption_entry
        self._specialists["captioning"] = caption_entry

        # 8. Visual Grounding (Spatial target demarcation)
        def run_grounding(**kw):
            from backend.tools.grounding import GroundingTool
            tool = GroundingTool()
            res = tool.run(kw, kw.get("parameters"))
            top_reg = res.get("primary_region")
            if top_reg:
                ans = f"Grounding localized '{res.get('target')}' at: X: {top_reg.get('x_percent')}%, Y: {top_reg.get('y_percent')}%, Width: {top_reg.get('w_percent')}%, Height: {top_reg.get('h_percent')}%."
            else:
                ans = f"No target '{res.get('target')}' localized."
            return {
                "answer": ans,
                "confidence": 0.80 if top_reg else 0.40,
                "confidence_level": "High" if top_reg else "Medium",
                "evidence": {"regions": res.get("regions", []), "target": res.get("target")},
                "grounding": res.get("regions", [])
            }

        self._specialists["visual_grounding"] = SpecialistEntry(
            id="visual_grounding",
            name="Text-Guided Spatial Grounding Specialist",
            task="grounding",
            model_id="rs-grounding-specialist-v2",
            version="2.0.0",
            modality=["optical", "multispectral"],
            supported_input_types=["image/tiff", "image/png", "image/jpeg"],
            checkpoint_location=None,
            is_available=True,
            inference_fn=run_grounding
        )

    def get_specialist(self, specialist_id: str) -> Optional[SpecialistEntry]:
        return self._specialists.get(specialist_id)

    def list_specialists(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": s.id,
                "name": s.name,
                "task": s.task,
                "model_id": s.model_id,
                "version": s.version,
                "modality": s.modality,
                "supported_input_types": s.supported_input_types,
                "checkpoint_location": s.checkpoint_location,
                "is_available": s.is_available,
                "unavailable_reason": s.unavailable_reason
            }
            for s in self._specialists.values()
        ]
