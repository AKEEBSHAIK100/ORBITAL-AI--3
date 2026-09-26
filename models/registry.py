"""
ORBITAL-AI — Specialist Model Registry.
Declares all 8 domain specialist engines, their tasks, versions, modalities,
checkpoint locations, runtime availability criteria, and controlled unavailable states.
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
    base_model: Optional[str] = None
    model_type: Optional[str] = None
    is_remote_sensing_adapted: bool = False
    is_remote_sensing_expert: bool = False
    adaptation_scope: Optional[str] = None
    benchmark_accuracy_claim: Optional[str] = None
    confidence: Optional[float] = None
    confidence_status: str = "calibrated"

    @property
    def model(self) -> str:
        return self.model_id

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

        ben_runtime_available = False
        try:
            from backend.services.ben_classifier import BENClassifier
            ben_runtime_available = bool(BENClassifier.get_instance()._available)
        except Exception:
            ben_runtime_available = False

        self._specialists["land_cover"] = SpecialistEntry(
            id="land_cover",
            name="BigEarthNet v2.0 Corine Land-Cover Classifier",
            task="land_cover",
            model_id="BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
            version="0.2.0",
            modality=["multispectral", "optical"],
            supported_input_types=["image/tiff", "image/png", "image/jpeg"],
            checkpoint_location="huggingface:BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0" if ben_runtime_available else None,
            is_available=ben_runtime_available,
            unavailable_reason=None if ben_runtime_available else "BigEarthNet runtime classifier is not loaded/verified in this process.",
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
                "confidence": None,
                "confidence_level": "UNAVAILABLE",
                "evidence": res
            }

        self._specialists["change_detection"] = SpecialistEntry(
            id="change_detection",
            name="Bi-Temporal Change Detection Engine",
            task="change_detection",
            model_id="classical-cv-change-detector-v2",
            version="2.0.0-baseline",
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
            name="Classical Bi-Temporal Change VQA Baseline",
            task="change_vqa",
            model_id="classical-cv-change-vqa-baseline-v2",
            version="2.0.0-baseline",
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
            name="Classical Optical-SAR Cross-Modal Baseline",
            task="optical_sar_fusion",
            model_id="classical-cv-fusion-engine-v2",
            version="2.0.0-baseline",
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
            checkpoint_location="backend/models/adapters/blip_vqa_rs_lora" if vqa_avail else None,
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
            checkpoint_location="backend/models/adapters/blip_rs_lora" if caption_avail else None,
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
                "confidence": None,
                "confidence_level": "UNAVAILABLE",
                "evidence": {"regions": res.get("regions", []), "target": res.get("target"), "targets": res.get("targets", [])},
                "grounding": res.get("regions", []),
                "targets": res.get("targets", [])
            }

        self._specialists["visual_grounding"] = SpecialistEntry(
            id="visual_grounding",
            name="Text-Guided Spatial Grounding Specialist",
            task="grounding",
            model_id="classical-cv-grounding-baseline-v2",
            version="2.0.0-baseline",
            modality=["optical", "multispectral"],
            supported_input_types=["image/tiff", "image/png", "image/jpeg"],
            checkpoint_location=None,
            is_available=True,
            inference_fn=run_grounding
        )

        # 9. Remote Sensing Generalist Multimodal Fallback Specialist (Unadapted VLM candidate)
        gen_weights_dir = self.workspace_root / "backend" / "models" / "generalist" / "qwen2_vl"
        gen_avail = gen_weights_dir.exists() and any(gen_weights_dir.glob("*.safetensors"))
        gen_unavail_reason = None if gen_avail else (
            "Base model 'Qwen/Qwen2-VL-2B-Instruct' or RS adapter is not installed locally. "
            "Automatic downloads are disabled to prevent unexpected multi-GB transfers."
        )

        def run_generalist(**kw):
            from backend.tools.generalist import GeneralistTool
            tool = GeneralistTool(weights_dir=gen_weights_dir)
            res = tool.run(kw, kw.get("parameters"))
            return {
                "status": res.get("status", "SUCCESS"),
                "answer": res.get("answer", ""),
                "confidence": res.get("confidence"),
                "confidence_level": res.get("confidence_status", "UNAVAILABLE"),
                "confidence_status": res.get("confidence_status", "not_calibrated"),
                "evidence": res.get("evidence"),
                "evidence_source": "generalist_vlm",
                "model": res.get("model", "Qwen/Qwen2-VL-2B-Instruct"),
                "base_model": res.get("base_model", "Qwen/Qwen2-VL-2B-Instruct"),
                "adapter": res.get("adapter"),
                "model_type": res.get("model_type", "general_multimodal_vlm"),
                "device": res.get("device", "cpu"),
                "inference_time_ms": res.get("inference_time_ms", 0.0),
                "provenance": res.get("provenance"),
                "warnings": res.get("warnings", []),
            }

        self._specialists["rs_generalist"] = SpecialistEntry(
            id="rs_generalist",
            name="Remote-Sensing Generalist Multimodal Fallback Specialist",
            task="general_vqa",
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            version="0.1.0-unadapted",
            modality=["optical", "multispectral", "sar"],
            supported_input_types=["image/tiff", "image/png", "image/jpeg"],
            checkpoint_location=str(gen_weights_dir) if gen_avail else None,
            is_available=gen_avail,
            unavailable_reason=gen_unavail_reason,
            inference_fn=run_generalist,
            base_model="Qwen/Qwen2-VL-2B-Instruct",
            model_type="general_multimodal_vlm",
            is_remote_sensing_adapted=False,
            is_remote_sensing_expert=False,
            adaptation_scope="unadapted_foundation_model",
            benchmark_accuracy_claim=None,
            confidence=None,
            confidence_status="not_calibrated",
        )

        # 10. AdaptLLM Remote-Sensing VLM Specialist (Domain-adapted VLM candidate)
        adaptllm_path_env = os.getenv("ADAPTLLM_MODEL_PATH")
        adaptllm_weights_dir = (
            Path(adaptllm_path_env).resolve()
            if adaptllm_path_env
            else (self.workspace_root / "backend" / "models" / "generalist" / "adaptllm")
        )
        adaptllm_enabled = (
            os.getenv("ENABLE_ADAPTLLM", "false").lower() in ("true", "1", "yes")
            or bool(adaptllm_path_env)
        )
        adaptllm_avail = (
            adaptllm_enabled
            and adaptllm_weights_dir.exists()
            and (any(adaptllm_weights_dir.glob("*.safetensors")) or any(adaptllm_weights_dir.glob("*.bin")))
        )
        adaptllm_unavail_reason = None if adaptllm_avail else (
            "AdaptLLM remote-sensing VLM weights are not installed or enabled."
        )

        def run_adaptllm(**kw):
            from backend.tools.adaptllm import AdaptLLMTool
            tool = AdaptLLMTool(weights_dir=adaptllm_weights_dir)
            res = tool.run(kw, kw.get("parameters"))
            return {
                "status": res.get("status", "SUCCESS"),
                "answer": res.get("answer", ""),
                "confidence": None,
                "confidence_level": "UNAVAILABLE",
                "confidence_status": "not_calibrated",
                "evidence": res.get("evidence"),
                "evidence_source": "adaptllm_vlm",
                "model": "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct",
                "base_model": "Qwen/Qwen2-VL-2B-Instruct",
                "adapter": None,
                "model_type": "remote_sensing_vlm",
                "is_remote_sensing_adapted": True,
                "is_remote_sensing_expert": True,
                "adaptation_scope": "remote_sensing_domain_post_training",
                "benchmark_accuracy_claim": None,
                "device": res.get("device", "cpu"),
                "inference_time_ms": res.get("inference_time_ms", 0.0),
                "provenance": res.get("provenance"),
                "warnings": res.get("warnings", []),
            }

        self._specialists["rs_adaptllm"] = SpecialistEntry(
            id="rs_adaptllm",
            name="AdaptLLM Remote-Sensing VLM Specialist",
            task="general_vqa",
            model_id="AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct",
            version="0.1.0-domain-adapted",
            modality=["optical", "multispectral", "sar"],
            supported_input_types=["image/tiff", "image/png", "image/jpeg"],
            checkpoint_location=str(adaptllm_weights_dir) if adaptllm_avail else None,
            is_available=adaptllm_avail,
            unavailable_reason=adaptllm_unavail_reason,
            inference_fn=run_adaptllm,
            base_model="Qwen/Qwen2-VL-2B-Instruct",
            model_type="remote_sensing_vlm",
            is_remote_sensing_adapted=True,
            is_remote_sensing_expert=True,
            adaptation_scope="remote_sensing_domain_post_training",
            benchmark_accuracy_claim=None,
            confidence=None,
            confidence_status="not_calibrated",
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
                "model": s.model_id,
                "base_model": s.base_model,
                "model_type": s.model_type,
                "version": s.version,
                "modality": s.modality,
                "supported_input_types": s.supported_input_types,
                "checkpoint_location": s.checkpoint_location,
                "is_available": s.is_available,
                "unavailable_reason": s.unavailable_reason,
                "is_remote_sensing_adapted": s.is_remote_sensing_adapted,
                "is_remote_sensing_expert": s.is_remote_sensing_expert,
                "adaptation_scope": s.adaptation_scope,
                "benchmark_accuracy_claim": s.benchmark_accuracy_claim,
                "confidence": s.confidence,
                "confidence_status": s.confidence_status,
            }
            for s in self._specialists.values()
        ]
