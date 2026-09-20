import time
from typing import Any, Dict, List, Optional
import numpy as np

from .base import BaseTool
from ..services.rs_adapters import RSAdapterRuntime

class VQATool(BaseTool):
    id = "rs_vqa_adapted"
    name = "Remote-Sensing Adapted Visual Question Answering Specialist"
    description = (
        "Answers natural-language questions regarding remote-sensing imagery using "
        "Salesforce/blip-vqa-base adapted with a BigEarthNet-derived VQA LoRA pilot adapter."
    )
    supported_tasks = ["vqa"]
    modalities = ["optical", "multispectral"]
    adapter = "BigEarthNet-derived VQA LoRA pilot adapter"
    domain_adaptation = (
        "Pilot domain adaptation on 551 BigEarthNet QA examples across 69 training patches "
        "and 144 QA examples across 18 validation patches (3 epochs). "
        "Pilot artifact only; no benchmark superiority claim (e.g. RSVQA)."
    )
    base_model = "Salesforce/blip-vqa-base"
    model_id = "rs-vqa-adapted-v1"
    provenance = {
        "base_model": "Salesforce/blip-vqa-base",
        "adapter": "BigEarthNet-derived VQA LoRA pilot adapter",
        "training_qa": 551,
        "training_patches": 69,
        "validation_qa": 144,
        "validation_patches": 18,
        "epochs": 3,
        "adaptation_scope": "pilot_domain_adaptation",
        "benchmark_accuracy_claim": None,
        "note": "Pilot adaptation artifact only; no benchmark superiority claim (e.g. RSVQA).",
    }
    permitted_parameters = {
        "max_new_tokens": 50,
        "include_supporting_evidence": True,
    }

    def __init__(
        self,
        runtime: Optional[RSAdapterRuntime] = None,
        ben_tool: Optional[Any] = None,
        building_tool: Optional[Any] = None
    ):
        self.runtime = runtime or RSAdapterRuntime.get_instance()
        self._ben_tool = ben_tool
        self._building_tool = building_tool

    def _get_ben_tool(self):
        if self._ben_tool is None:
            try:
                from .land_cover import BigEarthNetTool
                self._ben_tool = BigEarthNetTool()
            except Exception:
                self._ben_tool = None
        return self._ben_tool

    def _get_building_tool(self):
        if self._building_tool is None:
            try:
                from .building_detection import BuildingDetectionTool
                self._building_tool = BuildingDetectionTool()
            except Exception:
                self._building_tool = None
        return self._building_tool

    def run(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        query = inputs.get("query", inputs.get("question", "")).strip()
        img_bgr = inputs.get("image")

        if img_bgr is None:
            return {
                "status": "error",
                "answer": "Missing image for VQA analysis",
                "model_id": self.model_id,
                "adapter": self.adapter,
                "question": query,
                "evidence": {},
                "provenance": self.provenance,
                "inference_time_ms": 0.0,
                "confidence": None,
            }

        # 1. Primary execution path: BLIP-VQA + LoRA pilot adapter
        adapter_res = self.runtime.vqa(img_bgr, query)
        status = adapter_res.get("status", "error")
        answer = adapter_res.get("answer", "")
        inference_time_ms = adapter_res.get("inference_time_ms", (time.time() - t0) * 1000)

        evidence: Dict[str, Any] = {
            "device": adapter_res.get("device", self.runtime.device),
            "raw_output": answer,
        }

        # 2. For questions that require specialist spatial reasoning unavailable to BLIP-VQA,
        # sequence additional registered tools (such as building detection) as supporting evidence
        params = parameters or {}
        if params.get("include_supporting_evidence", True):
            q_lower = query.lower()
            if any(k in q_lower for k in ["building", "structure", "count", "rooftop", "footprint"]):
                existing_bldg = params.get("supporting_building_result")
                if isinstance(existing_bldg, dict) and existing_bldg.get("status") in ("success", "SUCCESS") and "building_count" in existing_bldg:
                    evidence["supporting_building_detection"] = {
                        "building_count": existing_bldg.get("building_count"),
                        "high_confidence_count": existing_bldg.get("high_confidence_count"),
                        "confidence_tier": existing_bldg.get("confidence_level"),
                        "reused": True,
                    }
                else:
                    bldg_tool = self._get_building_tool()
                    if bldg_tool is not None:
                        try:
                            bldg_res = bldg_tool.run({"image": img_bgr})
                            evidence["supporting_building_detection"] = {
                                "building_count": bldg_res.get("building_count"),
                                "high_confidence_count": bldg_res.get("high_confidence_count"),
                                "confidence_tier": bldg_res.get("confidence_level"),
                            }
                        except Exception as e:
                            evidence["supporting_building_detection"] = {"error": str(e)}

            # Optional supporting land-cover context
            existing_lc = params.get("supporting_land_cover_result")
            if isinstance(existing_lc, dict) and existing_lc.get("status") == "success" and "top_label" in existing_lc:
                evidence["supporting_land_cover"] = {
                    "top_label": existing_lc.get("top_label"),
                    "active_labels": [
                        l.get("name") if isinstance(l, dict) else l
                        for l in existing_lc.get("active_labels", [])
                    ],
                    "reused": True,
                }
            else:
                ben_tool = self._get_ben_tool()
                if ben_tool is not None:
                    try:
                        ben_res = ben_tool.run({"image": img_bgr})
                        evidence["supporting_land_cover"] = {
                            "top_label": ben_res.get("top_label"),
                            "active_labels": [
                                l.get("name") if isinstance(l, dict) else l
                                for l in ben_res.get("active_labels", [])
                            ],
                        }
                    except Exception:
                        pass

        # If adapted model is unavailable, return structured specialist-unavailable response
        if status == "specialist_unavailable":
            return {
                "status": "specialist_unavailable",
                "answer": adapter_res.get("answer", "VQA specialist unavailable"),
                "model_id": self.model_id,
                "adapter": self.adapter,
                "question": query,
                "evidence": evidence,
                "provenance": self.provenance,
                "inference_time_ms": inference_time_ms,
                "confidence": None,
                "confidence_status": "specialist_unavailable",
                "warnings": adapter_res.get("warnings", []),
            }

        return {
            "status": status,
            "answer": answer,
            "model_id": self.model_id,
            "adapter": self.adapter,
            "question": query,
            "evidence": evidence,
            "provenance": self.provenance,
            "inference_time_ms": inference_time_ms,
            "confidence": None,  # VLM output is uncalibrated
            "confidence_status": "not_calibrated",
            "warnings": adapter_res.get("warnings", []),
        }

    def to_spec(self) -> Dict[str, Any]:
        base_spec = super().to_spec()
        base_spec.update({
            "base_model": self.base_model,
            "provenance": self.provenance,
            "supported_modalities": self.modalities,
        })
        return base_spec
