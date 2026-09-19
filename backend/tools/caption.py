import time
from typing import Any, Dict, List, Optional
import numpy as np

from .base import BaseTool
from ..services.rs_adapters import RSAdapterRuntime

class CaptionTool(BaseTool):
    id = "rs_caption_adapted"
    name = "Remote-Sensing Adapted Captioning Specialist"
    description = (
        "Generates descriptive captions for remote-sensing imagery using "
        "Salesforce/blip-image-captioning-base adapted with a BigEarthNet-derived LoRA pilot adapter."
    )
    supported_tasks = ["caption"]
    modalities = ["optical", "multispectral"]
    adapter = "BigEarthNet-derived LoRA pilot adapter"
    domain_adaptation = (
        "Pilot domain adaptation on 87 BigEarthNet image-text pairs (3 epochs). "
        "Pilot artifact only; no benchmark accuracy claim (e.g. VRSBench)."
    )
    base_model = "Salesforce/blip-image-captioning-base"
    model_id = "rs-caption-adapted-v1"
    provenance = {
        "base_model": "Salesforce/blip-image-captioning-base",
        "adapter": "BigEarthNet-derived LoRA pilot adapter",
        "training_pairs": 87,
        "epochs": 3,
        "adaptation_scope": "pilot_domain_adaptation",
        "benchmark_accuracy_claim": None,
        "note": "Pilot adaptation artifact only; no benchmark accuracy claim (e.g. VRSBench).",
    }
    permitted_parameters = {
        "max_new_tokens": 60,
        "include_supporting_land_cover": True,
    }

    def __init__(self, runtime: Optional[RSAdapterRuntime] = None, ben_tool: Optional[Any] = None):
        self.runtime = runtime or RSAdapterRuntime.get_instance()
        self._ben_tool = ben_tool  # Lazy loaded if requested

    def _get_ben_tool(self):
        if self._ben_tool is None:
            try:
                from .land_cover import BigEarthNetTool
                self._ben_tool = BigEarthNetTool()
            except Exception:
                self._ben_tool = None
        return self._ben_tool

    def run(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        img_bgr = inputs.get("image")

        if img_bgr is None:
            return {
                "status": "error",
                "caption": None,
                "error": "Missing image for captioning",
                "model_id": self.model_id,
                "adapter": self.adapter,
                "evidence": {},
                "provenance": self.provenance,
                "inference_time_ms": 0.0,
                "confidence": None,
            }

        # 1. Primary execution path: BLIP + LoRA pilot adapter
        adapter_res = self.runtime.caption(img_bgr)
        status = adapter_res.get("status", "error")
        caption = adapter_res.get("caption")
        inference_time_ms = adapter_res.get("inference_time_ms", (time.time() - t0) * 1000)

        # Build evidence container
        evidence: Dict[str, Any] = {
            "device": adapter_res.get("device", self.runtime.device),
            "raw_output": adapter_res.get("answer"),
        }

        # 2. Retain BigEarthNet land-cover information ONLY as optional supporting evidence
        params = parameters or {}
        if params.get("include_supporting_land_cover", True):
            ben_tool = self._get_ben_tool()
            if ben_tool is not None:
                try:
                    ben_res = ben_tool.run({"image": img_bgr})
                    evidence["supporting_land_cover"] = {
                        "top_label": ben_res.get("top_label"),
                        "active_labels": [l.get("name") for l in ben_res.get("active_labels", [])],
                    }
                except Exception as e:
                    evidence["supporting_land_cover"] = {"error": f"Supporting classifier error: {e}"}

        # If adapted model is unavailable, return structured specialist-unavailable result.
        # DO NOT fabricate a caption using the old heuristic implementation.
        if status == "specialist_unavailable":
            return {
                "status": "specialist_unavailable",
                "caption": None,
                "answer": adapter_res.get("answer", "Caption specialist unavailable"),
                "model_id": self.model_id,
                "adapter": self.adapter,
                "evidence": evidence,
                "provenance": self.provenance,
                "inference_time_ms": inference_time_ms,
                "confidence": None,
                "confidence_status": "specialist_unavailable",
                "warnings": adapter_res.get("warnings", []),
            }

        return {
            "status": status,
            "caption": caption,
            "answer": caption,
            "model_id": self.model_id,
            "adapter": self.adapter,
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
