"""
ORBITAL-AI — Lightweight Runtime Capability & Provenance Inspector.

Inspects actual local filesystem and model registry availability without
triggering heavy PyTorch model weights loading or unannounced network downloads.
Provides truthful provenance and availability status for judges and UI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "backend" / "models"
CAPTION_ADAPTER_PATH = MODELS_DIR / "adapters" / "blip_rs_lora"
VQA_ADAPTER_PATH = MODELS_DIR / "adapters" / "blip_vqa_rs_lora"
BUILDING_MODEL_PATH = MODELS_DIR / "building_model.pt"
GENERALIST_WEIGHTS_DIR = MODELS_DIR / "generalist" / "qwen2_vl"


def inspect_capabilities() -> Dict[str, Any]:
    """
    Evaluates actual on-disk readiness of all specialists.
    Strictly safe at startup: zero tensor operations, zero network downloads.
    """
    # 1. Building detector (YOLO weights)
    bldg_available = (
        BUILDING_MODEL_PATH.exists()
        and BUILDING_MODEL_PATH.stat().st_size > 1000
    )

    # 2. BigEarthNet ResNet-50
    # A configured model ID is not proof of executable local readiness.
    ben_available = False

    # 3. Caption LoRA pilot adapter
    caption_available = (
        CAPTION_ADAPTER_PATH.exists()
        and (CAPTION_ADAPTER_PATH / "adapter_model.safetensors").exists()
    )

    # 4. VQA LoRA pilot adapter
    vqa_available = (
        VQA_ADAPTER_PATH.exists()
        and (VQA_ADAPTER_PATH / "adapter_model.safetensors").exists()
    )

    # 5. Classical Change Detection & Optical-SAR are executable baselines.
    # They are not presented as trained-model availability.

    # 6. Generalist Qwen2-VL-2B (Unadapted fallback VLM)
    qwen_weights_installed = (
        GENERALIST_WEIGHTS_DIR.exists()
        and any(GENERALIST_WEIGHTS_DIR.glob("*.safetensors"))
    )

    # 7. AdaptLLM Remote-Sensing VLM Specialist (Domain-adapted VLM candidate)
    adaptllm_path_env = os.getenv("ADAPTLLM_MODEL_PATH")
    adaptllm_weights_dir = (
        Path(adaptllm_path_env).resolve()
        if adaptllm_path_env
        else MODELS_DIR / "generalist" / "adaptllm"
    )
    adaptllm_enabled = (
        os.getenv("ENABLE_ADAPTLLM", "false").lower() in ("true", "1", "yes")
        or bool(adaptllm_path_env)
    )
    adaptllm_available = (
        adaptllm_enabled
        and adaptllm_weights_dir.exists()
        and (any(adaptllm_weights_dir.glob("*.safetensors")) or any(adaptllm_weights_dir.glob("*.bin")))
    )

    items: List[Dict[str, Any]] = [
        {
            "id": "building_detection",
            "name": "YOLO Building Footprint Segmentation",
            "model_id": "yolo-segmentation-building_model.pt",
            "type": "trained_instance_segmentation",
            "status": "available" if bldg_available else "unavailable",
            "display_status": "Available" if bldg_available else "Unavailable",
            "provenance": "Fine-tuned on satellite urban imagery for rooftop footprints",
            "benchmark_claim": None,
            "weights_present": bldg_available,
        },
        {
            "id": "land_cover",
            "name": "BIFOLD BigEarthNet ResNet-50",
            "model_id": "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
            "type": "trained_deep_classifier",
            "status": "available" if ben_available else "unavailable",
            "display_status": "Available" if ben_available else "Unavailable",
            "provenance": "Official BIFOLD BigEarthNet v2.0 Corine 19-class taxonomy (IGARSS 2025)",
            "benchmark_claim": None,
            "weights_present": ben_available,
        },
        {
            "id": "caption",
            "name": "BigEarthNet Adapted Caption LoRA",
            "model_id": "rs-caption-adapted-v1",
            "base_model": "Salesforce/blip-image-captioning-base",
            "type": "pilot_domain_adaptation",
            "status": "available" if caption_available else "unavailable",
            "display_status": "Available" if caption_available else "Unavailable",
            "provenance": "Pilot LoRA adapter on 87 BigEarthNet patch-text pairs (3 epochs)",
            "benchmark_claim": None,
            "weights_present": caption_available,
        },
        {
            "id": "vqa",
            "name": "BigEarthNet Adapted VQA LoRA",
            "model_id": "rs-vqa-adapted-v1",
            "base_model": "Salesforce/blip-vqa-base",
            "type": "pilot_domain_adaptation",
            "status": "available" if vqa_available else "unavailable",
            "display_status": "Available" if vqa_available else "Unavailable",
            "provenance": "Pilot LoRA adapter on 551 BigEarthNet QA pairs (3 epochs)",
            "benchmark_claim": None,
            "weights_present": vqa_available,
        },
        {
            "id": "change_detection",
            "name": "Classical Bi-Temporal Change Detection",
            "model_id": "classical-cv-change-detector-v2",
            "type": "classical_cv",
            "status": "available",
            "display_status": "Available (baseline)",
            "provenance": "Deterministic pixel differencing, morphological cleanup, and visible-band vegetation proxy",
            "benchmark_claim": None,
            "weights_present": True,
        },
        {
            "id": "optical_sar",
            "name": "Classical Optical–SAR Fusion Engine",
            "model_id": "classical-cv-fusion-engine-v2",
            "type": "classical_cv",
            "status": "available",
            "display_status": "Available (baseline)",
            "provenance": "Classical RGB proxies, raw SAR intensity telemetry, SSIM/NCC; no sensor-specific calibration",
            "benchmark_claim": None,
            "weights_present": True,
        },
        {
            "id": "rs_generalist",
            "name": "Generalist Qwen2-VL-2B",
            "model_id": "Qwen/Qwen2-VL-2B-Instruct",
            "type": "general_multimodal_vlm",
            "status": "weights_installed" if qwen_weights_installed else "not_installed",
            "display_status": "Weights Installed" if qwen_weights_installed else "Generalist Qwen2-VL: Not installed — safe unavailable state",
            "provenance": "General multimodal foundation VLM (unadapted to remote sensing)",
            "benchmark_claim": None,
            "weights_present": qwen_weights_installed,
            "safe_fallback": True,
            "note": (
                "Base model weights are not installed on disk. Automatic downloads are disabled "
                "to prevent unexpected multi-GB network transfers. Controlled safe fallback."
                if not qwen_weights_installed else "Installed locally."
            ),
        },
        {
            "id": "rs_adaptllm",
            "name": "AdaptLLM Remote-Sensing VLM Specialist",
            "model_id": "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct",
            "model": "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct",
            "base_model": "Qwen/Qwen2-VL-2B-Instruct",
            "type": "remote_sensing_vlm",
            "model_type": "remote_sensing_vlm",
            "status": "available" if adaptllm_available else "configured_unavailable",
            "display_status": "Available" if adaptllm_available else "configured but unavailable",
            "provenance": "Remote-sensing domain post-training on Qwen2-VL-2B-Instruct (uncalibrated)",
            "benchmark_claim": None,
            "benchmark_accuracy_claim": None,
            "weights_present": adaptllm_available,
            "is_remote_sensing_adapted": True,
            "is_remote_sensing_expert": True,
            "adaptation_scope": "remote_sensing_domain_post_training",
            "confidence": None,
            "confidence_status": "not_calibrated",
            "note": (
                "Optional remote-sensing domain-adapted VLM candidate. "
                "Configured but unavailable unless explicitly enabled and local weights are installed."
            ),
        }
    ]

    return {
        "status": "success",
        "service": "ORBITAL-AI Capability & Provenance Inspector",
        "version": "3.0.0",
        "capabilities": items,
        "summary": {
            "total": len(items),
            "available": sum(1 for i in items if i["status"] in ("available", "weights_installed")),
            "unavailable": sum(1 for i in items if i["status"] in ("unavailable", "not_installed", "configured_unavailable")),
            "qwen_status": "installed" if qwen_weights_installed else "not_installed_safe_fallback",
            "adaptllm_status": "installed" if adaptllm_available else "configured_but_unavailable",
        }
    }
