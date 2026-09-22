"""
SatQuery AI — Capability-Aware Agentic Router.
Routes remote-sensing queries based on:
1. Natural language intent (VQA, captioning, grounding, counting, change, fusion)
2. Number of images and temporal pairs
3. Sensor modalities (optical, SAR, multispectral)
4. Dynamic specialist model availability and input compatibility
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from models.registry import ModelRegistry


def classify_query_intent(
    query: str,
    image_count: int = 1,
    modalities: Optional[List[str]] = None
) -> str:
    """
    Deterministic query classifier: extracts intent from natural language question
    without LLM overhead, ensuring 100% auditable and reproducible routing.
    """
    modalities = modalities or ["optical"]
    q = query.lower().strip()

    # 1. SAR / Optical Fusion
    if (
        any("sar" in m.lower() for m in modalities) or
        "sar" in q or
        "radar" in q or
        "fusion" in q or
        "cross-modal" in q or
        "optical and sar" in q or
        "sar and optical" in q or
        "risat" in q or
        "sentinel-1" in q or
        ("built-up and water-covered" in q and ("optical" in q or "sar" in q or image_count >= 2))
    ):
        if image_count >= 2 or len(modalities) >= 2 or "sar" in q or "fusion" in q:
            return "sar_optical_fusion"

    # 2. Bi-Temporal Change Detection & Change-VQA
    if (
        image_count >= 2 or
        "change" in q or
        "compare" in q or
        "between these two dates" in q or
        "before and after" in q or
        "increased" in q or
        "decreased" in q or
        "remained unchanged" in q or
        "temporal" in q or
        "evolution" in q or
        "expansion between" in q or
        "what changed" in q
    ):
        if "what changed" in q or "increased" in q or "decreased" in q or "?" in q:
            return "change_vqa"
        return "change_detection"

    # 3. Building Footprint Detection & Counting
    if (
        "building" in q or
        "footprint" in q or
        "rooftop" in q or
        "structure" in q or
        "how many buildings" in q or
        "count buildings" in q or
        "audit buildings" in q or
        ("count" in q and ("urban" in q or "roof" in q or "house" in q))
    ):
        return "building_detection"

    # 4. Text-Guided Grounding
    if (
        "highlight" in q or
        "locate" in q or
        "find the" in q or
        "where is" in q or
        "pinpoint" in q or
        "bounding box" in q or
        "demarcate" in q or
        "water body referred to" in q or
        "identify major roads" in q or
        "target area" in q
    ):
        return "grounding"

    # 5. Scene Captioning
    if (
        "caption" in q or
        "describe the land-cover" in q or
        "major objects visible" in q or
        "scene overview" in q or
        "detailed description" in q or
        "vrsbench" in q
    ):
        return "caption"

    # 6. Land Cover / BigEarthNet Scene Classification
    if (
        "land cover" in q or
        "land-cover" in q or
        "classification" in q or
        "corine" in q or
        "bigearthnet" in q or
        "biome" in q or
        "terrain type" in q or
        "what type of land" in q
    ):
        return "land_cover"

    # 7. Open Remote-Sensing / Generalist Visual Questions
    if any(k in q for k in [
        "solar panel", "solar panels", "airplane", "airplanes", "aircraft",
        "shipping container", "shipping containers", "swimming pool", "swimming pools",
        "crane", "cranes", "helicopter", "helicopters", "storage tank", "oil tank"
    ]):
        return "general_vqa"

    # Default to general Remote Sensing VQA
    return "vqa"


def route_query_to_specialist(
    query: str,
    image_count: int = 1,
    modalities: Optional[List[str]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Selects the optimal specialist dynamically by inspecting intent,
    input constraints, and ModelRegistry capabilities.
    Returns (specialist_id, specialist_metadata).
    """
    intent = classify_query_intent(query, image_count=image_count, modalities=modalities)
    reg = ModelRegistry.get_instance()

    adaptllm_spec = reg.get_specialist("rs_adaptllm")
    open_candidates = ["rs_adaptllm"] if (adaptllm_spec and adaptllm_spec.is_available) else ["rs_generalist"]

    # Map intent to candidate specialists in priority order
    intent_to_specialist_map = {
        "sar_optical_fusion": ["optical_sar_fusion"],
        "change_vqa": ["change_vqa", "change_detection"],
        "change_detection": ["change_detection", "change_vqa"],
        "building_detection": ["building_detection"],
        "grounding": ["visual_grounding"],
        "caption": ["rs_caption_adapted", "captioning"],
        "land_cover": ["land_cover", "rs_vqa_adapted"],
        "vqa": ["rs_vqa_adapted", "rs_vqa", "land_cover"],
        "general_vqa": open_candidates,
        "open_question": open_candidates,
    }

    candidates = intent_to_specialist_map.get(
        intent,
        open_candidates if intent in ("general_vqa", "open_question") else ["rs_vqa_adapted"]
    )
    chosen_id = candidates[0]
    specialist = reg.get_specialist(chosen_id)

    # Fallback to secondary if primary unavailable (never silently replace AdaptLLM with generalist)
    if (not specialist or not specialist.is_available) and len(candidates) > 1 and chosen_id != "rs_adaptllm":
        alt = reg.get_specialist(candidates[1])
        if alt and alt.is_available:
            chosen_id = candidates[1]
            specialist = alt

    info = {
        "task": intent,
        "specialist_id": chosen_id,
        "specialist_name": specialist.name if specialist else chosen_id,
        "model_id": specialist.model_id if specialist else "unknown",
        "is_available": specialist.is_available if specialist else False,
        "unavailable_reason": specialist.unavailable_reason if specialist else None
    }
    return chosen_id, info
