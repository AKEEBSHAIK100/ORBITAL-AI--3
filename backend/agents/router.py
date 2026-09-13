from typing import List, Optional

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
        "terrain type" in q
    ):
        return "land_cover"

    # Default to general Remote Sensing VQA
    return "vqa"
