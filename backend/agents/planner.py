"""
Orbital-AI / SatQuery AI — Agentic Query Planner & Intent Router.

Deterministically interprets user natural-language queries into structured QueryPlans
specifying:
- intent
- required_images
- required_modalities
- required_tasks
- specialists
- execution_order
- evidence_requirements

Never fabricates answers or routes through a generic unspecialized VLM.
Strictly distinguishes supported remote-sensing tasks from unsupported questions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from ..schemas.analysis import QueryPlan


UNSUPPORTED_KEYWORDS = [
    "population", "people living", "demographic", "who lives", "how many people",
    "next year", "future", "tomorrow", "predict stock", "crime rate", "gdp",
    "weather forecast", "temperature tomorrow", "politics", "president", "election",
    "who is", "recipe", "write a poem", "history of", "tell me a joke"
]


def is_query_unsupported(query: str) -> bool:
    """Detects whether a query falls outside the remote-sensing task family."""
    q = query.lower().strip()
    return any(kw in q for kw in UNSUPPORTED_KEYWORDS)


def create_query_plan(
    query: str,
    image_count: int = 1,
    modalities: Optional[List[str]] = None
) -> QueryPlan:
    """
    Constructs a structured QueryPlan from user natural language query and input configuration.
    Adheres strictly to the 10 supported intents:
    - caption
    - vqa
    - land_cover
    - grounding
    - building_detection
    - change_detection
    - change_vqa
    - optical_sar_analysis
    - multi_task
    - unsupported
    """
    modalities = modalities or ["optical"]
    q = query.lower().strip()

    # ── 1. Query-Scope Control: Unsupported Query Detection ──────────────────
    if is_query_unsupported(q):
        alternatives = [
            "Count existing building footprints and estimate structural density",
            "Classify urban fabric and land-cover classes via BigEarthNet v2.0",
            "Perform bi-temporal change detection between two observation dates",
            "Measure hydrological surfaces and vegetation index proxies"
        ]
        reason = (
            "The requested conclusion (demographic, socio-economic, or future predictive analysis) "
            "cannot be verified or inferred from remote sensing imagery alone. Remote sensing specialists "
            "extract observable surface, spectral, and structural properties, not demographic census data."
        )
        return QueryPlan(
            intent="unsupported",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=[],
            specialists=[],
            execution_order=[],
            evidence_requirements=[],
            unsupported_reason=reason,
            supported_alternatives=alternatives
        )

    # ── 2. Multi-Task Queries ────────────────────────────────────────────────
    # Check for compound requests combining distinct specialist tasks
    has_caption_request = any(k in q for k in ["describe", "caption", "overview"])
    has_water_request = any(k in q for k in ["water", "river", "lake", "ocean", "pond"])
    has_building_request = any(k in q for k in ["building", "structure", "house", "footprint"])
    has_vegetation_request = any(k in q for k in ["vegetation", "canopy", "tree", "forest", "crop", "greenery"])
    has_change_request = any(k in q for k in ["change", "compare", "between these", "before and after", "new", "increased", "decreased"])

    # Example: "Compare these images and tell me whether vegetation increased and whether buildings were added."
    if has_change_request and (has_vegetation_request and has_building_request) and image_count >= 2:
        return QueryPlan(
            intent="change_detection",
            required_images=2,
            required_modalities=["optical"],
            required_tasks=["change_detection", "land_cover", "building_detection"],
            specialists=["change_detection", "land_cover", "building_detection"],
            execution_order=["change_detection", "land_cover", "building_detection"],
            evidence_requirements=[
                "Bi-temporal surface change mask and cluster statistics",
                "Vegetation and canopy spectral classification difference",
                "Building footprint count and structural addition assessment"
            ]
        )

    # Example: "Describe this scene and tell me whether there is water and whether buildings are present."
    if has_caption_request and has_water_request and has_building_request:
        return QueryPlan(
            intent="multi_task",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["caption", "vqa", "building_detection"],
            specialists=["rs_caption_adapted", "rs_vqa_adapted", "building_detection"],
            execution_order=["rs_caption_adapted", "rs_vqa_adapted", "building_detection"],
            evidence_requirements=[
                "Scene caption describing land-cover and dominant visible features",
                "VQA verification of hydrological presence",
                "Tiled building footprint extraction and structural count"
            ]
        )

    # Example: "Describe the image and tell me whether water is present"
    if has_caption_request and (has_water_request or "whether" in q or "tell me" in q):
        return QueryPlan(
            intent="multi_task",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["caption", "vqa"],
            specialists=["rs_caption_adapted", "rs_vqa_adapted"],
            execution_order=["rs_caption_adapted", "rs_vqa_adapted"],
            evidence_requirements=[
                "Scene description caption",
                "VQA verification of hydrological presence"
            ]
        )

    # ── 3. Optical-SAR Joint Analysis ────────────────────────────────────────
    # Examples:
    # "Compare these optical and SAR images"
    # "What differences can you see between the optical and SAR images?"
    # "What structures are visible across both modalities?"
    if (
        "optical and sar" in q or
        "sar and optical" in q or
        "both modalities" in q or
        ("optical" in q and "sar" in q) or
        "cross-modal" in q or
        "radar" in q or
        any("sar" in m.lower() for m in modalities)
    ):
        return QueryPlan(
            intent="optical_sar_analysis",
            required_images=2,
            required_modalities=["optical", "sar"],
            required_tasks=["optical_sar_analysis"],
            specialists=["optical_sar_fusion"],
            execution_order=["optical_sar_fusion"],
            evidence_requirements=[
                "Cross-modal structural similarity (SSIM) and correlation",
                "Radar backscatter intensity (dB) and speckle distribution",
                "Complementary optical-SAR feature decomposition"
            ]
        )

    # ── 4. Bi-Temporal Change & Change-VQA ────────────────────────────────────
    # Examples:
    # "What changed?" / "What changed between these images?"
    # "Has vegetation increased?"
    # "Are there new buildings?" / "Compare the two images and tell me if new buildings appeared"
    if (
        image_count >= 2 or
        has_change_request or
        "between these images" in q or
        "between the two images" in q or
        "new buildings" in q or
        "vegetation increased" in q
    ):
        # Case A: Change with building focus ("Are there new buildings?", "tell me if new buildings appeared")
        if has_building_request or "new building" in q or "buildings appeared" in q:
            return QueryPlan(
                intent="change_detection",
                required_images=2,
                required_modalities=["optical"],
                required_tasks=["change_detection", "building_detection"],
                specialists=["change_detection", "building_detection"],
                execution_order=["change_detection", "building_detection"],
                evidence_requirements=[
                    "Bi-temporal change detection surface difference",
                    "Building footprint detection and count comparison"
                ]
            )

        # Case B: Change with vegetation focus ("Has vegetation increased?")
        if has_vegetation_request or "vegetation increased" in q or "vegetation decreased" in q:
            return QueryPlan(
                intent="change_detection",
                required_images=2,
                required_modalities=["optical"],
                required_tasks=["change_detection", "land_cover"],
                specialists=["change_detection", "land_cover"],
                execution_order=["change_detection", "land_cover"],
                evidence_requirements=[
                    "Bi-temporal change detection surface alteration metrics",
                    "Land cover vegetation fraction and canopy change evidence"
                ]
            )

        # Case C: General Change-VQA ("What changed?", "What changed between these images?")
        if "what changed" in q or "?" in q:
            return QueryPlan(
                intent="change_vqa",
                required_images=2,
                required_modalities=["optical"],
                required_tasks=["change_detection", "change_vqa"],
                specialists=["change_detection", "change_vqa"],
                execution_order=["change_detection", "change_vqa"],
                evidence_requirements=[
                    "Bi-temporal difference metrics and cluster localization",
                    "Change-VQA natural language alteration reasoning"
                ]
            )

        # Case D: Standard change detection
        return QueryPlan(
            intent="change_detection",
            required_images=2,
            required_modalities=["optical"],
            required_tasks=["change_detection"],
            specialists=["change_detection"],
            execution_order=["change_detection"],
            evidence_requirements=[
                "Bi-temporal change percentage and spatial alteration mask"
            ]
        )

    # ── 5. Building Detection & Counting ─────────────────────────────────────
    # "How many buildings?" → building_detection
    if (
        "how many buildings" in q or
        "count buildings" in q or
        "how many structures" in q or
        (has_building_request and ("how many" in q or "count" in q or "number of" in q))
    ):
        return QueryPlan(
            intent="building_detection",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["building_detection"],
            specialists=["building_detection"],
            execution_order=["building_detection"],
            evidence_requirements=[
                "Tiled YOLO structural footprint count with confidence stratification"
            ]
        )

    # ── 6. Spatial Grounding ─────────────────────────────────────────────────
    # "Where are the buildings?" → visual_grounding + building_detection
    if (
        "where are the buildings" in q or
        "locate buildings" in q or
        "where are buildings" in q
    ):
        return QueryPlan(
            intent="grounding",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["grounding", "building_detection"],
            specialists=["visual_grounding", "building_detection"],
            execution_order=["visual_grounding", "building_detection"],
            evidence_requirements=[
                "Text-guided spatial grounding bounding coordinates",
                "Structural footprint instance polygons"
            ]
        )

    if (
        "where is" in q or
        "where are" in q or
        "locate" in q or
        "find the" in q or
        "highlight" in q or
        "pinpoint" in q or
        "bounding box" in q
    ):
        return QueryPlan(
            intent="grounding",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["grounding"],
            specialists=["visual_grounding"],
            execution_order=["visual_grounding"],
            evidence_requirements=[
                "Normalized bounding box coordinates of targeted spatial feature"
            ]
        )

    # ── 7. Land Cover Classification ─────────────────────────────────────────
    # "What type of land is present?" / "What type of land is present here?"
    # → land_cover + VQA
    if (
        "what type of land" in q or
        "land cover" in q or
        "land-cover" in q or
        "terrain type" in q or
        "corine" in q or
        "bigearthnet" in q or
        "biome" in q or
        "type of terrain" in q
    ):
        return QueryPlan(
            intent="land_cover",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["land_cover", "vqa"],
            specialists=["land_cover", "rs_vqa_adapted"],
            execution_order=["land_cover", "rs_vqa_adapted"],
            evidence_requirements=[
                "BigEarthNet v2.0 Corine 19-class semantic land-cover labels",
                "Remote-sensing adapted visual question answering scene details"
            ]
        )

    # ── 8. Scene Captioning ──────────────────────────────────────────────────
    # "Describe this image" / "Describe this image."
    if (
        "describe this image" in q or
        "describe the image" in q or
        "describe image" in q or
        "caption" in q or
        "scene overview" in q
    ):
        return QueryPlan(
            intent="caption",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["caption"],
            specialists=["rs_caption_adapted"],
            execution_order=["rs_caption_adapted"],
            evidence_requirements=[
                "Descriptive scene caption adapted on remote sensing corpus"
            ]
        )

    # ── 9. Building presence VQA ─────────────────────────────────────────────
    # "Are there buildings?" → adapted VQA + building detection when spatial/count evidence is needed
    if has_building_request and ("are there" in q or "is there" in q or "any buildings" in q):
        return QueryPlan(
            intent="vqa",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["vqa", "building_detection"],
            specialists=["rs_vqa_adapted", "building_detection"],
            execution_order=["rs_vqa_adapted", "building_detection"],
            evidence_requirements=[
                "VQA verification of structural presence",
                "Tiled building footprint extraction for spatial/count evidence"
            ]
        )

    # ── 10. General / Hydrological VQA ───────────────────────────────────────
    # "Is there water?"
    return QueryPlan(
        intent="vqa",
        required_images=1,
        required_modalities=["optical"],
        required_tasks=["vqa"],
        specialists=["rs_vqa_adapted"],
        execution_order=["rs_vqa_adapted"],
        evidence_requirements=[
            "Adapted VQA reasoning on remote sensing observation"
        ]
    )
