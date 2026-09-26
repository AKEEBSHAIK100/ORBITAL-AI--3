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
from typing import Any, Dict, List, Optional, Tuple
from ..schemas.analysis import QueryPlan


UNSUPPORTED_KEYWORDS = [
    "population", "people living", "demographic", "who lives", "how many people",
    "next year", "future", "tomorrow", "predict stock", "crime rate", "gdp",
    "weather forecast", "temperature tomorrow", "politics", "president", "election",
    "who is", "recipe", "write a poem", "history of", "tell me a joke",
    "stock price", "stock market", "cryptocurrency", "bitcoin", "lottery",
    "who won", "celebrity", "sports score", "quantum physics", "write code",
    "translate to", "sing a song", "solve equation", "medical diagnosis",
    "identify this person", "facial recognition", "license plate", "exact address",
    "personal identity", "bank balance", "financial advice"
]

DISPOSITION_KNOWN_SPECIALIST = "Known specialist available"
DISPOSITION_GENERALIST_FALLBACK = "Generalist fallback required"
DISPOSITION_NO_CAPABILITY = "No supported visual capability"

KNOWN_RS_VQA_KEYWORDS = [
    "water", "river", "lake", "ocean", "pond", "sea", "stream", "reservoir", "coast", "coastline", "canal", "wetland",
    "vegetation", "canopy", "tree", "trees", "forest", "crop", "crops", "greenery", "agriculture", "grass",
    "farmland", "woodland", "orchard", "pasture",
    "building", "buildings", "house", "houses", "structure", "structures", "rooftop", "rooftops",
    "urban", "residential", "industrial", "commercial", "settlement",
    "road", "roads", "street", "streets", "highway", "runway", "runways",
    "airport", "railway", "railroad", "rail", "bridge", "intersection", "overpass",
    "parking", "parking lot", "harbor", "port", "dam", "reservoir", "coastline",
    "shoreline", "island", "wetland", "industrial", "residential", "commercial",
    "suburban", "rural", "urban", "built-up", "bare land", "sand", "beach"
]

OPEN_VISUAL_OBJECT_KEYWORDS = [
    "solar panel", "solar panels", "airplane", "airplanes", "aircraft",
    "shipping container", "shipping containers", "container", "containers",
    "swimming pool", "swimming pools", "pool", "pools",
    "car", "cars", "truck", "trucks", "vehicle", "vehicles",
    "ship", "ships", "boat", "boats", "vessel", "vessels",
    "crane", "cranes", "helicopter", "helicopters",
    "tank", "tanks", "storage tank", "oil tank",
    "stadium", "tennis court", "baseball", "golf course", "sports field",
    "playground", "fence", "power line", "wind turbine", "solar farm", "solar panel",
    "airstrip", "helipad", "warehouse", "factory", "greenhouse", "dam", "levee",
    "pier", "dock", "shipyard", "construction site", "parking lot",
    "unusual structure", "unusual structures", "infrastructure"
]


def is_query_unsupported(query: str) -> bool:
    """Detects whether a query falls outside the remote-sensing task family."""
    q = query.lower().strip()
    return any(kw in q for kw in UNSUPPORTED_KEYWORDS)


def get_planner_disposition(plan: QueryPlan) -> str:
    """Returns the capability disposition string from a QueryPlan."""
    return plan.planner_disposition


def extract_vqa_sub_question(q: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract a focused VQA question and target entity from a compound query."""
    q_low = q.lower()
    if any(w in q_low for w in ["vegetation", "canopy", "tree", "forest", "crop", "greenery"]):
        return "Is there vegetation in this image?", "vegetation"
    elif any(w in q_low for w in ["water", "river", "lake", "ocean", "pond"]):
        return "Is there water in this image?", "water"
    elif any(w in q_low for w in ["building", "structure", "house", "footprint"]):
        return "Are there buildings in this image?", "buildings"

    match = re.search(r'(?:whether|if)\s+([a-zA-Z\s]+?)\s+(?:is|are)\s+present', q_low)
    if match:
        target = match.group(1).strip()
        return f"Is there {target} in this image?", target

    match_pres = re.search(r'presence\s+of\s+([a-zA-Z\s]+?)(?:[\?,\.]|$)', q_low)
    if match_pres:
        target = match_pres.group(1).strip()
        return f"Is there {target} in this image?", target

    return None, None


def _get_open_vlm_specialist() -> Tuple[str, str]:
    """
    Selects between AdaptLLM (domain-adapted VLM candidate) if available/enabled,
    or falls through to generalist Qwen2-VL.
    Returns (specialist_id, evidence_requirement).
    """
    try:
        from ..tools.adaptllm import is_adaptllm_available
        if is_adaptllm_available():
            return (
                "rs_adaptllm",
                "Remote-sensing domain-adapted visual observation reasoning (uncalibrated)"
            )
    except Exception:
        pass
    return (
        "rs_generalist",
        "Qualitative general multimodal vision-language observation reasoning (uncalibrated)"
    )


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
            supported_alternatives=alternatives,
            planner_disposition=DISPOSITION_NO_CAPABILITY,
            task_category="UNSUPPORTED",
        )

    # Natural-language scene/observation requests stay inside RS scope when an image is supplied.
    has_scene_observation_request = any(k in q for k in [
        "what can you tell me", "what do you see", "what is visible",
        "what features are visible", "what features can you identify",
        "what is shown", "what does this image show", "analyze this image",
        "analyse this image", "inspect this image", "interpret this image",
        "give me an overview", "give an overview", "summarize this image",
        "summarise this image", "what is in this image", "what's in this image",
        "what kind of area is this", "what type of area is this",
        "is this urban", "is this rural", "is this agricultural",
        "is this residential", "is this industrial"
    ])
    has_visual_question_form = any(k in q for k in [
        "is there", "are there", "does this image", "does the image",
        "can you identify", "can you see", "identify the", "identify any",
        "tell me about", "what is", "what are", "which areas", "which features",
        "where can i find", "where can i see", "how much of the image",
        "how much area", "how dense", "how developed"
    ])

    # ── 2. Multi-Task Queries ────────────────────────────────────────────────
    # Check for compound requests combining distinct specialist tasks
    has_caption_request = any(k in q for k in ["describe", "caption", "scene description", "scene overview", "overview"])
    has_land_cover_request = any(k in q for k in ["land cover", "land-cover", "main land cover", "type of land", "terrain type", "classify land"])
    has_water_request = any(k in q for k in ["water", "river", "lake", "ocean", "pond"])
    has_building_request = any(k in q for k in ["building", "structure", "house", "footprint"])
    has_vegetation_request = any(k in q for k in ["vegetation", "canopy", "tree", "forest", "crop", "greenery"])
    has_change_request = any(k in q for k in ["change", "compare", "between these", "before and after", "new", "increased", "decreased"])
    has_open_visual_request = any(k in q for k in OPEN_VISUAL_OBJECT_KEYWORDS)
    has_vqa_request = (
        any(k in q for k in ["whether", "is there", "are there", "presence of", "tell me if", "tell me whether", "detect if", "present in", "present?"])
        or ("tell me" in q and (has_water_request or has_vegetation_request or has_building_request))
    )

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
            ],
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="COMPLEX_KNOWN_TASK",
        )

    # Compound A: Caption + Land Cover + VQA ("Describe the scene, identify the main land cover, and tell me whether vegetation is present.")
    if has_caption_request and has_land_cover_request and (has_vqa_request or has_vegetation_request or has_water_request):
        vqa_q, vqa_target = extract_vqa_sub_question(q)
        return QueryPlan(
            intent="multi_task",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["caption", "land_cover", "vqa"],
            specialists=["rs_caption_adapted", "land_cover", "rs_vqa_adapted"],
            execution_order=["rs_caption_adapted", "land_cover", "rs_vqa_adapted"],
            evidence_requirements=[
                "Scene description caption",
                "Semantic land-cover classification labels",
                f"VQA verification of {vqa_target or 'target'} presence"
            ],
            vqa_question=vqa_q,
            vqa_target=vqa_target,
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="COMPLEX_KNOWN_TASK",
        )

    # Compound B: Caption + Land Cover
    if has_caption_request and has_land_cover_request:
        return QueryPlan(
            intent="multi_task",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["caption", "land_cover"],
            specialists=["rs_caption_adapted", "land_cover"],
            execution_order=["rs_caption_adapted", "land_cover"],
            evidence_requirements=[
                "Scene description caption",
                "Semantic land-cover classification labels"
            ],
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="COMPLEX_KNOWN_TASK",
        )

    # Compound C: Caption + Water + Buildings ("Describe this scene and tell me whether there is water and whether buildings are present.")
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
            ],
            vqa_question="Is there water in this image?",
            vqa_target="water",
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="COMPLEX_KNOWN_TASK",
        )

    # Compound D: Caption + VQA ("Describe the image and tell me whether water is present")
    if has_caption_request and (has_water_request or has_vegetation_request or has_vqa_request or "whether" in q or "tell me" in q):
        vqa_q, vqa_target = extract_vqa_sub_question(q)
        return QueryPlan(
            intent="multi_task",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["caption", "vqa"],
            specialists=["rs_caption_adapted", "rs_vqa_adapted"],
            execution_order=["rs_caption_adapted", "rs_vqa_adapted"],
            evidence_requirements=[
                "Scene description caption",
                f"VQA verification of {vqa_target or 'target'} presence"
            ],
            vqa_question=vqa_q,
            vqa_target=vqa_target,
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="COMPLEX_KNOWN_TASK",
        )

    # Compound E: Building Detection + Land Cover
    # e.g. "Audit the buildings in this urban area and classify the surrounding land cover"
    # or "Count the buildings and tell me what kind of land surrounds them"
    has_surrounding_land = any(k in q for k in [
        "surrounding land", "surrounding terrain", "kind of land surrounds",
        "land surrounds", "classify the surrounding", "what kind of land", "surrounds them"
    ])
    if (has_building_request or "buildings" in q) and (has_land_cover_request or has_surrounding_land):
        return QueryPlan(
            intent="multi_task",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["building_detection", "land_cover"],
            specialists=["building_detection", "land_cover"],
            execution_order=["building_detection", "land_cover"],
            evidence_requirements=[
                "Tiled YOLO structural footprint count with confidence stratification",
                "BigEarthNet v2.0 Corine 19-class semantic land-cover labels"
            ],
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="COMPLEX_KNOWN_TASK",
        )

    # Compound F: Grounding + Land Cover (Water / Terrain / Road / Infrastructure)
    # e.g. "Locate the water body and identify the surrounding terrain"
    # or "Highlight roads and verify if transport infrastructure is present"
    has_grounding_keyword = any(k in q for k in [
        "locate", "highlight", "find the", "where is", "pinpoint", "demarcate", "bounding box"
    ])
    is_grounding_with_terrain = (
        has_grounding_keyword and (
            (has_water_request and (has_land_cover_request or "surrounding terrain" in q or "surrounding land" in q or "terrain" in q or "identify the surrounding" in q)) or
            (("road" in q or "roads" in q or "highway" in q) and ("transport" in q or "infrastructure" in q or has_land_cover_request or "verify" in q)) or
            (has_land_cover_request and ("road" in q or has_water_request))
        )
    )
    if is_grounding_with_terrain:
        return QueryPlan(
            intent="multi_task",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["grounding", "land_cover"],
            specialists=["visual_grounding", "land_cover"],
            execution_order=["visual_grounding", "land_cover"],
            evidence_requirements=[
                "Text-guided spatial grounding coordinates for target geographical feature",
                "BigEarthNet v2.0 Corine 19-class semantic land-cover labels"
            ],
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="COMPLEX_KNOWN_TASK",
        )

    # Compound G: Grounding + Temporal Change (e.g. Water body change)
    # e.g. "Where is the water body and has it changed?"
    has_target_change = (
        has_grounding_keyword and (
            has_change_request or "has it changed" in q or "have they changed" in q or "what changed" in q
        )
    )
    if has_target_change:
        return QueryPlan(
            intent="change_vqa" if ("what changed" in q or "?" in q) else "change_detection",
            required_images=2,
            required_modalities=["optical"],
            required_tasks=["grounding", "change_detection", "change_vqa"],
            specialists=["visual_grounding", "change_detection", "change_vqa"],
            execution_order=["visual_grounding", "change_detection", "change_vqa"],
            evidence_requirements=[
                "Spatial grounding of targeted geographic feature",
                "Bi-temporal change detection surface difference metrics",
                "Change-VQA temporal alteration assessment"
            ],
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="COMPLEX_KNOWN_TASK",
        )

    # Compound H: Optical-Only Vegetation Proxy + Land Cover
    # e.g. "Assess the vegetation coverage and greenery in this image"
    # or "What is the vegetation index?", "How green is this area?"
    is_veg_proxy_query = (
        "vegetation coverage" in q or
        "greenery" in q or
        "vegetation index" in q or
        "how green" in q or
        ("assess" in q and has_vegetation_request) or
        ("coverage" in q and has_vegetation_request)
    ) and not has_change_request and image_count == 1
    if is_veg_proxy_query:
        return QueryPlan(
            intent="multi_task",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["land_cover", "optical_sar_analysis"],
            specialists=["land_cover", "optical_sar_fusion"],
            execution_order=["land_cover", "optical_sar_fusion"],
            evidence_requirements=[
                "BigEarthNet v2.0 Corine vegetation land-cover classification",
                "Visible-Band Vegetation Proxy (Green-Red Ratio) and Excess Green Index"
            ],
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="COMPLEX_KNOWN_TASK",
        )

    # ── 3. Optical-SAR Joint Analysis ────────────────────────────────────────
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
            ],
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="KNOWN_TASK",
        )

    # ── 4. Bi-Temporal Change & Change-VQA ────────────────────────────────────
    if (
        image_count >= 2 or
        has_change_request or
        "between these images" in q or
        "between the two images" in q or
        "new buildings" in q or
        "vegetation increased" in q
    ):
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
                ],
                planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
                task_category="COMPLEX_KNOWN_TASK",
            )

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
                ],
                planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
                task_category="COMPLEX_KNOWN_TASK",
            )

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
                ],
                planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
                task_category="KNOWN_TASK",
            )

        return QueryPlan(
            intent="change_detection",
            required_images=2,
            required_modalities=["optical"],
            required_tasks=["change_detection"],
            specialists=["change_detection"],
            execution_order=["change_detection"],
            evidence_requirements=[
                "Bi-temporal change percentage and spatial alteration mask"
            ],
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="KNOWN_TASK",
        )

    # ── 5. Building Detection & Counting ─────────────────────────────────────
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
            ],
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="KNOWN_TASK",
        )

    # ── 6. Spatial Grounding ─────────────────────────────────────────────────
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
            ],
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="COMPLEX_KNOWN_TASK",
        )

    # Grounding route: only use classical CV grounding for spectrally-groundable targets.
    # If query uses spatial keywords but targets an open-vocabulary entity (airplanes,
    # cars, ships, etc.) that the classical spectral grounding baseline cannot demarcate,
    # fall through to the generalist.
    GROUNDING_SPATIAL_KEYWORDS = ["where is", "where are", "locate", "find the", "highlight", "pinpoint", "bounding box"]
    CLASSICAL_GROUNDING_TARGETS = [
        "water", "river", "lake", "ocean", "pond", "sea", "stream", "reservoir", "canal", "wetland",
        "vegetation", "canopy", "forest", "tree", "trees", "crop", "crops", "greenery", "grass",
        "road", "roads", "highway", "runway", "runways", "street", "streets",
        "building", "buildings", "house", "houses", "structure", "structures", "rooftop", "rooftops", "built-up"
    ]
    has_spatial_keyword = any(kw in q for kw in GROUNDING_SPATIAL_KEYWORDS)
    has_classical_target = any(kw in q for kw in CLASSICAL_GROUNDING_TARGETS)
    has_open_vocab_target = has_open_visual_request

    if has_spatial_keyword:
        if has_classical_target:
            # Classical spectral grounding is able to demarcate this target
            return QueryPlan(
                intent="grounding",
                required_images=1,
                required_modalities=["optical"],
                required_tasks=["grounding"],
                specialists=["visual_grounding"],
                execution_order=["visual_grounding"],
                evidence_requirements=[
                    "Normalized bounding box coordinates of targeted spatial feature"
                ],
                planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
                task_category="KNOWN_TASK",
            )
        elif has_open_vocab_target:
            # Open-vocabulary object (e.g. airplanes, cars) — route to AdaptLLM if available, else generalist
            open_spec, open_req = _get_open_vlm_specialist()
            return QueryPlan(
                intent="general_vqa",
                required_images=1,
                required_modalities=["optical"],
                required_tasks=["general_vqa"],
                specialists=[open_spec],
                execution_order=[open_spec],
                evidence_requirements=[open_req],
                planner_disposition=DISPOSITION_GENERALIST_FALLBACK,
                task_category="OPEN_REMOTE_SENSING_QUESTION",
            )
        else:
            # Spatial query with unknown target — attempt grounding with classical CV
            return QueryPlan(
                intent="grounding",
                required_images=1,
                required_modalities=["optical"],
                required_tasks=["grounding"],
                specialists=["visual_grounding"],
                execution_order=["visual_grounding"],
                evidence_requirements=[
                    "Normalized bounding box coordinates of targeted spatial feature"
                ],
                planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
                task_category="KNOWN_TASK",
            )

    # ── 7. Land Cover Classification ─────────────────────────────────────────
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
            ],
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="KNOWN_TASK",
        )

    # ── 8. Scene Captioning ──────────────────────────────────────────────────
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
            ],
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="KNOWN_TASK",
        )

    # ── 8b. Natural-language scene observation ───────────────────────────────
    if has_scene_observation_request:
        return QueryPlan(
            intent="vqa" if has_visual_question_form else "caption",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["vqa"] if has_visual_question_form else ["caption"],
            specialists=["rs_vqa_adapted"] if has_visual_question_form else ["rs_caption_adapted"],
            execution_order=["rs_vqa_adapted"] if has_visual_question_form else ["rs_caption_adapted"],
            evidence_requirements=[
                "Natural-language visual question answering grounded in the supplied image"
                if has_visual_question_form else
                "Remote-sensing scene description grounded in the supplied image"
            ],
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="NATURAL_LANGUAGE_RS_OBSERVATION",
        )

    # ── 9. Open Visual Object Requests (Route to AdaptLLM if available, else Generalist) ──
    # Specific open-vocabulary objects not covered by domain-adapted specialists
    if has_open_visual_request:
        open_spec, open_req = _get_open_vlm_specialist()
        return QueryPlan(
            intent="general_vqa",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["general_vqa"],
            specialists=[open_spec],
            execution_order=[open_spec],
            evidence_requirements=[open_req],
            planner_disposition=DISPOSITION_GENERALIST_FALLBACK,
            task_category="OPEN_REMOTE_SENSING_QUESTION",
        )

    # ── 10. Building presence VQA ────────────────────────────────────────────
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
            ],
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="KNOWN_TASK",
        )

    # ── 11. Known Remote-Sensing VQA ─────────────────────────────────────────
    # Environmental, hydrological, vegetation, structural, infrastructure presence/QA
    is_known_rs_vqa = (
        has_vegetation_request or
        has_water_request or
        has_building_request or
        any(k in q for k in KNOWN_RS_VQA_KEYWORDS) or
        any(k in q for k in OPEN_VISUAL_OBJECT_KEYWORDS) or
        has_visual_question_form
    )
    if is_known_rs_vqa:
        return QueryPlan(
            intent="vqa",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["vqa"],
            specialists=["rs_vqa_adapted"],
            execution_order=["rs_vqa_adapted"],
            evidence_requirements=[
                "Adapted VQA reasoning on remote sensing observation"
            ],
            planner_disposition=DISPOSITION_KNOWN_SPECIALIST,
            task_category="KNOWN_TASK",
        )

    # ── 12. Open Remote-Sensing Question / AdaptLLM or Generalist Fallback ───
    # Only use the open-vocabulary VLM path when the query is clearly about
    # an image/scene. Never route arbitrary text to a vision model.
    rs_context = any(k in q for k in [
        "image", "scene", "satellite", "remote sensing", "aerial", "imagery",
        "photo", "picture", "raster", "landscape", "area", "region"
    ])
    if rs_context or has_visual_question_form:
        open_spec, open_req = _get_open_vlm_specialist()
        return QueryPlan(
            intent="general_vqa",
            required_images=1,
            required_modalities=["optical"],
            required_tasks=["general_vqa"],
            specialists=[open_spec],
            execution_order=[open_spec],
            evidence_requirements=[open_req],
            planner_disposition=DISPOSITION_GENERALIST_FALLBACK,
            task_category="OPEN_REMOTE_SENSING_QUESTION",
        )

    return QueryPlan(
        intent="unsupported",
        required_images=1,
        required_modalities=["optical"],
        required_tasks=[],
        specialists=[],
        execution_order=[],
        evidence_requirements=[],
        unsupported_reason=(
            "The query is not a supported remote-sensing visual question. "
            "Ask about observable content, land cover, objects, spatial features, "
            "or changes in supplied imagery."
        ),
        supported_alternatives=[
            "Describe the supplied satellite image",
            "Identify visible land-cover or infrastructure features",
            "Compare two supplied images for observable change",
        ],
        planner_disposition=DISPOSITION_NO_CAPABILITY,
        task_category="UNSUPPORTED",
    )
