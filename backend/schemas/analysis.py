from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ImageInput(BaseModel):
    data: str = Field(..., description="Base64 encoded image or URL")
    filename: Optional[str] = None
    modality: Optional[str] = Field("optical", description="optical, sar, or multispectral")


class AnalyzeRequest(BaseModel):
    query: str = Field("Analyze satellite imagery", description="Natural language question or instruction")
    image: Optional[str] = Field(None, description="Primary satellite image (base64 or URL)")
    secondary_image: Optional[str] = Field(None, description="Optional secondary image for bi-temporal or optical-SAR fusion")
    images: Optional[List[str]] = Field(None, description="List of image data strings")
    modality: Optional[str] = "optical"
    secondary_modality: Optional[str] = None
    task_type: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


# ─── 1. Query Plan ────────────────────────────────────────────────────────────

class QueryPlan(BaseModel):
    intent: str = Field(..., description="caption, vqa, land_cover, grounding, building_detection, change_detection, change_vqa, optical_sar_analysis, multi_task, unsupported")
    required_images: int = Field(1, description="1 or 2 images required")
    required_modalities: List[str] = Field(default_factory=lambda: ["optical"], description="e.g. ['optical'], ['optical', 'sar']")
    required_tasks: List[str] = Field(default_factory=list, description="Remote sensing tasks to be executed")
    specialists: List[str] = Field(default_factory=list, description="Registered specialist IDs")
    execution_order: List[str] = Field(default_factory=list, description="Sequence of specialist or validation steps")
    evidence_requirements: List[str] = Field(default_factory=list, description="Description of evidence needed to answer query")
    unsupported_reason: Optional[str] = None
    supported_alternatives: Optional[List[str]] = None
    vqa_question: Optional[str] = Field(None, description="Focused question for VQA specialist in multi-task workflows")
    vqa_target: Optional[str] = Field(None, description="Target entity/phenomenon queried by VQA, e.g. vegetation, water, buildings")
    planner_disposition: str = Field(
        "Known specialist available",
        description="'Known specialist available' | 'Generalist fallback required' | 'No supported visual capability'"
    )
    task_category: str = Field(
        "KNOWN_TASK",
        description="KNOWN_TASK | COMPLEX_KNOWN_TASK | UNSUPPORTED | OPEN_REMOTE_SENSING_QUESTION"
    )


# ─── 6. Specialist Evidence Object ────────────────────────────────────────────

class SpecialistEvidenceObject(BaseModel):
    task: str = Field(..., description="Task name e.g. building_detection, land_cover, change_detection")
    result: Any = Field(..., description="Core output or text result from specialist")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Detailed metrics, counts, or findings")
    source: str = Field(..., description="Specialist tool ID")
    model: str = Field(..., description="Model identifier or engine name")
    provenance: Any = Field(None, description="Training provenance, checkpoints, or citation")
    confidence: Optional[float] = Field(None, description="Calibrated score or None if uncalibrated")
    confidence_status: str = Field("not_calibrated", description="calibrated, not_calibrated, unavailable")
    warnings: List[str] = Field(default_factory=list, description="Tool-specific operational warnings")


# ─── Building Detection Schemas ───────────────────────────────────────────────

class BuildingDetectionItem(BaseModel):
    id: str
    confidence: float
    confidence_tier: str  # high, medium, low
    bbox: List[float]  # [x1, y1, x2, y2]
    bbox_pct: List[float]  # [x_pct, y_pct, w_pct, h_pct]
    polygon: List[List[float]]
    polygon_pct: List[List[float]]
    centroid: List[float]
    centroid_pct: List[float]
    area: float
    is_partial: bool
    touches_border: bool = False


class BuildingAnalysisResponse(BaseModel):
    success: bool = True
    image_dimensions: Dict[str, int]
    tiles_processed: int
    raw_detections_count: int
    merged_detections_count: int
    building_count: int
    high_confidence_count: int
    medium_confidence_count: int
    low_confidence_count: int
    partial_count: int
    confidence: Optional[float] = None
    confidence_level: Optional[str] = None
    validation_status: str
    validation: Dict[str, Any]
    detections: List[BuildingDetectionItem]
    geojson: Optional[Dict[str, Any]] = None


# ─── 9. Observable Execution Trace ───────────────────────────────────────────

class TraceStep(BaseModel):
    step: int
    tool: str
    description: str
    input_summary: str
    output_summary: str
    duration_ms: float
    status: str
    success: bool = True
    confidence_source: str = "none"
    parameters: Optional[Dict[str, Any]] = None


class ObservableTrace(BaseModel):
    agent_version: str = "OrbitalAI-Agent-v3.0"
    task_type: str
    tools_invoked: List[str]
    steps: List[TraceStep]
    total_duration_ms: float
    input_validation: Dict[str, Any]
    model_registry_entry: Dict[str, Any]


# ─── Visual Grounding Schemas ────────────────────────────────────────────────

class GroundingRegion(BaseModel):
    x_percent: float
    y_percent: float
    w_percent: float
    h_percent: float


class GroundingItem(BaseModel):
    target: str
    region: GroundingRegion
    confidence: float
    label: str


# ─── 10. Final Response Structure ─────────────────────────────────────────────

class UnifiedAnalysisResponse(BaseModel):
    status: str = Field("SUCCESS", description="SUCCESS | VALIDATION_ERROR | SPECIALIST_UNAVAILABLE | UNSUPPORTED_QUERY | ERROR")
    answer: str = Field(..., description="Synthesized natural language answer strictly grounded in specialist evidence")
    query_plan: Optional[QueryPlan] = Field(None, description="Structured query interpretation and execution sequence")
    evidence: Any = Field(default_factory=list, description="Structured evidence objects from each executed specialist")
    visual_evidence: Dict[str, Any] = Field(default_factory=dict, description="Visual annotations: bounding boxes, footprints, change masks, telemetry")
    warnings: List[str] = Field(default_factory=list, description="Operational warnings or unverified co-registration notices")
    confidence: Optional[Union[float, Dict[str, Any]]] = None
    confidence_status: Optional[str] = Field("not_calibrated", description="calibrated, not_calibrated, unavailable")
    execution_trace: ObservableTrace

    # Backwards compatibility fields for existing UI components
    success: bool = True
    task_type: Optional[str] = None
    confidence_level: Optional[str] = None
    confidence_source: Optional[str] = None
    tools_used: List[str] = Field(default_factory=list)
    input_modality: Optional[str] = "optical"
    timestamp: Optional[str] = None
    execution_time_ms: Optional[float] = 0.0
    building_analysis: Optional[BuildingAnalysisResponse] = None
    grounding: Optional[List[GroundingItem]] = None
    change_map: Optional[Dict[str, Any]] = None
    fusion_metrics: Optional[Dict[str, Any]] = None
    land_cover: Optional[Dict[str, Any]] = None
    mode: Optional[str] = None
