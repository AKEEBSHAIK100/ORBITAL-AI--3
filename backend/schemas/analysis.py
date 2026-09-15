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
    confidence: float
    confidence_level: str
    validation_status: str
    validation: Dict[str, Any]
    detections: List[BuildingDetectionItem]
    geojson: Optional[Dict[str, Any]] = None

class TraceStep(BaseModel):
    step: int
    tool: str
    description: str
    input_summary: str
    output_summary: str
    duration_ms: float
    status: str
    success: bool = True
    confidence_source: str = "heuristic"
    parameters: Optional[Dict[str, Any]] = None

class ObservableTrace(BaseModel):
    agent_version: str = "SatQuery-Agent-v3.0"
    task_type: str
    tools_invoked: List[str]
    steps: List[TraceStep]
    total_duration_ms: float
    input_validation: Dict[str, Any]
    model_registry_entry: Dict[str, Any]

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

class UnifiedAnalysisResponse(BaseModel):
    success: bool = True
    task_type: str
    answer: str
    confidence: float
    confidence_level: str  # High, Medium, Low
    tools_used: List[str]
    input_modality: str
    timestamp: str
    execution_time_ms: float
    evidence: Dict[str, Any] = Field(default_factory=dict)
    execution_trace: ObservableTrace
    warnings: List[str] = Field(default_factory=list)
    building_analysis: Optional[BuildingAnalysisResponse] = None
    grounding: Optional[List[GroundingItem]] = None
    change_map: Optional[Dict[str, Any]] = None
    fusion_metrics: Optional[Dict[str, Any]] = None
    land_cover: Optional[Dict[str, Any]] = None
    mode: Optional[str] = None  # "model" | "synthetic_fallback" | "demo_scene"
