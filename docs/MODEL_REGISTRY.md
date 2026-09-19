# Model Registry — SatQuery AI / ORBITAL-AI

## Overview

The Model Registry (models/registry.py) is the single source of truth for
all specialist engines available in SatQuery AI.  It declares each specialist,
checks checkpoint availability at startup, and provides controlled fallback
responses when a specialist is unavailable.

---

## Registry API

`python
from models.registry import ModelRegistry

reg = ModelRegistry.get_instance()

# List all registered specialists
specialists = reg.list_specialists()

# Get a single specialist by ID
spec = reg.get_specialist("rs_vlm_vqa")

# Execute or return SPECIALIST_UNAVAILABLE
result = spec.execute_or_fallback(image=img_array, query="What land cover is visible?")
`

---

## Registered Specialists

| ID | Name | Task | Checkpoint | Status |
|---|---|---|---|---|
| uilding_seg | YOLOv8 Building Segmentation | uilding_detection | ackend/models/building_model.pt | **AVAILABLE** (54.7 MB, ships with project) |
| land_cover_cls | BigEarthNet ResNet-50 Land Cover Classifier | land_cover_classification | Auto-downloaded via HuggingFace Hub (BIFOLD/BigEarthNetv2-0) | **AVAILABLE** (requires internet at first run) |
| s_vlm_caption | Remote-Sensing VLM Caption Specialist | captioning | checkpoints/rs_vlm_adapter/ | **UNAVAILABLE** until Colab training completes |
| s_vlm_vqa | Remote-Sensing VLM VQA Specialist | qa | checkpoints/rs_vlm_adapter/ | **UNAVAILABLE** until Colab training completes |
| s_grounding | Remote-Sensing Visual Grounding | grounding | checkpoints/rs_vlm_adapter/ | **UNAVAILABLE** (heuristic fallback active) |
| change_detector | Bi-Temporal Change Detector | change_detection | Classical CV pipeline (no checkpoint) | **AVAILABLE** |
| change_vqa | CDVQA Change Question Answering | change_vqa | checkpoints/rs_vlm_adapter/ | **UNAVAILABLE** until adapter installed |
| sar_optical_fusion | Optical–SAR Sensor Fusion | sar_optical_fusion | Classical CV pipeline (no checkpoint) | **AVAILABLE** |

---

## Specialist Entry Schema

Each specialist is declared as a SpecialistEntry dataclass:

`python
@dataclass
class SpecialistEntry:
    id: str                        # Unique identifier
    name: str                      # Human-readable name
    task: str                      # Task category
    model_id: str                  # HuggingFace or local model identifier
    version: str                   # Semantic version
    modality: List[str]            # ["optical"], ["sar"], or ["optical", "sar"]
    supported_input_types: List[str]   # ["image", "image_pair", "text"]
    checkpoint_location: Optional[str] # Path to weights file or HF hub ID
    is_available: bool             # Computed at startup
    unavailable_reason: Optional[str]  # Shown in SPECIALIST_UNAVAILABLE response
    inference_fn: Optional[Callable]   # Real inference function or None
`

---

## Availability Check Logic

Availability is determined at ModelRegistry.get_instance() time (i.e. at
FastAPI startup) by the _check_availability() method in each specialist entry.
The check is purely filesystem-based for local checkpoints:

`python
adapter_path = Path("checkpoints/rs_vlm_adapter/adapter_config.json")
is_available = adapter_path.exists()
`

For HuggingFace Hub models, the registry checks the local cache directory.

---

## SPECIALIST_UNAVAILABLE Response Schema

Whenever a specialist is unavailable, execute_or_fallback() returns:

`json
{
  "status": "SPECIALIST_UNAVAILABLE",
  "answer": "Analysis unavailable: <reason>",
  "confidence": null,
  "confidence_level": "UNAVAILABLE",
  "evidence": null,
  "warnings": ["<reason>"],
  "model": "<specialist name>",
  "model_version": "<version>",
  "task": "<task>",
  "specialist_id": "<id>"
}
`

---

## Adding a New Specialist

1. Define an inference_fn in models/<task>/:

`python
def my_specialist_fn(image: np.ndarray, query: str, **kwargs) -> dict:
    ...
    return {"answer": "...", "confidence": 0.87, "confidence_level": "High"}
`

2. Register it in ModelRegistry._build_registry():

`python
SpecialistEntry(
    id="my_specialist",
    name="My Custom Specialist",
    task="custom_task",
    model_id="org/model-name",
    version="0.1.0",
    modality=["optical"],
    supported_input_types=["image", "text"],
    checkpoint_location="checkpoints/my_model/",
    is_available=Path("checkpoints/my_model/config.json").exists(),
    inference_fn=my_specialist_fn,
)
`

3. Add the task to ackend/agents/router.py intent classifier.

4. Add a test case in 	ests/test_model_adaptation.py.

---

## API Endpoint

`
GET /api/models
GET /api/model-status
`

Returns the full list of specialists with their is_available state, version,
and unavailable reason (if applicable).
