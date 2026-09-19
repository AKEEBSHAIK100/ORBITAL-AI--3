# SatQuery AI / ORBITAL-AI — Model Adaptation Audit

**Document Version:** 1.0.0  
**Audit Timestamp:** 2026-09-18  
**Scope:** Complete Codebase, Models, Inference Pipelines, Evaluation Interfaces & Benchmark Provenance

---

## 1. Executive Summary

This audit establishes the ground truth of the SatQuery AI / ORBITAL-AI system. Its purpose is to clearly demarcate:
1. What is genuinely executed via neural weights / real inference,
2. What is executed via deterministic computer vision / signal heuristics,
3. What is placeholder / mock / template generation,
4. What is missing to make this a true domain-adapted Remote Sensing Vision-Language Model (VLM) system.

As mandated by project integrity rules, **no benchmark figures, model checkpoints, or training completions are fabricated**.

---

## 2. Component-by-Component Identification & Status

### A. Frontend
- **Location:** `src/App.tsx`, `src/components/`, `src/pages/`
- **Architecture:** React 19 + Vite + Tailwind CSS v4 + Three.js globe.
- **Current State:** Functional UI with analysis console, satellite telemetry displays, evaluation criteria modal (`EvaluationCriteriaModal.tsx`), and trace visualization modal (`AgentTraceModal.tsx`).
- **Benchmark presentation:** Displays benchmark descriptions (RSVQA, VRSBench, CDVQA, ISRO/SAC) accurately as specifications, but does not claim live local scores.
- **Status:** **WORKING**. No unnecessary rewrite required.

### B. Node / Express API & Backend Bridge
- **Location:** `server.ts`, `api/` (`analyze.ts`, `buildings.ts`, `classify.ts`, `fuse.ts`, `compare.ts`)
- **Architecture:** Express 5 / Vercel serverless TypeScript bridge forwarding to Python FastAPI backend (`http://localhost:8000`) or serving fallback routines if the Python service is offline.
- **Status:** **WORKING**.

### C. Python FastAPI Backend
- **Location:** `backend/main.py`, `backend/routers/`
- **Registered Routes:**
  - `POST /api/analyze` (Unified multi-specialist endpoint)
  - `POST /api/analyze/buildings` (Dedicated building segmentation)
  - `POST /api/analyze/grounding` (Grounding endpoint)
  - `POST /api/analyze/change` (Bi-temporal change detection)
  - `POST /api/analyze/optical-sar` (SAR-optical fusion)
  - `POST /api/analyze/vqa` (VQA endpoint)
  - `POST /api/classify` (Land-cover classification)
  - `GET /health`, `/api/health`, `/api/models`, `/api/tools`
- **Status:** **WORKING & OPERATIONAL**.

### D. Agent Router & Orchestration
- **Location:** `backend/agents/router.py` & `lib/agentController.ts`
- **Current Mechanism:** Rule-based keyword matching on natural language query, image count, and modalities.
- **Status:** Functional deterministic router, but currently does not dynamically query specialist model capabilities, availability status, or device constraints from a unified Specialist Model Registry.

### E. Existing Model Wrappers & Inference Reality

| Component | Implementation File | Method / Underlying Engine | Reality / Integrity Status |
|---|---|---|---|
| **Building Detection & Counting** | `backend/services/building_detector.py`, `backend/tools/building_detection.py` | Ultralytics YOLOv8 instance segmentation (`building_model.pt`, 54.7 MB) with 512px sliding-window tiling and polygon IoU deduplication. | **REAL INFERENCE**. Model weights exist locally. Tiling, connected components, and polygon extraction work on CPU/CUDA. |
| **Land-Cover Classification** | `backend/services/ben_classifier.py`, `backend/tools/land_cover.py` | `BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0` (reBEN IGARSS 2025) via `configilm` + PyTorch. | **REAL INFERENCE**. Pretrained weights downloaded and cached. Operates zero-shot feature extraction on 120x120px patches. Note: Uses RGB broadcast approximation when 10-band Sentinel-2 TIFF is not provided. |
| **Remote-Sensing VQA** | `backend/tools/vqa.py` | Calls `BigEarthNetTool` + `BuildingDetectionTool` + spectral heuristics, and generates a formatted template string answer. | **TEMPLATE / HEURISTIC**. No fine-tuned vision-language model is answering questions directly. It is a composite script, not an adapted VLM. |
| **Visual Grounding** | `backend/tools/grounding.py` | Color thresholding (blue/green/gray ratios) + OpenCV Canny edge detector + contour bounding boxes. | **HEURISTIC / CLASSICAL CV**. Not a learned visual grounding specialist (e.g. no bounding box prediction head or multimodal spatial tokens). |
| **Bi-Temporal Change Detection** | `backend/tools/change_detection.py` | OpenCV absolute radiometric differencing (`cv2.absdiff`) + morphological closing + connected components + Excess Green Index (`ExG`). | **CLASSICAL CV BASELINE**. Real mathematical operations, but zero learned temporal attention or vision-language reasoning. |
| **Optical–SAR Fusion** | `backend/tools/optical_sar.py`, `backend/services/fusion_service.py` | OpenCV grayscale conversion, SSIM, cross-correlation, speckle index, Sobel filter, synthetic radar noise if only 1 image provided. | **CLASSICAL CV BASELINE**. Mathematical signal extraction. If 1 image is supplied, it generates a synthetic radar proxy (honestly disclosed in headers). Sensor-agnostic generic pipeline needs hardening to reject un-coregistered inputs. |
| **Building Evaluation** | `backend/evaluation/building_eval.py`, `backend/services/validation.py` | Calculates Precision, Recall, F1, and IoU when `ground_truth_bboxes` or `ground_truth_count` is passed by caller. If not provided, returns `has_ground_truth: false`. | **REAL EVALUATOR**. Transparently disclaims when ground truth is absent. Does not invent scores. |

### F. Database Layer
- **Location:** `backend/db/` (`connection.py`, `session_repo.py`) & `db/` (PostgreSQL / `pg` connection pool, migration scripts).
- **Status:** **OPERATIONAL**. Non-blocking fallback if PostgreSQL is not active; executes session logging and audit recording when configured. Must be preserved without large dataset blob storage.

### G. Existing Evaluation Scripts & Benchmark Results
- **Location:** `backend/evaluation/building_eval.py` is the only evaluation script present.
- **Missing:**
  - No VRSBench evaluation pipeline (`evaluate_captioning.py`, `evaluate_grounding.py`, `evaluate_vqa.py`, `metrics.py`).
  - No RSVQA evaluation pipeline (`evaluate.py`, `metrics.py`, `adapter.py`).
  - No CDVQA evaluation pipeline (`evaluate.py`, `metrics.py`).
  - No benchmark results storage directory (`evaluation/results/`).
- **Status:** Currently, benchmark figures are not fabricated, but actual evaluation scripts on public benchmarks do not yet exist.

### H. Requirements & Environment
- **Python:** 3.14.7 on Windows.
- **Hardware:** CPU-only development environment (CUDA: False).
- **Installed Packages:** `torch` 2.14.0, `ultralytics` 8.4.138, `transformers` 4.57.6, `timm` 0.9.16, `opencv-python` 5.0.0, `fastapi`, `pydantic`.
- **Missing Local Dependencies:** `peft`, `bitsandbytes`, `accelerate` (cannot train LoRA on CPU laptop efficiently).

### I. Docker Configuration
- **Status:** No Dockerfile or docker-compose.yml currently exists.

### J. Model & Checkpoint Directories
- `backend/models/building_model.pt` (54.7 MB, valid YOLOv8 weights).
- No LoRA adapter checkpoints exist yet (`checkpoints/` directory not created).

---

## 3. Detailed Component Classification Matrix

| Component | Requires Training? | Zero-Shot Capable? | Requires Dataset? | Requires GPU? | Current Implementation State |
|---|---|---|---|---|---|
| **BigEarthNet Adapter (LoRA)** | Yes (LoRA/QLoRA) | No (Base VLM can run zero-shot, but adaptation requires fine-tuning) | Yes (BigEarthNet-S2 / reBEN) | Yes (Training: CUDA GPU / Colab; Inference: CPU/remote) | Needs Creation (`training/`) |
| **VRSBench Evaluator** | No (eval script) | Evaluates zero-shot or adapted model | Yes (when running evaluation) | Recommended for large split | Needs Creation (`evaluation/vrsbench/`) |
| **RSVQA Evaluator** | No (eval script) | Evaluates zero-shot or adapted model | Yes (when running evaluation) | Recommended | Needs Creation (`evaluation/rsvqa/`) |
| **CDVQA Specialist & Evaluator** | No (eval script + bi-temporal pipeline) | Classical CV operates zero-shot; learned CDVQA requires adapter | Yes (for evaluation) | Optional for CV; Required for VLM | Needs Creation (`models/change/`, `evaluation/cdvqa/`) |
| **Optical-SAR Fusion Engine** | No (multimodal alignment & telemetry) | Yes (classical signal processing & cross-sensor telemetry) | No (accepts GeoTIFF / sensor pairs) | CPU sufficient | Needs Hardening (`models/fusion/`) |
| **Building Instance Segmentation** | Already trained (`building_model.pt`) | Yes (runs on aerial imagery) | No for inference; Yes for formal benchmark evaluation | Runs on CPU, accelerated on GPU | Working in `backend/services/building_detector.py` |
| **Specialist Model Registry** | No (software infrastructure) | Yes | No | No | Needs Formal Creation (`models/registry.py`) |
| **Agentic Capability Router** | No (routing logic) | Yes | No | No | Needs Update to use Model Registry & availability |

---

## 4. Gaps to Close in Adaptation & Evaluation Plan

1. **Dataset Adapter & Registry System (`Phase B`):**
   - No unified registry currently manages datasets or checks if local files exist, their splits, metadata, and licenses.
   - Must handle `NOT_DOWNLOADED` status gracefully without blocking application startup.

2. **BigEarthNet Adaptation Pipeline (`Phase C` & `Phase D`):**
   - Need real instruction-data generation from BigEarthNet annotations.
   - Need parameter-efficient fine-tuning script (`train_lora.py`) using configurable base model (`BASE_MODEL_ID`).
   - Need Google Colab notebook (`training/colab/train_remote_sensing_lora.ipynb`) because local development machine is CPU-only.
   - Local application must load trained adapter if supplied, or connect to remote endpoint, or degrade gracefully with `SPECIALIST_UNAVAILABLE`.

3. **Standard Benchmark Evaluators (`Phase E`, `Phase F`, `Phase G`):**
   - Need official evaluation protocol scripts for VRSBench (captioning, grounding, VQA).
   - Need RSVQA evaluation script.
   - Need CDVQA bi-temporal evaluation script.
   - If dataset not present, return `"status": "dataset_not_available"` with no fabricated numbers.
   - Write genuine runs to `evaluation/results/` with standard JSON schema.

4. **Sensor-Agnostic Co-Registration & Optical-SAR Fusion (`Phase H`):**
   - Multi-sensor GeoTIFF validator checking CRS, spatial dimensions, resolution, and pixel alignment.
   - Strict rejection of un-coregistered inputs with clear trace explanation.

5. **Specialist Model Registry & Controlled Errors (`Phase I`, `Phase J`, `Phase K`, `Phase L`):**
   - Standardized output structure with real confidence or `null` / `UNAVAILABLE`.
   - Execution trace with operational steps only.

6. **Admin / Developer Model Status Interface (`Phase O`):**
   - Visual dashboard in UI showing dataset presence, model availability, adapter status, and real evaluation results.

7. **Sensor-Agnostic Hidden Evaluation Readiness (`Phase P`):**
   - Support generic TIFF/GeoTIFF (Cartosat, RISAT, Sentinel, airborne) through identical validation interface without mocking ISRO data.

8. **Comprehensive Documentation & Unit Tests (`Phase Q`, `Phase R`):**
   - Full docs in `docs/` and test suites in `tests/`.
