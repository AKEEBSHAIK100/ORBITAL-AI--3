# SatQuery AI — Remote-Sensing Vision-Language Model Benchmark

## Overview
This module provides an isolated, reproducible benchmarking harness to evaluate remote-sensing-native Vision-Language Models (VLMs) against the SatQuery generalist baseline:

1. **GeoChat-7B** (Candidate A — Remote-Sensing Native Visual-Language Reasoning)
2. **SkyEyeGPT** (Candidate B — Remote-Sensing Multimodal Instruction Tuning)
3. **Qwen2-VL-2B-Instruct** (Baseline Candidate C — General Multimodal Foundation Model)

> [!IMPORTANT]
> **Evaluation Phase Only**:
> This package does **not** alter production routing, agentic planning, specialist priority, or frontend behaviors.
> No multi-GB model checkpoints are downloaded automatically during application startup or unit testing.

---

## Benchmark Structure

```
backend/evaluation/
    __init__.py              # Package exports
    schemas.py               # Normalized evaluation dataclasses & schemas
    model_adapters.py        # Unified RemoteSensingVLMAdapter implementations
    question_set.py          # Deterministic 40-question test suite & image registry
    evaluator.py             # Sequential runner with single-candidate lifecycle
    report.py                # Neutral Markdown and JSON comparison reporter
    rs_vlm_benchmark.py      # CLI runner
    building_eval.py         # SpaceNet building detection evaluation (pre-existing)
    README.md                # This documentation
```

---

## 40-Question Evaluation Suite
The evaluation suite contains exactly 40 deterministic questions across 20 remote-sensing categories (2 questions each):
1. **Scene Description** (`RS-SCN-01`, `RS-SCN-02`)
2. **Land Cover** (`RS-LND-01`, `RS-LND-02`)
3. **Vegetation** (`RS-VEG-01`, `RS-VEG-02`)
4. **Water** (`RS-WTR-01`, `RS-WTR-02`)
5. **Agriculture** (`RS-AGR-01`, `RS-AGR-02`)
6. **Buildings** (`RS-BLD-01`, `RS-BLD-02`)
7. **Infrastructure** (`RS-INF-01`, `RS-INF-02`)
8. **Roads** (`RS-ROD-01`, `RS-ROD-02`)
9. **Ships** (`RS-SHP-01`, `RS-SHP-02`)
10. **Aircraft** (`RS-AIR-01`, `RS-AIR-02`)
11. **Vehicles** (`RS-VEH-01`, `RS-VEH-02`)
12. **Spatial Location** (`RS-LOC-01`, `RS-LOC-02`)
13. **Spatial Relationships** (`RS-REL-01`, `RS-REL-02`)
14. **Urban Scene Reasoning** (`RS-URB-01`, `RS-URB-02`)
15. **Rural Scene Reasoning** (`RS-RUR-01`, `RS-RUR-02`)
16. **Captioning** (`RS-CAP-01`, `RS-CAP-02`)
17. **Visual Grounding** (`RS-GND-01`, `RS-GND-02`)
18. **SAR Understanding** (`RS-SAR-01`, `RS-SAR-02`)
19. **Temporal Change Reasoning** (`RS-CHG-01`, `RS-CHG-02`)
20. **Open-Vocabulary RS Reasoning** (`RS-OPN-01`, `RS-OPN-02`)

### Imagery Used
- Optical aerial: `backend/data/default_aerial.jpg`
- Bi-temporal LEVIR-CD pair: `backend/data/levir_cd/A/test_2_0000_0000.png` & `backend/data/levir_cd/B/test_2_0000_0000.png`
- Optical & SAR unaligned pair: `backend/data/optical_sar/optical.tif` & `backend/data/optical_sar/sar.tif` (with explicit geospatial dimension checks; no silent resizing)

---

## Running the Benchmark CLI

### 1. List Candidate Models & Provenance
```powershell
python -m backend.evaluation.rs_vlm_benchmark --list-models
```

### 2. Check Weight Availability & Environment
```powershell
python -m backend.evaluation.rs_vlm_benchmark --check
```

### 3. Dry-Run Evaluation (No Heavy Weights)
```powershell
python -m backend.evaluation.rs_vlm_benchmark --all --dry-run
```

### 4. Evaluate Specific Candidate
```powershell
python -m backend.evaluation.rs_vlm_benchmark --model geochat_7b
python -m backend.evaluation.rs_vlm_benchmark --model skyeyegpt
python -m backend.evaluation.rs_vlm_benchmark --model qwen2_vl_2b
```

### 5. Generate Availability & Readiness Report
```powershell
python -m backend.evaluation.rs_vlm_benchmark --report --format md
```

---

## Installing Weights for Actual Bake-Off Inference

To run live inference with candidates:

### GeoChat-7B
1. Download weights from HuggingFace (`MBZUAI/geochat-7b`).
2. Place files into `backend/models/evaluation/geochat_7b/` or set `GEOCHAT_WEIGHTS_PATH`.

### SkyEyeGPT
1. Download weights from HuggingFace (`Sun-Y/SkyEyeGPT`).
2. Place files into `backend/models/evaluation/skyeyegpt/` or set `SKYEYEGPT_WEIGHTS_PATH`.

### Qwen2-VL-2B-Instruct
1. Download weights from HuggingFace (`Qwen/Qwen2-VL-2B-Instruct`).
2. Place files into `backend/models/generalist/qwen2_vl/` or set `QWEN2_VL_WEIGHTS_PATH`.

---

## Manual Review & Hallucination Grading
All generated responses must be graded on the 0-3 scale:
- `correctness`: (0 = incorrect, 1 = partially useful, 2 = good, 3 = strong)
- `relevance`: (0-3)
- `completeness`: (0-3)
- `rs_terminology`: (0-3)
- `spatial_correctness`: (0-3)
- `hallucination_observed`: (bool)
- `unsupported_claim`: (bool)
- `uncertainty_appropriate`: (bool)
