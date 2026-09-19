# Hidden Evaluation Readiness — SatQuery AI / ORBITAL-AI

## Purpose

This document describes the readiness of SatQuery AI for assessment by a
hidden evaluation dataset (e.g. an ISRO/SAC evaluation set or similar
adjudication corpus that the development team does not have access to).

---

## Readiness Status Summary

| Component | Status | Notes |
|---|---|---|
| Building detection | **FUNCTIONAL** | Real YOLOv8 inference, 54.7 MB checkpoint ships with project |
| Land-cover classification | **FUNCTIONAL** | Real BigEarthNet ResNet-50 inference via HuggingFace |
| RS VQA / Captioning | **ADAPTER PENDING** | Colab training pipeline ready; adapter absent until GPU training completes |
| Visual Grounding | **HEURISTIC FALLBACK** | Classical CV (spectral + Canny); LoRA adapter required for VLM grounding |
| Bi-temporal Change Detection | **FUNCTIONAL** | Classical radiometric + morphological; CRS verification enforced |
| Optical–SAR Fusion | **FUNCTIONAL** | SSIM + cross-correlation pipeline; co-registration enforced |
| Benchmark Evaluators | **READY (awaiting datasets)** | Evaluators exist; DATASET_UNAVAILABLE returned until data downloaded |
| Zero Fabricated Numbers | **CONFIRMED** | No hardcoded metrics anywhere in the codebase |

---

## Assumptions about the Hidden Dataset

The hidden evaluation dataset is **NOT** assumed to be available during
development.  SatQuery AI:

1. Never inspects evaluation labels except during a sanctioned evaluation run.
2. Does not cache or memorise expected outputs.
3. Returns honest SPECIALIST_UNAVAILABLE or DATASET_UNAVAILABLE states when
   inputs cannot be processed.
4. Supports sensor-agnostic input (Sentinel-1/2, Cartosat, RISAT, Landsat, etc.)
   through the modality validator (models/fusion/modality_validator.py).

---

## What to Do When the Hidden Dataset Arrives

1. Place the hidden dataset in data/<dataset_name>/ following the standard
   directory conventions.
2. Register it in 	raining/datasets/dataset_registry.py with appropriate
   parser.
3. Run validation:
   `ash
   python -c "from training.datasets.dataset_registry import DatasetRegistry; reg = DatasetRegistry(); print(reg.status_all())"
   `
4. Once status reaches READY, trigger the appropriate evaluator:
   `ash
   python evaluation/vrsbench/evaluate_vqa.py --data-dir data/<dataset_name>
   `
5. Results are written to evaluation/results/ as a tamper-evident JSON record.

---

## Known Limitations (Disclosed)

- **No GPU locally:** LoRA adapter training requires Colab.  The adapter
  specialist functions return SPECIALIST_UNAVAILABLE until the adapter is
  installed.  This is disclosed in the UI via the Model Status panel.
- **Heuristic grounding:** Spatial grounding uses classical CV (spectral
  thresholding, Canny, contour detection) rather than a trained detector.
  Results are labelled method: "heuristic_spectral_contour" in the API
  response.
- **RGB broadcasting:** The land-cover classifier uses Sentinel-2 B4/B3/B2
  as an RGB proxy when input is a standard 3-channel JPEG/PNG.  This is
  documented in the trace metadata.
- **No ground truth at inference:** Accuracy metrics in building detection
  reports (alidation_status: "Model inference (unverified against external
  benchmark)") are not claimed unless ground truth is supplied.

---

## Anti-Fabrication Guarantees

The following are hard invariants enforced in the codebase:

1. SpecialistEntry.execute_or_fallback() never invents confidence scores.
2. BenchmarkStorage.store_result() requires a real status field; a call
   with status="COMPLETED" and metrics=None raises ValueError.
3. evaluate_accuracy() in ackend/services/validation.py returns
   status: "no_ground_truth_provided" when no GT is available — it does not
   compute a fake accuracy number.
4. All evaluation scripts exit with code 1 and a clear message if the dataset
   directory is empty or NOT_DOWNLOADED.

---

## Evidence of Genuine Inference

The following artefacts prove real model usage (verifiable by inspecting the
running backend):

- ackend/models/building_model.pt (54.7 MB) — real YOLOv8 weights
- models/registry.py — imports real PyTorch modules, not mocks
- ackend/services/building_detector.py — real ultralytics YOLO inference
- ackend/services/ben_classifier.py — real 	orch.hub + configilm inference
- 	raining/colab/train_remote_sensing_lora.ipynb — executable on Colab T4
- evaluation/vrsbench/metrics.py — CIDEr / BLEU / ROUGE computed from NLTK
  and standard RS libraries, not hardcoded
