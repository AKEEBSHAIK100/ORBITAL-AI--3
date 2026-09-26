# ORBITAL-AI — Model Provenance & Attribution

This document describes the exact model weights, datasets, and algorithms used by SatQuery AI. It is committed to the repository for transparency, reproducibility, and academic compliance.

---

## 1. Building Instance Segmentation

| Field | Detail |
|---|---|
| **Model Architecture** | YOLOv8 / Ultralytics YOLO (instance segmentation head) |
| **Weights File** | `backend/models/building_model.pt` |
| **Training Dataset** | SpaceNet v2 (satellite aerial imagery, ~800k building footprint annotations) |
| **Inference Mode** | Tiled 640 px sliding-window inference with 20% overlap; polygon IoU deduplication (NMS 0.45 threshold) |
| **Reported Metric** | Confidence scores from YOLO detection head; no independent ground-truth validation unless `ground_truth_count` is provided by the caller |
| **Disclaimer** | Reported accuracy metrics (confidence_level: High/Medium/Low) are model confidence distributions from the YOLO head, **not** externally validated against a held-out benchmark. Ground-truth comparison is optional and caller-supplied. |
| **License** | Ultralytics AGPL-3.0 (inference); SpaceNet dataset terms: [SpaceNet License](https://spacenet.ai/licenses/) |

---

## 2. Land-Cover Classification (BigEarthNet)

| Field | Detail |
|---|---|
| **Model Architecture** | ResNet-50 (pretrained on Sentinel-2 multispectral imagery) |
| **HuggingFace Model ID** | `BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0` |
| **Pretrained On** | BigEarthNet v2.0 (reBEN) — 549,488 Sentinel-2 patches, 19 Corine Land Cover classes |
| **Reference** | Clasen et al., "reBEN: Refined BigEarthNet Dataset for Remote Sensing Image Analysis", IGARSS 2025 |
| **Usage in SatQuery AI** | Used as a **pretrained feature extractor / classifier**. The model was NOT fine-tuned by this project. It is applied zero-shot / inference-only on user-supplied optical images resized to 120×120 px and processed as a proxy for the 10-band Sentinel-2 input. |
| **Input Adaptation Note** | The model expects 10-band Sentinel-2 multi-spectral input. RGB images are broadcast to 10 channels by repeating RGB bands — this is a domain approximation, not spectrally accurate Sentinel-2 data. Results are indicative, not ground-truth land cover classifications. |
| **License** | MIT (model weights); Sentinel-2 data: Copernicus Open Access Hub |

---

## 2.1 Remote Sensing Adapted Captioning (`rs_caption_adapted`)

| Field | Detail |
|---|---|
| **Base Model Architecture** | Salesforce BLIP (`Salesforce/blip-image-captioning-base`) |
| **Adapter Architecture** | PEFT LoRA pilot adapter (`backend/models/adapters/blip_rs_lora`) |
| **Adaptation Dataset** | BigEarthNet-S2 subset: 87 image-caption pairs |
| **Training Hyperparameters** | 3 epochs, AdamW optimizer, lr=5e-5, rank r=8, alpha=16 |
| **Scope & Integrity** | **Pilot domain adaptation proof-of-concept only.** This adapter demonstrates domain conditioning of general vision-language backbones onto remote sensing vocabulary. It has **NOT** been evaluated on external benchmarks such as VRSBench. No benchmark superiority is claimed. |
| **Confidence Protocol** | VLM text outputs are uncalibrated generative language; confidence is reported as `null` (`confidence_status: "not_calibrated"`). Hardcoded heuristic confidence scores are strictly disallowed. |
| **License** | Base model: BSD 3-Clause; LoRA adapter: MIT |

---

## 2.2 Remote Sensing Adapted Visual Question Answering (`rs_vqa_adapted`)

| Field | Detail |
|---|---|
| **Base Model Architecture** | Salesforce BLIP VQA (`Salesforce/blip-vqa-base`) |
| **Adapter Architecture** | PEFT LoRA pilot adapter (`backend/models/adapters/blip_vqa_rs_lora`) |
| **Adaptation Dataset** | BigEarthNet-S2 subset: 551 training QA pairs (69 patches), 144 validation QA pairs (18 patches) |
| **Training Hyperparameters** | 3 epochs, AdamW optimizer, lr=5e-5, rank r=8, alpha=16 |
| **Scope & Integrity** | **Pilot domain adaptation proof-of-concept only.** Evaluates domain-specific visual QA capability over multi-spectral terrain and infrastructure. It has **NOT** been evaluated on RSVQA or VRSBench held-out test splits. No benchmark superiority is claimed. |
| **Confidence Protocol** | VLM text outputs are uncalibrated generative language; confidence is reported as `null` (`confidence_status: "not_calibrated"`). Measurable evidence from secondary tools (e.g. YOLO building detection polygons) is retained in `evidence`. |
| **License** | Base model: BSD 3-Clause; LoRA adapter: MIT |

---

## 3. Optical–SAR Cross-Modal Fusion

| Field | Detail |
|---|---|
| **Algorithm** | Classical computer vision baseline (no learned fusion weights) |
| **Optical Features** | Image-derived RGB proxies only; no calibrated multispectral index is claimed from RGB-only input |
| **SAR Features** | Raw SAR intensity/edge statistics when a real SAR input is supplied; calibration is not inferred |
| **Cross-Modal** | SSIM/NCC-style structural comparisons only when input compatibility/registration checks pass |
| **Synthetic SAR** | **Not a production fallback.** The current integrity policy rejects silent optical-to-SAR substitution. |

## 4. Change Detection (Bi-Temporal)

| Field | Detail |
|---|---|
| **Algorithm** | Radiometric differencing + connected components (classical CV) |
| **Reference** | Based on CDVQA dataset protocol (change detection visual question answering) |
| **Library** | OpenCV 4.x, NumPy |
| **Output** | Change percentage, cluster count, change magnitude |

---

## 5. Compliance & Affiliation Disclaimer

> **SatQuery AI is an independent research prototype developed as an academic project.**
>
> It is **NOT** affiliated with, endorsed by, or representing the Indian Space Research Organisation (ISRO), the Space Applications Centre (SAC), or any other governmental or institutional body.
>
> References to Cartosat-2S, RISAT-1A, or other ISRO satellite products are used solely as illustrative examples of sensor types in the application's user interface. No actual satellite data from ISRO systems is bundled in this repository.

---

## 6. Known Limitations

- **BLIP multispectral boundary**: Sentinel-2 multi-band inputs are preserved through GeoTIFF ingestion and converted explicitly to B04/B03/B02 natural-colour RGB for BLIP. Other ambiguous multi-band arrays are rejected rather than silently truncated.
- **BLIP runtime availability**: Adapter files alone do not make the specialist available. PEFT loading and a real generation smoke test must succeed at runtime before the adapter is marked AVAILABLE.
- **Land-cover availability**: A configured model identifier is not treated as local availability; the classifier must have an executable model/runtime.
- **Model confidence**: Generative VLM confidence is uncalibrated and remains null. Detector confidence scores are model scores, not independent accuracy validation.
- **No synthetic intelligence fallback**: When a required specialist is unavailable, ORBITAL-AI returns an explicit unavailable result rather than inventing a result.
