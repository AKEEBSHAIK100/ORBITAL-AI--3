# SatQuery AI — Model Provenance & Attribution

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

## 3. Optical–SAR Cross-Modal Fusion

| Field | Detail |
|---|---|
| **Algorithm** | Classical computer vision (no learned weights) |
| **Optical Features** | NDVI vegetation proxy (red/near-IR channel ratio approximation on RGB), water fraction (blue channel dominance), built-up fraction (complement) |
| **SAR Features** | Mean backscatter (dB), standard deviation backscatter, speckle index (σ/μ), Sobel edge density |
| **Cross-Modal** | Structural Similarity Index (SSIM), normalized cross-correlation, complementarity index |
| **Library** | OpenCV 4.x |
| **Synthetic SAR Proxy** | When no real SAR image is provided, a synthetic radar approximation is generated from optical gradient magnitude + speckle noise (physically calibrated but not geometrically registered SAR). Responses flagged with `"mode": "synthetic_proxy"`. |

---

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

- **RGB as proxy for multispectral**: BigEarthNet expects 10-band Sentinel-2 imagery. RGB-only input is a significant approximation.
- **Model availability**: `building_model.pt` is not included in this repository due to file size. Place a YOLO weights file trained on aerial building footprint data at `backend/models/building_model.pt` or set the `BUILDING_MODEL_PATH` environment variable.
- **Synthetic fallbacks**: When the specialist vision model is unavailable, the system may return heuristic estimates labelled with `"mode": "synthetic_fallback"`. These are explicitly disclosed in API responses and the UI.
- **SAR mode**: The application simulates SAR analysis from single optical images when no real SAR data is provided. This is disclosed via `"sar_synthetic": true` in fusion API responses.
