# ORBITAL-AI — Implementation Status

Updated: 2026-09-26

This file is the living implementation status for the remote-sensing prototype. A capability is marked **done** only when the repository contains the implementation and its runtime contract is explicit. A model artifact on disk is not treated as executable availability until its runtime can be verified.

## Core routing and honesty
- **Done:** deterministic natural-language task routing with unsupported-query boundaries.
- **Done:** specialist runtime failures propagate as unavailable; no synthetic answer is generated when required evidence is missing.
- **Done:** confidence is not invented when a specialist does not provide calibrated confidence.
- **Done:** audit/execution trace support is retained in the orchestration path.

## Single-image analysis
- **Done:** real BLIP caption/VQA adapter paths exist with BigEarthNet-derived LoRA artifacts.
- **Done:** BLIP adapter configs target verified attention modules and reject the stale `dense` target.
- **Done:** PEFT loading plus real generation smoke tests are required before BLIP is marked AVAILABLE.
- **Done:** ambiguous multispectral input is rejected instead of silently dropping bands.
- **Done:** Sentinel-2 10-band input can be explicitly converted to B04/B03/B02 natural-colour RGB for BLIP.
- **Scoped-down:** BLIP outputs are uncalibrated generative text; no benchmark accuracy is claimed.
- **Blocked for runtime verification:** the Git repository intentionally excludes `adapter_model.safetensors`; both current BLIP adapter directories contain config files only. Real pilot weights must be mounted/copied through the documented adapter paths before PEFT + inference verification can occur.

## GeoTIFF / sensor integrity
- **Done:** multi-band GeoTIFF/TIFF ingestion preserves H×W×bands arrays.
- **Done:** raster metadata includes band count, CRS/geotransform when supplied, sensor tags when supplied, and band descriptions when supplied.
- **Done:** ordinary RGB images retain the legacy BGR path for OpenCV consumers.
- **Done:** TIFF decode failure does not silently fall back to RGB.

## Land cover
- **Done:** classifier failure is surfaced as unavailable; no heuristic land-cover label is fabricated.
- **Scoped-down:** actual pretrained BigEarthNet execution depends on the model/dependency being installed and runtime-verified.

## Buildings / grounding
- **Done:** building/grounding outputs no longer invent confidence, validation status, targets, or coordinates when evidence is absent.
- **Scoped-down:** detector confidence is a model score, not an externally validated accuracy percentage.

## Bi-temporal change
- **Done:** classical change-analysis baseline with input compatibility checks.
- **Done:** no silent image resizing to force mismatched pairs.
- **Scoped-down:** vegetation/structural outputs are image-derived proxies, not independently validated land-change ground truth.

## Optical + SAR
- **Done:** real SAR input is required for joint optical-SAR analysis; silent optical-to-SAR substitution is not permitted.
- **Done:** registration/alignment claims are conditional on available metadata/verification.
- **Scoped-down:** classical fusion telemetry is not sensor-calibrated SAR backscatter.

## External specialist path
- **Scoped-down:** free Hugging Face ZeroGPU remote-sensing specialist is an external runtime dependency, not an ORBITAL-AI-owned model benchmark.
- **Unavailable until runtime success:** local AdaptLLM/Qwen2-VL availability is based on executable weights/runtime, not model-name configuration alone.

## Validation status
- Added focused multispectral/BLIP integrity regression coverage; tests have not yet been executed in this environment. Earlier reported BLIP test counts are historical and must not be treated as current validation.
- Latest full-project test/build gate: **not yet rerun after the current integrity changes**.

## Explicit non-goals
- No fabricated model outputs.
- No fake confidence or benchmark scores.
- No synthetic SAR presented as real SAR.
- No claim of calibrated VLM confidence.
- No deployment in the current work phase.
