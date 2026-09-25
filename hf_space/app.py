import io
import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import requests
import spaces
import torch
from PIL import Image, ImageChops, ImageFilter
from peft import PeftModel
from transformers import (
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    BlipProcessor,
)

REPO_RAW = "https://raw.githubusercontent.com/AKEEBSHAIK100/ORBITAL-AI--3/main/backend/models/adapters"
CACHE_ROOT = Path("/tmp/orbital_adapters")

_vqa_processor = None
_vqa_model = None
_caption_processor = None
_caption_model = None


def _download_adapter(name: str) -> Path:
    target = CACHE_ROOT / name
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        path = target / filename
        if not path.exists():
            response = requests.get(f"{REPO_RAW}/{name}/{filename}", timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
    return target


def _load_vqa():
    global _vqa_processor, _vqa_model
    if _vqa_model is not None:
        return _vqa_processor, _vqa_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32
    )
    _vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    base = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base", torch_dtype=dtype
    )
    adapter = PeftModel.from_pretrained(base, str(_download_adapter("blip_vqa_rs_lora")))
    _vqa_model = adapter.to(device).eval()
    return _vqa_processor, _vqa_model


def _load_caption():
    global _caption_processor, _caption_model
    if _caption_model is not None:
        return _caption_processor, _caption_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32
    )
    _caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    base = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=dtype
    )
    adapter = PeftModel.from_pretrained(base, str(_download_adapter("blip_rs_lora")))
    _caption_model = adapter.to(device).eval()
    return _caption_processor, _caption_model


def _image(value: Any) -> Image.Image:
    if value is None:
        raise ValueError("An image is required.")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    return Image.open(value).convert("RGB")


def _unavailable(task: str, reason: str) -> dict:
    return {
        "ok": False,
        "task": task,
        "answer": f"{task} is unavailable in the current worker: {reason}",
        "confidence": None,
        "confidence_status": "unavailable",
        "model": None,
    }


@spaces.GPU(duration=120)
def vqa(image: Any, question: str) -> dict:
    started = time.perf_counter()
    image = _image(image)
    question = (question or "").strip()
    if not question:
        raise ValueError("A question is required.")

    processor, model = _load_vqa()
    inputs = processor(images=image, text=question, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=50)
    answer = processor.decode(output[0], skip_special_tokens=True).strip()

    return {
        "ok": True,
        "task": "vqa",
        "answer": answer,
        "model": "Salesforce/blip-vqa-base + BigEarthNet-derived LoRA",
        "adapter": "blip_vqa_rs_lora",
        "remote_sensing_adapted": True,
        "confidence": None,
        "confidence_status": "not_calibrated",
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@spaces.GPU(duration=120)
def caption(image: Any) -> dict:
    started = time.perf_counter()
    image = _image(image)
    processor, model = _load_caption()
    inputs = processor(images=image, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=60)
    text = processor.decode(output[0], skip_special_tokens=True).strip()

    return {
        "ok": True,
        "task": "caption",
        "answer": text,
        "model": "Salesforce/blip-image-captioning-base + BigEarthNet-derived LoRA",
        "adapter": "blip_rs_lora",
        "remote_sensing_adapted": True,
        "confidence": None,
        "confidence_status": "not_calibrated",
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@spaces.GPU(duration=120)
def analyze(image: Any, query: str) -> dict:
    query = (query or "").strip()
    if not query:
        raise ValueError("A query is required.")
    q = query.lower()

    if any(k in q for k in ("describe", "caption", "scene description", "scene overview")):
        result = caption(image)
        result["routed_by"] = "worker_task_router"
        return result

    result = vqa(image, query)
    result["routed_by"] = "worker_task_router"
    return result


def _gray_array(value: Any) -> np.ndarray:
    img = _image(value).convert("L")
    return np.asarray(img, dtype=np.float32)


def change_analysis(before: Any, after: Any) -> dict:
    started = time.perf_counter()
    if before is None or after is None:
        return _unavailable("change_detection", "two corresponding images are required")

    a = _gray_array(before)
    b = _gray_array(after)
    if a.shape != b.shape:
        return _unavailable("change_detection", f"dimension mismatch: {a.shape} vs {b.shape}")

    diff = np.abs(a - b)
    threshold = 32.0
    changed = diff > threshold

    # Small morphological opening using PIL's minimum/maximum filters.
    mask = Image.fromarray((changed * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    changed = np.asarray(mask, dtype=np.uint8) > 0

    fraction = float(changed.mean())
    return {
        "ok": True,
        "task": "change_detection",
        "method": "classical grayscale absolute difference + threshold + morphological opening",
        "change_fraction_percent": round(fraction * 100, 2),
        "threshold": threshold,
        "confidence": None,
        "confidence_status": "not_calibrated",
        "geospatial_compatibility": "unverified",
        "note": "This is a classical baseline, not a trained CDVQA/change model. CRS, geotransform and GSD are unavailable from these image inputs.",
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def fusion_analysis(optical: Any, sar: Any) -> dict:
    started = time.perf_counter()
    if optical is None or sar is None:
        return _unavailable("optical_sar_analysis", "optical and SAR images are required")

    opt = _gray_array(optical)
    radar = _gray_array(sar)
    if opt.shape != radar.shape:
        return {
            "ok": False,
            "task": "optical_sar_analysis",
            "answer": "Optical-SAR joint analysis unavailable because the supplied images are not pixel-dimension matched.",
            "confidence": None,
            "confidence_status": "unavailable",
            "geospatial_compatibility": "rejected_dimension_mismatch",
            "optical_shape": list(opt.shape),
            "sar_shape": list(radar.shape),
        }

    optical_mean = float(opt.mean())
    sar_mean = float(radar.mean())
    sar_std = float(radar.std())

    return {
        "ok": True,
        "task": "optical_sar_analysis",
        "method": "classical co-array intensity statistics",
        "optical_mean_intensity": round(optical_mean, 3),
        "sar_mean_intensity": round(sar_mean, 3),
        "sar_std_intensity": round(sar_std, 3),
        "confidence": None,
        "confidence_status": "not_calibrated",
        "geospatial_compatibility": "unverified",
        "note": "This baseline does not claim radiometric SAR calibration, CRS co-registration, or neural optical-SAR fusion.",
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def health() -> dict:
    return {
        "ok": True,
        "service": "ORBITAL-AI ZeroGPU worker",
        "specialists": [
            "rs_vqa_adapted",
            "rs_caption_adapted",
            "change_detection_classical",
            "optical_sar_classical",
        ],
        "adaptllm": "not loaded in this worker",
    }


with gr.Blocks(title="ORBITAL-AI Remote-Sensing Worker") as demo:
    gr.Markdown("# ORBITAL-AI Remote-Sensing Worker")
    gr.Markdown(
        "Free ZeroGPU worker for the adapted VQA/caption specialists plus explicitly-labelled classical baselines."
    )

    with gr.Tab("Agent"):
        agent_image = gr.Image(type="filepath", label="Optical / aerial image")
        agent_query = gr.Textbox(label="Natural-language query")
        agent_output = gr.JSON(label="Result")
        gr.Button("Run agent").click(
            analyze, [agent_image, agent_query], agent_output, api_name="analyze"
        )

    with gr.Tab("VQA"):
        vqa_image = gr.Image(type="filepath", label="Image")
        vqa_question = gr.Textbox(label="Question")
        vqa_output = gr.JSON(label="Result")
        gr.Button("Run VQA").click(vqa, [vqa_image, vqa_question], vqa_output, api_name="vqa")

    with gr.Tab("Caption"):
        caption_image = gr.Image(type="filepath", label="Image")
        caption_output = gr.JSON(label="Result")
        gr.Button("Generate Caption").click(
            caption, caption_image, caption_output, api_name="caption"
        )

    with gr.Tab("Change"):
        before_image = gr.Image(type="filepath", label="Earlier image")
        after_image = gr.Image(type="filepath", label="Later image")
        change_output = gr.JSON(label="Result")
        gr.Button("Analyze Change").click(
            change_analysis, [before_image, after_image], change_output, api_name="change"
        )

    with gr.Tab("Optical + SAR"):
        optical_image = gr.Image(type="filepath", label="Optical image")
        sar_image = gr.Image(type="filepath", label="SAR image")
        fusion_output = gr.JSON(label="Result")
        gr.Button("Run Joint Analysis").click(
            fusion_analysis, [optical_image, sar_image], fusion_output, api_name="fusion"
        )

    gr.api(health, api_name="health")

demo.launch(max_file_size="20mb")
