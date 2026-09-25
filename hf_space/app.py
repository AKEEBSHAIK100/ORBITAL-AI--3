import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import requests
import spaces
import torch
from PIL import Image
from peft import PeftModel
from transformers import BlipForQuestionAnswering, BlipForConditionalGeneration, BlipProcessor

REPO_RAW = "https://raw.githubusercontent.com/AKEEBSHAIK100/ORBITAL-AI--3/main/backend/models/adapters"
CACHE_ROOT = Path("/tmp/orbital_adapters")


def _download_adapter(name: str) -> Path:
    target = CACHE_ROOT / name
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        path = target / filename
        if not path.exists():
            url = f"{REPO_RAW}/{name}/{filename}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
    return target


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

_vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
_vqa_base = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base", torch_dtype=dtype
)
_vqa_adapter = PeftModel.from_pretrained(
    _vqa_base, str(_download_adapter("blip_vqa_rs_lora"))
)
_vqa_model = _vqa_adapter.to(device).eval()

_caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
_caption_base = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base", torch_dtype=dtype
)
_caption_adapter = PeftModel.from_pretrained(
    _caption_base, str(_download_adapter("blip_rs_lora"))
)
_caption_model = _caption_adapter.to(device).eval()


def _image(value: Any) -> Image.Image:
    if value is None:
        raise ValueError("An image is required.")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    return Image.open(value).convert("RGB")


@spaces.GPU(duration=90)
def vqa(image: Any, question: str) -> dict:
    image = _image(image)
    question = (question or "").strip()
    if not question:
        raise ValueError("A question is required.")
    inputs = _vqa_processor(images=image, text=question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = _vqa_model.generate(**inputs, max_new_tokens=50)
    answer = _vqa_processor.decode(output[0], skip_special_tokens=True).strip()
    return {
        "ok": True,
        "task": "vqa",
        "answer": answer,
        "model": "Salesforce/blip-vqa-base + BigEarthNet-derived LoRA",
        "confidence": None,
        "confidence_status": "not_calibrated",
    }


@spaces.GPU(duration=90)
def caption(image: Any) -> dict:
    image = _image(image)
    inputs = _caption_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = _caption_model.generate(**inputs, max_new_tokens=60)
    text = _caption_processor.decode(output[0], skip_special_tokens=True).strip()
    return {
        "ok": True,
        "task": "caption",
        "answer": text,
        "model": "Salesforce/blip-image-captioning-base + BigEarthNet-derived LoRA",
        "confidence": None,
        "confidence_status": "not_calibrated",
    }


with gr.Blocks(title="ORBITAL-AI RS Backend") as demo:
    gr.Markdown("# ORBITAL-AI Remote-Sensing Backend")
    gr.Markdown("Programmatic RS inference service. No fabricated metrics or confidence.")
    with gr.Tab("VQA"):
        vqa_image = gr.Image(type="filepath", label="Image")
        vqa_question = gr.Textbox(label="Question")
        vqa_output = gr.JSON(label="Result")
        gr.Button("Run VQA").click(vqa, [vqa_image, vqa_question], vqa_output, api_name="vqa")
    with gr.Tab("Caption"):
        caption_image = gr.Image(type="filepath", label="Image")
        caption_output = gr.JSON(label="Result")
        gr.Button("Generate Caption").click(caption, caption_image, caption_output, api_name="caption")

demo.launch()
