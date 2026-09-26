"""
Train the ORBITAL-AI BLIP remote-sensing LoRA adapters.

Expected JSONL records:
Caption:
  {"image": "/path/to/image.npy|jpg|png", "caption": "..."}
VQA:
  {"image": "/path/to/image.npy|jpg|png", "question": "...", "answer": "..."}

For .npy Sentinel-2 arrays, an optional sidecar metadata JSON may be supplied
with --metadata-json or each record may contain:
  {"metadata": {"band_names": [...], "sensor": "Sentinel-2", ...}}

This script intentionally fails closed:
- it trains BLIP, not Qwen2-VL;
- it targets only verified BLIP attention modules;
- it requires real output files;
- it reloads the adapter through PEFT before reporting success;
- it does not create placeholder safetensors files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from backend.services.rs_adapters import preprocess_s2_to_blip_rgb


def load_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        records.append(row)
    if not records:
        raise ValueError(f"No training records found in {path}")
    return records


def load_image(path: str, metadata: Dict[str, Any] | None = None) -> Image.Image:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
        if arr.ndim == 3 and arr.shape[2] > 3:
            image, _ = preprocess_s2_to_blip_rgb(arr, metadata)
            return image
        if arr.ndim == 3 and arr.shape[2] == 3:
            return Image.fromarray(arr.astype(np.uint8), mode="RGB")
        raise ValueError(f"Unsupported .npy shape for BLIP training: {arr.shape}")
    return Image.open(p).convert("RGB")


class CaptionDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = load_image(row["image"], row.get("metadata"))
        enc = self.processor(images=image, text=row["caption"], return_tensors="pt", padding="max_length", truncation=True)
        return {k: v.squeeze(0) for k, v in enc.items()}


class VQADataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = load_image(row["image"], row.get("metadata"))
        enc = self.processor(images=image, text=row["question"], return_tensors="pt")
        labels = self.processor.tokenizer(
            row["answer"], return_tensors="pt", padding="max_length", truncation=True
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return {k: v.squeeze(0) for k, v in enc.items()}


def train(task: str, model_id: str, data_path: Path, output_dir: Path, epochs: int, lr: float):
    if not torch.cuda.is_available():
        raise RuntimeError("BLIP LoRA training requires CUDA. Use a GPU/Colab runtime.")

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        BlipForConditionalGeneration,
        BlipForQuestionAnswering,
        BlipProcessor,
        Trainer,
        TrainingArguments,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    processor = BlipProcessor.from_pretrained(model_id)

    if task == "caption":
        model = BlipForConditionalGeneration.from_pretrained(model_id)
        dataset = CaptionDataset(load_records(data_path), processor)
    else:
        model = BlipForQuestionAnswering.from_pretrained(model_id)
        dataset = VQADataset(load_records(data_path), processor)

    peft_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["qkv"],
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    cfg = output_dir / "adapter_config.json"
    weights = output_dir / "adapter_model.safetensors"
    if not cfg.is_file() or not weights.is_file() or weights.stat().st_size == 0:
        raise RuntimeError("Training finished without a real PEFT config and non-empty adapter_model.safetensors.")

    # Reload through PEFT before declaring the artifact usable.
    from peft import PeftModel
    base = (
        BlipForConditionalGeneration.from_pretrained(model_id)
        if task == "caption"
        else BlipForQuestionAnswering.from_pretrained(model_id)
    )
    reloaded = PeftModel.from_pretrained(base, str(output_dir))
    reloaded.eval()
    print(f"VERIFIED: {task} BLIP LoRA adapter reloaded successfully from {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["caption", "vqa"], required=True)
    parser.add_argument("--data", required=True, help="JSONL training records")
    parser.add_argument("--output", required=True, help="Adapter output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()

    model_id = args.model_id or (
        "Salesforce/blip-image-captioning-base"
        if args.task == "caption"
        else "Salesforce/blip-vqa-base"
    )
    train(args.task, model_id, Path(args.data), Path(args.output), args.epochs, args.lr)


if __name__ == "__main__":
    main()
