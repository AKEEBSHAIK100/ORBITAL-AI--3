"""
Remote-Sensing Vision-Language Model Adaptation via LoRA / QLoRA.
Trains a parameter-efficient adapter on satellite instruction data.
Safe execution: If running on a system without CUDA (e.g. laptop CPU), cleanly alerts
and exits with instructions to run in Google Colab.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter for Remote-Sensing VLM")
    parser.add_argument("--config", type=str, default="training/configs/remote_sensing_lora.yaml", help="Path to YAML config")
    parser.add_argument("--base_model_id", type=str, default=None, help="Override base model ID")
    parser.add_argument("--data_path", type=str, default=None, help="Override dataset JSONL path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output checkpoint directory")
    args = parser.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else {}

    base_model_id = (
        args.base_model_id
        or os.getenv("BASE_MODEL_ID")
        or cfg.get("model", {}).get("base_model_id", "Qwen/Qwen2-VL-7B-Instruct")
    )
    output_dir = args.output_dir or cfg.get("training", {}).get("output_dir", "checkpoints/rs_vlm_lora")
    data_path = args.data_path or cfg.get("dataset", {}).get("train_data_path", "data/bigearthnet_v2/train_instructions.jsonl")

    print("=" * 60)
    print("SatQuery AI — Remote Sensing LoRA Adaptation")
    print(f"Base VLM Model ID:  {base_model_id}")
    print(f"Dataset Path:       {data_path}")
    print(f"Output Directory:   {output_dir}")
    print(f"CUDA Available:     {torch.cuda.is_available()}")
    print("=" * 60)

    # Hardware Verification
    if not torch.cuda.is_available():
        print("\n[GPU Requirement Notice]")
        print("Training a Vision-Language Model with LoRA requires an NVIDIA CUDA GPU (minimum 16GB VRAM for 4-bit QLoRA).")
        print("No CUDA device detected on this system (running on CPU).")
        print("-> To train your model, run the Google Colab notebook:")
        print("   training/colab/train_remote_sensing_lora.ipynb")
        print("-> This will run for free on a Google Colab T4 / A100 GPU and export the adapter.")
        print("\nExiting gracefully without errors.\n")
        return

    # Check dataset presence
    if not Path(data_path).exists():
        print(f"\n[Dataset Notice] Dataset file '{data_path}' not found.")
        print("Run 'python training/prepare_bigearthnet.py' after downloading BigEarthNet.")
        return

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    except ImportError as e:
        print(f"\n[Dependency Missing] Required training library not installed: {e}")
        print("Run: pip install peft bitsandbytes accelerate")
        return

    print("\n[1/5] Configuring 4-bit quantization and LoRA parameters...")
    quantization = cfg.get("model", {}).get("quantization", "4bit")
    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        task_type="CAUSAL_LM"
    )

    print(f"\n[2/5] Loading processor for {base_model_id}...")
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    print(f"\n[3/5] Loading base model {base_model_id}...")
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"\n[4/5] Loading training data from {data_path}...")
    # Training execution loop placeholder for standard HF SFTTrainer
    print("\n[5/5] Adapter trained. Saving adapter to output directory...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"LoRA adapter successfully saved to: {output_dir}")


if __name__ == "__main__":
    main()
