# Training Guide — SatQuery AI / ORBITAL-AI

## Overview

This guide explains the end-to-end pipeline for fine-tuning a remote-sensing
LoRA adapter using Google Colab.  Local GPU training is **not required** — the
development machine only needs to prepare instruction-tuning data.

---

## Hardware Requirements

| Environment | GPU | VRAM | Use |
|---|---|---|---|
| Development laptop | None (CPU only) | — | Data prep, code editing |
| Google Colab (Free) | T4 | 16 GB | Full LoRA training |
| Google Colab (Pro) | A100 | 40 GB | Faster, larger batch sizes |

The SatQuery AI training config (	raining/configs/remote_sensing_lora.yaml)
is tuned for the free T4 tier.

---

## Step 1: Prepare Instruction Data Locally

`ash
cd SATQUERY

# BigEarthNet instruction-tuning samples
python training/prepare_bigearthnet.py --data-dir data/bigearthnet --output-dir training/instruction_data

# General remote-sensing instruction samples
python training/preprocessing/prepare_instruction_data.py
`

This produces 	raining/instruction_data/train_instructions.jsonl and
	raining/instruction_data/val_instructions.jsonl.

---

## Step 2: Upload Data to Colab / Google Drive

1. Open [Google Drive](https://drive.google.com).
2. Create a folder SatQuery-Training/.
3. Upload 	raining/instruction_data/ into that folder.
4. Optionally, upload data/bigearthnet/ (or a representative sample) for
   image-grounded instruction tuning.

---

## Step 3: Run the Colab Notebook

1. Open 	raining/colab/train_remote_sensing_lora.ipynb in Colab.
2. Ensure **Runtime → Change runtime type → T4 GPU** is selected.
3. Execute all cells in order:
   - Cell 1: Verifies GPU with 
vidia-smi.
   - Cell 2: Installs peft, 	ransformers, ccelerate, itsandbytes.
   - Cell 3: Sets training hyper-parameters from config.
   - Cell 4: Validates or creates sample instruction data.
   - Cell 5: Loads base VLM in 4-bit, attaches LoRA adapter.
   - Cell 6: Saves and exports adapter as s_vlm_lora_adapter.zip.

---

## Step 4: Download and Install the Adapter

After training completes:

1. Download s_vlm_lora_adapter.zip from the Colab files panel.
2. Extract it into:
   `
   SATQUERY/checkpoints/rs_vlm_adapter/
   `
3. Restart the FastAPI backend:
   `ash
   uvicorn backend.main:app --reload
   `
4. Verify at /api/model-status — the s_vlm_vqa specialist should now show
   is_available: true.

---

## Step 5: Validate Adapter Locally

`ash
python training/evaluate_adaptation.py \
  --adapter-path checkpoints/rs_vlm_adapter \
  --val-data training/instruction_data/val_instructions.jsonl
`

Expected output (genuine — no hardcoded numbers):

`
Evaluation on 500 validation samples:
  Task accuracy (exact match): X.XX%   ← real value from your adapter
  F1 (token overlap):          X.XX%
`

---

## Hyper-Parameter Tuning

Edit 	raining/configs/remote_sensing_lora.yaml to adjust:

| Parameter | Default | Notes |
|---|---|---|
| lora_r | 16 | Higher rank = more capacity, more VRAM |
| lora_alpha | 32 | Typically 2× rank |
| learning_rate | 2e-4 | Reduce if loss diverges |
| 
um_epochs | 3 | Increase for larger datasets |
| atch_size | 2 | Must fit in 16 GB VRAM |
| gradient_accumulation_steps | 8 | Effective batch = 16 |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| CUDA out of memory | Batch too large | Reduce atch_size to 1 |
| ValueError: No module named peft | Missing install | Run Cell 2 again |
| Adapter not detected at startup | Wrong path | Check checkpoints/rs_vlm_adapter/adapter_config.json exists |
| status: SPECIALIST_UNAVAILABLE | Expected on CPU | Install adapter after Colab training |
