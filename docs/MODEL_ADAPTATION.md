# Model Adaptation — SatQuery AI / ORBITAL-AI

## Overview

SatQuery AI uses **parameter-efficient fine-tuning (LoRA/QLoRA)** to adapt a
pretrained Vision-Language Model (VLM) to the remote-sensing domain.  This
document explains the adaptation strategy, the base model selection rationale,
the LoRA configuration, and how to verify the resulting adapter in inference.

---

## 1. Adaptation Strategy

### Why Domain Adaptation?

General-purpose VLMs (trained on internet imagery) perform poorly on satellite
data because:

- Satellite images use non-standard spectral bands (NIR, SWIR, SAR backscatter).
- Scene scale is radically different from ground-level photographs.
- Geographic objects (buildings, fields, water bodies) look fundamentally
  different from above.
- Task vocabulary ("land cover", "CRS", "backscatter coefficient", "patch") is
  domain-specific.

### Approach: LoRA / QLoRA Adapter

We apply **Low-Rank Adaptation (LoRA)** on top of a frozen base VLM.  This:

- Keeps the base model parameters unchanged (prevents catastrophic forgetting).
- Adds only ~0.1–1% of total parameters as trainable adapter matrices.
- Allows the adapter to be swapped without re-downloading the base model.
- Enables GPU-efficient training with 4-bit quantisation (QLoRA).

---

## 2. Base VLM

| Field | Value |
|---|---|
| **Default Model** | `Qwen/Qwen2-VL-7B-Instruct` |
| **Alternative** | `microsoft/Florence-2-large` |
| **Architecture** | Vision-Language (image encoder + language decoder) |
| **Parameters** | ~7B (Qwen2-VL) |
| **Hugging Face Hub** | https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct |

The base model can be changed by editing `training/configs/remote_sensing_lora.yaml`
(field: `base_model_id`).

---

## 3. LoRA Configuration

Located at `training/configs/remote_sensing_lora.yaml`:

```yaml
base_model_id: Qwen/Qwen2-VL-7B-Instruct
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj]
task_type: CAUSAL_LM
quantization: 4bit  # nf4 via bitsandbytes
batch_size: 2
gradient_accumulation_steps: 8
num_epochs: 3
learning_rate: 2e-4
```

These values balance GPU memory constraints (16 GB T4) with convergence
stability for multi-label satellite scene understanding.

---

## 4. Training Data

The adapter is trained on instruction-tuning samples derived from:

| Dataset | Task type | Approximate size |
|---|---|---|
| BigEarthNet-S2 (reBEN) | Land-cover identification, label presence | ~590 000 patches |
| VRSBench | Captioning, grounding, VQA | ~37 000 image pairs |
| RSVQA-LR / RSVQA-HR | Presence/count/comparison VQA | ~772 000 QA pairs |
| CDVQA | Bi-temporal change VQA | ~30 000 pairs |

All instruction samples must be prepared locally before uploading to Colab:

```bash
python training/prepare_bigearthnet.py
python training/preprocessing/prepare_instruction_data.py
```

See `docs/DATASET_SETUP.md` for download and format details.

---

## 5. Adapter Checkpoint Location

After training (see `docs/TRAINING_GUIDE.md`), place the adapter contents in:

```
SATQUERY/
  checkpoints/
    rs_vlm_adapter/
      adapter_model.safetensors
      adapter_config.json
      tokenizer.json
      ...
```

The backend registry (`models/registry.py`) checks for
`checkpoints/rs_vlm_adapter/adapter_config.json` at startup to determine
adapter availability.

---

## 6. Fallback Behaviour (No Adapter)

If the adapter checkpoint is absent the registry returns a controlled response:

```json
{
  "status": "SPECIALIST_UNAVAILABLE",
  "answer": "Analysis unavailable: LoRA adapter checkpoint not installed.",
  "confidence": null,
  "confidence_level": "UNAVAILABLE"
}
```

No mock metrics are fabricated.

---

## 7. Continuous Improvement

Once evaluation results are recorded in `evaluation/results/`, performance can
be tracked over adapter versions.  Each evaluation run stamps:

- `run_id` (UUID)
- `adapter_version`
- `dataset`
- `metrics` (only if dataset is `READY`)
- `status` (`NOT_RUN` / `DATASET_UNAVAILABLE` / `COMPLETED` / `FAILED`)

See `docs/EVALUATION_GUIDE.md` for details.
