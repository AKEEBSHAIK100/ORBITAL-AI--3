# ORBITAL-AI BLIP remote-sensing adapters

These directories contain the **adapter configuration only**:

- `blip_rs_lora/` — BLIP image-captioning pilot adapter
- `blip_vqa_rs_lora/` — BLIP VQA pilot adapter

The corresponding `adapter_model.safetensors` files are intentionally not stored in Git. The repository `.gitignore` excludes `*.safetensors` because model weights are large.

## Runtime contract

ORBITAL-AI marks an adapter executable only after all of the following are true:

1. `adapter_config.json` exists.
2. `adapter_model.safetensors` exists and is non-empty.
3. The BLIP base model can load.
4. PEFT can load the adapter into that base model.
5. The runtime inference path completes successfully.

File presence alone does **not** mark the specialist as available.

## Supplying the pilot weights

Mount or copy the real adapter weights into the corresponding directories, or set:

- `RS_CAPTION_ADAPTER_PATH`
- `RS_VQA_ADAPTER_PATH`

Do not create placeholder, truncated, or synthetic weight files. If the real weights are unavailable, the API intentionally reports the specialist as unavailable.

The current pilot provenance is documented in `backend/services/rs_adapters.py`; no benchmark accuracy or superiority claim is made.
