---
title: ORBITAL-AI Remote Sensing Backend
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.28.0"
python_version: "3.12"
suggested_hardware: "zero-gpu"
app_file: app.py
---

# ORBITAL-AI Remote-Sensing Inference Backend

ZeroGPU-backed Gradio Server for the ORBITAL-AI custom frontend.

The service exposes programmatic endpoints for real remote-sensing adapted VQA and captioning. It never returns fabricated confidence or benchmark metrics.

For deployment, copy the contents of this directory to a Hugging Face **Gradio ZeroGPU Space**.
