# Dataset Setup — SatQuery AI / ORBITAL-AI

## Overview

This document describes how to download, validate, and register the datasets
used for training and evaluation in SatQuery AI.  **The web application and
FastAPI backend start normally even when all datasets are absent.**

---

## Important: Non-Blocking Design

The dataset registry (	raining/datasets/dataset_registry.py) uses a five-state
status model:

| Status | Meaning |
|---|---|
| NOT_DOWNLOADED | Directory placeholder exists; data not yet fetched |
| AVAILABLE | Files found but not validated |
| INVALID | Validation failed (wrong format, corrupt files) |
| PROCESSING | Currently being validated or prepared |
| READY | Validated and ready for training / evaluation |

Evaluation endpoints return status: "dataset_not_available" — never fabricated
numbers — when the dataset is not READY.

---

## 1. Dataset Directory Layout

`
SATQUERY/
  data/
    bigearthnet/            ← BigEarthNet-S2 patch directories
    bigearthnet_txt/        ← Metadata text files (.gitkeep placeholder)
    vrsbench/               ← VRSBench JSON annotations + image directories
    rsvqa/                  ← RSVQA JSON + image directories
    cdvqa/                  ← CDVQA bi-temporal image pairs + annotations
`

All directories currently contain .gitkeep placeholders.

---

## 2. BigEarthNet-S2 (reBEN)

**Purpose:** Land-cover classification training and validation.

**Source:** https://bigearth.net

`ash
# Download with the official script or manually via the website
# Unzip into data/bigearthnet/
# Expected structure: data/bigearthnet/<patch_id>/*.tif + *.json

# Validate the download
python -c "from training.datasets.bigearthnet import BigEarthNetDataset; d = BigEarthNetDataset('data/bigearthnet'); print(d.validate())"
`

**Licence:** Community Data Licence Agreement – Permissive (CDLA-Permissive-1.0)

---

## 3. VRSBench

**Purpose:** Visual remote-sensing captioning, grounding, and VQA evaluation.

**Source:** https://github.com/lx709/VRSBench

`ash
# Download annotations and images, extract into data/vrsbench/
# Expected: data/vrsbench/annotations/*.json + data/vrsbench/images/

python -c "from training.datasets.vrsbench import VRSBenchDataset; d = VRSBenchDataset('data/vrsbench'); print(d.validate())"
`

---

## 4. RSVQA

**Purpose:** Remote-sensing visual question answering (presence, count, comparison).

**Source:** https://zenodo.org/record/6344334 (LR) and https://zenodo.org/record/6344367 (HR)

`ash
# Download and extract into data/rsvqa/
python -c "from training.datasets.rsvqa import RSVQADataset; d = RSVQADataset('data/rsvqa'); print(d.validate())"
`

---

## 5. CDVQA

**Purpose:** Bi-temporal change VQA for the change-detection specialist.

**Source:** https://github.com/YZHJesse/CDVQA

`ash
# Download and extract into data/cdvqa/
python -c "from training.datasets.cdvqa import CDVQADataset; d = CDVQADataset('data/cdvqa'); print(d.validate())"
`

---

## 6. Check All Dataset Statuses

`python
from training.datasets.dataset_registry import DatasetRegistry
reg = DatasetRegistry()
for name, info in reg.status_all().items():
    print(f"{name}: {info['status']}")
`

Expected output (before any downloads):
`
bigearthnet: NOT_DOWNLOADED
vrsbench: NOT_DOWNLOADED
rsvqa: NOT_DOWNLOADED
cdvqa: NOT_DOWNLOADED
`

---

## 7. Auto-Detection at Backend Start

ackend/main.py and ackend/routers/analyze.py import DatasetRegistry at
startup.  If all datasets are absent, startup completes normally.  The
/api/system/status endpoint returns the live registry state.

**Never run dataset downloads automatically during 
pm install or app startup.**
