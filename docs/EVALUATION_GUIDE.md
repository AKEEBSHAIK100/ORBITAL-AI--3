# Evaluation Guide — SatQuery AI / ORBITAL-AI

## Overview

SatQuery AI includes formal benchmark evaluation runners for three remote-sensing
evaluation suites.  **All evaluators produce genuine metrics only when the
required dataset is present and validated.**  When a dataset is unavailable, the
evaluator returns status: "dataset_not_available" and stores that record in
evaluation/results/ — no scores are fabricated.

---

## Benchmark Suites Supported

| Suite | Tasks | Location |
|---|---|---|
| **VRSBench** | Captioning, grounding IoU, VQA | evaluation/vrsbench/ |
| **RSVQA** | Presence, count, comparison | evaluation/rsvqa/ |
| **CDVQA** | Bi-temporal change VQA | evaluation/cdvqa/ |

---

## 1. VRSBench

### Captioning Evaluation

`ash
python evaluation/vrsbench/evaluate_captioning.py \
  --data-dir data/vrsbench \
  --adapter-path checkpoints/rs_vlm_adapter
`

Metrics computed (via evaluation/vrsbench/metrics.py):
- **CIDEr** — consensus-based image description evaluation
- **BLEU-4** — 4-gram precision overlap
- **ROUGE-L** — longest common subsequence recall

### Grounding Evaluation

`ash
python evaluation/vrsbench/evaluate_grounding.py \
  --data-dir data/vrsbench \
  --adapter-path checkpoints/rs_vlm_adapter
`

Metric: **Grounding IoU@0.5** (fraction of predicted bounding boxes with
IoU ≥ 0.5 against ground truth).

### VQA Evaluation

`ash
python evaluation/vrsbench/evaluate_vqa.py \
  --data-dir data/vrsbench \
  --adapter-path checkpoints/rs_vlm_adapter
`

Metrics:
- **Exact Match (EM)** — token-normalised exact answer match
- **Token F1** — token overlap F1 between predicted and reference answers

---

## 2. RSVQA

`ash
python evaluation/rsvqa/evaluate.py \
  --data-dir data/rsvqa \
  --adapter-path checkpoints/rs_vlm_adapter \
  --split lr   # or: hr
`

Metrics (evaluation/rsvqa/metrics.py):
- **Presence Accuracy** — yes/no question accuracy
- **Count Accuracy** — numeric count accuracy within tolerance ±1
- **Comparison Accuracy** — greater/less/equal comparison accuracy
- **Overall Accuracy** — weighted mean

---

## 3. CDVQA

`ash
python evaluation/cdvqa/evaluate.py \
  --data-dir data/cdvqa \
  --adapter-path checkpoints/rs_vlm_adapter
`

Metrics (evaluation/cdvqa/metrics.py):
- **Binary Change Accuracy** — changed / unchanged classification
- **Directional Change F1** — macro-F1 for increase / decrease / unchanged

---

## 4. Evaluation Results Storage

All runs append a JSON record to evaluation/results/<suite>_<timestamp>.json:

`json
{
  "run_id": "c1e2d3f4-...",
  "suite": "vrsbench",
  "task": "captioning",
  "adapter_version": "rs_vlm_lora_v1",
  "dataset_status": "READY",
  "status": "COMPLETED",
  "metrics": {
    "cider": 0.743,
    "bleu4": 0.312,
    "rouge_l": 0.511
  },
  "timestamp": "2026-09-15T14:22:00Z",
  "num_samples": 2251
}
`

If the dataset is not READY:

`json
{
  "status": "DATASET_UNAVAILABLE",
  "metrics": null
}
`

---

## 5. Viewing Results via the API

`ash
curl http://localhost:8000/api/evaluation/results
`

Returns all stored evaluation records from evaluation/results/.

---

## 6. Rules for Valid Evaluation

- Do **not** evaluate on training data.
- Do **not** hard-code or copy benchmark numbers from papers.
- Do **not** run evaluation before the dataset status is READY.
- Do **not** claim evaluation results from a dataset that is NOT_DOWNLOADED.
- Every metric in the results file must be computed from real model outputs.
