"""
VRSBench Official Metric Calculators.
Computes BLEU-4, ROUGE-L, Grounding IoU@0.5, and VQA Exact Match / F1.
Pure Python implementations ensuring deterministic calculation without external API dependencies.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple


def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in clean.split() if w]


# ── 1. Captioning Metrics (BLEU & ROUGE-L) ───────────────────────────────────

def compute_ngram_precisions(reference_tokens: Sequence[str], candidate_tokens: Sequence[str], n: int = 4) -> List[float]:
    precisions = []
    for i in range(1, n + 1):
        cand_ngrams = [tuple(candidate_tokens[j:j+i]) for j in range(len(candidate_tokens) - i + 1)]
        ref_ngrams = [tuple(reference_tokens[j:j+i]) for j in range(len(reference_tokens) - i + 1)]

        if not cand_ngrams:
            precisions.append(0.0)
            continue

        cand_counts = Counter(cand_ngrams)
        ref_counts = Counter(ref_ngrams)

        clipped = sum(min(cand_counts[ng], ref_counts.get(ng, 0)) for ng in cand_counts)
        precisions.append(clipped / len(cand_ngrams))
    return precisions


def compute_bleu4(reference: str, candidate: str) -> float:
    ref_tok = tokenize(reference)
    cand_tok = tokenize(candidate)

    if not ref_tok or not cand_tok:
        return 0.0

    precisions = compute_ngram_precisions(ref_tok, cand_tok, n=4)
    if any(p == 0.0 for p in precisions):
        return 0.0

    log_mean = sum(0.25 * math.log(p) for p in precisions)
    bp = math.exp(min(0.0, 1.0 - len(ref_tok) / len(cand_tok)))
    return round(bp * math.exp(log_mean) * 100.0, 2)


def compute_rouge_l(reference: str, candidate: str) -> float:
    ref_tok = tokenize(reference)
    cand_tok = tokenize(candidate)
    m, n = len(ref_tok), len(cand_tok)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_tok[i] == cand_tok[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    lcs = dp[m][n]
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    f1 = (2 * prec * rec) / (prec + rec)
    return round(f1 * 100.0, 2)


# ── 2. Visual Grounding Metrics (IoU & Accuracy@0.5) ──────────────────────────

def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Computes IoU between [x1, y1, x2, y2]."""
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return (inter / union) if union > 0 else 0.0


def compute_grounding_metrics(predictions: List[List[float]], targets: List[List[float]], threshold: float = 0.5) -> Dict[str, float]:
    if not predictions or not targets:
        return {"mean_iou": 0.0, "acc_at_05": 0.0, "sample_count": 0}

    total_iou = 0.0
    hits = 0
    count = min(len(predictions), len(targets))

    for pred, gt in zip(predictions[:count], targets[:count]):
        iou = compute_box_iou(pred, gt)
        total_iou += iou
        if iou >= threshold:
            hits += 1

    return {
        "mean_iou": round((total_iou / count) * 100.0, 2),
        "acc_at_05": round((hits / count) * 100.0, 2),
        "sample_count": count
    }


# ── 3. VQA Metrics (Exact Match & F1) ────────────────────────────────────────

def compute_vqa_accuracy(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    if not predictions or not ground_truths:
        return {"exact_match": 0.0, "macro_f1": 0.0, "sample_count": 0}

    em_hits = 0
    total_f1 = 0.0
    count = min(len(predictions), len(ground_truths))

    for pred, gt in zip(predictions[:count], ground_truths[:count]):
        p_clean = " ".join(tokenize(pred))
        g_clean = " ".join(tokenize(gt))

        if p_clean == g_clean:
            em_hits += 1

        # Token F1
        p_tokens = Counter(p_clean.split())
        g_tokens = Counter(g_clean.split())
        common = sum((p_tokens & g_tokens).values())

        if len(p_tokens) == 0 and len(g_tokens) == 0:
            total_f1 += 1.0
        elif len(p_tokens) == 0 or len(g_tokens) == 0 or common == 0:
            total_f1 += 0.0
        else:
            prec = common / len(p_tokens)
            rec = common / len(g_tokens)
            total_f1 += (2 * prec * rec) / (prec + rec)

    return {
        "exact_match": round((em_hits / count) * 100.0, 2),
        "macro_f1": round((total_f1 / count) * 100.0, 2),
        "sample_count": count
    }
