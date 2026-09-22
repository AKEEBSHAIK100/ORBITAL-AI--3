"""
SatQuery AI — Standalone RS-VLM Google Colab Evaluation Runner.

Target Model:   AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct
Baseline Model: Qwen/Qwen2-VL-2B-Instruct
Benchmark:      Deterministic 40-Question RS Evaluation Suite (20 categories)

Zero production code modifications.
Zero local weight downloads.
Designed for execution on a Google Colab T4 GPU instance.
"""

import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ─── 1. Image Loading & Preprocessing ─────────────────────────────────────────

def load_image_rgb(path: str) -> Optional[Image.Image]:
    """Loads an image and converts it safely to RGB."""
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        try:
            import cv2
            cv_img = cv2.imread(path)
            if cv_img is not None:
                rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
        except Exception:
            pass
    return None

def make_side_by_side_composite(img_a: Image.Image, img_b: Image.Image) -> Image.Image:
    """
    Composites two images side-by-side without silent resizing or distorting aspect ratios.
    Matches SatQuery AI's geospatial safety protocol for multi-image reasoning.
    """
    max_h = max(img_a.height, img_b.height)
    canvas = Image.new("RGB", (img_a.width + img_b.width, max_h), color=(0, 0, 0))
    canvas.paste(img_a, (0, 0))
    canvas.paste(img_b, (img_a.width, 0))
    return canvas

def resolve_image_samples(base_data_dir: str) -> Dict[str, Any]:
    """Resolves the evaluation imagery from the workspace data directory."""
    data_dir = Path(base_data_dir)
    
    aerial_path = str(data_dir / "default_aerial.jpg")
    levir_a_path = str(data_dir / "levir_cd" / "A" / "test_2_0000_0000.png")
    levir_b_path = str(data_dir / "levir_cd" / "B" / "test_2_0000_0000.png")
    opt_tif_path = str(data_dir / "optical_sar" / "optical.tif")
    sar_tif_path = str(data_dir / "optical_sar" / "sar.tif")

    return {
        "sample_optical_aerial": {
            "type": "single",
            "image": load_image_rgb(aerial_path),
            "path": aerial_path,
        },
        "sample_levir_cd_pair": {
            "type": "pair",
            "image_a": load_image_rgb(levir_a_path),
            "image_b": load_image_rgb(levir_b_path),
            "path_a": levir_a_path,
            "path_b": levir_b_path,
        },
        "sample_sar_single": {
            "type": "single",
            "image": load_image_rgb(sar_tif_path),
            "path": sar_tif_path,
        },
        "sample_optical_sar_pair": {
            "type": "pair",
            "image_a": load_image_rgb(opt_tif_path),
            "image_b": load_image_rgb(sar_tif_path),
            "path_a": opt_tif_path,
            "path_b": sar_tif_path,
        },
    }

# ─── 2. Model Lifecycle & Memory Management ────────────────────────────────────

def load_vlm(model_id: str):
    """
    Loads a Qwen2-VL family model into GPU memory in bfloat16/float16.
    Estimated VRAM footprint: ~4.9 GB.
    """
    print(f"Loading {model_id} on CUDA...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cuda",
    )
    model.eval()
    peak_vram = round(torch.cuda.memory_allocated() / (1024**2), 2)
    print(f"Loaded {model_id} successfully. Allocated VRAM: {peak_vram} MB")
    return model, processor

def unload_vlm(model, processor):
    """
    Strictly unloads the model and flushes CUDA cache so only ONE candidate
    resides in VRAM at any time.
    """
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    current_vram = round(torch.cuda.memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else 0
    print(f"Model unloaded. Remaining VRAM: {current_vram} MB\n")

# ─── 3. Single Inference Runner ────────────────────────────────────────────────

# Maximum pixel budget per image fed to the vision encoder.
# 256 patches × (28×28) = 200,704 pixels ≈ 448×448 px.
# Keeping this under ~300 patches prevents KV-cache OOM on a 16 GB T4
# when generating even a short sequence (the KV cache scales with n_tokens²).
_SMOKE_MAX_PIXELS: int = 200_704  # 256 × 28 × 28


def run_single_inference(
    model,
    processor,
    image: Image.Image,
    question_text: str,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """
    Runs multimodal inference for one image and one prompt.
    Records wall-clock latency and peak CUDA memory.

    Memory-safety: the image pixel budget is capped at _SMOKE_MAX_PIXELS so
    that the vision encoder never generates more than ~256 vision tokens.  All
    temporary CUDA tensors are explicitly freed after decoding.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    inputs = None
    output_ids = None
    try:
        # Construct message format (image placed first per AdaptLLM specification)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question_text},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Cap vision-token count: pass max_pixels to the processor so it
        # downscales the image internally rather than generating thousands of
        # patch tokens that blow up the KV cache on a 16 GB T4.
        inputs = processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
            max_pixels=_SMOKE_MAX_PIXELS,
        ).to("cuda")

        # inference_mode() is strictly lighter than no_grad (disables the
        # version counter entirely) and is the correct context for pure inference.
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        answer = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        peak_mem_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else None

        return {
            "status": "success",
            "answer": answer,
            "latency_ms": latency_ms,
            "peak_mem_mb": peak_mem_mb,
            "error": None,
        }
    except Exception as e:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "status": "error",
            "answer": f"Inference error: {str(e)}",
            "latency_ms": latency_ms,
            "peak_mem_mb": None,
            "error": str(e),
        }
    finally:
        # Always release CUDA tensors so VRAM is reclaimed before the next call.
        del inputs, output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ─── 4. Full Evaluation Execution ──────────────────────────────────────────────

def evaluate_model_suite(
    model,
    processor,
    model_id: str,
    model_name: str,
    is_rs_adapted: bool,
    samples: Dict[str, Any],
    questions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Evaluates all questions sequentially against a loaded model."""
    results = []
    print(f"\n--- Running 40-Question Benchmark for {model_name} ---")

    for idx, q in enumerate(questions, 1):
        sample = samples.get(q["sample_id"])
        if sample is None:
            results.append({
                "question_id": q["question_id"],
                "category": q["category"],
                "category_name": q["category_name"],
                "model_id": model_id,
                "model_name": model_name,
                "is_remote_sensing_adapted": is_rs_adapted,
                "status": "error",
                "answer": f"Sample {q['sample_id']} not found.",
                "latency_ms": 0.0,
                "peak_mem_mb": None,
                "error": "Missing image sample",
                "human_review_status": "manual_review_required",
            })
            continue

        # Prepare image (single vs paired composite)
        if q["input_type"] == "pair":
            img_a = sample.get("image_a")
            img_b = sample.get("image_b")
            if img_a is None or img_b is None:
                img = None
            else:
                img = make_side_by_side_composite(img_a, img_b)
        else:
            img = sample.get("image")

        if img is None:
            results.append({
                "question_id": q["question_id"],
                "category": q["category"],
                "category_name": q["category_name"],
                "model_id": model_id,
                "model_name": model_name,
                "is_remote_sensing_adapted": is_rs_adapted,
                "status": "error",
                "answer": "Image data unreadable.",
                "latency_ms": 0.0,
                "peak_mem_mb": None,
                "error": "Image loading failure",
                "human_review_status": "manual_review_required",
            })
            continue

        out = run_single_inference(model, processor, img, q["question_text"])
        res = {
            "question_id": q["question_id"],
            "category": q["category"],
            "category_name": q["category_name"],
            "model_id": model_id,
            "model_name": model_name,
            "is_remote_sensing_adapted": is_rs_adapted,
            "status": out["status"],
            "answer": out["answer"],
            "latency_ms": out["latency_ms"],
            "peak_mem_mb": out["peak_mem_mb"],
            "error": out["error"],
            "human_review_status": "manual_review_required",
        }
        results.append(res)
        print(f"  [{idx:02d}/40] {q['question_id']} ({q['category']}) -> {out['status']} ({out['latency_ms']} ms)")

    return results

# ─── 5. Summary Reporter ───────────────────────────────────────────────────────

def generate_summary(results: List[Dict[str, Any]], model_id: str, model_name: str) -> Dict[str, Any]:
    model_res = [r for r in results if r["model_id"] == model_id]
    total = len(model_res)
    success = sum(1 for r in model_res if r["status"] == "success")
    failed = sum(1 for r in model_res if r["status"] != "success")
    
    latencies = [r["latency_ms"] for r in model_res if r["status"] == "success" and r["latency_ms"] is not None]
    avg_lat = round(statistics.mean(latencies), 2) if latencies else None
    med_lat = round(statistics.median(latencies), 2) if latencies else None
    
    peak_vrams = [r["peak_mem_mb"] for r in model_res if r.get("peak_mem_mb") is not None]
    max_vram = max(peak_vrams) if peak_vrams else None

    categories = sorted(list(set(r["category"] for r in model_res)))
    paired_success = all(r["status"] == "success" for r in model_res if "pair" in r["question_id"].lower() or r["category"] in ["cross_sensor_registration", "temporal_reasoning"])
    errors = [r["error"] for r in model_res if r["error"] is not None]

    return {
        "model_id": model_id,
        "model_name": model_name,
        "total_questions": total,
        "successful_questions": success,
        "failed_questions": failed,
        "avg_latency_ms": avg_lat,
        "median_latency_ms": med_lat,
        "peak_vram_mb": max_vram,
        "categories_covered": len(categories),
        "errors": errors,
        "paired_image_success": paired_success,
    }

# ─── 6. Main Orchestrator ──────────────────────────────────────────────────────

def run_bakeoff(base_data_dir: str = "backend/data", questions_path: Optional[str] = None, output_file: str = "satquery_rsvlm_colab_results.json"):
    assert torch.cuda.is_available(), "CUDA is not available. Please run in a Google Colab GPU (T4) runtime."
    
    print("===========================================================================")
    print("SatQuery AI — Remote-Sensing VLM Colab Evaluation")
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    print(f"Allocated VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("===========================================================================\n")

    # Load questions
    if questions_path and os.path.exists(questions_path):
        with open(questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
    else:
        # Fallback to backend question_set if running in repo
        from backend.evaluation.question_set import get_40_question_suite
        questions = [q.to_dict() for q in get_40_question_suite()]

    assert len(questions) == 40, f"Expected exactly 40 questions, got {len(questions)}"
    print(f"Loaded 40 benchmark questions across 20 remote-sensing categories.")

    # Resolve images
    samples = resolve_image_samples(base_data_dir)
    print("Resolved evaluation imagery (optical aerial, LEVIR pair, SAR, Optical-SAR).")

    # ─── STAGE 1: Smoke Tests ─────────────────────────────────────────────────
    print("\n--- STAGE 1: Smoke Tests ---")
    smoke_q = questions[0]  # RS-SCN-01
    smoke_img = samples["sample_optical_aerial"]["image"]
    assert smoke_img is not None, "Optical aerial sample image failed to load."

    # Smoke Test: AdaptLLM
    print("\n[Smoke Test 1/2] Loading AdaptLLM-RS-Qwen2-VL-2B...")
    adapt_model, adapt_proc = load_vlm("AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct")
    # max_new_tokens=32 keeps the KV-cache footprint minimal for the smoke test
    smoke_adapt = run_single_inference(adapt_model, adapt_proc, smoke_img, smoke_q["question_text"], max_new_tokens=32)
    print(f"AdaptLLM Smoke Status: {smoke_adapt['status']} | Latency: {smoke_adapt['latency_ms']} ms | Peak VRAM: {smoke_adapt['peak_mem_mb']} MB")
    assert smoke_adapt["status"] == "success", f"AdaptLLM smoke test failed: {smoke_adapt['error']}"
    unload_vlm(adapt_model, adapt_proc)

    # Smoke Test: Qwen2-VL-2B Baseline
    print("[Smoke Test 2/2] Loading Baseline Qwen2-VL-2B-Instruct...")
    base_model, base_proc = load_vlm("Qwen/Qwen2-VL-2B-Instruct")
    # max_new_tokens=32 keeps the KV-cache footprint minimal for the smoke test
    smoke_base = run_single_inference(base_model, base_proc, smoke_img, smoke_q["question_text"], max_new_tokens=32)
    print(f"Qwen2-VL Baseline Smoke Status: {smoke_base['status']} | Latency: {smoke_base['latency_ms']} ms | Peak VRAM: {smoke_base['peak_mem_mb']} MB")
    assert smoke_base["status"] == "success", f"Baseline smoke test failed: {smoke_base['error']}"
    unload_vlm(base_model, base_proc)

    print("\n>>> BOTH SMOKE TESTS PASSED. Proceeding to full 40-question sequential evaluation.\n")

    # ─── STAGE 2: Full Sequential Evaluation ──────────────────────────────────
    all_results = []

    # 1. AdaptLLM
    adapt_model, adapt_proc = load_vlm("AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct")
    adapt_results = evaluate_model_suite(
        adapt_model, adapt_proc,
        model_id="adaptllm_rs_2b",
        model_name="AdaptLLM-RS-Qwen2-VL-2B",
        is_rs_adapted=True,
        samples=samples,
        questions=questions,
    )
    all_results.extend(adapt_results)
    unload_vlm(adapt_model, adapt_proc)

    # 2. Qwen2-VL Baseline
    base_model, base_proc = load_vlm("Qwen/Qwen2-VL-2B-Instruct")
    base_results = evaluate_model_suite(
        base_model, base_proc,
        model_id="qwen2_vl_2b_baseline",
        model_name="Qwen2-VL-2B-Instruct",
        is_rs_adapted=False,
        samples=samples,
        questions=questions,
    )
    all_results.extend(base_results)
    unload_vlm(base_model, base_proc)

    # ─── STAGE 3: Export Results ──────────────────────────────────────────────
    export_data = {
        "benchmark_name": "SatQuery AI — Remote-Sensing VLM Bake-Off",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "gpu_name": torch.cuda.get_device_name(0),
            "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "torch_version": torch.__version__,
        },
        "candidates": [
            {"model_id": "adaptllm_rs_2b", "model_name": "AdaptLLM-RS-Qwen2-VL-2B", "is_rs_adapted": True},
            {"model_id": "qwen2_vl_2b_baseline", "model_name": "Qwen2-VL-2B-Instruct", "is_rs_adapted": False},
        ],
        "total_questions": len(questions),
        "results": all_results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f"\nSuccessfully exported benchmark results to: {output_file}")

    # ─── STAGE 4: Print Summary Table ─────────────────────────────────────────
    summary_adapt = generate_summary(all_results, "adaptllm_rs_2b", "AdaptLLM-RS-Qwen2-VL-2B")
    summary_base = generate_summary(all_results, "qwen2_vl_2b_baseline", "Qwen2-VL-2B-Instruct")

    print("\n" + "=" * 80)
    print("RS-VLM BAKE-OFF BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<30} | {'AdaptLLM-RS-2B (Candidate)':<24} | {'Qwen2-VL-2B (Baseline)':<24}")
    print("-" * 80)
    adapt_succ = f"{summary_adapt['successful_questions']}/40"
    base_succ = f"{summary_base['successful_questions']}/40"
    adapt_fail = f"{summary_adapt['failed_questions']}/40"
    base_fail = f"{summary_base['failed_questions']}/40"
    adapt_avg_lat = f"{summary_adapt['avg_latency_ms']} ms" if summary_adapt['avg_latency_ms'] is not None else "N/A"
    base_avg_lat = f"{summary_base['avg_latency_ms']} ms" if summary_base['avg_latency_ms'] is not None else "N/A"
    adapt_med_lat = f"{summary_adapt['median_latency_ms']} ms" if summary_adapt['median_latency_ms'] is not None else "N/A"
    base_med_lat = f"{summary_base['median_latency_ms']} ms" if summary_base['median_latency_ms'] is not None else "N/A"
    adapt_vram = f"{summary_adapt['peak_vram_mb']} MB" if summary_adapt['peak_vram_mb'] is not None else "N/A"
    base_vram = f"{summary_base['peak_vram_mb']} MB" if summary_base['peak_vram_mb'] is not None else "N/A"
    adapt_cats = f"{summary_adapt['categories_covered']}/20"
    base_cats = f"{summary_base['categories_covered']}/20"
    adapt_paired = str(summary_adapt['paired_image_success'])
    base_paired = str(summary_base['paired_image_success'])
    adapt_errs = str(len(summary_adapt['errors']))
    base_errs = str(len(summary_base['errors']))

    print(f"{'Successful Questions / 40':<30} | {adapt_succ:<24} | {base_succ:<24}")
    print(f"{'Failed Questions / 40':<30} | {adapt_fail:<24} | {base_fail:<24}")
    print(f"{'Average Latency':<30} | {adapt_avg_lat:<24} | {base_avg_lat:<24}")
    print(f"{'Median Latency':<30} | {adapt_med_lat:<24} | {base_med_lat:<24}")
    print(f"{'Peak VRAM':<30} | {adapt_vram:<24} | {base_vram:<24}")
    print(f"{'Categories Covered':<30} | {adapt_cats:<24} | {base_cats:<24}")
    print(f"{'Paired Imagery Executed':<30} | {adapt_paired:<24} | {base_paired:<24}")
    print(f"{'Loading / Processor Errors':<30} | {adapt_errs:<24} | {base_errs:<24}")
    print("=" * 80)
    print("NOTE: All results marked 'manual_review_required'. No subjective quality claimed.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "backend/data"
    q_path = sys.argv[2] if len(sys.argv) > 2 else "scratch/questions_40.json"
    out_file = sys.argv[3] if len(sys.argv) > 3 else "satquery_rsvlm_colab_results.json"
    run_bakeoff(data_dir, q_path, out_file)
